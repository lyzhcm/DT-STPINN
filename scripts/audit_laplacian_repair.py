"""Repair-audit prototype for the graph heat-conduction Laplacian.

This script does not train.  It audits whether raw VTU files contain FEM mesh
information, builds a local quadratic weighted least-squares (GMLS-style)
pointwise Laplacian prototype from the existing graph neighborhoods, compares it
against the current graph Laplacian on analytic/manufactured fields, and (if the
prototype passes the basic polynomial tests) compares old/new operators in the
same heat-source residual calculation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.training_diagnostics import load_graph, write_csv  # noqa: E402
from scripts.training_diagnostics_multwindow import DEFAULT_TRAIN_BLOCKS, expand_blocks  # noqa: E402
from src.config import Config  # noqa: E402
from src.physics.differentiation import spatial_laplacian  # noqa: E402
from src.physics.heat_source import gaussian_heat_source  # noqa: E402

try:  # mesh audit only; do not make the rest of the script depend on meshio.
    import meshio  # type: ignore
except Exception:  # pragma: no cover
    meshio = None


@dataclass
class GMLSStencil:
    rows: np.ndarray
    cols: np.ndarray
    weights: np.ndarray
    valid: np.ndarray
    neighbor_count: np.ndarray
    rank: np.ndarray
    cond: np.ndarray
    radius: np.ndarray
    hops_used: np.ndarray


def finite_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float("nan")
        return float(x.detach().cpu().float().reshape(-1)[0].item())
    return float(x)


def rms_np(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.nanmean(x * x)))


def mae_np(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.nanmean(np.abs(x)))


def percentile_np(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def summarize_error(values: np.ndarray, truth: np.ndarray | float, mask: np.ndarray) -> dict[str, float]:
    idx = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if np.ndim(truth) == 0:
        err = values[idx] - float(truth)
        tv = np.full(err.shape, float(truth), dtype=np.float64)
    else:
        t = np.asarray(truth, dtype=np.float64)
        idx = idx & np.isfinite(t)
        err = values[idx] - t[idx]
        tv = t[idx]
    return {
        "count": int(idx.sum()),
        "pred_mean": float(np.nanmean(values[idx])) if idx.sum() else float("nan"),
        "truth_mean": float(np.nanmean(tv)) if idx.sum() else float("nan"),
        "err_mean": float(np.nanmean(err)) if idx.sum() else float("nan"),
        "err_mae": mae_np(err),
        "err_rmse": rms_np(err),
        "err_p95_abs": percentile_np(np.abs(err), 95),
        "err_max_abs": float(np.nanmax(np.abs(err))) if idx.sum() else float("nan"),
    }


def build_adjacency(edge_index: np.ndarray, n: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    row = edge_index[0].astype(np.int64, copy=False)
    col = edge_index[1].astype(np.int64, copy=False)
    for r, c in zip(row, col):
        if 0 <= r < n and 0 <= c < n and r != c:
            adj[int(r)].append(int(c))
    for i in range(n):
        if len(adj[i]) > 1:
            adj[i] = sorted(set(adj[i]))
    return adj


def expand_neighbors(adj: list[list[int]], center: int, max_hops: int, min_neighbors: int) -> tuple[list[int], int]:
    visited = {center}
    frontier = {center}
    neigh: set[int] = set()
    used = 0
    for hop in range(1, max_hops + 1):
        next_frontier: set[int] = set()
        for u in frontier:
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    next_frontier.add(v)
                    neigh.add(v)
        frontier = next_frontier
        used = hop
        if len(neigh) >= min_neighbors:
            break
        if not frontier:
            break
    neigh.discard(center)
    return sorted(neigh), used


def design_matrix(s: np.ndarray) -> np.ndarray:
    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    return np.column_stack(
        [
            x,
            y,
            z,
            0.5 * x * x,
            0.5 * y * y,
            0.5 * z * z,
            x * y,
            x * z,
            y * z,
        ]
    )


def build_gmls_stencil(
    coords: np.ndarray,
    edge_index: np.ndarray,
    *,
    min_neighbors: int = 20,
    max_hops: int = 2,
    rank_tol: float = 1.0e-10,
    cond_max: float = 1.0e8,
    weight_power: float = 2.0,
) -> GMLSStencil:
    """Build pointwise Laplacian stencils from local quadratic WLS fits.

    For each node i, fit f_j - f_i on scaled offsets s=(x_j-x_i)/h using the
    quadratic Taylor basis [s, 0.5*s^2, cross terms].  The Laplacian is the trace
    of the fitted Hessian divided by h^2.  Invalid/rank-deficient neighborhoods
    are marked rather than silently trusted.
    """
    n = coords.shape[0]
    adj = build_adjacency(edge_index, n)
    out_rows: list[int] = []
    out_cols: list[int] = []
    out_weights: list[float] = []
    valid = np.zeros(n, dtype=bool)
    counts = np.zeros(n, dtype=np.int32)
    ranks = np.zeros(n, dtype=np.int16)
    conds = np.full(n, np.inf, dtype=np.float64)
    radii = np.zeros(n, dtype=np.float64)
    hops_used = np.zeros(n, dtype=np.int16)
    lap_selector = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float64)

    for i in range(n):
        neigh, used = expand_neighbors(adj, i, max_hops=max_hops, min_neighbors=min_neighbors)
        hops_used[i] = used
        counts[i] = len(neigh)
        if len(neigh) < 9:
            continue
        r = coords[np.asarray(neigh, dtype=np.int64)] - coords[i]
        dist = np.linalg.norm(r, axis=1)
        h = float(np.max(dist)) if dist.size else 0.0
        radii[i] = h
        if not np.isfinite(h) or h <= 0.0:
            continue
        s = r / h
        A = design_matrix(s)
        # Smooth compact-like positive weights; exact polynomial reproduction is
        # retained when the weighted design matrix has full column rank.
        w = np.exp(-np.power(dist / h, weight_power))
        sw = np.sqrt(w)
        Aw = A * sw[:, None]
        try:
            sv = np.linalg.svd(Aw, compute_uv=False)
        except np.linalg.LinAlgError:
            continue
        if sv.size == 0 or sv[0] <= 0.0:
            continue
        rank = int(np.sum(sv > sv[0] * rank_tol))
        ranks[i] = rank
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0.0 else float("inf")
        conds[i] = cond
        if rank < 9 or (not np.isfinite(cond)) or cond > cond_max:
            continue
        # beta = pinv(Aw) @ (sqrt(w) * (f_neigh - f_i)); lap = selector beta / h^2.
        pinv = np.linalg.pinv(Aw, rcond=rank_tol)
        coeff = (lap_selector @ pinv) * sw / (h * h)
        if not np.all(np.isfinite(coeff)):
            continue
        valid[i] = True
        out_rows.extend([i] * len(neigh))
        out_cols.extend(neigh)
        out_weights.extend(coeff.astype(np.float64).tolist())

    return GMLSStencil(
        rows=np.asarray(out_rows, dtype=np.int64),
        cols=np.asarray(out_cols, dtype=np.int64),
        weights=np.asarray(out_weights, dtype=np.float64),
        valid=valid,
        neighbor_count=counts,
        rank=ranks,
        cond=conds,
        radius=radii,
        hops_used=hops_used,
    )


def apply_gmls(stencil: GMLSStencil, f: np.ndarray, n: int) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    out = np.full(n, np.nan, dtype=np.float64)
    if stencil.rows.size:
        contrib = stencil.weights * (f[stencil.cols] - f[stencil.rows])
        sums = np.bincount(stencil.rows, weights=contrib, minlength=n)
        out[stencil.valid] = sums[stencil.valid]
    return out


def old_laplacian_np(f: np.ndarray, coords: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        ft = torch.from_numpy(np.asarray(f, dtype=np.float32))
        ct = torch.from_numpy(np.asarray(coords, dtype=np.float32))
        et = torch.from_numpy(np.asarray(edge_index, dtype=np.int64))
        out = spatial_laplacian(ft, ct, et)
    return out.detach().cpu().numpy().astype(np.float64)


def rotation_matrix_z(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def make_fields(coords: np.ndarray) -> list[dict[str, Any]]:
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    a = 20.0
    b = 15.0
    fields: list[dict[str, Any]] = [
        {"field": "constant", "values": np.ones(coords.shape[0]) * 123.4, "truth": 0.0, "kind": "poly0"},
        {"field": "x", "values": x, "truth": 0.0, "kind": "linear"},
        {"field": "y", "values": y, "truth": 0.0, "kind": "linear"},
        {"field": "z", "values": z, "truth": 0.0, "kind": "linear"},
        {"field": "x2", "values": x * x, "truth": 2.0, "kind": "quadratic"},
        {"field": "y2", "values": y * y, "truth": 2.0, "kind": "quadratic"},
        {"field": "z2", "values": z * z, "truth": 2.0, "kind": "quadratic"},
        {"field": "xy", "values": x * y, "truth": 0.0, "kind": "quadratic_cross"},
        {"field": "xz", "values": x * z, "truth": 0.0, "kind": "quadratic_cross"},
        {"field": "yz", "values": y * z, "truth": 0.0, "kind": "quadratic_cross"},
        {"field": "x2_y2_z2", "values": x * x + y * y + z * z, "truth": 6.0, "kind": "quadratic"},
        {
            "field": "manufactured_sin_cos_z2",
            "values": np.sin(a * x) + np.cos(b * y) + z * z,
            "truth": -a * a * np.sin(a * x) - b * b * np.cos(b * y) + 2.0,
            "kind": "manufactured",
        },
    ]
    return fields


def mesh_audit(vtu_dir: Path, outdir: Path, graph_cache_path: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = sorted(vtu_dir.glob("*.vtu"))
    if not files:
        rows.append({"item": "vtu_files", "available": False, "evidence": f"No *.vtu under {vtu_dir}", "impact": "Cannot audit raw FEM mesh."})
        return rows
    if meshio is None:
        rows.append({"item": "meshio", "available": False, "evidence": "meshio import failed", "impact": "Cannot inspect raw cell connectivity in this environment."})
        return rows

    # Inspect first, one heated middle-ish file, and final file for stable structure / evolving Live data.
    sample_indices = sorted(set([0, min(21, len(files) - 1), len(files) - 1]))
    first_mesh = None
    for si in sample_indices:
        fp = files[si]
        mesh = meshio.read(str(fp))
        if first_mesh is None:
            first_mesh = mesh
        rows.append({"item": "sample_file", "available": True, "evidence": f"{fp.name}, size={fp.stat().st_size}", "impact": "VTU file readable by meshio."})
        rows.append({"item": f"points_{fp.name}", "available": True, "evidence": f"shape={mesh.points.shape}, dtype={mesh.points.dtype}, min={np.nanmin(mesh.points, axis=0).tolist()}, max={np.nanmax(mesh.points, axis=0).tolist()}", "impact": "Nodal point coordinates available."})
        cell_desc = []
        for cb in mesh.cells:
            conn = np.asarray(cb.data)
            cell_desc.append(f"{cb.type}:{tuple(conn.shape)}")
        rows.append({"item": f"cells_{fp.name}", "available": bool(cell_desc), "evidence": "; ".join(cell_desc), "impact": "Raw element connectivity and element type are present; FEM assembly is possible in principle."})
        rows.append({"item": f"point_data_{fp.name}", "available": bool(mesh.point_data), "evidence": ", ".join(mesh.point_data.keys()), "impact": "Nodal Temperature/Live/Boundary fields are available."})
        rows.append({"item": f"cell_data_{fp.name}", "available": bool(mesh.cell_data), "evidence": ", ".join(mesh.cell_data.keys()), "impact": "Cell-wise Live/Style/Cpu_id and other arrays exist; no explicit material property table found."})
        for key in ["Live", "Style", "Cpu_id"]:
            if key in mesh.cell_data:
                parts = []
                for bi, arr in enumerate(mesh.cell_data[key]):
                    a = np.asarray(arr).reshape(-1)
                    vals, counts = np.unique(a, return_counts=True)
                    if len(vals) <= 20:
                        parts.append(f"block{bi}:" + ",".join(f"{float(v):g}={int(c)}" for v, c in zip(vals, counts)))
                    else:
                        parts.append(f"block{bi}:min={float(np.nanmin(a)):g},max={float(np.nanmax(a)):g},n={a.size}")
                rows.append({"item": f"cell_data_{key}_{fp.name}", "available": True, "evidence": "; ".join(parts), "impact": "Useful for activation/regions, but semantic mapping still needs FEM protocol confirmation."})

    # Loader/cache retention audit.
    rows.append({"item": "VTULoader.cells", "available": True, "evidence": "src/data/vtu_loader.py stores cells in VTUData.cells", "impact": "Raw parse keeps connectivity."})
    rows.append({"item": "DynamicGraph.cache.cells", "available": False, "evidence": "DynamicGraph.state_dict/cache keys contain edge_index but no cells/cell_data", "impact": "Current cached training graph cannot assemble FEM weak form without re-reading VTU."})
    rows.append({"item": "current_graph_edges", "available": True, "evidence": "cache metadata use_mesh_edges=True; edge_index is extracted from mesh cells", "impact": "Existing graph neighborhoods are mesh-derived, so GMLS prototype can use them as local neighborhoods."})
    return rows


def geometry_masks(coords_mm: np.ndarray, stencil: GMLSStencil) -> dict[str, np.ndarray]:
    deg_nonzero = stencil.neighbor_count > 0
    mn = coords_mm.min(axis=0)
    mx = coords_mm.max(axis=0)
    tol = 1.0e-6
    geom_boundary = deg_nonzero & np.any((np.abs(coords_mm - mn) <= tol) | (np.abs(coords_mm - mx) <= tol), axis=1)
    interior = deg_nonzero & ~geom_boundary
    pathological = deg_nonzero & (~stencil.valid | (stencil.cond > 1.0e5))
    return {
        "all_nonzero_degree": deg_nonzero,
        "gmls_valid": stencil.valid,
        "geometric_interior": interior,
        "geometric_boundary": geom_boundary,
        "pathological_or_invalid": pathological,
    }


def run_operator_tests(coords_m: np.ndarray, coords_mm: np.ndarray, edge_index: np.ndarray, stencil: GMLSStencil) -> list[dict[str, Any]]:
    masks = geometry_masks(coords_mm, stencil)
    rows: list[dict[str, Any]] = []

    cases: list[tuple[str, np.ndarray, GMLSStencil | None]] = [("base_m", coords_m, stencil)]

    offset = np.array([0.37, -0.21, 0.08], dtype=np.float64)  # meters; intentionally large but harmless for centered stencils.
    cases.append(("translated_m", coords_m + offset, build_gmls_stencil(coords_m + offset, edge_index)))

    R = rotation_matrix_z(math.radians(37.0))
    coords_rot = coords_m @ R.T
    cases.append(("rotated_m", coords_rot, build_gmls_stencil(coords_rot, edge_index)))

    for case, c, st in cases:
        assert st is not None
        for fdef in make_fields(c):
            values = np.asarray(fdef["values"], dtype=np.float64)
            truth = fdef["truth"]
            old = old_laplacian_np(values, c, edge_index)
            new = apply_gmls(st, values, c.shape[0])
            for op_name, pred in [("old_graph_laplacian", old), ("gmls_quadratic_wls", new)]:
                for group, mask in masks.items() if case == "base_m" else {"gmls_valid": st.valid}.items():
                    stats = summarize_error(pred, truth, mask)
                    rows.append({
                        "case": case,
                        "operator": op_name,
                        "field": fdef["field"],
                        "kind": fdef["kind"],
                        "group": group,
                        **stats,
                    })

    # Unit consistency: compute same physical field f=x_m^2+y_m^2+z_m^2 using mm coordinates.
    st_mm = build_gmls_stencil(coords_mm, edge_index)
    x_m_from_mm = coords_mm[:, 0] * 1.0e-3
    y_m_from_mm = coords_mm[:, 1] * 1.0e-3
    z_m_from_mm = coords_mm[:, 2] * 1.0e-3
    values = x_m_from_mm**2 + y_m_from_mm**2 + z_m_from_mm**2
    # Operators with coords in mm return derivative per mm^2; convert to per m^2 by multiplying 1e6.
    old_mm = old_laplacian_np(values, coords_mm, edge_index) * 1.0e6
    new_mm = apply_gmls(st_mm, values, coords_mm.shape[0]) * 1.0e6
    for op_name, pred in [("old_graph_laplacian", old_mm), ("gmls_quadratic_wls", new_mm)]:
        stats = summarize_error(pred, 6.0, st_mm.valid if "gmls" in op_name else (st_mm.neighbor_count > 0))
        rows.append({
            "case": "unit_mm_coords_physical_x2y2z2_converted_to_per_m2",
            "operator": op_name,
            "field": "x_m2_y_m2_z_m2",
            "kind": "unit_consistency",
            "group": "valid_or_nonzero_degree",
            **stats,
        })
    return rows


def tensor_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().flatten()
    if x.numel() == 0:
        return {f"{prefix}_{k}": float("nan") for k in ["mean", "mean_abs", "rms", "p95_abs", "max_abs"]}
    ax = x.abs()
    return {
        f"{prefix}_mean": finite_float(x.mean()),
        f"{prefix}_mean_abs": finite_float(ax.mean()),
        f"{prefix}_rms": finite_float(torch.sqrt((x * x).mean())),
        f"{prefix}_p95_abs": finite_float(torch.quantile(ax, 0.95)),
        f"{prefix}_max_abs": finite_float(ax.max()),
    }


def choose_windows(graph: Any, cfg: Config, requested: list[int] | None, max_windows: int) -> list[int]:
    if requested:
        return [int(x) for x in requested[:max_windows]]
    candidates = [118, 120, 160, 208, 210, 421, 424, 634, 637, 847, 850, 1061, 1064, 1274, 1277]
    for s in expand_blocks(DEFAULT_TRAIN_BLOCKS):
        if s not in candidates:
            candidates.append(s)
    out = []
    for s in candidates:
        target = int(s) + int(cfg.data.window_size) + int(cfg.data.predict_steps) - 1
        if 0 < target < len(graph.times):
            out.append(int(s))
        if len(out) >= max_windows:
            break
    return out


def compare_true_residual(
    graph: Any,
    cfg: Config,
    stencil: GMLSStencil,
    windows: list[int],
    *,
    pde_power_W: float,
    pde_efficiency: float,
    pde_radius_m: float,
    pde_depth_m: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rho = float(cfg.material.density)
    Cp = float(cfg.material.specific_heat)
    k = float(cfg.material.thermal_conductivity)
    solidus = float(cfg.material.solidus_temp)
    coord_scale = float(cfg.physics.coordinate_scale_to_m)
    time_scale = float(cfg.physics.time_scale_to_s)
    coords_m_t = graph.coords.detach().float() * coord_scale
    coords_m_np = coords_m_t.detach().cpu().numpy().astype(np.float64)
    edge_index = graph.edge_index.detach().cpu().numpy().astype(np.int64)

    for start in windows:
        target = int(start) + int(cfg.data.window_size) + int(cfg.data.predict_steps) - 1
        prev = target - 1
        raw_t = finite_float(graph.times[target])
        raw_p = finite_float(graph.times[prev])
        dt = (raw_t - raw_p) * time_scale
        if dt <= 0:
            continue
        T_t = graph.temperatures[target].detach().float()
        T_prev = graph.temperatures[prev].detach().float()
        valid = graph.get_active_mask(target).detach().bool()
        hot = valid & (T_t >= solidus)
        background = valid & ~hot
        laser_mm = graph._laser_positions[target].detach().float()
        q = gaussian_heat_source(
            coords_m_t,
            laser_mm * coord_scale,
            power=pde_power_W,
            efficiency=pde_efficiency,
            radius=pde_radius_m,
            depth=pde_depth_m,
        )
        storage = rho * Cp * (T_t - T_prev) / dt
        old_lap_t = spatial_laplacian(T_t, coords_m_t, graph.edge_index).detach().float()
        new_lap_np = apply_gmls(stencil, T_t.detach().cpu().numpy().astype(np.float64), coords_m_np.shape[0])
        new_lap_t = torch.from_numpy(new_lap_np).to(dtype=torch.float32)
        gvalid = torch.from_numpy(stencil.valid).bool()
        masks = {
            "valid": valid & gvalid,
            "hot": hot & gvalid,
            "background": background & gvalid,
        }
        old_cond = k * old_lap_t
        new_cond = k * new_lap_t
        old_res = storage - old_cond - q.cpu()
        new_res = storage - new_cond - q.cpu()
        for group, mask in masks.items():
            if int(mask.sum()) == 0:
                rows.append({"window_start": start, "target_step": target, "group": group, "count": 0})
                continue
            row: dict[str, Any] = {
                "window_start": start,
                "target_step": target,
                "group": group,
                "count": int(mask.sum()),
                "hot_nodes_total": int(hot.sum()),
                "gmls_valid_nodes": int(gvalid.sum()),
                "target_temp_max": finite_float(T_t[mask].max()),
            }
            for name, arr in [
                ("storage", storage[mask]),
                ("q_endpoint", q.cpu()[mask]),
                ("old_k_lap", old_cond[mask]),
                ("new_k_lap", new_cond[mask]),
                ("old_residual", old_res[mask]),
                ("new_residual", new_res[mask]),
            ]:
                row.update(tensor_stats(name, arr))
            row["new_over_old_residual_rms"] = row["new_residual_rms"] / row["old_residual_rms"] if row.get("old_residual_rms", 0.0) else float("nan")
            row["new_over_old_conduction_rms"] = row["new_k_lap_rms"] / row["old_k_lap_rms"] if row.get("old_k_lap_rms", 0.0) else float("nan")
            rows.append(row)
    return rows


def write_mesh_audit_md(outdir: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# VTU 单元信息与离散路线审计\n",
        "本文件由 `scripts/audit_laplacian_repair.py` 生成；本轮不训练、不调权重。\n",
        "## 原始 VTU / 当前缓存可用信息\n",
        "| 项目 | 是否可用 | 证据 | 对离散路线影响 |",
        "|---|---:|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r.get('item','')} | {'是' if r.get('available') else '否'} | {str(r.get('evidence','')).replace('|','/')} | {str(r.get('impact','')).replace('|','/')} |")
    lines.extend([
        "\n## 路线选择\n",
        "- 原始 VTU 中存在 `tetra` 与 `hexahedron` 单元连接和 cell data，因此**长期更可信路线应是 FEM 弱形式**：构建质量矩阵/导热刚度矩阵/体热源载荷/边界载荷后检查热平衡。\n",
        "- 但当前训练缓存只保留 `edge_index`，没有 cell connectivity/cell_data；同时边界载荷、材料区域语义和 FEM 时间积分仍未核准。因此本脚本先实现**局部二次加权最小二乘（GMLS 类）点值 Laplacian 原型**，用于替换旧点值图 Laplacian 的一致性诊断。\n",
        "- 旧 `spatial_laplacian` 保留为对照；本脚本不把 GMLS 原型直接写入训练损失。\n",
    ])
    (outdir / "mesh_data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(outdir: Path, manifest: dict[str, Any], stencil: GMLSStencil, test_rows: list[dict[str, Any]], residual_rows: list[dict[str, Any]]) -> None:
    def find_row(op: str, field: str, case: str = "base_m", group: str = "gmls_valid") -> dict[str, Any] | None:
        for r in test_rows:
            if r.get("operator") == op and r.get("field") == field and r.get("case") == case and r.get("group") == group:
                return r
        return None

    key_fields = ["x", "y", "z", "x2", "y2", "z2", "xy", "xz", "yz", "x2_y2_z2", "manufactured_sin_cos_z2"]
    lines = [
        "# Laplacian 修复原型审计摘要\n",
        "本轮目标：检查 VTU 单元信息、选择离散路线、完成解析场/制造解测试，并在相同热源下对比新旧算子。没有训练模型。\n",
        "## GMLS stencil 诊断\n",
        f"- 节点数：{manifest['num_nodes']}；边数（有向）：{manifest['num_edges']}。\n",
        f"- GMLS 有效节点：{int(stencil.valid.sum())}/{len(stencil.valid)}；无效节点：{int((~stencil.valid).sum())}。\n",
        f"- 邻域点数：median={float(np.median(stencil.neighbor_count)):.1f}, p95={percentile_np(stencil.neighbor_count,95):.1f}, max={int(stencil.neighbor_count.max())}。\n",
        f"- 条件数（有效节点）：median={percentile_np(stencil.cond[stencil.valid],50):.2f}, p95={percentile_np(stencil.cond[stencil.valid],95):.2f}, max={float(np.nanmax(stencil.cond[stencil.valid])):.2f}。\n",
        "\n## 关键解析场结果（base_m / GMLS valid）\n",
        "| field | old RMSE | old mean | GMLS RMSE | GMLS mean | theory mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for f in key_fields:
        old = find_row("old_graph_laplacian", f)
        new = find_row("gmls_quadratic_wls", f)
        if old and new:
            lines.append(
                f"| {f} | {old.get('err_rmse', float('nan')):.6g} | {old.get('pred_mean', float('nan')):.6g} | "
                f"{new.get('err_rmse', float('nan')):.6g} | {new.get('pred_mean', float('nan')):.6g} | {new.get('truth_mean', float('nan')):.6g} |"
            )
    # Residual summary aggregate by group.
    lines.extend([
        "\n## 相同 heat source 下真实场残差对比（endpoint Q，训练窗口样本）\n",
        "注意：这不是 FEM 协议最终验证，只是隔离“算子变化”的对比；热源/材料/边界/时间积分仍按当前代码默认。\n",
        "| group | windows | old residual RMS mean | new residual RMS mean | new/old mean | old kLap RMS mean | new kLap RMS mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in residual_rows:
        if r.get("count", 0):
            by_group.setdefault(str(r.get("group")), []).append(r)
    for group, rows in by_group.items():
        old_r = np.array([float(r.get("old_residual_rms", np.nan)) for r in rows])
        new_r = np.array([float(r.get("new_residual_rms", np.nan)) for r in rows])
        ratio = np.array([float(r.get("new_over_old_residual_rms", np.nan)) for r in rows])
        old_c = np.array([float(r.get("old_k_lap_rms", np.nan)) for r in rows])
        new_c = np.array([float(r.get("new_k_lap_rms", np.nan)) for r in rows])
        lines.append(f"| {group} | {len(rows)} | {np.nanmean(old_r):.6g} | {np.nanmean(new_r):.6g} | {np.nanmean(ratio):.6g} | {np.nanmean(old_c):.6g} | {np.nanmean(new_c):.6g} |")

    lines.extend([
        "\n## 当前结论边界\n",
        "1. 旧图 Laplacian 在实际网格上不满足线性/二次多项式一致性；继续暂停旧 PDE 分支是必要的。\n",
        "2. GMLS 原型在可解邻域上应显著改善多项式再现；若制造解误差仍偏大，应先看边界/病态邻域和邻域尺度，而不是恢复训练。\n",
        "3. 原始 VTU 单元信息完整，FEM 弱形式仍是最终物理路线候选；但当前训练缓存没有保留 cell data，且边界/材料/时间积分协议未核准。\n",
        "4. 即便新算子改善真实场残差，也不能自动证明 PDE 可恢复训练；还需热源、边界、相变/温变材料和时间积分协议审计。\n",
    ])
    (outdir / "laplacian_repair_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vtu-dir", default=r"F:\VTU")
    parser.add_argument("--cache-dir", default="data/processed")
    parser.add_argument("--config", default="configs/feature_e0_baseline.yaml")
    parser.add_argument("--output-dir", default="results/training_diagnostics/physics_protocol_audit/operator_repair")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--min-neighbors", type=int, default=20)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--windows", nargs="*", type=int)
    parser.add_argument("--pde-power-W", type=float, default=900.0)
    parser.add_argument("--pde-efficiency", type=float, default=0.7)
    parser.add_argument("--pde-radius-m", type=float, default=2.0e-3)
    parser.add_argument("--pde-depth-m", type=float, default=1.0e-3)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    vtu_dir = Path(args.vtu_dir)
    mesh_rows = mesh_audit(vtu_dir, outdir)
    write_csv(outdir / "mesh_data_audit.csv", mesh_rows)
    write_mesh_audit_md(outdir, mesh_rows)

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    graph = load_graph(args.vtu_dir, args.cache_dir, cfg, device, args.no_cache)
    graph.to(torch.device("cpu"))

    coords_mm = graph.coords.detach().cpu().numpy().astype(np.float64)
    coord_scale = float(cfg.physics.coordinate_scale_to_m)
    coords_m = coords_mm * coord_scale
    edge_index = graph.edge_index.detach().cpu().numpy().astype(np.int64)

    t0 = time.time()
    stencil = build_gmls_stencil(
        coords_m,
        edge_index,
        min_neighbors=args.min_neighbors,
        max_hops=args.max_hops,
    )
    stencil_time = time.time() - t0

    stencil_rows = []
    for group_name, mask in geometry_masks(coords_mm, stencil).items():
        idx = np.asarray(mask, dtype=bool)
        stencil_rows.append({
            "group": group_name,
            "count": int(idx.sum()),
            "neighbor_count_min": int(np.min(stencil.neighbor_count[idx])) if idx.any() else 0,
            "neighbor_count_median": float(np.median(stencil.neighbor_count[idx])) if idx.any() else float("nan"),
            "neighbor_count_p95": percentile_np(stencil.neighbor_count[idx], 95) if idx.any() else float("nan"),
            "rank_min": int(np.min(stencil.rank[idx])) if idx.any() else 0,
            "rank_median": float(np.median(stencil.rank[idx])) if idx.any() else float("nan"),
            "cond_median": percentile_np(stencil.cond[idx], 50) if idx.any() else float("nan"),
            "cond_p95": percentile_np(stencil.cond[idx], 95) if idx.any() else float("nan"),
            "cond_max": float(np.nanmax(stencil.cond[idx & np.isfinite(stencil.cond)])) if np.any(idx & np.isfinite(stencil.cond)) else float("nan"),
            "radius_m_median": percentile_np(stencil.radius[idx], 50) if idx.any() else float("nan"),
            "hops_used_median": float(np.median(stencil.hops_used[idx])) if idx.any() else float("nan"),
        })
    write_csv(outdir / "gmls_stencil_diagnostics.csv", stencil_rows)

    test_rows = run_operator_tests(coords_m, coords_mm, edge_index, stencil)
    write_csv(outdir / "laplacian_operator_tests.csv", test_rows)

    windows = choose_windows(graph, cfg, args.windows, args.max_windows)
    residual_rows = compare_true_residual(
        graph,
        cfg,
        stencil,
        windows,
        pde_power_W=args.pde_power_W,
        pde_efficiency=args.pde_efficiency,
        pde_radius_m=args.pde_radius_m,
        pde_depth_m=args.pde_depth_m,
    )
    write_csv(outdir / "true_field_residual_old_vs_gmls.csv", residual_rows)

    manifest = {
        "script": "scripts/audit_laplacian_repair.py",
        "vtu_dir": str(vtu_dir),
        "config": str(args.config),
        "output_dir": str(outdir),
        "num_nodes": int(coords_m.shape[0]),
        "num_edges": int(edge_index.shape[1]),
        "coordinate_scale_to_m": coord_scale,
        "gmls_min_neighbors": int(args.min_neighbors),
        "gmls_max_hops": int(args.max_hops),
        "gmls_build_seconds": stencil_time,
        "gmls_valid_nodes": int(stencil.valid.sum()),
        "windows": windows,
        "pde_power_W": args.pde_power_W,
        "pde_efficiency": args.pde_efficiency,
        "pde_radius_m": args.pde_radius_m,
        "pde_depth_m": args.pde_depth_m,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary_md(outdir, manifest, stencil, test_rows, residual_rows)
    print(outdir / "laplacian_repair_summary.md", flush=True)


if __name__ == "__main__":
    main()
