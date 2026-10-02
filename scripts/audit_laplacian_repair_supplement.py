"""Supplementary audits for the Laplacian repair prototype.

Adds checks that were intentionally not part of training:
1) synthetic grid-scale convergence for a smooth manufactured field;
2) true-vs-temperature-hold residual ordering with old and GMLS operators;
3) hotspot gradient/perturbation direction under the same endpoint heat source.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.audit_laplacian_repair import (  # noqa: E402
    build_gmls_stencil,
    choose_windows,
    old_laplacian_np,
    apply_gmls,
    summarize_error,
    write_summary_md,
)
from scripts.training_diagnostics import load_graph, write_csv  # noqa: E402
from src.config import Config  # noqa: E402
from src.physics.differentiation import spatial_laplacian  # noqa: E402
from src.physics.heat_source import gaussian_heat_source  # noqa: E402


def finite_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().cpu().reshape(-1)[0].item()) if x.numel() else float("nan")
    return float(x)


def tensor_rms(x: torch.Tensor) -> float:
    return finite_float(torch.sqrt((x.detach().float().flatten() ** 2).mean())) if x.numel() else float("nan")


def make_grid_edges(n: int, *, connectivity: int = 26) -> np.ndarray:
    idx = np.arange(n**3, dtype=np.int64).reshape(n, n, n)
    offsets = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if di == dj == dk == 0:
                    continue
                man = abs(di) + abs(dj) + abs(dk)
                if connectivity == 6 and man != 1:
                    continue
                offsets.append((di, dj, dk))
    rows = []
    cols = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                u = idx[i, j, k]
                for di, dj, dk in offsets:
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < n and 0 <= jj < n and 0 <= kk < n:
                        rows.append(u)
                        cols.append(idx[ii, jj, kk])
    return np.vstack([np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)])


def synthetic_grid_scale_tests(outdir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Keep domain fixed and refine h.  Interior mask excludes two grid layers so
    # GMLS neighborhoods are complete enough for a fair trend check.
    for n in [9, 13, 17, 21]:
        xs = np.linspace(0.0, 1.0, n)
        X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
        coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)
        edge = make_grid_edges(n, connectivity=26)
        stencil = build_gmls_stencil(coords, edge, min_neighbors=20, max_hops=1)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        a = 2.1
        b = 1.7
        values = np.sin(a * x) + np.cos(b * y) + z * z
        truth = -a * a * np.sin(a * x) - b * b * np.cos(b * y) + 2.0
        old = old_laplacian_np(values, coords, edge)
        new = apply_gmls(stencil, values, coords.shape[0])
        grid_i = np.repeat(np.arange(n), n * n)
        grid_j = np.tile(np.repeat(np.arange(n), n), n)
        grid_k = np.tile(np.arange(n), n * n)
        masks = {
            "all_valid": stencil.valid,
            "interior_margin2": stencil.valid & (grid_i >= 2) & (grid_i <= n - 3) & (grid_j >= 2) & (grid_j <= n - 3) & (grid_k >= 2) & (grid_k <= n - 3),
        }
        for opname, pred in [("old_graph_laplacian", old), ("gmls_quadratic_wls", new)]:
            for group, mask in masks.items():
                stats = summarize_error(pred, truth, mask)
                rows.append({
                    "n_per_axis": n,
                    "h": 1.0 / (n - 1),
                    "operator": opname,
                    "field": "sin(2.1x)+cos(1.7y)+z2",
                    "group": group,
                    "gmls_valid_nodes": int(stencil.valid.sum()),
                    **stats,
                })
    write_csv(outdir / "synthetic_grid_scale_tests.csv", rows)
    return rows


def gmls_laplacian_torch(f: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, weights: torch.Tensor, n: int) -> torch.Tensor:
    fv = f.reshape(-1)
    contrib = weights * (fv[cols] - fv[rows])
    out = torch.zeros(n, dtype=fv.dtype, device=fv.device)
    out.index_add_(0, rows, contrib)
    return out


def operator_laplacian_torch(op: str, T: torch.Tensor, coords_m: torch.Tensor, edge_index: torch.Tensor, gmls_tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    if op == "old_graph_laplacian":
        return spatial_laplacian(T, coords_m, edge_index)
    return gmls_laplacian_torch(T, gmls_tensors["rows"], gmls_tensors["cols"], gmls_tensors["weights"], coords_m.shape[0])


def residual_and_gradient_audit(graph: Any, cfg: Config, outdir: Path, windows: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coord_scale = float(cfg.physics.coordinate_scale_to_m)
    coords_m = graph.coords.detach().float() * coord_scale
    edge_index = graph.edge_index.detach().long()
    stencil = build_gmls_stencil(coords_m.cpu().numpy().astype(np.float64), edge_index.cpu().numpy().astype(np.int64), min_neighbors=20, max_hops=2)
    gvalid = torch.from_numpy(stencil.valid).bool()
    gmls_tensors = {
        "rows": torch.from_numpy(stencil.rows).long(),
        "cols": torch.from_numpy(stencil.cols).long(),
        "weights": torch.from_numpy(stencil.weights).float(),
    }

    rho = float(cfg.material.density)
    Cp = float(cfg.material.specific_heat)
    k = float(cfg.material.thermal_conductivity)
    solidus = float(cfg.material.solidus_temp)
    time_scale = float(cfg.physics.time_scale_to_s)
    residual_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []

    for start in windows:
        target = int(start) + int(cfg.data.window_size) + int(cfg.data.predict_steps) - 1
        prev = target - 1
        dt = (finite_float(graph.times[target]) - finite_float(graph.times[prev])) * time_scale
        if dt <= 0:
            continue
        T_true = graph.temperatures[target].detach().float().cpu()
        T_prev = graph.temperatures[prev].detach().float().cpu()
        valid = graph.get_active_mask(target).detach().bool().cpu() & gvalid
        hot = valid & (T_true >= solidus)
        laser_mm = graph._laser_positions[target].detach().float().cpu()
        q = gaussian_heat_source(coords_m.cpu(), laser_mm * coord_scale, power=900.0, efficiency=0.7, radius=2.0e-3, depth=1.0e-3).detach().float().cpu()
        storage_true = rho * Cp * (T_true - T_prev) / dt
        zero_storage = torch.zeros_like(storage_true)
        for op in ["old_graph_laplacian", "gmls_quadratic_wls"]:
            lap_true = operator_laplacian_torch(op, T_true, coords_m.cpu(), edge_index.cpu(), gmls_tensors).detach()
            lap_hold = operator_laplacian_torch(op, T_prev, coords_m.cpu(), edge_index.cpu(), gmls_tensors).detach()
            res_true = storage_true - k * lap_true - q
            res_hold = zero_storage - k * lap_hold - q
            for group_name, mask in {"valid": valid, "hot": hot, "background": valid & ~hot}.items():
                if int(mask.sum()) == 0:
                    residual_rows.append({"window_start": start, "target_step": target, "operator": op, "group": group_name, "count": 0})
                    continue
                true_rms = tensor_rms(res_true[mask])
                hold_rms = tensor_rms(res_hold[mask])
                residual_rows.append({
                    "window_start": start,
                    "target_step": target,
                    "operator": op,
                    "group": group_name,
                    "count": int(mask.sum()),
                    "hot_nodes": int(hot.sum()),
                    "true_residual_rms": true_rms,
                    "hold_residual_rms": hold_rms,
                    "hold_over_true_rms": hold_rms / true_rms if true_rms else float("nan"),
                    "hold_smaller_than_true": bool(hold_rms < true_rms),
                })

            if int(hot.sum()) > 0:
                # Autograd at the true field under valid-node PDE loss.
                T_var = T_true.clone().requires_grad_(True)
                lap = operator_laplacian_torch(op, T_var, coords_m.cpu(), edge_index.cpu(), gmls_tensors)
                storage = rho * Cp * (T_var - T_prev) / dt
                res = storage - k * lap - q
                loss = (res[valid] ** 2).mean()
                loss.backward()
                grad_hot = T_var.grad.detach()[hot]
                base_loss = finite_float(loss)
                row = {
                    "window_start": start,
                    "target_step": target,
                    "operator": op,
                    "hot_nodes": int(hot.sum()),
                    "base_loss": base_loss,
                    "hot_grad_mean": finite_float(grad_hot.mean()),
                    "hot_grad_positive_frac": finite_float((grad_hot > 0).float().mean()),
                    "hot_grad_negative_frac": finite_float((grad_hot < 0).float().mean()),
                    "hot_grad_rms": tensor_rms(grad_hot),
                }
                for delta in [-10.0, -50.0, 10.0, 50.0]:
                    Tp = T_true.clone()
                    Tp[hot] += float(delta)
                    lapp = operator_laplacian_torch(op, Tp, coords_m.cpu(), edge_index.cpu(), gmls_tensors).detach()
                    storagep = rho * Cp * (Tp - T_prev) / dt
                    resp = storagep - k * lapp - q
                    new_loss = finite_float((resp[valid] ** 2).mean())
                    key = f"perturb_{int(delta):+d}C"
                    row[f"{key}_loss"] = new_loss
                    row[f"{key}_loss_change_frac"] = (new_loss - base_loss) / base_loss if base_loss else float("nan")
                grad_rows.append(row)
    write_csv(outdir / "true_vs_hold_residual_old_vs_gmls.csv", residual_rows)
    write_csv(outdir / "hotspot_gradient_perturbation_old_vs_gmls.csv", grad_rows)
    return residual_rows, grad_rows


def write_supplement_summary(outdir: Path, grid_rows: list[dict[str, Any]], residual_rows: list[dict[str, Any]], grad_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Laplacian 修复补充审计：尺度趋势、温度保持与热点梯度\n",
        "本补充仍不训练模型；只在旧算子和 GMLS 原型之间做机制对照。\n",
        "## 1. Synthetic grid-scale manufactured-solution trend\n",
        "| operator | group | n | h | RMSE | MAE | count |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in grid_rows:
        if r.get("group") == "interior_margin2":
            lines.append(f"| {r['operator']} | {r['group']} | {r['n_per_axis']} | {float(r['h']):.5f} | {float(r['err_rmse']):.6g} | {float(r['err_mae']):.6g} | {r['count']} |")
    lines += [
        "\n## 2. 真实升温场 vs 温度保持预测的残差排序\n",
        "| operator | group | windows | hold smaller count | mean hold/true RMS |",
        "|---|---|---:|---:|---:|",
    ]
    for op in ["old_graph_laplacian", "gmls_quadratic_wls"]:
        for group in ["valid", "hot", "background"]:
            rows = [r for r in residual_rows if r.get("operator") == op and r.get("group") == group and int(r.get("count", 0)) > 0]
            if not rows:
                continue
            ratios = np.array([float(r["hold_over_true_rms"]) for r in rows], dtype=float)
            hold_smaller = sum(str(r.get("hold_smaller_than_true")).lower() == "true" or r.get("hold_smaller_than_true") is True for r in rows)
            lines.append(f"| {op} | {group} | {len(rows)} | {hold_smaller} | {np.nanmean(ratios):.6g} |")
    lines += [
        "\n## 3. 热点处 PDE loss 梯度方向与扰动\n",
        "正梯度表示按梯度下降会降低该热点温度。\n",
        "| operator | windows | mean positive grad frac | mean -10C loss change | mean -50C loss change | mean +10C loss change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for op in ["old_graph_laplacian", "gmls_quadratic_wls"]:
        rows = [r for r in grad_rows if r.get("operator") == op]
        if not rows:
            continue
        pos = np.array([float(r["hot_grad_positive_frac"]) for r in rows])
        m10 = np.array([float(r["perturb_-10C_loss_change_frac"]) for r in rows])
        m50 = np.array([float(r["perturb_-50C_loss_change_frac"]) for r in rows])
        p10 = np.array([float(r["perturb_+10C_loss_change_frac"]) for r in rows])
        lines.append(f"| {op} | {len(rows)} | {np.nanmean(pos):.6g} | {np.nanmean(m10):.6g} | {np.nanmean(m50):.6g} | {np.nanmean(p10):.6g} |")
    lines += [
        "\n## 结论边界\n",
        "- Synthetic grid trend 只验证点值 GMLS 原型的数值一致性，不替代 FEM 弱形式。\n",
        "- 温度保持残差排序和热点梯度只说明当前近似 PDE loss 的局部偏好；真实场不要求梯度为零。\n",
        "- 若 GMLS 后仍存在保持温度残差更小或热点降温倾向，应继续核查 heat source、边界、相变/潜热、温变材料和时间积分。\n",
    ]
    (outdir / "laplacian_repair_supplement_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vtu-dir", default=r"F:\VTU")
    parser.add_argument("--cache-dir", default="data/processed")
    parser.add_argument("--config", default="configs/feature_e0_baseline.yaml")
    parser.add_argument("--output-dir", default="results/training_diagnostics/physics_protocol_audit/operator_repair")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--windows", nargs="*", type=int)
    args = parser.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    grid_rows = synthetic_grid_scale_tests(outdir)
    cfg = Config.from_yaml(args.config)
    graph = load_graph(args.vtu_dir, args.cache_dir, cfg, torch.device(args.device), no_cache=False)
    graph.to(torch.device("cpu"))
    windows = choose_windows(graph, cfg, args.windows, args.max_windows)
    residual_rows, grad_rows = residual_and_gradient_audit(graph, cfg, outdir, windows)
    manifest = {
        "script": "scripts/audit_laplacian_repair_supplement.py",
        "windows": windows,
        "outputs": [
            "synthetic_grid_scale_tests.csv",
            "true_vs_hold_residual_old_vs_gmls.csv",
            "hotspot_gradient_perturbation_old_vs_gmls.csv",
            "laplacian_repair_supplement_summary.md",
        ],
    }
    (outdir / "supplement_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_supplement_summary(outdir, grid_rows, residual_rows, grad_rows)
    print(outdir / "laplacian_repair_supplement_summary.md")


if __name__ == "__main__":
    main()
