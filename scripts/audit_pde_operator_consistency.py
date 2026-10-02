"""Audit graph Laplacian consistency and current PDE hotspot gradient direction.

No model training is performed.  The script has two diagnostics:

1) Analytic-field Laplacian tests on the actual graph/mesh.
2) True-field PDE loss perturbation and gradient-direction tests near hotspots.
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
from torch_scatter import scatter_mean

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.training_diagnostics import load_graph, write_csv  # noqa: E402
from scripts.training_diagnostics_multwindow import DEFAULT_TRAIN_BLOCKS, expand_blocks  # noqa: E402
from src.config import Config  # noqa: E402
from src.physics.heat_source import gaussian_heat_source  # noqa: E402


def fval(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float("nan")
        return float(x.detach().cpu().item())
    return float(x)


def stats(prefix: str, x: torch.Tensor, expected: float | None = None) -> dict[str, Any]:
    x = x.detach().float().flatten()
    out: dict[str, Any] = {f"{prefix}_count": int(x.numel())}
    if x.numel() == 0:
        for k in ["mean", "mean_abs", "rms", "median", "p95_abs", "max_abs", "bias"]:
            out[f"{prefix}_{k}"] = float("nan")
        return out
    ax = x.abs()
    out.update({
        f"{prefix}_mean": fval(x.mean()),
        f"{prefix}_mean_abs": fval(ax.mean()),
        f"{prefix}_rms": fval(torch.sqrt((x * x).mean())),
        f"{prefix}_median": fval(torch.quantile(x, 0.5)),
        f"{prefix}_p95_abs": fval(torch.quantile(ax, 0.95)),
        f"{prefix}_max_abs": fval(ax.max()),
    })
    if expected is not None:
        e = x - float(expected)
        ae = e.abs()
        out.update({
            f"{prefix}_bias": fval(e.mean()),
            f"{prefix}_err_mean_abs": fval(ae.mean()),
            f"{prefix}_err_rms": fval(torch.sqrt((e * e).mean())),
            f"{prefix}_err_p95_abs": fval(torch.quantile(ae, 0.95)),
            f"{prefix}_err_max_abs": fval(ae.max()),
        })
    return out


def laplacian_eps(values: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor, eps: float) -> torch.Tensor:
    f = values.reshape(-1)
    row, col = edge_index[0], edge_index[1]
    df = f[col] - f[row]
    d2 = (coords[col] - coords[row]).pow(2).sum(dim=1) + float(eps)
    contrib = df / d2
    lap = scatter_mean(contrib, row, dim=0, dim_size=coords.shape[0])
    return lap * (2 * coords.shape[1])


def masked_laplacian_eps(values: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor, eps: float, active: torch.Tensor) -> torch.Tensor:
    """Variant that only uses active-active edges, for audit comparison only."""
    row0, col0 = edge_index[0], edge_index[1]
    keep = active[row0] & active[col0]
    row, col = row0[keep], col0[keep]
    f = values.reshape(-1)
    df = f[col] - f[row]
    d2 = (coords[col] - coords[row]).pow(2).sum(dim=1) + float(eps)
    contrib = df / d2
    lap = scatter_mean(contrib, row, dim=0, dim_size=coords.shape[0])
    return lap * (2 * coords.shape[1])


def analytic_fields(coords: torch.Tensor) -> list[tuple[str, torch.Tensor, float]]:
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    return [
        ("constant", torch.ones_like(x) * 123.4, 0.0),
        ("x", x, 0.0),
        ("y", y, 0.0),
        ("z", z, 0.0),
        ("x2", x * x, 2.0),
        ("r2", x * x + y * y + z * z, 6.0),
    ]


def group_masks(graph: Any, step: int) -> dict[str, torch.Tensor]:
    valid = graph.get_active_mask(step).detach().bool()
    b = graph.boundary.detach()
    coords = graph.coords.detach().float()
    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values
    geom_boundary_0p2 = ((coords - mins).abs() <= 0.2).any(dim=1) | ((coords - maxs).abs() <= 0.2).any(dim=1)
    geom_boundary_0p5 = ((coords - mins).abs() <= 0.5).any(dim=1) | ((coords - maxs).abs() <= 0.5).any(dim=1)
    return {
        "all_nodes": torch.ones_like(valid, dtype=torch.bool),
        "valid": valid,
        "valid_geom_interior_0p2mm": valid & (~geom_boundary_0p2),
        "valid_geom_boundary_0p2mm": valid & geom_boundary_0p2,
        "valid_geom_interior_0p5mm": valid & (~geom_boundary_0p5),
        "valid_geom_boundary_0p5mm": valid & geom_boundary_0p5,
        "valid_boundary_zero": valid & (b == 0),
        "valid_boundary_nonzero": valid & (b != 0),
        "valid_boundary_positive": valid & (b > 0),
        "valid_boundary_negative": valid & (b < 0),
    }


def run_laplacian_tests(graph: Any, cfg: Config, outdir: Path, windows: list[int], eps_values: list[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    edge_index = graph.edge_index
    row, col = edge_index[0], edge_index[1]
    target_steps = [s + cfg.data.window_size + cfg.data.predict_steps - 1 for s in windows]
    ref_step = target_steps[0]

    for step in target_steps:
        valid = graph.get_active_mask(step).detach().bool()
        total_edges = int(row.numel())
        vv = valid[row] & valid[col]
        vi = valid[row] & ~valid[col]
        iv = ~valid[row] & valid[col]
        edge_rows.append({
            "target_step": step,
            "valid_nodes": int(valid.sum().detach().cpu()),
            "invalid_nodes": int((~valid).sum().detach().cpu()),
            "total_edges": total_edges,
            "valid_to_valid_edges": int(vv.sum().detach().cpu()),
            "valid_to_invalid_edges": int(vi.sum().detach().cpu()),
            "invalid_to_valid_edges": int(iv.sum().detach().cpu()),
            "valid_row_edges": int(valid[row].sum().detach().cpu()),
            "valid_row_to_invalid_fraction": fval(vi.sum().float() / valid[row].sum().float().clamp_min(1)),
        })

    # Analytic tests are static.  Use the first audited target step for active masks.
    masks = group_masks(graph, ref_step)
    units = [
        ("mm", graph.coords.detach().double(), 0.0),
        ("m", graph.coords.detach().double() * float(cfg.physics.coordinate_scale_to_m), 0.0),
    ]
    for unit_name, coords, _ in units:
        for eps in eps_values:
            for edge_mode in ["all_edges", "active_active_edges"]:
                for field, values, expected in analytic_fields(coords):
                    if edge_mode == "all_edges":
                        lap = laplacian_eps(values, coords, edge_index, eps)
                    else:
                        active = masks["valid"]
                        lap = masked_laplacian_eps(values, coords, edge_index, eps, active)
                    for group, mask in masks.items():
                        if edge_mode == "active_active_edges" and group == "all_nodes":
                            continue
                        vals = lap[mask]
                        rowd: dict[str, Any] = {
                            "unit": unit_name,
                            "eps": eps,
                            "edge_mode": edge_mode,
                            "target_step_for_masks": ref_step,
                            "field": field,
                            "theoretical_laplacian": expected,
                            "group": group,
                        }
                        rowd.update(stats("lap", vals, expected=expected))
                        rows.append(rowd)
    write_csv(outdir / "laplacian_analytic_tests.csv", rows)
    write_csv(outdir / "laplacian_edge_activity.csv", edge_rows)
    return rows, edge_rows


def interpolate_laser_pos_from_segments(graph: Any, raw_time: float) -> torch.Tensor:
    st = graph._laser_path_segment_start_times
    en = graph._laser_path_segment_end_times
    starts = graph._laser_path_segment_starts
    ends = graph._laser_path_segment_ends
    t = torch.tensor(float(raw_time), device=st.device, dtype=st.dtype)
    active = torch.nonzero((t >= st) & (t <= en), as_tuple=False).flatten()
    if active.numel() > 0:
        i = int(active[0].detach().cpu().item())
        frac = ((t - st[i]) / (en[i] - st[i]).clamp_min(1.0e-12)).clamp(0.0, 1.0)
        return starts[i] + frac * (ends[i] - starts[i])
    ds, is_ = (st - t).abs().min(dim=0)
    de, ie = (en - t).abs().min(dim=0)
    if fval(ds) < fval(de):
        return starts[int(is_.detach().cpu().item())]
    return ends[int(ie.detach().cpu().item())]


def interval_q(graph: Any, coords_m: torch.Tensor, raw_prev: float, raw_target: float, coord_scale: float, args: argparse.Namespace) -> torch.Tensor:
    q = torch.zeros(coords_m.shape[0], device=coords_m.device, dtype=coords_m.dtype)
    n = max(1, int(args.num_q_samples))
    for i in range(n):
        raw = raw_prev + (i + 0.5) / n * (raw_target - raw_prev)
        pos = interpolate_laser_pos_from_segments(graph, raw).to(coords_m.device) * coord_scale
        q = q + gaussian_heat_source(coords_m, pos, power=args.pde_power_W, efficiency=args.pde_efficiency, radius=args.pde_radius_m, depth=args.pde_depth_m)
    return q / n


def pde_loss_and_terms(T_pred: torch.Tensor, T_prev: torch.Tensor, coords_m: torch.Tensor, edge_index: torch.Tensor, dt_s: float,
                       q: torch.Tensor, mask: torch.Tensor, rho: float, Cp: float, k: float, residual_scale: float, eps: float) -> tuple[torch.Tensor, dict[str, float]]:
    lap = laplacian_eps(T_pred, coords_m, edge_index, eps)
    storage = rho * Cp * (T_pred.reshape(-1) - T_prev.reshape(-1)) / max(dt_s, 1.0e-12)
    residual = storage - k * lap - q
    rn = residual[mask] / residual_scale
    loss = (rn * rn).mean()
    return loss, {
        "loss": fval(loss),
        "residual_rms": fval(torch.sqrt((residual[mask] * residual[mask]).mean())),
        "residual_norm_rms": fval(torch.sqrt((rn * rn).mean())),
    }


def run_hotspot_gradient_tests(graph: Any, cfg: Config, outdir: Path, windows: list[int], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rho = float(cfg.material.density)
    Cp = float(cfg.material.specific_heat)
    k = float(cfg.material.thermal_conductivity)
    solidus = float(cfg.material.solidus_temp)
    coord_scale = float(cfg.physics.coordinate_scale_to_m)
    time_scale = float(cfg.physics.time_scale_to_s)
    residual_scale = rho * Cp * float(cfg.physics.pde_temperature_scale) / float(cfg.physics.pde_time_scale)
    coords_m = graph.coords.detach().float() * coord_scale
    edge_index = graph.edge_index

    for start in windows:
        step = int(start) + cfg.data.window_size + cfg.data.predict_steps - 1
        prev = step - 1
        raw_t = fval(graph.times[step])
        raw_prev = fval(graph.times[prev])
        dt_s = (raw_t - raw_prev) * time_scale
        valid = graph.get_active_mask(step).detach().bool()
        T_true = graph.temperatures[step].detach().float()
        T_prev = graph.temperatures[prev].detach().float()
        hot = valid & (T_true >= solidus)
        if int(hot.sum().detach().cpu()) == 0:
            rows.append({"window_start": start, "target_step": step, "hot_nodes": 0, "note": "no hot nodes"})
            continue
        laser = graph._laser_positions[step].detach().float() * coord_scale
        q_endpoint = gaussian_heat_source(coords_m, laser, power=args.pde_power_W, efficiency=args.pde_efficiency, radius=args.pde_radius_m, depth=args.pde_depth_m)
        q_int = interval_q(graph, coords_m, raw_prev, raw_t, coord_scale, args)
        for q_name, q in [("endpoint", q_endpoint), ("interval_avg", q_int)]:
            T_var = T_true.clone().detach().requires_grad_(True)
            base_loss, base_terms = pde_loss_and_terms(T_var, T_prev, coords_m, edge_index, dt_s, q, valid, rho, Cp, k, residual_scale, args.laplacian_eps)
            base_loss.backward()
            grad = T_var.grad.detach().float()
            gh = grad[hot]
            # Positive grad means gradient descent lowers temperature.
            hot_down_fraction = fval((gh > 0).float().mean())
            hot_up_fraction = fval((gh < 0).float().mean())
            near = valid & (torch.cdist(graph.coords.detach().float()[hot][:, :], graph.coords.detach().float()[:, :]).min(dim=0).values <= args.hot_neighbor_radius_mm)
            # cdist may be expensive but hot node count is tiny in selected windows.
            for delta in args.deltas_C:
                for direction, sign in [("hot_decrease", -1.0), ("hot_increase", 1.0)]:
                    Tp = T_true.clone().detach()
                    Tp[hot] = Tp[hot] + sign * float(delta)
                    loss_p, terms_p = pde_loss_and_terms(Tp, T_prev, coords_m, edge_index, dt_s, q, valid, rho, Cp, k, residual_scale, args.laplacian_eps)
                    row = {
                        "window_start": start,
                        "target_step": step,
                        "q_mode": q_name,
                        "hot_nodes": int(hot.sum().detach().cpu()),
                        "valid_nodes": int(valid.sum().detach().cpu()),
                        "target_hot_mean": fval(T_true[hot].mean()),
                        "target_hot_max": fval(T_true[hot].max()),
                        "base_loss_norm_mse": base_terms["loss"],
                        "base_residual_rms": base_terms["residual_rms"],
                        "grad_hot_mean": fval(gh.mean()),
                        "grad_hot_abs_mean": fval(gh.abs().mean()),
                        "grad_hot_rms": fval(torch.sqrt((gh * gh).mean())),
                        "grad_hot_positive_fraction_push_down": hot_down_fraction,
                        "grad_hot_negative_fraction_push_up": hot_up_fraction,
                        "grad_hot_min": fval(gh.min()),
                        "grad_hot_max": fval(gh.max()),
                        "grad_valid_abs_mean": fval(grad[valid].abs().mean()),
                        "grad_near_hot_abs_mean": fval(grad[near].abs().mean()) if int(near.sum().detach().cpu()) else float("nan"),
                        "delta_C": float(delta),
                        "perturbation": direction,
                        "perturbed_loss_norm_mse": terms_p["loss"],
                        "loss_change": terms_p["loss"] - base_terms["loss"],
                        "loss_change_fraction": (terms_p["loss"] - base_terms["loss"]) / base_terms["loss"] if base_terms["loss"] else float("nan"),
                        "perturbed_residual_rms": terms_p["residual_rms"],
                    }
                    rows.append(row)
    write_csv(outdir / "hotspot_pde_gradient_perturbation.csv", rows)
    return rows


def write_summary(outdir: Path, lap_rows: list[dict[str, Any]], edge_rows: list[dict[str, Any]], grad_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    def select_lap(unit: str, eps: float, edge_mode: str, group: str, field: str) -> dict[str, Any] | None:
        for r in lap_rows:
            if r["unit"] == unit and float(r["eps"]) == float(eps) and r["edge_mode"] == edge_mode and r["group"] == group and r["field"] == field:
                return r
        return None

    lines: list[str] = []
    lines.append("# 图 Laplacian 与 PDE 热点梯度诊断\n\n")
    lines.append("本轮不训练模型；目标是验证当前图微分算子和 PDE 损失方向，而不是调热源权重。\n\n")
    lines.append("## 配置\n\n")
    for k, v in manifest.items():
        lines.append(f"- {k}: `{v}`\n")
    lines.append("\n## 1. 解析场 Laplacian 测试摘要\n\n")
    lines.append("下表为当前实现等价设置：坐标转米、`eps=1e-8`、all_edges、valid 节点。\n\n")
    lines.append("| field | theory | mean | MAE | RMSE | p95 abs err | max abs err |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for field in ["constant", "x", "y", "z", "x2", "r2"]:
        r = select_lap("m", manifest["laplacian_eps"], "all_edges", "valid", field)
        if r:
            lines.append(f"| {field} | {float(r['theoretical_laplacian']):.3g} | {float(r['lap_mean']):.6g} | {float(r['lap_err_mean_abs']):.6g} | {float(r['lap_err_rms']):.6g} | {float(r['lap_err_p95_abs']):.6g} | {float(r['lap_err_max_abs']):.6g} |\n")
    lines.append("\n### Geometric interior/boundary split (m coords, eps=1e-8, all_edges, r2)\n\n")
    lines.append("| group | count | mean | RMSE err | p95 abs err | max abs err |\n|---|---:|---:|---:|---:|---:|\n")
    for group in ["valid_geom_interior_0p2mm", "valid_geom_boundary_0p2mm", "valid_geom_interior_0p5mm", "valid_geom_boundary_0p5mm", "valid_boundary_positive", "valid_boundary_negative"]:
        r = select_lap("m", manifest["laplacian_eps"], "all_edges", group, "r2")
        if r:
            lines.append(f"| {group} | {int(float(r['lap_count']))} | {float(r['lap_mean']):.6g} | {float(r['lap_err_rms']):.6g} | {float(r['lap_err_p95_abs']):.6g} | {float(r['lap_err_max_abs']):.6g} |\n")

    lines.append("\n### eps 与单位观察\n\n")
    for unit in ["mm", "m"]:
        for eps in [0.0, 1e-12, 1e-10, 1e-8]:
            r = select_lap(unit, eps, "all_edges", "valid", "r2")
            if r:
                lines.append(f"- unit={unit}, eps={eps:g}, field=r2: mean={float(r['lap_mean']):.6g}, RMSE err={float(r['lap_err_rms']):.6g}, p95 abs err={float(r['lap_err_p95_abs']):.6g}\n")
    lines.append("\n固定小量 `+1e-8` 若用于米坐标，相当于 0.1 mm 长度平方量级；解析场结果需据此判断是否已经扭曲二次场输出。\n\n")

    lines.append("## 2. 无效节点/边参与审计\n\n")
    if edge_rows:
        avg_vi_frac = float(np.mean([float(r["valid_row_to_invalid_fraction"]) for r in edge_rows]))
        max_vi_frac = float(np.max([float(r["valid_row_to_invalid_fraction"]) for r in edge_rows]))
        lines.append(f"- 被审计窗口中，valid-row 边连接到 invalid col 的平均比例: {avg_vi_frac:.6f}，最大比例: {max_vi_frac:.6f}。\n")
        lines.append("- 当前 `spatial_laplacian` 先用全图边计算，再只在 residual 处套 mask；若该比例非零，真实温度场的无效节点会参与有效节点导热近似。\n\n")

    lines.append("## 3. 真实热点附近 PDE 梯度/扰动方向\n\n")
    valid_grad = [r for r in grad_rows if r.get("hot_nodes", "0") not in ("", 0, "0") and r.get("perturbation") == "hot_decrease" and r.get("q_mode") == "endpoint"]
    if valid_grad:
        down_better = sum(float(r["loss_change"]) < 0 for r in valid_grad)
        frac_push = float(np.mean([float(r["grad_hot_positive_fraction_push_down"]) for r in valid_grad]))
        lines.append(f"- endpoint Q 下，热点降温扰动使 PDE loss 下降的记录数: {down_better}/{len(valid_grad)}。\n")
        lines.append(f"- endpoint Q 下，热点节点梯度为正、即梯度下降会降低热点温度的平均比例: {frac_push:.3f}。\n")
    lines.append("\n逐窗口/不同扰动幅度详见 `hotspot_pde_gradient_perturbation.csv`。正梯度表示在只看 PDE loss 时，梯度下降会把该节点温度往下推。\n\n")

    lines.append("## 输出文件\n\n")
    lines.append("- `laplacian_analytic_tests.csv`：解析场 Laplacian 测试。\n")
    lines.append("- `laplacian_edge_activity.csv`：valid/invalid 边参与统计。\n")
    lines.append("- `hotspot_pde_gradient_perturbation.csv`：热点升/降温扰动和 PDE 梯度方向。\n")
    lines.append("- `operator_gradient_audit_summary.md`：本摘要。\n")
    (outdir / "operator_gradient_audit_summary.md").write_text("".join(lines), encoding="utf-8")


def parse_eps(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/feature_e1_laser_distance.yaml")
    p.add_argument("--vtu_dir", default="F:/VTU")
    p.add_argument("--cache_dir", default="data/processed")
    p.add_argument("--laser_xml", default="configs/laser_paths/5_block_fem_additive_z_scan.xml")
    p.add_argument("--output_dir", type=Path, default=Path("results/training_diagnostics/physics_protocol_audit/operator_gradient_audit"))
    p.add_argument("--windows", nargs="*", type=int, default=[118, 120, 160, 208, 210, 421, 424, 634, 637, 847, 850, 1061, 1064, 1274, 1277])
    p.add_argument("--eps_values", default="0,1e-12,1e-10,1e-8")
    p.add_argument("--laplacian_eps", type=float, default=1e-8)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--pde_power_W", type=float, default=900.0)
    p.add_argument("--pde_efficiency", type=float, default=0.7)
    p.add_argument("--pde_radius_m", type=float, default=2e-3)
    p.add_argument("--pde_depth_m", type=float, default=1e-3)
    p.add_argument("--num_q_samples", type=int, default=24)
    p.add_argument("--deltas_C", nargs="*", type=float, default=[10.0, 50.0])
    p.add_argument("--hot_neighbor_radius_mm", type=float, default=1.0)
    args = p.parse_args()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = Config.from_yaml(args.config)
    cfg.data.laser_xml_path = args.laser_xml
    cfg.data.laser_path_mode = "additive_z_scan"
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu"))
    graph = load_graph(args.vtu_dir, args.cache_dir, cfg, device, args.no_cache)
    windows = [int(x) for x in args.windows]
    eps_values = parse_eps(args.eps_values)

    manifest = {
        "config": str(args.config),
        "vtu_dir": str(args.vtu_dir),
        "laser_xml": str(args.laser_xml),
        "windows": windows,
        "device": str(device),
        "coordinate_scale_to_m": float(cfg.physics.coordinate_scale_to_m),
        "physics_time_scale_to_s": float(cfg.physics.time_scale_to_s),
        "laser_path_time_scale_to_s": float(cfg.data.laser_path_time_scale_to_s),
        "laplacian_eps": float(args.laplacian_eps),
        "eps_values": eps_values,
        "pde_power_W": float(args.pde_power_W),
        "pde_efficiency": float(args.pde_efficiency),
        "pde_radius_m": float(args.pde_radius_m),
        "pde_depth_m": float(args.pde_depth_m),
        "deltas_C": args.deltas_C,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    lap_rows, edge_rows = run_laplacian_tests(graph, cfg, outdir, windows, eps_values)
    grad_rows = run_hotspot_gradient_tests(graph, cfg, outdir, windows, args)
    write_summary(outdir, lap_rows, edge_rows, grad_rows, manifest)
    print(outdir / "operator_gradient_audit_summary.md", flush=True)


if __name__ == "__main__":
    main()

