"""Audit true VTU temperature fields against the current graph-PDE approximation.

This script does not train any model.  It evaluates the residual

    rho Cp (T_t - T_{t-1}) / dt - k * Lap_G(T_t) - Q

on selected training windows, compares endpoint laser heat input with an
interval-averaged heat input, and writes CSV/PNG/Markdown diagnostics.
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

from scripts.training_diagnostics import load_graph, write_csv  # noqa: E402
from scripts.training_diagnostics_multwindow import (  # noqa: E402
    DEFAULT_TRAIN_BLOCKS,
    expand_blocks,
    original_train_starts,
)
from src.config import Config  # noqa: E402
from src.physics.differentiation import spatial_laplacian  # noqa: E402
from src.physics.heat_source import gaussian_heat_source  # noqa: E402


def finite_float(x: torch.Tensor | float | int) -> float:
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float("nan")
        return float(x.detach().float().cpu().item())
    return float(x)


def tensor_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    if x.numel() == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_mean_abs": float("nan"),
            f"{prefix}_rms": float("nan"),
            f"{prefix}_median_abs": float("nan"),
            f"{prefix}_p95_abs": float("nan"),
            f"{prefix}_max_abs": float("nan"),
        }
    ax = x.abs()
    return {
        f"{prefix}_mean": finite_float(x.mean()),
        f"{prefix}_mean_abs": finite_float(ax.mean()),
        f"{prefix}_rms": finite_float(torch.sqrt((x * x).mean())),
        f"{prefix}_median_abs": finite_float(torch.quantile(ax, 0.50)),
        f"{prefix}_p95_abs": finite_float(torch.quantile(ax, 0.95)),
        f"{prefix}_max_abs": finite_float(ax.max()),
    }


def group_row(
    *,
    window_start: int,
    target_step: int,
    group: str,
    mask: torch.Tensor,
    T_t: torch.Tensor,
    T_prev: torch.Tensor,
    storage: torch.Tensor,
    conduction: torch.Tensor,
    q_endpoint: torch.Tensor,
    q_interval: torch.Tensor,
    residual_endpoint: torch.Tensor,
    residual_interval: torch.Tensor,
    residual_hold_endpoint: torch.Tensor,
    residual_hold_interval: torch.Tensor,
    residual_scale: float,
    boundary: torch.Tensor,
) -> dict[str, Any]:
    idx = mask.detach().bool()
    row: dict[str, Any] = {
        "window_start": window_start,
        "target_step": target_step,
        "group": group,
        "count": int(idx.sum().detach().cpu()),
    }
    if row["count"] == 0:
        return row
    temp = T_t[idx]
    dtemp = (T_t - T_prev)[idx]
    row.update(
        {
            "target_temp_mean": finite_float(temp.mean()),
            "target_temp_max": finite_float(temp.max()),
            "delta_temp_mean": finite_float(dtemp.mean()),
            "delta_temp_max": finite_float(dtemp.max()),
            "boundary_positive_count": int(((boundary > 0) & idx).sum().detach().cpu()),
            "boundary_nonzero_count": int(((boundary != 0) & idx).sum().detach().cpu()),
        }
    )
    for name, arr in [
        ("storage", storage[idx]),
        ("conduction_k_lap", conduction[idx]),
        ("q_endpoint", q_endpoint[idx]),
        ("q_interval_avg", q_interval[idx]),
        ("residual_endpoint", residual_endpoint[idx]),
        ("residual_interval", residual_interval[idx]),
        ("residual_hold_endpoint", residual_hold_endpoint[idx]),
        ("residual_hold_interval", residual_hold_interval[idx]),
        ("residual_endpoint_norm", residual_endpoint[idx] / residual_scale),
        ("residual_interval_norm", residual_interval[idx] / residual_scale),
        ("residual_hold_endpoint_norm", residual_hold_endpoint[idx] / residual_scale),
        ("residual_hold_interval_norm", residual_hold_interval[idx] / residual_scale),
    ]:
        row.update(tensor_stats(name, arr))
    row["interval_vs_endpoint_residual_rms_ratio"] = (
        row.get("residual_interval_rms", float("nan")) / row.get("residual_endpoint_rms", float("nan"))
        if row.get("residual_endpoint_rms", 0.0) not in (0.0, float("nan"))
        else float("nan")
    )
    row["hold_endpoint_vs_true_endpoint_rms_ratio"] = (
        row.get("residual_hold_endpoint_rms", float("nan")) / row.get("residual_endpoint_rms", float("nan"))
        if row.get("residual_endpoint_rms", 0.0) not in (0.0, float("nan"))
        else float("nan")
    )
    return row


def interpolate_laser_pos_from_segments(graph: Any, raw_time: float) -> torch.Tensor:
    """Interpolate laser position in mm using DynamicGraph segment tables."""
    device = graph.coords.device
    st = graph._laser_path_segment_start_times
    en = graph._laser_path_segment_end_times
    starts = graph._laser_path_segment_starts
    ends = graph._laser_path_segment_ends
    t = torch.tensor(float(raw_time), device=st.device, dtype=st.dtype)
    active = torch.nonzero((t >= st) & (t <= en), as_tuple=False).flatten()
    if active.numel() > 0:
        i = int(active[0].detach().cpu().item())
        denom = (en[i] - st[i]).clamp_min(1.0e-12)
        frac = ((t - st[i]) / denom).clamp(0.0, 1.0)
        return (starts[i] + frac * (ends[i] - starts[i])).to(device)

    # Dwell/off-scan gap: clamp to the nearest segment endpoint.  This is a
    # diagnostic fallback; off-scan handling should be checked against FEM if used.
    dist_start = (st - t).abs()
    dist_end = (en - t).abs()
    ds, is_ = dist_start.min(dim=0)
    de, ie = dist_end.min(dim=0)
    if float(ds.detach().cpu()) < float(de.detach().cpu()):
        return starts[int(is_.detach().cpu().item())].to(device)
    return ends[int(ie.detach().cpu().item())].to(device)


def interval_average_q(
    graph: Any,
    coords_m: torch.Tensor,
    raw_prev: float,
    raw_target: float,
    *,
    num_samples: int,
    power: float,
    efficiency: float,
    radius: float,
    depth: float,
    coord_scale_to_m: float,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    q_sum = torch.zeros(coords_m.shape[0], device=coords_m.device, dtype=coords_m.dtype)
    samples: list[dict[str, float]] = []
    span = float(raw_target) - float(raw_prev)
    n = max(1, int(num_samples))
    for i in range(n):
        raw = float(raw_prev) + (i + 0.5) / n * span
        pos_mm = interpolate_laser_pos_from_segments(graph, raw)
        pos_m = pos_mm * coord_scale_to_m
        q = gaussian_heat_source(
            coords_m,
            pos_m,
            power=power,
            efficiency=efficiency,
            radius=radius,
            depth=depth,
        )
        q_sum += q
        if i in {0, n // 2, n - 1}:
            samples.append({
                "sample_i": i,
                "raw_time": raw,
                "laser_x_mm": finite_float(pos_mm[0]),
                "laser_y_mm": finite_float(pos_mm[1]),
                "laser_z_mm": finite_float(pos_mm[2]),
            })
    return q_sum / n, samples


def choose_windows(graph: Any, cfg: Config, requested: list[int] | None, max_windows: int) -> list[int]:
    if requested:
        return [int(x) for x in requested[:max_windows]]
    candidates = expand_blocks(DEFAULT_TRAIN_BLOCKS)
    # Add a few starts around transitions if present in original train split.
    extras = [118, 120, 160, 208, 210, 421, 424, 634, 637, 847, 850, 1061, 1064, 1274, 1277]
    seen = []
    for s in extras + candidates:
        if s not in seen:
            seen.append(s)
    rows = []
    for s in seen:
        tgt = s + cfg.data.window_size + cfg.data.predict_steps - 1
        if tgt >= len(graph.times) or tgt <= 0:
            continue
        g = graph.get_graph_at(tgt)
        v = g.y.detach().float().flatten()[g.mask.detach().bool().flatten()]
        hot = int((v >= cfg.material.solidus_temp).sum().detach().cpu())
        rows.append((s, hot, float(v.max().detach().cpu())))
    # Keep the hand-picked temporal coverage order, capped to max_windows.
    return [s for s, _h, _mx in rows[:max_windows]]


def make_plots(outdir: Path, group_rows: list[dict[str, Any]], window_plot_payloads: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        (outdir / "plots_unavailable.txt").write_text(str(exc), encoding="utf-8")
        return

    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Term RMS by window for all valid nodes.
    rows = [r for r in group_rows if r.get("group") == "valid"]
    if rows:
        xs = [r["target_step"] for r in rows]
        plt.figure(figsize=(10, 5))
        for key, label in [
            ("storage_rms", "rhoCp dT/dt"),
            ("conduction_k_lap_rms", "k Lap_G T"),
            ("q_endpoint_rms", "Q endpoint"),
            ("q_interval_avg_rms", "Q interval avg"),
            ("residual_endpoint_rms", "R endpoint"),
            ("residual_interval_rms", "R interval avg Q"),
        ]:
            plt.plot(xs, [r.get(key, np.nan) for r in rows], marker="o", label=label)
        plt.yscale("log")
        plt.xlabel("target step")
        plt.ylabel("RMS magnitude")
        plt.title("True-field PDE term magnitudes by window")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / "term_rms_by_window.png", dpi=180)
        plt.close()

    # Spatial maps for selected windows.
    for p in window_plot_payloads:
        coords = p["coords"]
        valid = p["valid"]
        x = coords[valid, 0]
        y = coords[valid, 1]
        temp = p["T_t"][valid]
        res = p["residual_norm"][valid]
        q = p["q_interval"][valid]
        step = p["target_step"]
        start = p["window_start"]
        # Subsample deterministically for readability.
        n = x.shape[0]
        if n > 35000:
            idx = np.linspace(0, n - 1, 35000).astype(np.int64)
            x, y, temp, res, q = x[idx], y[idx], temp[idx], res[idx], q[idx]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        specs = [
            (temp, "T target (C)", "inferno"),
            (np.log10(np.abs(res) + 1e-12), "log10 |R_endpoint norm|", "viridis"),
            (np.log10(np.abs(q) + 1.0), "log10(Q_interval+1)", "magma"),
        ]
        for ax, (c, title, cmap) in zip(axes, specs):
            sc = ax.scatter(x, y, c=c, s=1.2, cmap=cmap, linewidths=0)
            ax.scatter([p["laser_x"]], [p["laser_y"]], marker="x", s=40, c="cyan")
            ax.set_title(title)
            ax.set_xlabel("x mm")
            ax.set_ylabel("y mm")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"window {start}, target step {step}")
        fig.savefig(plot_dir / f"window_{start}_step_{step}_spatial.png", dpi=180)
        plt.close(fig)


def write_summary(
    outdir: Path,
    manifest: dict[str, Any],
    window_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    protocol_path: Path,
) -> None:
    valid_rows = [r for r in group_rows if r.get("group") == "valid"]
    hot_rows = [r for r in group_rows if r.get("group") == "hot"]
    nohot_count = sum(1 for r in valid_rows if int(r.get("hot_nodes", 0) or 0) == 0)

    def avg(key: str, rows: list[dict[str, Any]]) -> float:
        vals = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    lines: list[str] = []
    lines.append("# 真实温度场 PDE 残差审计\n\n")
    lines.append("本审计不训练模型；仅将原训练集窗口的真实 VTU 温度代入当前图 PDE 近似。\n\n")
    lines.append("## 协议\n\n")
    lines.append(f"- config: `{manifest['config']}`\n")
    lines.append(f"- VTU: `{manifest['vtu_dir']}`\n")
    lines.append(f"- XML: `{manifest['laser_xml']}`\n")
    lines.append(f"- windows: {manifest['windows']}\n")
    lines.append(f"- coordinate scale: {manifest['coordinate_scale_to_m']} m/mm; PDE time scale: {manifest['physics_time_scale_to_s']} s/raw; path time scale: {manifest['laser_path_time_scale_to_s']} s/raw\n")
    lines.append(f"- heat source in audit/current PDE: Gaussian, power={manifest['pde_power_W']} W, efficiency={manifest['pde_efficiency']}, radius={manifest['pde_radius_m']} m, depth={manifest['pde_depth_m']} m\n")
    lines.append(f"- 参数协议审计文档: `{protocol_path.as_posix()}`\n\n")
    lines.append("## 关键量级摘要（valid 组，逐窗口均值）\n\n")
    lines.append("| term | RMS mean | mean_abs mean |\n|---|---:|---:|\n")
    for key, label in [
        ("storage", "rho Cp dT/dt"),
        ("conduction_k_lap", "k Lap_G T"),
        ("q_endpoint", "Q endpoint"),
        ("q_interval_avg", "Q interval avg"),
        ("residual_endpoint", "R endpoint"),
        ("residual_interval", "R interval avg Q"),
        ("residual_hold_endpoint", "R hold endpoint"),
    ]:
        lines.append(f"| {label} | {avg(key + '_rms', valid_rows):.6e} | {avg(key + '_mean_abs', valid_rows):.6e} |\n")
    lines.append("\n## 终点热源 vs 区间平均热源\n\n")
    better_interval = sum(
        1 for r in valid_rows
        if float(r.get("residual_interval_rms", float("inf"))) < float(r.get("residual_endpoint_rms", float("inf")))
    )
    lines.append(f"- valid 组中，区间平均 Q 的真实场残差 RMS 小于终点 Q 的窗口数: {better_interval}/{len(valid_rows)}。\n")
    lines.append(f"- 平均相邻输出激光移动距离: {avg('laser_move_mm', window_rows):.3f} mm。\n")
    lines.append("- 该比较只说明当前 PDE 近似的解释能力差异，不能单独证明 FEM 采用区间平均热源。\n\n")
    lines.append("## 保持温度预测的残差偏好检查\n\n")
    hold_better = sum(
        1 for r in valid_rows
        if float(r.get("residual_hold_endpoint_rms", float("inf"))) < float(r.get("residual_endpoint_rms", float("inf")))
    )
    lines.append(f"- 在 endpoint Q 下，T(t)=T(t-1) 的残差 RMS 小于真实升温场残差的窗口数: {hold_better}/{len(valid_rows)}。\n")
    lines.append("- 如果该数值较高，说明当前 PDE 项可能偏好低升温/保持温度，需要先核对热源和时间尺度。\n\n")
    lines.append("## 热点组摘要\n\n")
    lines.append(f"- 含热点窗口数: {len([r for r in hot_rows if r.get('count',0)])}/{len(valid_rows)}；无热点窗口数: {nohot_count}。\n")
    lines.append(f"- 热点组 endpoint 残差 RMS 均值: {avg('residual_endpoint_rms', hot_rows):.6e}；valid 组均值: {avg('residual_endpoint_rms', valid_rows):.6e}。\n")
    lines.append(f"- 热点组 interval 残差 RMS 均值: {avg('residual_interval_rms', hot_rows):.6e}。\n\n")
    lines.append("## 输出文件\n\n")
    lines.append("- `true_field_residual_by_window.csv`：逐窗口整体信息、移动距离、热点计数。\n")
    lines.append("- `true_field_residual_by_group.csv`：valid/hot/background/boundary 分组项量级与残差。\n")
    lines.append("- `laser_interval_samples.csv`：区间平均热源采样位置抽样。\n")
    lines.append("- `plots/term_rms_by_window.png` 与 `plots/window_*_spatial.png`：量级图和空间图。\n\n")
    lines.append("## 审慎解释\n\n")
    lines.append("当前图 Laplacian 与 FEM 算子并不相同，且相变、温变材料参数、真实边界条件等机制未必包含在简化 PDE 中；因此不要求残差为零。本审计用于排除尺度错误、热源不匹配和明显不合理的约束偏好。\n")
    (outdir / "true_field_residual_summary.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/feature_e1_laser_distance.yaml")
    p.add_argument("--vtu_dir", default="F:/VTU")
    p.add_argument("--cache_dir", default="data/processed")
    p.add_argument("--laser_xml", default="configs/laser_paths/5_block_fem_additive_z_scan.xml")
    p.add_argument("--output_dir", type=Path, default=Path("results/training_diagnostics/physics_protocol_audit/true_field_residual"))
    p.add_argument("--windows", nargs="*", type=int, default=None)
    p.add_argument("--max_windows", type=int, default=15)
    p.add_argument("--num_q_samples", type=int, default=24)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--pde_power_W", type=float, default=900.0)
    p.add_argument("--pde_efficiency", type=float, default=0.7)
    p.add_argument("--pde_radius_m", type=float, default=2.0e-3)
    p.add_argument("--pde_depth_m", type=float, default=1.0e-3)
    p.add_argument("--max_spatial_plots", type=int, default=8)
    args = p.parse_args()

    outdir: Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = Config.from_yaml(args.config)
    cfg.data.laser_xml_path = args.laser_xml
    cfg.data.laser_path_mode = "additive_z_scan"

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu"))
    graph = load_graph(args.vtu_dir, args.cache_dir, cfg, device, args.no_cache)

    windows = choose_windows(graph, cfg, args.windows, args.max_windows)
    rho = float(cfg.material.density)
    Cp = float(cfg.material.specific_heat)
    k = float(cfg.material.thermal_conductivity)
    solidus = float(cfg.material.solidus_temp)
    coord_scale = float(cfg.physics.coordinate_scale_to_m)
    physics_time_scale = float(cfg.physics.time_scale_to_s)
    residual_scale = rho * Cp * float(cfg.physics.pde_temperature_scale) / float(cfg.physics.pde_time_scale)

    coords_m = graph.coords.detach().float() * coord_scale
    edge_index = graph.edge_index
    boundary = graph.boundary.detach().float()

    window_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    plot_payloads: list[dict[str, Any]] = []

    for start in windows:
        target_step = int(start) + int(cfg.data.window_size) + int(cfg.data.predict_steps) - 1
        prev_step = target_step - 1
        raw_target = finite_float(graph.times[target_step])
        raw_prev = finite_float(graph.times[prev_step])
        dt_s = (raw_target - raw_prev) * physics_time_scale
        if dt_s <= 0:
            raise RuntimeError(f"non-positive dt_s for start {start}: {dt_s}")

        T_t = graph.temperatures[target_step].detach().float()
        T_prev = graph.temperatures[prev_step].detach().float()
        valid = graph.get_active_mask(target_step).detach().bool()
        hot = valid & (T_t >= solidus)
        background = valid & ~hot
        boundary_pos = valid & (boundary > 0)
        boundary_nonzero = valid & (boundary != 0)

        laser_target_mm = graph._laser_positions[target_step].detach().float()
        laser_prev_mm = graph._laser_positions[prev_step].detach().float()
        laser_move_mm = finite_float(torch.linalg.norm(laser_target_mm - laser_prev_mm))
        q_endpoint = gaussian_heat_source(
            coords_m,
            laser_target_mm * coord_scale,
            power=args.pde_power_W,
            efficiency=args.pde_efficiency,
            radius=args.pde_radius_m,
            depth=args.pde_depth_m,
        )
        q_interval, q_samples = interval_average_q(
            graph,
            coords_m,
            raw_prev,
            raw_target,
            num_samples=args.num_q_samples,
            power=args.pde_power_W,
            efficiency=args.pde_efficiency,
            radius=args.pde_radius_m,
            depth=args.pde_depth_m,
            coord_scale_to_m=coord_scale,
        )
        for row in q_samples:
            sample_rows.append({"window_start": start, "target_step": target_step, **row})

        lap = spatial_laplacian(T_t, coords_m, edge_index)
        lap_hold = spatial_laplacian(T_prev, coords_m, edge_index)
        storage = rho * Cp * (T_t - T_prev) / dt_s
        conduction = k * lap
        conduction_hold = k * lap_hold
        residual_endpoint = storage - conduction - q_endpoint
        residual_interval = storage - conduction - q_interval
        residual_hold_endpoint = -conduction_hold - q_endpoint
        residual_hold_interval = -conduction_hold - q_interval

        near_endpoint = valid & (torch.linalg.norm(graph.coords.detach().float() - laser_target_mm, dim=1) <= 2.0)
        window_row = {
            "window_start": start,
            "prev_step": prev_step,
            "target_step": target_step,
            "raw_prev": raw_prev,
            "raw_target": raw_target,
            "dt_raw": raw_target - raw_prev,
            "dt_s_pde": dt_s,
            "dt_s_path": (raw_target - raw_prev) * float(cfg.data.laser_path_time_scale_to_s),
            "laser_prev_x_mm": finite_float(laser_prev_mm[0]),
            "laser_prev_y_mm": finite_float(laser_prev_mm[1]),
            "laser_prev_z_mm": finite_float(laser_prev_mm[2]),
            "laser_target_x_mm": finite_float(laser_target_mm[0]),
            "laser_target_y_mm": finite_float(laser_target_mm[1]),
            "laser_target_z_mm": finite_float(laser_target_mm[2]),
            "laser_move_mm": laser_move_mm,
            "laser_move_over_pde_radius": laser_move_mm / (args.pde_radius_m / coord_scale),
            "laser_move_over_xml_front_radius_0p4mm": laser_move_mm / 0.4,
            "laser_move_over_xml_body_radius_0p1mm": laser_move_mm / 0.1,
            "valid_nodes": int(valid.sum().detach().cpu()),
            "hot_nodes": int(hot.sum().detach().cpu()),
            "background_nodes": int(background.sum().detach().cpu()),
            "boundary_positive_nodes": int(boundary_pos.sum().detach().cpu()),
            "boundary_nonzero_nodes": int(boundary_nonzero.sum().detach().cpu()),
            "near_endpoint_nodes_radius2mm": int(near_endpoint.sum().detach().cpu()),
            "target_temp_max": finite_float(T_t[valid].max()),
            "target_temp_mean": finite_float(T_t[valid].mean()),
            "delta_temp_max": finite_float((T_t - T_prev)[valid].max()),
            "delta_temp_min": finite_float((T_t - T_prev)[valid].min()),
            "q_endpoint_max": finite_float(q_endpoint[valid].max()),
            "q_interval_max": finite_float(q_interval[valid].max()),
        }
        window_rows.append(window_row)

        group_defs = [
            ("valid", valid),
            ("hot", hot),
            ("background", background),
            ("boundary_positive", boundary_pos),
            ("boundary_nonzero", boundary_nonzero),
            ("near_endpoint_2mm", near_endpoint),
        ]
        for group, m in group_defs:
            row = group_row(
                window_start=start,
                target_step=target_step,
                group=group,
                mask=m,
                T_t=T_t,
                T_prev=T_prev,
                storage=storage,
                conduction=conduction,
                q_endpoint=q_endpoint,
                q_interval=q_interval,
                residual_endpoint=residual_endpoint,
                residual_interval=residual_interval,
                residual_hold_endpoint=residual_hold_endpoint,
                residual_hold_interval=residual_hold_interval,
                residual_scale=residual_scale,
                boundary=boundary,
            )
            row["hot_nodes"] = int(hot.sum().detach().cpu())
            row["laser_move_mm"] = laser_move_mm
            group_rows.append(row)

        if len(plot_payloads) < args.max_spatial_plots:
            plot_payloads.append(
                {
                    "window_start": start,
                    "target_step": target_step,
                    "coords": graph.coords.detach().float().cpu().numpy(),
                    "valid": valid.detach().cpu().numpy().astype(bool),
                    "T_t": T_t.detach().cpu().numpy(),
                    "residual_norm": (residual_endpoint / residual_scale).detach().cpu().numpy(),
                    "q_interval": q_interval.detach().cpu().numpy(),
                    "laser_x": finite_float(laser_target_mm[0]),
                    "laser_y": finite_float(laser_target_mm[1]),
                }
            )

        print(f"audited window {start} -> target {target_step}, hot={window_row['hot_nodes']}, move={laser_move_mm:.3f} mm", flush=True)

    write_csv(outdir / "true_field_residual_by_window.csv", window_rows)
    write_csv(outdir / "true_field_residual_by_group.csv", group_rows)
    write_csv(outdir / "laser_interval_samples.csv", sample_rows)
    manifest = {
        "config": str(args.config),
        "vtu_dir": str(args.vtu_dir),
        "cache_dir": str(args.cache_dir),
        "laser_xml": str(args.laser_xml),
        "windows": windows,
        "device": str(device),
        "coordinate_scale_to_m": coord_scale,
        "physics_time_scale_to_s": physics_time_scale,
        "laser_path_time_scale_to_s": float(cfg.data.laser_path_time_scale_to_s),
        "laser_path_time_offset_s": float(cfg.data.laser_path_time_offset_s),
        "rho": rho,
        "Cp": Cp,
        "k": k,
        "solidus": solidus,
        "residual_scale": residual_scale,
        "pde_power_W": args.pde_power_W,
        "pde_efficiency": args.pde_efficiency,
        "pde_radius_m": args.pde_radius_m,
        "pde_depth_m": args.pde_depth_m,
        "num_q_samples": args.num_q_samples,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    make_plots(outdir, group_rows, plot_payloads)
    protocol_path = outdir.parent / "heat_source_protocol_audit.md"
    write_summary(outdir, manifest, window_rows, group_rows, protocol_path)
    print(outdir / "true_field_residual_summary.md", flush=True)


if __name__ == "__main__":
    main()
