"""Stage 4 paired supervision diagnostics for DT-STPINN.

For each E0/E1/E2, this script:
1) trains normal full-graph MSE to update 500 from a fixed seed;
2) branches from the exact update-500 model+optimizer state;
3) continues A: normal MSE to update 2000;
4) continues B: balanced hot/neighbor/background supervision to update 2000.

It keeps the full graph as input and changes only the supervised nodes/loss for B.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Keep matplotlib cache in writable workspace if plotting is enabled.
os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.training_diagnostics import (  # noqa: E402
    EXPERIMENTS,
    Config,
    DTSTPINN,
    disable_dropout,
    forward_loss,
    grad_norm,
    load_graph,
    make_dataset,
    metrics_from_pred,
    param_delta,
    set_seed,
    write_csv,
)


def clone_state_to_cpu(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)


def fixed_background_indices(target: torch.Tensor, valid: torch.Tensor, excluded: torch.Tensor,
                             solidus: float, seed: int, per_bin: int) -> tuple[torch.Tensor, dict[str, int]]:
    t = target.detach().float().flatten().cpu()
    v = valid.detach().bool().flatten().cpu()
    ex = excluded.detach().bool().flatten().cpu()
    available = v & (~ex)
    bins = [
        ("lt_100", None, 100.0),
        ("100_500", 100.0, 500.0),
        ("500_1000", 500.0, 1000.0),
        ("1000_solidus", 1000.0, solidus),
    ]
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    selected: list[torch.Tensor] = []
    counts: dict[str, int] = {}
    for name, lo, hi in bins:
        mask = available.clone()
        if lo is not None:
            mask &= t >= float(lo)
        if hi is not None:
            mask &= t < float(hi)
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        counts[f"available_{name}"] = int(idx.numel())
        if idx.numel() == 0:
            counts[f"selected_{name}"] = 0
            continue
        if idx.numel() > per_bin:
            perm = torch.randperm(idx.numel(), generator=rng)[:per_bin]
            idx = idx[perm]
        selected.append(idx)
        counts[f"selected_{name}"] = int(idx.numel())
    if not selected:
        raise RuntimeError("No background nodes selected")
    bg = torch.cat(selected).unique(sorted=True)
    counts["selected_total"] = int(bg.numel())
    return bg, counts


def build_groups(sample: dict[str, Any], solidus: float, liquidus: float,
                 seed: int, background_per_bin: int) -> dict[str, Any]:
    target = sample["target"].detach().float().flatten()
    valid = sample["target_mask"].detach().bool().flatten()
    edge_index = sample["edge_index"].detach().long()
    device = target.device
    hot_mask = valid & (target >= solidus)
    hot_idx = torch.nonzero(hot_mask, as_tuple=False).flatten()
    if hot_idx.numel() == 0:
        raise RuntimeError("Selected window has no solidus hotspot nodes")
    neighbor_mask = torch.zeros_like(valid, dtype=torch.bool)
    src, dst = edge_index[0], edge_index[1]
    edge_touch_hot = hot_mask[src] | hot_mask[dst]
    touched = torch.cat([src[edge_touch_hot], dst[edge_touch_hot]]).unique()
    neighbor_mask[touched] = True
    neighbor_mask &= valid & (~hot_mask)
    neighbor_idx = torch.nonzero(neighbor_mask, as_tuple=False).flatten()
    excluded = hot_mask | neighbor_mask
    bg_cpu, bg_counts = fixed_background_indices(target, valid, excluded, solidus, seed, background_per_bin)
    bg_idx = bg_cpu.to(device)

    # Exact full-temperature bucket counts over all valid nodes.
    t_valid = target[valid]
    bucket_counts = {
        "valid_total": int(valid.sum().detach().cpu()),
        "hot_solidus_total": int((t_valid >= solidus).sum().detach().cpu()),
        "liquidus_total": int((t_valid >= liquidus).sum().detach().cpu()),
        "lt_100": int((t_valid < 100.0).sum().detach().cpu()),
        "100_500": int(((t_valid >= 100.0) & (t_valid < 500.0)).sum().detach().cpu()),
        "500_1000": int(((t_valid >= 500.0) & (t_valid < 1000.0)).sum().detach().cpu()),
        "1000_solidus": int(((t_valid >= 1000.0) & (t_valid < solidus)).sum().detach().cpu()),
        "solidus_liquidus": int(((t_valid >= solidus) & (t_valid < liquidus)).sum().detach().cpu()),
        "ge_liquidus": int((t_valid >= liquidus).sum().detach().cpu()),
    }
    return {
        "hot_idx": hot_idx,
        "neighbor_idx": neighbor_idx,
        "background_idx": bg_idx,
        "hot_mask": hot_mask,
        "neighbor_mask": neighbor_mask,
        "background_counts": bg_counts,
        "bucket_counts": bucket_counts,
    }


def balanced_loss(pred: torch.Tensor, target: torch.Tensor, groups: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    p = pred.float()
    t = target.float()
    if p.numel() == t.numel():
        p = p.reshape_as(t)
    p = p.flatten()
    t = t.flatten()
    losses = []
    comps: dict[str, float] = {}
    for name, key in [("hot", "hot_idx"), ("neighbor", "neighbor_idx"), ("background", "background_idx")]:
        idx = groups[key]
        if idx.numel() == 0:
            continue
        mse = ((p[idx] - t[idx]) ** 2).mean()
        losses.append(mse)
        comps[f"loss_{name}"] = float(mse.detach().cpu())
    if not losses:
        raise RuntimeError("No non-empty groups for balanced loss")
    loss = sum(losses) / len(losses)
    comps["loss_balanced"] = float(loss.detach().cpu())
    return loss, comps


def extra_position_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                           coords: torch.Tensor, groups: dict[str, Any], solidus: float) -> dict[str, Any]:
    p = pred.detach().float()
    t = target.detach().float()
    if p.numel() == t.numel():
        p = p.reshape_as(t)
    p = p.flatten()
    t = t.flatten()
    valid = mask.detach().bool().flatten()
    hot_idx = groups["hot_idx"]
    bg_mask = valid.clone()
    bg_mask[hot_idx] = False
    valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
    p_valid = p[valid]
    peak_local = torch.argmax(p_valid)
    peak_idx = valid_idx[peak_local]
    hot_coords = coords[hot_idx].detach().float()
    peak_coord = coords[peak_idx].detach().float()
    dist = torch.linalg.norm(hot_coords - peak_coord.view(1, -1), dim=1)
    nearest = float(dist.min().detach().cpu()) if dist.numel() else float("nan")
    peak_is_hot = bool((peak_idx == hot_idx).any().detach().cpu())
    pred_hot_on_true = p[hot_idx]
    true_hot = t[hot_idx]
    fp_count = int(((p >= solidus) & (t < solidus) & valid).sum().detach().cpu())
    tp_count = int(((p >= solidus) & (t >= solidus) & valid).sum().detach().cpu())
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) else 0.0
    bg_abs = (p[bg_mask] - t[bg_mask]).abs()
    return {
        "true_hot_pred_min": float(pred_hot_on_true.min().detach().cpu()),
        "true_hot_pred_mean": float(pred_hot_on_true.mean().detach().cpu()),
        "true_hot_pred_max": float(pred_hot_on_true.max().detach().cpu()),
        "true_hot_true_min": float(true_hot.min().detach().cpu()),
        "true_hot_true_mean": float(true_hot.mean().detach().cpu()),
        "true_hot_true_max": float(true_hot.max().detach().cpu()),
        "true_hot_mae_exact": float((pred_hot_on_true - true_hot).abs().mean().detach().cpu()),
        "background_mae_excluding_hot": float(bg_abs.mean().detach().cpu()),
        "full_fp_count": fp_count,
        "full_tp_count": tp_count,
        "full_precision": precision,
        "pred_peak_node": int(peak_idx.detach().cpu()),
        "pred_peak_x": float(peak_coord[0].detach().cpu()),
        "pred_peak_y": float(peak_coord[1].detach().cpu()),
        "pred_peak_z": float(peak_coord[2].detach().cpu()),
        "pred_peak_nearest_true_hot_dist_mm": nearest,
        "pred_peak_is_true_hot_node": peak_is_hot,
    }


def eval_full(model: DTSTPINN, sample: dict[str, Any], cfg: Config, groups: dict[str, Any],
              update: int, branch: str, loss_mode: str, loss_value: float | None,
              loss_components: dict[str, float] | None, last_grad_norm: float | None,
              init_params: list[torch.Tensor]) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        _loss, _out, pred = forward_loss(model, sample, False, None)
    model.train()
    row: dict[str, Any] = {
        "update": update,
        "branch": branch,
        "loss_mode": loss_mode,
        "train_loss_current_mode": loss_value,
        "grad_norm_pre_clip": last_grad_norm,
        "lr": None,
        "amp_enabled": False,
        "dropout": 0.0,
        "temperature_only_loss": True,
        "physics_loss_enabled": False,
        "smooth_loss_enabled": False,
        "hot_group_n": int(groups["hot_idx"].numel()),
        "neighbor_group_n": int(groups["neighbor_idx"].numel()),
        "background_group_n": int(groups["background_idx"].numel()),
    }
    if loss_components:
        row.update(loss_components)
    row.update(metrics_from_pred(pred, sample["target"], sample.get("target_mask"), cfg.material.solidus_temp, cfg.material.liquidus_temp, None, prefix=""))
    row.update(extra_position_metrics(pred, sample["target"], sample["target_mask"], sample["coords"], groups, cfg.material.solidus_temp))
    dl2, dmax = param_delta(model, init_params)
    row["param_delta_l2_from_init"] = dl2
    row["param_delta_max_from_init"] = dmax
    return row


def train_segment(model: DTSTPINN, opt: torch.optim.Optimizer, sample: dict[str, Any], cfg: Config,
                  groups: dict[str, Any], start_update: int, end_update: int, branch: str,
                  loss_mode: str, log_every: int, init_params: list[torch.Tensor],
                  rows: list[dict[str, Any]]) -> None:
    for update in range(start_update + 1, end_update + 1):
        opt.zero_grad(set_to_none=True)
        if loss_mode == "mse":
            loss, _out, pred = forward_loss(model, sample, False, None)
            comps = {"loss_mse": float(loss.detach().cpu())}
        elif loss_mode == "balanced":
            graph_seq = sample["graph_sequence"]
            dt = sample.get("dt", 1.0)
            if isinstance(dt, torch.Tensor):
                dt = float(dt.detach().cpu().item())
            out = model(graph_seq, dt=dt)
            pred = out["T_pred"]
            loss, comps = balanced_loss(pred, sample["target"], groups)
        else:
            raise ValueError(loss_mode)
        loss.backward()
        gnorm = grad_norm(model)
        if cfg.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        opt.step()
        if update % log_every == 0 or update == end_update:
            row = eval_full(model, sample, cfg, groups, update, branch, loss_mode,
                            float(loss.detach().cpu()), comps, gnorm, init_params)
            row["lr"] = opt.param_groups[0]["lr"]
            rows.append(row)
            print(
                f"{branch} {loss_mode} update {update}/{end_update} "
                f"rmse={row['rmse']:.3f} hot_mae={row['true_hot_mae_exact']:.3f} "
                f"recall={row['solidus_recall']:.3f} pred_max={row['pred_max']:.1f} "
                f"fp={row['full_fp_count']} dist={row['pred_peak_nearest_true_hot_dist_mm']:.3f}",
                flush=True,
            )


def build_model_optimizer(cfg: Config, device: torch.device) -> tuple[DTSTPINN, torch.optim.Optimizer]:
    model = DTSTPINN(cfg, cfg.material).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    return model, opt


def run_experiment(exp: str, graph: Any, args: argparse.Namespace, device: torch.device,
                   selected_start: int, common_groups: dict[str, Any] | None,
                   outdir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    set_seed(args.seed)
    cfg = Config.from_yaml(EXPERIMENTS[exp])
    if args.laser_xml:
        cfg.data.laser_xml_path = args.laser_xml
        cfg.data.laser_path_mode = "additive_z_scan"
    disable_dropout(cfg)
    cfg.training.use_amp = False
    ds = make_dataset(graph, cfg, [selected_start])
    cfg.model.node_feature_dim = ds.input_feature_dim
    sample = ds[0]
    model, opt = build_model_optimizer(cfg, device)
    init_params = [p.detach().float().cpu().clone() for p in model.parameters()]
    groups = common_groups or build_groups(sample, cfg.material.solidus_temp, cfg.material.liquidus_temp, args.seed, args.background_per_bin)
    # Ensure group indices are on this graph/device.
    for k in ["hot_idx", "neighbor_idx", "background_idx"]:
        groups[k] = groups[k].to(sample["target"].device)
    rows: list[dict[str, Any]] = []
    row0 = eval_full(model, sample, cfg, groups, 0, "common", "mse", None, None, None, init_params)
    row0["experiment"] = exp
    row0["input_feature_dim"] = ds.input_feature_dim
    row0["target_laser_feature_dim"] = ds.target_laser_feature_dim
    row0["lr"] = cfg.training.lr
    rows.append(row0)
    # Common normal-MSE warmup to branch point.
    train_segment(model, opt, sample, cfg, groups, 0, args.branch_update, "common", "mse", args.log_every, init_params, rows)
    for r in rows:
        r.setdefault("experiment", exp)
        r.setdefault("input_feature_dim", ds.input_feature_dim)
        r.setdefault("target_laser_feature_dim", ds.target_laser_feature_dim)
    branch_model_state = clone_state_to_cpu(model.state_dict())
    branch_opt_state = clone_state_to_cpu(opt.state_dict())
    ckpt_path = outdir / f"stage4_{exp}_branch_update{args.branch_update}.pt"
    torch.save({
        "experiment": exp,
        "update": args.branch_update,
        "model_state_dict": branch_model_state,
        "optimizer_state_dict": branch_opt_state,
        "config_path": str(EXPERIMENTS[exp]),
        "window_start": selected_start,
    }, ckpt_path)

    # Branch A: continue normal MSE.
    model_a, opt_a = build_model_optimizer(cfg, device)
    model_a.load_state_dict(branch_model_state)
    opt_a.load_state_dict(branch_opt_state)
    rows_a: list[dict[str, Any]] = []
    train_segment(model_a, opt_a, sample, cfg, groups, args.branch_update, args.total_updates, "A_mse", "mse", args.log_every, init_params, rows_a)
    for r in rows_a:
        r["experiment"] = exp
        r["input_feature_dim"] = ds.input_feature_dim
        r["target_laser_feature_dim"] = ds.target_laser_feature_dim
    # Branch B: continue balanced supervision.
    model_b, opt_b = build_model_optimizer(cfg, device)
    model_b.load_state_dict(branch_model_state)
    opt_b.load_state_dict(branch_opt_state)
    rows_b: list[dict[str, Any]] = []
    train_segment(model_b, opt_b, sample, cfg, groups, args.branch_update, args.total_updates, "B_balanced", "balanced", args.log_every, init_params, rows_b)
    for r in rows_b:
        r["experiment"] = exp
        r["input_feature_dim"] = ds.input_feature_dim
        r["target_laser_feature_dim"] = ds.target_laser_feature_dim
    all_rows = rows + rows_a + rows_b
    write_csv(outdir / f"stage4_curve_{exp}.csv", all_rows)
    summary = {
        "experiment": exp,
        "input_feature_dim": ds.input_feature_dim,
        "target_laser_feature_dim": ds.target_laser_feature_dim,
        "branch_checkpoint": str(ckpt_path),
    }
    for label, subset in [("common500", [r for r in all_rows if r["branch"] == "common" and int(r["update"]) == args.branch_update]),
                          ("A2000", [r for r in all_rows if r["branch"] == "A_mse" and int(r["update"]) == args.total_updates]),
                          ("B2000", [r for r in all_rows if r["branch"] == "B_balanced" and int(r["update"]) == args.total_updates])]:
        if subset:
            r = subset[-1]
            for key in ["rmse", "mae", "true_hot_mae_exact", "solidus_recall", "solidus_precision", "solidus_f1", "solidus_tp", "solidus_fp", "solidus_fn", "pred_max", "target_max", "full_fp_count", "pred_peak_nearest_true_hot_dist_mm", "pred_peak_is_true_hot_node", "background_mae_excluding_hot", "true_hot_pred_mean", "true_hot_pred_max"]:
                summary[f"{label}_{key}"] = r.get(key)
    # Cleanup
    del model, opt, model_a, opt_a, model_b, opt_b
    if device.type == "cuda":
        torch.cuda.empty_cache()
    group_info = {
        "bucket_counts": groups["bucket_counts"],
        "background_counts": groups["background_counts"],
        "hot_group_n": int(groups["hot_idx"].numel()),
        "neighbor_group_n": int(groups["neighbor_idx"].numel()),
        "background_group_n": int(groups["background_idx"].numel()),
    }
    return all_rows, summary, group_info


def build_report(outdir: Path, summaries: list[dict[str, Any]], group_info: dict[str, Any], args: argparse.Namespace) -> None:
    def ff(x: Any, n: int = 3) -> str:
        try:
            if x is None or x == "":
                return ""
            if isinstance(x, bool):
                return str(x)
            return f"{float(x):.{n}f}"
        except Exception:
            return str(x)
    lines: list[str] = []
    lines.append("# Stage 4 paired supervision diagnostics\n\n")
    lines.append(f"窗口：train start `{args.window_start}` / target step `425`；common branch update `{args.branch_update}`；总 update `{args.total_updates}`。\n\n")
    lines.append("## 节点分组与有效节点精确统计\n\n")
    bc = group_info["bucket_counts"]
    lines.append(f"有效节点总数：**{bc['valid_total']}**。其中 `<100°C` 为 **{bc['lt_100']}**，不是有效节点总数。\n\n")
    lines.append("| bucket | count |\n|---|---:|\n")
    for k in ["lt_100", "100_500", "500_1000", "1000_solidus", "solidus_liquidus", "ge_liquidus", "hot_solidus_total", "liquidus_total"]:
        lines.append(f"| {k} | {bc[k]} |\n")
    lines.append("\n监督分组：\n\n")
    lines.append(f"- 热点组：{group_info['hot_group_n']}\n")
    lines.append(f"- 一跳邻域组：{group_info['neighbor_group_n']}\n")
    lines.append(f"- 固定背景组：{group_info['background_group_n']}\n\n")
    lines.append("## 2000 update 配对结果\n\n")
    lines.append("| Exp | A RMSE | B RMSE | A hot MAE | B hot MAE | A Recall/P/F1 | B Recall/P/F1 | A pred max | B pred max | A FP | B FP | A peak dist mm | B peak dist mm |\n")
    lines.append("|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|\n")
    for s in summaries:
        lines.append(
            f"| {s['experiment']} | {ff(s.get('A2000_rmse'),2)} | {ff(s.get('B2000_rmse'),2)} | "
            f"{ff(s.get('A2000_true_hot_mae_exact'),1)} | {ff(s.get('B2000_true_hot_mae_exact'),1)} | "
            f"{ff(s.get('A2000_solidus_recall'))}/{ff(s.get('A2000_solidus_precision'))}/{ff(s.get('A2000_solidus_f1'))} | "
            f"{ff(s.get('B2000_solidus_recall'))}/{ff(s.get('B2000_solidus_precision'))}/{ff(s.get('B2000_solidus_f1'))} | "
            f"{ff(s.get('A2000_pred_max'),1)} | {ff(s.get('B2000_pred_max'),1)} | "
            f"{s.get('A2000_full_fp_count')} | {s.get('B2000_full_fp_count')} | "
            f"{ff(s.get('A2000_pred_peak_nearest_true_hot_dist_mm'),3)} | {ff(s.get('B2000_pred_peak_nearest_true_hot_dist_mm'),3)} |\n"
        )
    lines.append("\n说明：A=普通全图 MSE；B=从同一个 update-500 checkpoint 分叉后的 1/3 热点 + 1/3 邻域 + 1/3 背景平衡监督。\n\n")
    lines.append("完整曲线：`stage4_curve_E0.csv`、`stage4_curve_E1.csv`、`stage4_curve_E2.csv`；汇总：`stage4_summary.csv`；节点分组：`stage4_node_groups.json`。\n")
    (outdir / "stage4_report.md").write_text("".join(lines), encoding="utf-8")


def plot_curves(outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped: {exc}", flush=True)
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [("rmse", "Full RMSE"), ("true_hot_mae_exact", "True-hot MAE"), ("solidus_recall", "Solidus Recall"), ("pred_max", "Pred max")]
    colors = {"E0": "C0", "E1": "C1", "E2": "C2"}
    styles = {"common": ":", "A_mse": "-", "B_balanced": "--"}
    for exp in ["E0", "E1", "E2"]:
        p = outdir / f"stage4_curve_{exp}.csv"
        if not p.exists():
            continue
        rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
        for branch in ["common", "A_mse", "B_balanced"]:
            br = [r for r in rows if r["branch"] == branch]
            if not br:
                continue
            xs = [int(r["update"]) for r in br]
            for ax, (key, title) in zip(axes.ravel(), specs):
                ys = [float(r[key]) for r in br]
                ax.plot(xs, ys, styles[branch], color=colors[exp], label=f"{exp}-{branch}")
                ax.set_title(title)
                ax.set_xlabel("optimizer updates")
                ax.grid(True, alpha=0.3)
    axes.ravel()[3].axhline(1604.85, color="r", ls="--", lw=1, label="solidus")
    for ax in axes.ravel():
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "stage4_paired_learning_curves.png", dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage 4 paired supervision diagnostics")
    parser.add_argument("--vtu_dir", default="F:/VTU")
    parser.add_argument("--cache_dir", default="data/processed")
    parser.add_argument("--laser_xml", default="configs/laser_paths/5_block_fem_additive_z_scan.xml")
    parser.add_argument("--output_dir", default="results/training_diagnostics/stage4_paired_supervision")
    parser.add_argument("--experiments", nargs="+", default=["E0", "E1", "E2"], choices=list(EXPERIMENTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--graph_device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--window_start", type=int, default=421)
    parser.add_argument("--branch_update", type=int, default=500)
    parser.add_argument("--total_updates", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--background_per_bin", type=int, default=512)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu"))
    graph_device = device if args.graph_device == "auto" else torch.device(args.graph_device)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    ref_cfg = Config.from_yaml(EXPERIMENTS[args.experiments[0]])
    if args.laser_xml:
        ref_cfg.data.laser_xml_path = args.laser_xml
        ref_cfg.data.laser_path_mode = "additive_z_scan"
    graph = load_graph(args.vtu_dir, args.cache_dir, ref_cfg, graph_device, args.no_cache)

    # Build fixed groups once from E0-style target graph; targets/edges are identical across E0/E1/E2.
    tmp_cfg = Config.from_yaml(EXPERIMENTS["E0"])
    if args.laser_xml:
        tmp_cfg.data.laser_xml_path = args.laser_xml
        tmp_cfg.data.laser_path_mode = "additive_z_scan"
    tmp_ds = make_dataset(graph, tmp_cfg, [args.window_start])
    tmp_sample = tmp_ds[0]
    common_groups = build_groups(tmp_sample, tmp_cfg.material.solidus_temp, tmp_cfg.material.liquidus_temp, args.seed, args.background_per_bin)
    group_info_for_json = {
        "window_start": args.window_start,
        "target_step": int(tmp_sample["target_step"]),
        "target_time": float(tmp_sample["target_time"]),
        "bucket_counts": common_groups["bucket_counts"],
        "background_counts": common_groups["background_counts"],
        "hot_group_n": int(common_groups["hot_idx"].numel()),
        "neighbor_group_n": int(common_groups["neighbor_idx"].numel()),
        "background_group_n": int(common_groups["background_idx"].numel()),
        "hot_indices": common_groups["hot_idx"].detach().cpu().tolist(),
        "neighbor_indices_count": int(common_groups["neighbor_idx"].numel()),
        "background_indices": common_groups["background_idx"].detach().cpu().tolist(),
    }
    (outdir / "stage4_node_groups.json").write_text(json.dumps(group_info_for_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = vars(args).copy()
    manifest.update({"device": str(device), "graph_device": str(graph_device)})
    (outdir / "stage4_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summaries: list[dict[str, Any]] = []
    group_info = None
    for exp in args.experiments:
        print(f"=== Stage4 {exp} ===", flush=True)
        _rows, summary, gi = run_experiment(exp, graph, args, device, args.window_start, common_groups, outdir)
        summaries.append(summary)
        group_info = gi
        write_csv(outdir / "stage4_summary.csv", summaries)
    build_report(outdir, summaries, group_info or group_info_for_json, args)
    plot_curves(outdir)
    print(outdir / "stage4_report.md", flush=True)


if __name__ == "__main__":
    main()
