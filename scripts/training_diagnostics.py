"""Training-chain diagnostics for E0/E1/E2 laser-feature ablations.

Stages implemented here are intentionally diagnostic-only:
- fixed train-window parameter update audit (stage 2)
- single-window temperature-only overfit (stage 3)

They keep XML/time/features from the selected experiment configs, initialize from
scratch, and write results under results/training_diagnostics by default.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import graph_cache_path, load_graph_cache, save_graph_cache
from src.config import Config
from src.data.dataset import DEDTemporalDataset
from src.data.preprocessing import load_or_build_split_indices
from src.data.vtu_loader import VTULoader
from src.graph_builder.dynamic_graph import DynamicGraph
from src.model import DTSTPINN

EXPERIMENTS = {
    "E0": Path("configs/feature_e0_baseline.yaml"),
    "E1": Path("configs/feature_e1_laser_distance.yaml"),
    "E2": Path("configs/feature_e2_scan_arrival.yaml"),
}

BIN_SPECS = [
    ("lt_100", None, 100.0),
    ("100_500", 100.0, 500.0),
    ("500_1000", 500.0, 1000.0),
    ("1000_solidus", 1000.0, "solidus"),
    ("solidus_liquidus", "solidus", "liquidus"),
    ("ge_liquidus", "liquidus", None),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float("nan")
        return float(x.detach().cpu().item())
    return float(x)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def resolve_bound(bound: float | str | None, solidus: float, liquidus: float) -> float | None:
    if bound is None:
        return None
    if bound == "solidus":
        return solidus
    if bound == "liquidus":
        return liquidus
    return float(bound)


def tensor_stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
    v = values.detach().float()
    if v.numel() == 0:
        return {f"{prefix}_min": float("nan"), f"{prefix}_mean": float("nan"), f"{prefix}_std": float("nan"), f"{prefix}_max": float("nan")}
    return {
        f"{prefix}_min": float(v.min().cpu()),
        f"{prefix}_mean": float(v.mean().cpu()),
        f"{prefix}_std": float(v.std(unbiased=False).cpu()),
        f"{prefix}_max": float(v.max().cpu()),
    }


def metrics_from_pred(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None,
                      solidus: float, liquidus: float, loss: float | None = None,
                      prefix: str = "") -> dict[str, Any]:
    p = pred.detach().float()
    t0 = target.detach().float()
    if p.numel() == t0.numel():
        p = p.reshape_as(t0)
    p = p.flatten()
    t = t0.flatten()
    if mask is not None:
        m = mask.detach().bool().flatten()
        p = p[m]
        t = t[m]
    err = p - t
    sq = err * err
    abs_err = err.abs()
    true_hot = t >= solidus
    pred_hot = p >= solidus
    tp = int((true_hot & pred_hot).sum().cpu())
    fp = int((~true_hot & pred_hot).sum().cpu())
    fn = int((true_hot & ~pred_hot).sum().cpu())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    total_sq = float(sq.sum().cpu())
    row: dict[str, Any] = {
        f"{prefix}loss": loss,
        f"{prefix}valid_nodes": int(t.numel()),
        f"{prefix}mae": float(abs_err.mean().cpu()) if abs_err.numel() else float("nan"),
        f"{prefix}rmse": float(torch.sqrt(sq.mean()).cpu()) if sq.numel() else float("nan"),
        f"{prefix}hot_mae": float(abs_err[true_hot].mean().cpu()) if true_hot.any() else float("nan"),
        f"{prefix}peak_pred": float(p.max().cpu()) if p.numel() else float("nan"),
        f"{prefix}peak_true": float(t.max().cpu()) if t.numel() else float("nan"),
        f"{prefix}peak_abs_error": float((p.max() - t.max()).abs().cpu()) if p.numel() and t.numel() else float("nan"),
        f"{prefix}solidus_precision": precision,
        f"{prefix}solidus_recall": recall,
        f"{prefix}solidus_f1": f1,
        f"{prefix}solidus_tp": tp,
        f"{prefix}solidus_fp": fp,
        f"{prefix}solidus_fn": fn,
        f"{prefix}sqerr_total": total_sq,
    }
    row.update(tensor_stats(f"{prefix}pred", p))
    row.update(tensor_stats(f"{prefix}target", t))
    for name, lo_raw, hi_raw in BIN_SPECS:
        lo = resolve_bound(lo_raw, solidus, liquidus)
        hi = resolve_bound(hi_raw, solidus, liquidus)
        bmask = torch.ones_like(t, dtype=torch.bool)
        if lo is not None:
            bmask &= t >= lo
        if hi is not None:
            bmask &= t < hi
        b_count = int(bmask.sum().cpu())
        b_sq = float(sq[bmask].sum().cpu()) if b_count else 0.0
        row[f"{prefix}bin_{name}_n"] = b_count
        row[f"{prefix}bin_{name}_sqerr"] = b_sq
        row[f"{prefix}bin_{name}_sqerr_share"] = b_sq / total_sq if total_sq > 0 else 0.0
    return row


def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float()
            total += float((g * g).sum().cpu())
    return math.sqrt(total)


def param_delta(model: torch.nn.Module, before: list[torch.Tensor]) -> tuple[float, float]:
    total = 0.0
    max_abs = 0.0
    with torch.no_grad():
        for p, b in zip(model.parameters(), before):
            d = (p.detach().float().cpu() - b).abs()
            if d.numel():
                total += float((d * d).sum())
                max_abs = max(max_abs, float(d.max()))
    return math.sqrt(total), max_abs


def clone_params_cpu(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().float().cpu().clone() for p in model.parameters()]


def disable_dropout(config: Config) -> None:
    if hasattr(config.model, "spatial"):
        config.model.spatial.dropout = 0.0
    if hasattr(config.model, "temporal"):
        config.model.temporal.dropout = 0.0


def make_dataset(graph: DynamicGraph, config: Config, time_indices: list[int]) -> DEDTemporalDataset:
    return DEDTemporalDataset(
        graph,
        window_size=config.data.window_size,
        predict_steps=config.data.predict_steps,
        time_indices=time_indices,
        use_target_laser_features=config.data.use_target_laser_features,
        laser_feature_radius_mm=config.data.laser_feature_radius_mm,
        laser_feature_along_radius_mm=config.data.laser_feature_along_radius_mm,
        laser_feature_depth_mm=config.data.laser_feature_depth_mm,
        laser_feature_time_scale_to_s=config.data.laser_feature_time_scale_to_s,
        laser_feature_include_laser_coordinates=config.data.laser_feature_include_laser_coordinates,
        laser_feature_include_scan_geometry=config.data.laser_feature_include_scan_geometry,
        laser_feature_include_sweep=config.data.laser_feature_include_sweep,
        laser_feature_include_exposure=config.data.laser_feature_include_exposure,
        laser_feature_exposure_source=config.data.laser_feature_exposure_source,
        laser_feature_exposure_past_steps=config.data.laser_feature_exposure_past_steps,
        laser_feature_exposure_future_steps=config.data.laser_feature_exposure_future_steps,
        laser_feature_exposure_time_decay_s=config.data.laser_feature_exposure_time_decay_s,
        laser_feature_exposure_use_segments=config.data.laser_feature_exposure_use_segments,
        laser_feature_include_exposure_split=config.data.laser_feature_include_exposure_split,
        laser_feature_include_arrival_time=config.data.laser_feature_include_arrival_time,
        laser_feature_arrival_time_decay_s=config.data.laser_feature_arrival_time_decay_s,
        laser_feature_include_neighbor_temp=config.data.laser_feature_include_neighbor_temp,
        laser_feature_include_neighbor_hot_stats=config.data.laser_feature_include_neighbor_hot_stats,
        laser_feature_neighbor_hot_threshold=config.data.laser_feature_neighbor_hot_threshold,
        laser_feature_include_neighbor_warm_stats=config.data.laser_feature_include_neighbor_warm_stats,
        laser_feature_neighbor_warm_threshold=config.data.laser_feature_neighbor_warm_threshold,
        laser_feature_include_body_source=config.data.laser_feature_include_body_source,
        laser_body_radius_mm=config.data.laser_body_radius_mm,
        laser_body_height_mm=config.data.laser_body_height_mm,
        laser_body_radius_front_mm=config.data.laser_body_radius_front_mm,
        laser_body_radius_back_mm=config.data.laser_body_radius_back_mm,
        laser_body_coeff_front=config.data.laser_body_coeff_front,
        laser_body_coeff_back=config.data.laser_body_coeff_back,
        laser_feature_include_path_arrival=config.data.laser_feature_include_path_arrival,
        laser_feature_path_arrival_time_decay_s=config.data.laser_feature_path_arrival_time_decay_s,
        laser_feature_path_arrival_gate_mode=config.data.laser_feature_path_arrival_gate_mode,
        laser_feature_path_arrival_neighbor_tracks=config.data.laser_feature_path_arrival_neighbor_tracks,
        laser_feature_include_path_phase=config.data.laser_feature_include_path_phase,
        laser_feature_include_path_coordinates=config.data.laser_feature_include_path_coordinates,
        laser_feature_include_path_timing=config.data.laser_feature_include_path_timing,
        laser_feature_include_path_body_support=config.data.laser_feature_include_path_body_support,
        laser_feature_include_path_endpoint=config.data.laser_feature_include_path_endpoint,
        laser_feature_endpoint_radius_mm=config.data.laser_feature_endpoint_radius_mm,
        laser_feature_endpoint_time_decay_s=config.data.laser_feature_endpoint_time_decay_s,
        laser_feature_include_path_wake=config.data.laser_feature_include_path_wake,
        laser_feature_wake_cross_radius_mm=config.data.laser_feature_wake_cross_radius_mm,
        laser_feature_wake_tail_decay_mm=config.data.laser_feature_wake_tail_decay_mm,
        laser_feature_wake_lead_decay_mm=config.data.laser_feature_wake_lead_decay_mm,
        laser_feature_wake_time_decay_s=config.data.laser_feature_wake_time_decay_s,
    )


def load_graph(vtu_dir: str, cache_dir: str, config: Config, graph_device: torch.device,
               no_cache: bool = False) -> DynamicGraph:
    loader = VTULoader(vtu_dir)
    if loader.num_steps == 0:
        raise FileNotFoundError(f"No Data-*.vtu files found in {vtu_dir}")
    cache_path = graph_cache_path(cache_dir, vtu_dir, loader, config)
    if (not no_cache) and cache_path.exists():
        print(f"Loading graph cache: {cache_path}", flush=True)
        graph = load_graph_cache(cache_path, config.material)
    else:
        print("Building graph cache (this can take a while)...", flush=True)
        vtu_data = loader.parse_sequence(verbose=True)
        graph = DynamicGraph(
            vtu_data,
            material_props=config.material,
            k_neighbors=config.data.k_neighbors,
            use_mesh_edges=config.data.use_mesh_edges,
        )
        del vtu_data
        gc.collect()
        if not no_cache:
            save_graph_cache(cache_path, graph, {"vtu_dir": str(Path(vtu_dir).resolve())})
    graph.apply_laser_path_config(config.data)
    graph.to(graph_device)
    return graph


def to_device_sample(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    # Graph tensors are already on graph_device. This only handles scalar tensors if needed.
    out = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def forward_loss(model: DTSTPINN, sample: dict[str, Any], amp_enabled: bool,
                 amp_dtype: torch.dtype | None) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    graph_seq = sample["graph_sequence"]
    dt = sample.get("dt", 1.0)
    if isinstance(dt, torch.Tensor):
        dt = float(dt.detach().cpu().item())
    target = sample["target"]
    mask = sample.get("target_mask")
    with autocast("cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_enabled):
        output = model(graph_seq, dt=dt)
        pred = output["T_pred"]
        pred_for_loss = pred.float()
        target_for_loss = target.float()
        if pred_for_loss.numel() == target_for_loss.numel():
            pred_for_loss = pred_for_loss.reshape_as(target_for_loss)
        valid = mask.bool() if mask is not None else torch.ones_like(target_for_loss, dtype=torch.bool)
        diff = pred_for_loss[valid] - target_for_loss[valid]
        loss = (diff * diff).mean()
    return loss, output, pred


def infer_amp_dtype(name: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    name = str(name or "auto").lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name == "auto":
        major, _minor = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.bfloat16


def run_stage2_for_exp(exp: str, config_path: Path, graph: DynamicGraph, args: argparse.Namespace,
                       device: torch.device, selected_start: int) -> dict[str, Any]:
    set_seed(args.seed)
    cfg = Config.from_yaml(config_path)
    if args.laser_xml:
        cfg.data.laser_xml_path = args.laser_xml
        cfg.data.laser_path_mode = "additive_z_scan"
    ds = make_dataset(graph, cfg, [selected_start])
    cfg.model.node_feature_dim = ds.input_feature_dim
    model = DTSTPINN(cfg, cfg.material).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    amp_dtype = infer_amp_dtype(cfg.training.amp_dtype, device)
    amp_enabled = bool(cfg.training.use_amp and device.type == "cuda")
    scaler = GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    sample = to_device_sample(ds[0], device)
    before_params = clone_params_cpu(model)
    opt.zero_grad(set_to_none=True)
    loss0, _out0, pred0 = forward_loss(model, sample, amp_enabled, amp_dtype)
    scale_before = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
    scaler.scale(loss0).backward()
    scaler.unscale_(opt)
    gnorm = grad_norm(model)
    if cfg.training.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
    ret = scaler.step(opt)
    scaler.update()
    scale_after = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        loss1, _out1, pred1 = forward_loss(model, sample, amp_enabled, amp_dtype)
    delta_l2, delta_max = param_delta(model, before_params)
    skipped = (delta_l2 == 0.0) or (scaler.is_enabled() and scale_after < scale_before)
    row: dict[str, Any] = {
        "experiment": exp,
        "stage": "stage2_single_update",
        "window_start": selected_start,
        "target_step": int(sample.get("target_step")),
        "target_time": as_float(sample.get("target_time")),
        "input_feature_dim": ds.input_feature_dim,
        "target_laser_feature_dim": ds.target_laser_feature_dim,
        "lr": cfg.training.lr,
        "weight_decay": cfg.training.weight_decay,
        "grad_clip": cfg.training.grad_clip,
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else "none",
        "amp_scale_before": scale_before,
        "amp_scale_after": scale_after,
        "amp_step_skipped_inferred": skipped,
        "grad_norm_pre_clip": gnorm,
        "param_delta_l2_after_1_update": delta_l2,
        "param_delta_max_after_1_update": delta_max,
        "loss_before": float(loss0.detach().cpu()),
        "loss_after_1_update": float(loss1.detach().cpu()),
        "loss_delta": float(loss1.detach().cpu() - loss0.detach().cpu()),
    }
    row.update(metrics_from_pred(pred0, sample["target"], sample.get("target_mask"), cfg.material.solidus_temp, cfg.material.liquidus_temp, float(loss0.detach().cpu()), prefix="before_"))
    row.update(metrics_from_pred(pred1, sample["target"], sample.get("target_mask"), cfg.material.solidus_temp, cfg.material.liquidus_temp, float(loss1.detach().cpu()), prefix="after1_"))
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def run_stage3_for_exp(exp: str, config_path: Path, graph: DynamicGraph, args: argparse.Namespace,
                       device: torch.device, selected_start: int, outdir: Path) -> dict[str, Any]:
    set_seed(args.seed)
    cfg = Config.from_yaml(config_path)
    if args.laser_xml:
        cfg.data.laser_xml_path = args.laser_xml
        cfg.data.laser_path_mode = "additive_z_scan"
    disable_dropout(cfg)
    cfg.training.use_amp = False
    ds = make_dataset(graph, cfg, [selected_start])
    cfg.model.node_feature_dim = ds.input_feature_dim
    model = DTSTPINN(cfg, cfg.material).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.overfit_lr, weight_decay=cfg.training.weight_decay)
    sample = to_device_sample(ds[0], device)
    initial_params = clone_params_cpu(model)
    curve: list[dict[str, Any]] = []
    log_updates = set([0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 300, 400, args.overfit_updates])
    log_updates.update(range(args.log_every, args.overfit_updates + 1, args.log_every))

    def eval_row(update: int, last_grad_norm: float | None = None) -> dict[str, Any]:
        model.eval()
        with torch.no_grad():
            loss, _out, pred = forward_loss(model, sample, False, None)
        model.train()
        dl2, dmax = param_delta(model, initial_params)
        row: dict[str, Any] = {
            "experiment": exp,
            "stage": "stage3_single_window_overfit",
            "update": update,
            "epoch_equivalent_single_window": update,
            "window_start": selected_start,
            "target_step": int(sample.get("target_step")),
            "target_time": as_float(sample.get("target_time")),
            "input_feature_dim": ds.input_feature_dim,
            "target_laser_feature_dim": ds.target_laser_feature_dim,
            "lr": args.overfit_lr,
            "weight_decay": cfg.training.weight_decay,
            "dropout": 0.0,
            "temperature_only_loss": True,
            "physics_loss_enabled": False,
            "smooth_loss_enabled": False,
            "amp_enabled": False,
            "amp_step_skipped_inferred": False,
            "grad_norm_pre_clip": last_grad_norm,
            "param_delta_l2_from_init": dl2,
            "param_delta_max_from_init": dmax,
        }
        row.update(metrics_from_pred(pred, sample["target"], sample.get("target_mask"), cfg.material.solidus_temp, cfg.material.liquidus_temp, float(loss.detach().cpu()), prefix=""))
        return row

    curve.append(eval_row(0, None))
    last_gnorm = None
    for update in range(1, args.overfit_updates + 1):
        opt.zero_grad(set_to_none=True)
        loss, _out, _pred = forward_loss(model, sample, False, None)
        loss.backward()
        last_gnorm = grad_norm(model)
        if cfg.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        opt.step()
        if update in log_updates:
            curve.append(eval_row(update, last_gnorm))
            print(f"{exp} update {update}/{args.overfit_updates} loss={curve[-1]['loss']:.6g} rmse={curve[-1]['rmse']:.3f} peak_err={curve[-1]['peak_abs_error']:.3f}", flush=True)
    curve_path = outdir / f"stage3_curve_{exp}.csv"
    write_csv(curve_path, curve)
    summary = dict(curve[-1])
    summary["initial_loss"] = curve[0]["loss"]
    summary["initial_rmse"] = curve[0]["rmse"]
    summary["initial_hot_mae"] = curve[0]["hot_mae"]
    summary["initial_peak_abs_error"] = curve[0]["peak_abs_error"]
    summary["loss_reduction_pct"] = (curve[0]["loss"] - curve[-1]["loss"]) / curve[0]["loss"] * 100.0 if curve[0]["loss"] else float("nan")
    summary["rmse_reduction_pct"] = (curve[0]["rmse"] - curve[-1]["rmse"]) / curve[0]["rmse"] * 100.0 if curve[0]["rmse"] else float("nan")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def select_window(graph: DynamicGraph, cfg: Config, split_indices: str, requested_start: int | None) -> int:
    if requested_start is not None:
        return int(requested_start)
    train_idx, _val_idx, _test_idx, source = load_or_build_split_indices(
        graph.num_steps,
        train_ratio=cfg.data.train_split,
        val_ratio=cfg.data.val_split,
        split_indices_path=split_indices,
    )
    ds = make_dataset(graph, cfg, train_idx)
    best_i = None
    best_max = -float("inf")
    for i in range(len(ds)):
        mt = ds.target_max_temperature(i)
        if mt > best_max:
            best_i = i
            best_max = float(mt)
    if best_i is None:
        raise RuntimeError("No train windows available")
    start = int(ds.valid_starts[best_i])
    print(f"Selected train window start={start}, target_max={best_max:.3f}, source={source}", flush=True)
    return start


def build_markdown(outdir: Path, stage2_rows: list[dict[str, Any]], stage3_rows: list[dict[str, Any]],
                   selected_start: int) -> None:
    lines: list[str] = []
    lines.append("# Training diagnostics stages 2-3\n\n")
    lines.append(f"诊断窗口：train start `{selected_start}`；未使用测试集。\n\n")
    if stage2_rows:
        lines.append("## Stage 2 单窗口 1 次 optimizer update 审计\n\n")
        lines.append("| Exp | input dim | loss before | loss after | grad norm | param Δ L2 | AMP skipped | pred max before→after | true max | hot recall after |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|---:|\n")
        for r in stage2_rows:
            lines.append(
                f"| {r['experiment']} | {r['input_feature_dim']} | {r['loss_before']:.4g} | {r['loss_after_1_update']:.4g} | "
                f"{r['grad_norm_pre_clip']:.4g} | {r['param_delta_l2_after_1_update']:.4g} | {r['amp_step_skipped_inferred']} | "
                f"{r['before_pred_max']:.3f}→{r['after1_pred_max']:.3f} | {r['before_target_max']:.3f} | {r['after1_solidus_recall']:.3f} |\n"
            )
        lines.append("\nCSV：`stage2_single_update.csv`。\n\n")
    if stage3_rows:
        lines.append("## Stage 3 单窗口 temperature-only overfit\n\n")
        lines.append("设置：from scratch，dropout=0，关闭物理损失和平滑项，固定 LR，横轴为 optimizer update。\n\n")
        lines.append("| Exp | updates | loss init→final | RMSE init→final | MAE final | hot MAE init→final | peak err init→final | recall final | pred max final | true max | loss ↓ |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in stage3_rows:
            lines.append(
                f"| {r['experiment']} | {r['update']} | {r['initial_loss']:.4g}→{r['loss']:.4g} | "
                f"{r['initial_rmse']:.3f}→{r['rmse']:.3f} | {r['mae']:.3f} | "
                f"{r['initial_hot_mae']:.3f}→{r['hot_mae']:.3f} | {r['initial_peak_abs_error']:.3f}→{r['peak_abs_error']:.3f} | "
                f"{r['solidus_recall']:.3f} | {r['pred_max']:.3f} | {r['target_max']:.3f} | {r['loss_reduction_pct']:.1f}% |\n"
            )
        lines.append("\n学习曲线 CSV：`stage3_curve_E0.csv`、`stage3_curve_E1.csv`、`stage3_curve_E2.csv`。\n")
    (outdir / "stage2_stage3_summary.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training diagnostics stages 2-3")
    parser.add_argument("--vtu_dir", default="F:/VTU")
    parser.add_argument("--cache_dir", default="data/processed")
    parser.add_argument("--split_indices", default="tmp/protocol_checks/small_learning_train_only_split.json")
    parser.add_argument("--laser_xml", default="configs/laser_paths/5_block_fem_additive_z_scan.xml")
    parser.add_argument("--output_dir", default="results/training_diagnostics")
    parser.add_argument("--experiments", nargs="+", default=["E0", "E1", "E2"], choices=list(EXPERIMENTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--graph_device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--window_start", type=int, default=None)
    parser.add_argument("--stage2", action="store_true")
    parser.add_argument("--stage3", action="store_true")
    parser.add_argument("--overfit_updates", type=int, default=500)
    parser.add_argument("--overfit_lr", type=float, default=1.0e-3)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()
    if not args.stage2 and not args.stage3:
        args.stage2 = True
        args.stage3 = True
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
    selected_start = select_window(graph, ref_cfg, args.split_indices, args.window_start)
    manifest = {
        "selected_start": selected_start,
        "split_indices": args.split_indices,
        "vtu_dir": args.vtu_dir,
        "laser_xml": args.laser_xml,
        "device": str(device),
        "graph_device": str(graph_device),
        "experiments": args.experiments,
        "seed": args.seed,
        "overfit_updates": args.overfit_updates,
        "overfit_lr": args.overfit_lr,
    }
    (outdir / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stage2_rows: list[dict[str, Any]] = []
    if args.stage2:
        for exp in args.experiments:
            print(f"Stage2 {exp}", flush=True)
            row = run_stage2_for_exp(exp, EXPERIMENTS[exp], graph, args, device, selected_start)
            stage2_rows.append(row)
        write_csv(outdir / "stage2_single_update.csv", stage2_rows)
    stage3_rows: list[dict[str, Any]] = []
    if args.stage3:
        for exp in args.experiments:
            print(f"Stage3 {exp}", flush=True)
            row = run_stage3_for_exp(exp, EXPERIMENTS[exp], graph, args, device, selected_start, outdir)
            stage3_rows.append(row)
        write_csv(outdir / "stage3_overfit_summary.csv", stage3_rows)
    build_markdown(outdir, stage2_rows, stage3_rows, selected_start)
    print(outdir / "stage2_stage3_summary.md")


if __name__ == "__main__":
    main()
