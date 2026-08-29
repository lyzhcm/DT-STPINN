"""Temperature-binned checkpoint evaluation for DT-STPINN.

Loads a trained checkpoint, runs inference on the test split, and reports
per-bin metrics across six temperature ranges so that hotspot-blind models
can be diagnosed before any retraining.

Usage:
    python scripts/evaluate_checkpoint.py ^
      --config configs\paper1.yaml ^
      --checkpoint logs\paper1_temperature\best_model.pt ^
      --vtu_dir data\raw ^
      --output_dir results\paper1_temperature

Outputs:
    - Terminal table with per-bin metrics (MAE, RMSE, Bias, percentiles, max error)
    - High-temperature recall / precision / F1 / IoU above solidus
    - Per-timestep max-temperature comparison
    - Worst-case node diagnostic (input window, neighbors, laser distance, activation)
    - VTU files for the worst timestep (target, prediction, absolute error)
    - Full JSON report saved to <output_dir>/evaluation_report.json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.vtu_loader import VTULoader
from src.data.dataset import DEDTemporalDataset, collate_temporal_batch
from src.data.preprocessing import split_indices
from src.graph_builder.dynamic_graph import DynamicGraph
from src.model import DTSTPINN
from src.utils.laser_path import AdditiveZScanPath
from src.utils.visualization import write_temperature_vtu


def graph_cache_path(cache_dir: str | Path, vtu_dir: str | Path,
                     loader: VTULoader, config: Config) -> Path:
    cache_root = Path(cache_dir)
    h = hashlib.sha256()
    h.update(str(Path(vtu_dir).resolve()).encode("utf-8"))
    h.update(f"k={config.data.k_neighbors};mesh={config.data.use_mesh_edges};".encode("utf-8"))
    h.update(f"kmat={config.material.thermal_conductivity};".encode("utf-8"))

    for fp in loader.files:
        st = fp.stat()
        h.update(f"{fp.name}:{st.st_size}:{st.st_mtime_ns}\n".encode("utf-8"))

    return cache_root / f"dynamic_graph_{h.hexdigest()[:16]}.pt"


def load_graph_cache(path: Path, material_props):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    return DynamicGraph.from_cache_dict(state["graph"], material_props)


def save_graph_cache(path: Path, graph: DynamicGraph, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "metadata": metadata,
        "graph": graph.to_cache_dict(),
    }, path)

# ---------------------------------------------------------------------------
# Temperature bins: [low_bound, high_bound) with the last bin inclusive
# ---------------------------------------------------------------------------
BIN_DEFS = [
    ("<= 100 °C",       -float("inf"), 100.0),
    ("100–500 °C",       100.0,        500.0),
    ("500–1000 °C",      500.0,       1000.0),
    ("1000–1604.85 °C", 1000.0,       1604.85),
    ("1604.85–1654.85 °C (mushy)", 1604.85, 1654.85),
    (">= 1654.85 °C (liquid)",     1654.85, float("inf")),
]


def _bin_mask(target: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    if lo == -float("inf"):
        return target <= hi
    if hi == float("inf"):
        return target >= lo
    return (target >= lo) & (target < hi)


def _safe_q(arr: np.ndarray, q: float) -> float:
    """Quantile that returns NaN for empty arrays."""
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q, method="linear"))


def compute_bin_metrics(
    pred: np.ndarray, target: np.ndarray, solidus: float, liquidus: float
) -> dict:
    """Compute per-bin regression + detection metrics.

    Returns a dict keyed by bin label, each value a dict of metrics, plus
    an ``"all"`` entry for the pooled results.
    """
    results: dict[str, dict] = {}

    all_preds = []
    all_targets = []

    for label, lo, hi in BIN_DEFS:
        m = _bin_mask(torch.from_numpy(target), lo, hi).numpy()
        n = int(m.sum())
        if n == 0:
            results[label] = {"count": 0, "proportion": 0.0}
            continue

        p = pred[m]
        t = target[m]
        all_preds.append(p)
        all_targets.append(t)

        error = p - t
        abs_err = np.abs(error)

        results[label] = {
            "count": n,
            "proportion": float(n / len(target)),
            "MAE": float(np.mean(abs_err)),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "Bias": float(np.mean(error)),
            "AbsErrorP50": _safe_q(abs_err, 0.50),
            "AbsErrorP90": _safe_q(abs_err, 0.90),
            "AbsErrorP95": _safe_q(abs_err, 0.95),
            "AbsErrorP99": _safe_q(abs_err, 0.99),
            "MaxError": float(np.max(abs_err)) if n > 0 else float("nan"),
        }

    # --- high-temperature detection metrics (solidus as threshold) ---
    pred_t = torch.from_numpy(pred)
    targ_t = torch.from_numpy(target)
    true_high = targ_t >= solidus
    pred_high = pred_t >= solidus

    tp = (pred_high & true_high).sum().item()
    fp = (pred_high & ~true_high).sum().item()
    fn = (~pred_high & true_high).sum().item()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    intersection = (pred_high & true_high).sum().item()
    union = (pred_high | true_high).sum().item()
    iou = intersection / union if union > 0 else 0.0

    results["_detection"] = {
        "solidus_threshold": solidus,
        "liquidus_threshold": liquidus,
        "recall_above_solidus": recall,
        "precision_above_solidus": precision,
        "f1_above_solidus": f1,
        "iou_above_solidus": iou,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }

    # --- pooled "all" entry ---
    if all_preds:
        ap = np.concatenate(all_preds)
        at = np.concatenate(all_targets)
        ae = np.abs(ap - at)
        results["all"] = {
            "count": int(ap.size),
            "MAE": float(np.mean(ae)),
            "RMSE": float(np.sqrt(np.mean((ap - at) ** 2))),
            "Bias": float(np.mean(ap - at)),
            "AbsErrorP50": _safe_q(ae, 0.50),
            "AbsErrorP90": _safe_q(ae, 0.90),
            "AbsErrorP95": _safe_q(ae, 0.95),
            "AbsErrorP99": _safe_q(ae, 0.99),
            "MaxError": float(np.max(ae)),
        }

    return results


def compute_score_detection_metrics(
    score: np.ndarray, target: np.ndarray, target_threshold: float,
    score_threshold: float,
) -> dict:
    """Compute binary detection metrics from arbitrary scores/probabilities."""
    pred_high = score >= score_threshold
    true_high = target >= target_threshold

    tp = int(np.logical_and(pred_high, true_high).sum())
    fp = int(np.logical_and(pred_high, ~true_high).sum())
    fn = int(np.logical_and(~pred_high, true_high).sum())
    union = int(np.logical_or(pred_high, true_high).sum())

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / union if union > 0 else 0.0

    return {
        "target_threshold": float(target_threshold),
        "score_threshold": float(score_threshold),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "iou": float(iou),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def compute_regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute the canonical pooled regression metrics for experiment reports."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(target)
    if not np.any(valid):
        return {
            "count": 0,
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "Bias": float("nan"),
            "AbsErrorP50": float("nan"),
            "AbsErrorP90": float("nan"),
            "AbsErrorP95": float("nan"),
            "AbsErrorP99": float("nan"),
            "MaxError": float("nan"),
        }

    pred = pred[valid]
    target = target[valid]
    error = pred - target
    abs_error = np.abs(error)
    return {
        "count": int(pred.size),
        "MAE": float(np.mean(abs_error)),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "Bias": float(np.mean(error)),
        "AbsErrorP50": _safe_q(abs_error, 0.50),
        "AbsErrorP90": _safe_q(abs_error, 0.90),
        "AbsErrorP95": _safe_q(abs_error, 0.95),
        "AbsErrorP99": _safe_q(abs_error, 0.99),
        "MaxError": float(np.max(abs_error)),
    }


def flatten_detection_metrics(prefix: str, metrics: dict) -> dict:
    """Return Trainer-compatible names for a binary detection metric block."""
    return {
        f"TempRecallAbove{prefix}": metrics["recall"],
        f"TempPrecisionAbove{prefix}": metrics["precision"],
        f"TempF1Above{prefix}": metrics["f1"],
        f"TempIoUAbove{prefix}": metrics["iou"],
        f"TempTPAbove{prefix}": metrics["true_positive"],
        f"TempFPAbove{prefix}": metrics["false_positive"],
        f"TempFNAbove{prefix}": metrics["false_negative"],
    }


def time_scale_to_seconds(config) -> float:
    physics = getattr(config, "physics", None)
    return float(getattr(physics, "time_scale_to_s", 1.0e-3))


def build_path_model(config):
    mode = str(getattr(config.data, "laser_path_mode", "estimated"))
    if mode == "estimated":
        return None
    try:
        return AdditiveZScanPath.from_config(config)
    except Exception as exc:  # noqa: BLE001 - diagnostic output should survive partial configs.
        print(f"  [WARN] Could not build additive_z_scan path model: {exc}")
        return None


def laser_region_mask_for_record(rec: dict, config) -> torch.Tensor | None:
    """Mask valid nodes inside the target-step laser influence region."""
    target_laser_pos = rec.get("target_laser_pos")
    if target_laser_pos is None:
        return None

    coords = rec["coords"].detach().float().cpu()
    valid = rec["valid_mask"].detach().bool().cpu()
    laser = target_laser_pos.detach().float().reshape(1, 3).cpu()
    rel = coords - laser
    radius = float(getattr(config.data, "laser_feature_radius_mm", 0.4))
    depth = float(getattr(config.data, "laser_feature_depth_mm", 0.1))
    dxy = torch.linalg.vector_norm(rel[:, :2], dim=1)
    dz = rel[:, 2].abs()
    return valid & (dxy <= radius) & (dz <= depth)


def compute_laser_region_metrics(records: list[dict], config) -> dict:
    """Compute pooled metrics inside the target-step laser influence region."""
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for rec in records:
        mask = laser_region_mask_for_record(rec, config)
        if mask is None or not bool(mask.any()):
            continue
        valid_positions = torch.nonzero(rec["valid_mask"], as_tuple=False).reshape(-1)
        region_valid_idx = torch.nonzero(mask[valid_positions], as_tuple=False).reshape(-1)
        if region_valid_idx.numel() == 0:
            continue
        preds.append(rec["pred"][region_valid_idx].detach().cpu().numpy())
        targets.append(rec["target"][region_valid_idx].detach().cpu().numpy())

    if not preds:
        return {"count": 0}
    return compute_regression_metrics(np.concatenate(preds), np.concatenate(targets))


def path_features_for_node(path_model, config, coord_mm: np.ndarray,
                           raw_time: float) -> dict:
    """Return selected process-path features for one node/time pair."""
    if path_model is None:
        return {}
    try:
        features, columns = path_model.node_process_features(
            np.asarray(coord_mm, dtype=np.float64).reshape(1, 3),
            float(raw_time),
        )
    except Exception as exc:  # noqa: BLE001 - keep evaluation robust for old configs.
        return {"laser_path_error": str(exc)}

    row = features[0]
    values = {name: float(row[i]) for i, name in enumerate(columns)}
    return {
        "laser_position_mm": [
            values.get("laser_x_mm"),
            values.get("laser_y_mm"),
            values.get("laser_z_mm"),
        ],
        "distance_to_laser_mm": values.get("distance_to_laser_mm"),
        "current_along_mm": values.get("current_along_mm"),
        "current_cross_mm": values.get("current_cross_mm"),
        "line_along_mm": values.get("line_along_mm"),
        "line_cross_mm": values.get("line_cross_mm"),
        "time_to_arrival_s": values.get("time_to_arrival_s"),
        "arrival_raw_time": values.get("arrival_raw_time"),
        "layer_idx": values.get("layer_idx"),
        "track_physical": values.get("track_physical"),
        "track_program": values.get("track_program"),
        "direction_sign": values.get("direction_sign"),
        "in_track_neighborhood": values.get("in_track_neighborhood"),
    }


def collect_worst_cases(records: list[dict], config, *, top_k: int = 10) -> list[dict]:
    """Collect the worst pointwise errors with path-alignment diagnostics."""
    candidates: list[dict] = []
    path_model = build_path_model(config)
    seconds_scale = time_scale_to_seconds(config)

    optional_fields = [
        "hotspot_prob",
        "hotspot_specialist_temp",
        "hotspot_specialist_gate",
        "process_gate",
        "laser_target_heat_gate",
        "laser_body_heat_gate",
        "laser_path_body_heat_gate",
        "laser_path_wake_gate",
        "laser_sweep_heat_gate",
        "laser_arrival_gate",
        "laser_path_active_gate",
        "laser_path_pre_arrival_gate",
        "laser_path_post_arrival_gate",
        "laser_endpoint_gate",
        "laser_path_program_track_gate",
        "laser_path_elapsed_gate",
        "laser_path_time_until_gate",
        "laser_path_scanned_gate",
        "laser_path_cooling_tail_gate",
        "neighbor_hot_gate",
        "hotspot_laser_prior_temp",
        "hotspot_laser_prior_gate",
        "laser_residual_prior_gate",
        "laser_residual_learned_gate",
        "laser_residual_control_gate",
        "laser_residual_cold_start_gate",
        "laser_residual_gate",
        "laser_residual_delta",
        "laser_residual_boost",
        "cold_to_hot_gate",
    ]

    for rec in records:
        abs_err = (rec["pred"] - rec["target"]).abs()
        if abs_err.numel() == 0:
            continue
        local_k = min(top_k, int(abs_err.numel()))
        values, indices = torch.topk(abs_err, k=local_k)
        valid_positions = torch.nonzero(rec["valid_mask"], as_tuple=False).reshape(-1)
        for rank_value, valid_idx in zip(values.tolist(), indices.tolist()):
            node_idx = int(valid_positions[int(valid_idx)].item())
            coord = rec["coords"][node_idx].detach().float().cpu().numpy()
            raw_time = float(rec["target_time"])
            row = {
                "abs_error": float(rank_value),
                "prediction": float(rec["pred"][valid_idx].item()),
                "target": float(rec["target"][valid_idx].item()),
                "node_index": node_idx,
                "target_step": int(rec["target_step"]),
                "target_time_raw": raw_time,
                "target_time_s": raw_time * seconds_scale if raw_time > 0 else None,
                "coord_mm": [float(v) for v in coord.tolist()],
            }
            target_laser_pos = rec.get("target_laser_pos")
            if target_laser_pos is not None:
                laser = target_laser_pos.detach().float().cpu().numpy().reshape(3)
                row["target_laser_position_mm"] = [float(v) for v in laser.tolist()]
                row["target_laser_distance_mm"] = float(np.linalg.norm(coord - laser))
            row.update(path_features_for_node(path_model, config, coord, raw_time))
            for field in optional_fields:
                value = rec.get(field)
                if value is not None:
                    row[field] = float(value[valid_idx].item())
            candidates.append(row)

    candidates.sort(key=lambda item: item["abs_error"], reverse=True)
    return candidates[:top_k]


def find_neighbors(edge_index: torch.Tensor, center_node: int,
                   order: int = 2) -> dict[int, list[int]]:
    """Return 1st- and 2nd-order neighbour sets for a center node.

    Args:
        edge_index: [2, E] long tensor.
        center_node: node index.

    Returns:
        dict with keys ``1`` and ``2`` mapping to sorted lists of node indices.
    """
    ei = edge_index.cpu()
    adj: dict[int, set[int]] = {}
    src = ei[0].tolist()
    dst = ei[1].tolist()
    for s, d in zip(src, dst):
        adj.setdefault(s, set()).add(d)
        adj.setdefault(d, set()).add(s)

    first = sorted(adj.get(center_node, set()))
    if order < 2:
        return {1: first}

    second: set[int] = set()
    for n in first:
        second.update(adj.get(n, set()))
    second.discard(center_node)
    second.difference_update(first)
    return {1: first, 2: sorted(second)}


def collect_predictions(model, test_loader, device, dtype) -> list[dict]:
    """Run inference over the test set and collect per-sample results."""
    model.eval()
    records: list[dict] = []

    for batch in tqdm(test_loader, desc="Evaluating", dynamic_ncols=True):
        graph_seq = batch["graph_sequence"]
        if isinstance(graph_seq, list) and len(graph_seq) > 0 and isinstance(graph_seq[0], list):
            graph_seq = graph_seq[0]

        # Move graph sequence to device
        for d in graph_seq:
            for attr in ["x", "y", "edge_index", "edge_attr", "mask", "coords",
                         "boundary", "laser_pos"]:
                val = getattr(d, attr, None)
                if isinstance(val, torch.Tensor):
                    setattr(d, attr, val.to(device))

        target = batch["target"].to(device)
        mask = batch.get("target_mask")
        if mask is not None:
            mask = mask.to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
                output = model(graph_seq, dt=batch.get("dt", 1.0))
        T_pred = output["T_pred"].detach().float()
        hotspot_prob = None
        if output.get("hotspot_logit") is not None:
            hotspot_prob = torch.sigmoid(output["hotspot_logit"].detach().float())
        hotspot_specialist_temp = None
        if output.get("hotspot_specialist_temp") is not None:
            hotspot_specialist_temp = output["hotspot_specialist_temp"].detach().float()
        hotspot_specialist_gate = None
        if output.get("hotspot_specialist_gate") is not None:
            hotspot_specialist_gate = output["hotspot_specialist_gate"].detach().float()
        process_gate = None
        if output.get("process_gate") is not None:
            process_gate = output["process_gate"].detach().float()
        laser_target_heat_gate = None
        if output.get("laser_target_heat_gate") is not None:
            laser_target_heat_gate = output["laser_target_heat_gate"].detach().float()
        laser_body_heat_gate = None
        if output.get("laser_body_heat_gate") is not None:
            laser_body_heat_gate = output["laser_body_heat_gate"].detach().float()
        laser_path_body_heat_gate = None
        if output.get("laser_path_body_heat_gate") is not None:
            laser_path_body_heat_gate = output["laser_path_body_heat_gate"].detach().float()
        laser_path_wake_gate = None
        if output.get("laser_path_wake_gate") is not None:
            laser_path_wake_gate = output["laser_path_wake_gate"].detach().float()
        laser_sweep_heat_gate = None
        if output.get("laser_sweep_heat_gate") is not None:
            laser_sweep_heat_gate = output["laser_sweep_heat_gate"].detach().float()
        laser_arrival_gate = None
        if output.get("laser_arrival_gate") is not None:
            laser_arrival_gate = output["laser_arrival_gate"].detach().float()
        laser_path_active_gate = None
        if output.get("laser_path_active_gate") is not None:
            laser_path_active_gate = output["laser_path_active_gate"].detach().float()
        laser_path_pre_arrival_gate = None
        if output.get("laser_path_pre_arrival_gate") is not None:
            laser_path_pre_arrival_gate = output["laser_path_pre_arrival_gate"].detach().float()
        laser_path_post_arrival_gate = None
        if output.get("laser_path_post_arrival_gate") is not None:
            laser_path_post_arrival_gate = output["laser_path_post_arrival_gate"].detach().float()
        laser_endpoint_gate = None
        if output.get("laser_endpoint_gate") is not None:
            laser_endpoint_gate = output["laser_endpoint_gate"].detach().float()
        laser_path_program_track_gate = None
        if output.get("laser_path_program_track_gate") is not None:
            laser_path_program_track_gate = output["laser_path_program_track_gate"].detach().float()
        laser_path_elapsed_gate = None
        if output.get("laser_path_elapsed_gate") is not None:
            laser_path_elapsed_gate = output["laser_path_elapsed_gate"].detach().float()
        laser_path_time_until_gate = None
        if output.get("laser_path_time_until_gate") is not None:
            laser_path_time_until_gate = output["laser_path_time_until_gate"].detach().float()
        laser_path_scanned_gate = None
        if output.get("laser_path_scanned_gate") is not None:
            laser_path_scanned_gate = output["laser_path_scanned_gate"].detach().float()
        laser_path_cooling_tail_gate = None
        if output.get("laser_path_cooling_tail_gate") is not None:
            laser_path_cooling_tail_gate = output["laser_path_cooling_tail_gate"].detach().float()
        neighbor_hot_gate = None
        if output.get("neighbor_hot_gate") is not None:
            neighbor_hot_gate = output["neighbor_hot_gate"].detach().float()
        hotspot_laser_prior_temp = None
        if output.get("hotspot_laser_prior_temp") is not None:
            hotspot_laser_prior_temp = output["hotspot_laser_prior_temp"].detach().float()
        hotspot_laser_prior_gate = None
        if output.get("hotspot_laser_prior_gate") is not None:
            hotspot_laser_prior_gate = output["hotspot_laser_prior_gate"].detach().float()
        laser_residual_prior_gate = None
        if output.get("laser_residual_prior_gate") is not None:
            laser_residual_prior_gate = output["laser_residual_prior_gate"].detach().float()
        laser_residual_learned_gate = None
        if output.get("laser_residual_learned_gate") is not None:
            laser_residual_learned_gate = output["laser_residual_learned_gate"].detach().float()
        laser_residual_control_gate = None
        if output.get("laser_residual_control_gate") is not None:
            laser_residual_control_gate = output["laser_residual_control_gate"].detach().float()
        laser_residual_cold_start_gate = None
        if output.get("laser_residual_cold_start_gate") is not None:
            laser_residual_cold_start_gate = output["laser_residual_cold_start_gate"].detach().float()
        laser_residual_gate = None
        if output.get("laser_residual_gate") is not None:
            laser_residual_gate = output["laser_residual_gate"].detach().float()
        laser_residual_delta = None
        if output.get("laser_residual_delta") is not None:
            laser_residual_delta = output["laser_residual_delta"].detach().float()
        laser_residual_boost = None
        if output.get("laser_residual_boost") is not None:
            laser_residual_boost = output["laser_residual_boost"].detach().float()
        cold_to_hot_gate = None
        if output.get("cold_to_hot_gate") is not None:
            cold_to_hot_gate = output["cold_to_hot_gate"].detach().float()

        # Flatten
        pred_flat = T_pred.reshape(-1).cpu()
        target_flat = target.detach().float().reshape(-1).cpu()
        hot_prob_flat = hotspot_prob.reshape(-1).cpu() if hotspot_prob is not None else None
        spec_temp_flat = (
            hotspot_specialist_temp.reshape(-1).cpu()
            if hotspot_specialist_temp is not None else None
        )
        spec_gate_flat = (
            hotspot_specialist_gate.reshape(-1).cpu()
            if hotspot_specialist_gate is not None else None
        )
        process_gate_flat = (
            process_gate.reshape(-1).cpu()
            if process_gate is not None else None
        )
        target_heat_gate_flat = (
            laser_target_heat_gate.reshape(-1).cpu()
            if laser_target_heat_gate is not None else None
        )
        body_heat_gate_flat = (
            laser_body_heat_gate.reshape(-1).cpu()
            if laser_body_heat_gate is not None else None
        )
        path_body_heat_gate_flat = (
            laser_path_body_heat_gate.reshape(-1).cpu()
            if laser_path_body_heat_gate is not None else None
        )
        path_wake_gate_flat = (
            laser_path_wake_gate.reshape(-1).cpu()
            if laser_path_wake_gate is not None else None
        )
        sweep_heat_gate_flat = (
            laser_sweep_heat_gate.reshape(-1).cpu()
            if laser_sweep_heat_gate is not None else None
        )
        arrival_gate_flat = (
            laser_arrival_gate.reshape(-1).cpu()
            if laser_arrival_gate is not None else None
        )
        path_active_gate_flat = (
            laser_path_active_gate.reshape(-1).cpu()
            if laser_path_active_gate is not None else None
        )
        path_pre_arrival_gate_flat = (
            laser_path_pre_arrival_gate.reshape(-1).cpu()
            if laser_path_pre_arrival_gate is not None else None
        )
        path_post_arrival_gate_flat = (
            laser_path_post_arrival_gate.reshape(-1).cpu()
            if laser_path_post_arrival_gate is not None else None
        )
        endpoint_gate_flat = (
            laser_endpoint_gate.reshape(-1).cpu()
            if laser_endpoint_gate is not None else None
        )
        path_program_track_gate_flat = (
            laser_path_program_track_gate.reshape(-1).cpu()
            if laser_path_program_track_gate is not None else None
        )
        path_elapsed_gate_flat = (
            laser_path_elapsed_gate.reshape(-1).cpu()
            if laser_path_elapsed_gate is not None else None
        )
        path_time_until_gate_flat = (
            laser_path_time_until_gate.reshape(-1).cpu()
            if laser_path_time_until_gate is not None else None
        )
        path_scanned_gate_flat = (
            laser_path_scanned_gate.reshape(-1).cpu()
            if laser_path_scanned_gate is not None else None
        )
        path_cooling_tail_gate_flat = (
            laser_path_cooling_tail_gate.reshape(-1).cpu()
            if laser_path_cooling_tail_gate is not None else None
        )
        neighbor_gate_flat = (
            neighbor_hot_gate.reshape(-1).cpu()
            if neighbor_hot_gate is not None else None
        )
        prior_temp_flat = (
            hotspot_laser_prior_temp.reshape(-1).cpu()
            if hotspot_laser_prior_temp is not None else None
        )
        prior_gate_flat = (
            hotspot_laser_prior_gate.reshape(-1).cpu()
            if hotspot_laser_prior_gate is not None else None
        )
        residual_prior_gate_flat = (
            laser_residual_prior_gate.reshape(-1).cpu()
            if laser_residual_prior_gate is not None else None
        )
        residual_learned_gate_flat = (
            laser_residual_learned_gate.reshape(-1).cpu()
            if laser_residual_learned_gate is not None else None
        )
        residual_control_gate_flat = (
            laser_residual_control_gate.reshape(-1).cpu()
            if laser_residual_control_gate is not None else None
        )
        residual_cold_start_gate_flat = (
            laser_residual_cold_start_gate.reshape(-1).cpu()
            if laser_residual_cold_start_gate is not None else None
        )
        residual_gate_flat = (
            laser_residual_gate.reshape(-1).cpu()
            if laser_residual_gate is not None else None
        )
        residual_delta_flat = (
            laser_residual_delta.reshape(-1).cpu()
            if laser_residual_delta is not None else None
        )
        residual_boost_flat = (
            laser_residual_boost.reshape(-1).cpu()
            if laser_residual_boost is not None else None
        )
        cold_to_hot_gate_flat = (
            cold_to_hot_gate.reshape(-1).cpu()
            if cold_to_hot_gate is not None else None
        )
        coords = graph_seq[-1].coords.detach().float().cpu()

        # Build valid mask
        valid = torch.isfinite(pred_flat) & torch.isfinite(target_flat)
        if hot_prob_flat is not None:
            valid &= torch.isfinite(hot_prob_flat)
        if spec_temp_flat is not None:
            valid &= torch.isfinite(spec_temp_flat)
        if spec_gate_flat is not None:
            valid &= torch.isfinite(spec_gate_flat)
        if process_gate_flat is not None:
            valid &= torch.isfinite(process_gate_flat)
        if target_heat_gate_flat is not None:
            valid &= torch.isfinite(target_heat_gate_flat)
        if body_heat_gate_flat is not None:
            valid &= torch.isfinite(body_heat_gate_flat)
        if sweep_heat_gate_flat is not None:
            valid &= torch.isfinite(sweep_heat_gate_flat)
        if arrival_gate_flat is not None:
            valid &= torch.isfinite(arrival_gate_flat)
        if path_active_gate_flat is not None:
            valid &= torch.isfinite(path_active_gate_flat)
        if path_pre_arrival_gate_flat is not None:
            valid &= torch.isfinite(path_pre_arrival_gate_flat)
        if path_post_arrival_gate_flat is not None:
            valid &= torch.isfinite(path_post_arrival_gate_flat)
        if endpoint_gate_flat is not None:
            valid &= torch.isfinite(endpoint_gate_flat)
        if path_program_track_gate_flat is not None:
            valid &= torch.isfinite(path_program_track_gate_flat)
        if path_elapsed_gate_flat is not None:
            valid &= torch.isfinite(path_elapsed_gate_flat)
        if path_time_until_gate_flat is not None:
            valid &= torch.isfinite(path_time_until_gate_flat)
        if neighbor_gate_flat is not None:
            valid &= torch.isfinite(neighbor_gate_flat)
        if prior_temp_flat is not None:
            valid &= torch.isfinite(prior_temp_flat)
        if prior_gate_flat is not None:
            valid &= torch.isfinite(prior_gate_flat)
        if residual_prior_gate_flat is not None:
            valid &= torch.isfinite(residual_prior_gate_flat)
        if residual_learned_gate_flat is not None:
            valid &= torch.isfinite(residual_learned_gate_flat)
        if residual_control_gate_flat is not None:
            valid &= torch.isfinite(residual_control_gate_flat)
        if residual_cold_start_gate_flat is not None:
            valid &= torch.isfinite(residual_cold_start_gate_flat)
        if residual_gate_flat is not None:
            valid &= torch.isfinite(residual_gate_flat)
        if residual_delta_flat is not None:
            valid &= torch.isfinite(residual_delta_flat)
        if residual_boost_flat is not None:
            valid &= torch.isfinite(residual_boost_flat)
        if cold_to_hot_gate_flat is not None:
            valid &= torch.isfinite(cold_to_hot_gate_flat)
        if mask is not None:
            active = mask.detach().bool().reshape(-1).cpu()
            if active.numel() != target_flat.numel():
                repeats = target_flat.numel() // active.numel()
                active = active.repeat_interleave(repeats)
            valid &= active

        if not valid.any():
            continue

        target_step = batch.get("target_step", -1)
        if isinstance(target_step, torch.Tensor):
            target_step = int(target_step.item())
        target_time = batch.get("target_time", -1.0)
        if isinstance(target_time, torch.Tensor):
            target_time = float(target_time.item())
        target_laser_pos = batch.get("target_laser_pos")
        if isinstance(target_laser_pos, torch.Tensor):
            target_laser_pos = target_laser_pos.detach().float().reshape(-1).cpu()

        # Collect input window temperatures (ground truth for context)
        window_temps = []
        for g in graph_seq:
            y = g.y.detach().float().reshape(-1).cpu()
            window_temps.append(y)

        records.append({
            "pred": pred_flat[valid],
            "target": target_flat[valid],
            "hotspot_prob": hot_prob_flat[valid] if hot_prob_flat is not None else None,
            "hotspot_specialist_temp": spec_temp_flat[valid] if spec_temp_flat is not None else None,
            "hotspot_specialist_gate": spec_gate_flat[valid] if spec_gate_flat is not None else None,
            "process_gate": process_gate_flat[valid] if process_gate_flat is not None else None,
            "laser_target_heat_gate": target_heat_gate_flat[valid] if target_heat_gate_flat is not None else None,
            "laser_body_heat_gate": body_heat_gate_flat[valid] if body_heat_gate_flat is not None else None,
            "laser_path_body_heat_gate": path_body_heat_gate_flat[valid] if path_body_heat_gate_flat is not None else None,
            "laser_path_wake_gate": path_wake_gate_flat[valid] if path_wake_gate_flat is not None else None,
            "laser_sweep_heat_gate": sweep_heat_gate_flat[valid] if sweep_heat_gate_flat is not None else None,
            "laser_arrival_gate": arrival_gate_flat[valid] if arrival_gate_flat is not None else None,
            "laser_path_active_gate": path_active_gate_flat[valid] if path_active_gate_flat is not None else None,
            "laser_path_pre_arrival_gate": path_pre_arrival_gate_flat[valid] if path_pre_arrival_gate_flat is not None else None,
            "laser_path_post_arrival_gate": path_post_arrival_gate_flat[valid] if path_post_arrival_gate_flat is not None else None,
            "laser_endpoint_gate": endpoint_gate_flat[valid] if endpoint_gate_flat is not None else None,
            "laser_path_program_track_gate": path_program_track_gate_flat[valid] if path_program_track_gate_flat is not None else None,
            "laser_path_elapsed_gate": path_elapsed_gate_flat[valid] if path_elapsed_gate_flat is not None else None,
            "laser_path_time_until_gate": path_time_until_gate_flat[valid] if path_time_until_gate_flat is not None else None,
            "laser_path_scanned_gate": path_scanned_gate_flat[valid] if path_scanned_gate_flat is not None else None,
            "laser_path_cooling_tail_gate": path_cooling_tail_gate_flat[valid] if path_cooling_tail_gate_flat is not None else None,
            "neighbor_hot_gate": neighbor_gate_flat[valid] if neighbor_gate_flat is not None else None,
            "hotspot_laser_prior_temp": prior_temp_flat[valid] if prior_temp_flat is not None else None,
            "hotspot_laser_prior_gate": prior_gate_flat[valid] if prior_gate_flat is not None else None,
            "laser_residual_prior_gate": residual_prior_gate_flat[valid] if residual_prior_gate_flat is not None else None,
            "laser_residual_learned_gate": residual_learned_gate_flat[valid] if residual_learned_gate_flat is not None else None,
            "laser_residual_control_gate": residual_control_gate_flat[valid] if residual_control_gate_flat is not None else None,
            "laser_residual_cold_start_gate": residual_cold_start_gate_flat[valid] if residual_cold_start_gate_flat is not None else None,
            "laser_residual_gate": residual_gate_flat[valid] if residual_gate_flat is not None else None,
            "laser_residual_delta": residual_delta_flat[valid] if residual_delta_flat is not None else None,
            "laser_residual_boost": residual_boost_flat[valid] if residual_boost_flat is not None else None,
            "cold_to_hot_gate": cold_to_hot_gate_flat[valid] if cold_to_hot_gate_flat is not None else None,
            "coords": coords,
            "valid_mask": valid,
            "target_step": target_step,
            "target_time": target_time,
            "window_temps": window_temps,
            "boundary": graph_seq[-1].boundary.detach().cpu() if hasattr(graph_seq[-1], "boundary") else None,
            "input_laser_pos": graph_seq[-1].laser_pos.detach().cpu() if hasattr(graph_seq[-1], "laser_pos") else None,
            "target_laser_pos": target_laser_pos,
            "edge_index": graph_seq[-1].edge_index.detach().cpu(),
        })

    return records


def analyze_worst_case(records: list[dict], graph, config) -> dict:
    """Find the single worst prediction and return a detailed diagnostic."""
    worst = None
    worst_rec = None

    for rec in records:
        abs_err = (rec["pred"] - rec["target"]).abs()
        local_max = float(abs_err.max().item())
        if worst is None or local_max > worst["abs_error"]:
            idx = int(abs_err.argmax().item())
            # Map the index within the valid subset back to the global node index
            valid_positions = torch.nonzero(rec["valid_mask"], as_tuple=False).reshape(-1)
            node_idx = int(valid_positions[idx].item())
            worst = {
                "abs_error": local_max,
                "prediction": float(rec["pred"][idx].item()),
                "target": float(rec["target"][idx].item()),
                "node_index": node_idx,
                "target_step": rec["target_step"],
                "target_time": rec["target_time"],
                "target_time_s": rec["target_time"] * 1e-3 if rec["target_time"] > 0 else None,
                "coord_mm": rec["coords"][node_idx].tolist(),
            }
            if rec.get("hotspot_prob") is not None:
                worst["hotspot_probability"] = float(rec["hotspot_prob"][idx].item())
            if rec.get("hotspot_specialist_temp") is not None:
                worst["hotspot_specialist_temp"] = float(
                    rec["hotspot_specialist_temp"][idx].item()
                )
            if rec.get("hotspot_specialist_gate") is not None:
                worst["hotspot_specialist_gate"] = float(
                    rec["hotspot_specialist_gate"][idx].item()
                )
            if rec.get("process_gate") is not None:
                worst["process_gate"] = float(rec["process_gate"][idx].item())
            if rec.get("laser_target_heat_gate") is not None:
                worst["laser_target_heat_gate"] = float(
                    rec["laser_target_heat_gate"][idx].item()
                )
            if rec.get("laser_body_heat_gate") is not None:
                worst["laser_body_heat_gate"] = float(
                    rec["laser_body_heat_gate"][idx].item()
                )
            if rec.get("laser_path_body_heat_gate") is not None:
                worst["laser_path_body_heat_gate"] = float(
                    rec["laser_path_body_heat_gate"][idx].item()
                )
            if rec.get("laser_path_wake_gate") is not None:
                worst["laser_path_wake_gate"] = float(
                    rec["laser_path_wake_gate"][idx].item()
                )
            if rec.get("laser_sweep_heat_gate") is not None:
                worst["laser_sweep_heat_gate"] = float(
                    rec["laser_sweep_heat_gate"][idx].item()
                )
            if rec.get("laser_arrival_gate") is not None:
                worst["laser_arrival_gate"] = float(
                    rec["laser_arrival_gate"][idx].item()
                )
            if rec.get("laser_path_active_gate") is not None:
                worst["laser_path_active_gate"] = float(
                    rec["laser_path_active_gate"][idx].item()
                )
            if rec.get("laser_path_pre_arrival_gate") is not None:
                worst["laser_path_pre_arrival_gate"] = float(
                    rec["laser_path_pre_arrival_gate"][idx].item()
                )
            if rec.get("laser_path_post_arrival_gate") is not None:
                worst["laser_path_post_arrival_gate"] = float(
                    rec["laser_path_post_arrival_gate"][idx].item()
                )
            if rec.get("laser_endpoint_gate") is not None:
                worst["laser_endpoint_gate"] = float(
                    rec["laser_endpoint_gate"][idx].item()
                )
            if rec.get("laser_path_program_track_gate") is not None:
                worst["laser_path_program_track_gate"] = float(
                    rec["laser_path_program_track_gate"][idx].item()
                )
            if rec.get("laser_path_elapsed_gate") is not None:
                worst["laser_path_elapsed_gate"] = float(
                    rec["laser_path_elapsed_gate"][idx].item()
                )
            if rec.get("laser_path_time_until_gate") is not None:
                worst["laser_path_time_until_gate"] = float(
                    rec["laser_path_time_until_gate"][idx].item()
                )
            if rec.get("laser_path_scanned_gate") is not None:
                worst["laser_path_scanned_gate"] = float(
                    rec["laser_path_scanned_gate"][idx].item()
                )
            if rec.get("laser_path_cooling_tail_gate") is not None:
                worst["laser_path_cooling_tail_gate"] = float(
                    rec["laser_path_cooling_tail_gate"][idx].item()
                )
            if rec.get("neighbor_hot_gate") is not None:
                worst["neighbor_hot_gate"] = float(rec["neighbor_hot_gate"][idx].item())
            if rec.get("hotspot_laser_prior_temp") is not None:
                worst["hotspot_laser_prior_temp"] = float(
                    rec["hotspot_laser_prior_temp"][idx].item()
                )
            if rec.get("hotspot_laser_prior_gate") is not None:
                worst["hotspot_laser_prior_gate"] = float(
                    rec["hotspot_laser_prior_gate"][idx].item()
                )
            if rec.get("laser_residual_prior_gate") is not None:
                worst["laser_residual_prior_gate"] = float(
                    rec["laser_residual_prior_gate"][idx].item()
                )
            if rec.get("laser_residual_learned_gate") is not None:
                worst["laser_residual_learned_gate"] = float(
                    rec["laser_residual_learned_gate"][idx].item()
                )
            if rec.get("laser_residual_control_gate") is not None:
                worst["laser_residual_control_gate"] = float(
                    rec["laser_residual_control_gate"][idx].item()
                )
            if rec.get("laser_residual_cold_start_gate") is not None:
                worst["laser_residual_cold_start_gate"] = float(
                    rec["laser_residual_cold_start_gate"][idx].item()
                )
            if rec.get("laser_residual_gate") is not None:
                worst["laser_residual_gate"] = float(
                    rec["laser_residual_gate"][idx].item()
                )
            if rec.get("laser_residual_delta") is not None:
                worst["laser_residual_delta"] = float(
                    rec["laser_residual_delta"][idx].item()
                )
            if rec.get("laser_residual_boost") is not None:
                worst["laser_residual_boost"] = float(
                    rec["laser_residual_boost"][idx].item()
                )
            if rec.get("cold_to_hot_gate") is not None:
                worst["cold_to_hot_gate"] = float(
                    rec["cold_to_hot_gate"][idx].item()
                )
            worst_rec = rec

    if worst is None:
        return {}

    node_idx = worst["node_index"]
    target_step = worst["target_step"]
    coords_all = worst_rec["coords"]

    # --- input window temperatures for this node ---
    input_temps = {}
    for wi, wt in enumerate(worst_rec["window_temps"]):
        if node_idx < wt.shape[0]:
            input_temps[f"t-{len(worst_rec['window_temps']) - wi}"] = float(wt[node_idx].item())

    # --- activation status across window + target ---
    activation = {}
    for wi in range(len(worst_rec["window_temps"])):
        step = target_step - len(worst_rec["window_temps"]) + wi
        if 0 <= step < graph.num_steps:
            live_vec = graph.live[step]
            if node_idx < live_vec.shape[0]:
                activation[f"step_{step}"] = bool(live_vec[node_idx].item() > 0.5)

    # Target step activation
    if 0 <= target_step < graph.num_steps:
        live_vec = graph.live[target_step]
        if node_idx < live_vec.shape[0]:
            activation[f"step_{target_step}_target"] = bool(live_vec[node_idx].item() > 0.5)

    # Check if node just became active
    became_active = False
    if target_step > 0 and target_step < graph.num_steps:
        prev_live = graph.live[target_step - 1][node_idx].item() if node_idx < graph.live.shape[1] else 0
        curr_live = graph.live[target_step][node_idx].item() if node_idx < graph.live.shape[1] else 0
        became_active = prev_live < 0.5 and curr_live > 0.5

    # --- laser positions and distances ---
    input_laser_pos = worst_rec.get("input_laser_pos")
    target_laser_pos = worst_rec.get("target_laser_pos")
    input_laser_distance = None
    target_laser_distance = None
    if input_laser_pos is not None:
        node_coord = coords_all[node_idx]
        input_laser_distance = float(torch.norm(node_coord - input_laser_pos).item())
    if target_laser_pos is not None:
        node_coord = coords_all[node_idx]
        target_laser_distance = float(torch.norm(node_coord - target_laser_pos).item())

    # --- boundary label ---
    boundary_label = None
    if worst_rec["boundary"] is not None and node_idx < worst_rec["boundary"].shape[0]:
        boundary_label = int(worst_rec["boundary"][node_idx].item())

    # --- neighbours (1st and 2nd order) ---
    neighbors = find_neighbors(worst_rec["edge_index"], node_idx, order=2)

    # Neighbour temperatures at target step
    neighbor_temps_1st = {}
    if 0 <= target_step < graph.num_steps:
        all_T = graph.temperatures[target_step]
        for n in neighbors.get(1, [])[:20]:  # cap at 20
            if n < all_T.shape[0]:
                neighbor_temps_1st[str(n)] = float(all_T[n].item())

    neighbor_temps_2nd = {}
    if 0 <= target_step < graph.num_steps:
        all_T = graph.temperatures[target_step]
        for n in neighbors.get(2, [])[:20]:
            if n < all_T.shape[0]:
                neighbor_temps_2nd[str(n)] = float(all_T[n].item())

    worst["diagnostic"] = {
        "input_window_temperatures": input_temps,
        "activation_status": activation,
        "node_just_became_active": became_active,
        "input_laser_position_mm": input_laser_pos.tolist() if input_laser_pos is not None else None,
        "input_laser_distance_mm": input_laser_distance,
        "target_laser_position_mm": target_laser_pos.tolist() if target_laser_pos is not None else None,
        "target_laser_distance_mm": target_laser_distance,
        "boundary_label": boundary_label,
        "neighbor_count_1st_order": len(neighbors.get(1, [])),
        "neighbor_count_2nd_order": len(neighbors.get(2, [])),
        "neighbor_temperatures_1st_order_sample": neighbor_temps_1st,
        "neighbor_temperatures_2nd_order_sample": neighbor_temps_2nd,
    }

    return worst


def compute_per_timestep_max(preds_by_step: dict[int, np.ndarray],
                             targets_by_step: dict[int, np.ndarray],
                             times_by_step: dict[int, float]) -> list[dict]:
    """Compare true vs predicted max temperature for each time step."""
    rows = []
    for step in sorted(preds_by_step.keys()):
        p = preds_by_step[step]
        t = targets_by_step[step]
        time_s = times_by_step.get(step, 0.0) * 1e-3  # ms → s
        rows.append({
            "step": step,
            "time_s": time_s,
            "true_max": float(np.max(t)),
            "pred_max": float(np.max(p)),
            "max_error": float(np.max(p) - np.max(t)),
            "max_abs_error": float(np.abs(np.max(p) - np.max(t))),
        })
    return rows


def export_worst_vtu(records: list[dict], worst: dict, graph, output_dir: str):
    """Export VTU files for the time step containing the worst-case node."""
    if not worst:
        return

    target_step = worst["target_step"]
    # Find the record matching this time step
    rec = None
    for r in records:
        if r["target_step"] == target_step:
            rec = r
            break
    if rec is None:
        print(f"  [WARN] Could not find record for target step {target_step}")
        return

    coords = rec["coords"]
    n_nodes = coords.shape[0]

    # Reconstruct full-size tensors (including inactive nodes)
    pred_full = torch.full((n_nodes,), float("nan"))
    target_full = torch.full((n_nodes,), float("nan"))
    valid_indices = torch.nonzero(rec["valid_mask"], as_tuple=False).reshape(-1)
    for vi, global_idx in enumerate(valid_indices.tolist()):
        if vi < rec["pred"].shape[0]:
            pred_full[global_idx] = rec["pred"][vi]
            target_full[global_idx] = rec["target"][vi]

    abs_err_full = (pred_full - target_full).abs()

    cells = None
    if hasattr(graph, '_cells'):
        cells = graph._cells
    else:
        # Try to get cells from original VTU data
        pass

    os.makedirs(output_dir, exist_ok=True)

    base = f"step_{target_step:05d}"
    write_temperature_vtu(
        os.path.join(output_dir, f"{base}_target.vtu"),
        coords, target_full, cells,
    )
    write_temperature_vtu(
        os.path.join(output_dir, f"{base}_prediction.vtu"),
        coords, pred_full, cells,
    )
    write_temperature_vtu(
        os.path.join(output_dir, f"{base}_abs_error.vtu"),
        coords, abs_err_full, cells,
    )
    print(f"  VTU files saved to {output_dir}/{base}_*.vtu")


def main():
    parser = argparse.ArgumentParser(
        description="Temperature-binned checkpoint evaluation for DT-STPINN"
    )
    parser.add_argument("--config", type=str, default="configs/paper1.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--vtu_dir", type=str, default=None,
                        help="Override VTU directory (default: from config)")
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="data/processed")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument(
        "--graph_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Where to keep preprocessed graph tensors. auto follows --device when CUDA is used.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    config = Config.from_yaml(args.config)
    vtu_dir = args.vtu_dir or config.data.vtu_dir
    solidus = config.material.solidus_temp
    liquidus = config.material.liquidus_temp
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Config        : {args.config}")
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"VTU dir       : {vtu_dir}")
    print(f"Device        : {device}")
    print(f"Solidus       : {solidus} °C")
    print(f"Liquidus      : {liquidus} °C")
    print(f"Window size   : {config.data.window_size}")
    print()

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    print("Loading graph data ...")
    loader = VTULoader(vtu_dir)
    if loader.num_steps == 0:
        raise FileNotFoundError(f"No Data-*.vtu files found in {vtu_dir}")

    vtu_data = None
    cache_path = graph_cache_path(args.cache_dir, vtu_dir, loader, config)
    if not args.no_cache and cache_path.exists() and not args.rebuild_cache:
        print(f"  Loading preprocessed graph cache: {cache_path}")
        graph = load_graph_cache(cache_path, config.material)
    else:
        if args.no_cache:
            print("  Graph cache disabled by --no_cache.")
        elif args.rebuild_cache:
            print("  Rebuilding graph cache because --rebuild_cache was set.")
        else:
            print(f"  No graph cache found. It will be saved to: {cache_path}")

        print("  Loading VTU data ...")
        vtu_data = loader.parse_sequence(verbose=True)
        print(f"  {len(vtu_data)} time steps loaded.")

        graph = DynamicGraph(
            vtu_data,
            material_props=config.material,
            k_neighbors=config.data.k_neighbors,
            use_mesh_edges=config.data.use_mesh_edges,
        )
        del vtu_data
        gc.collect()

        if not args.no_cache:
            print(f"  Saving preprocessed graph cache: {cache_path}")
            save_graph_cache(cache_path, graph, {
                "vtu_dir": str(Path(vtu_dir).resolve()),
                "num_vtu_files": loader.num_steps,
                "k_neighbors": config.data.k_neighbors,
                "use_mesh_edges": config.data.use_mesh_edges,
            })

    graph.apply_laser_path_config(config.data)
    if config.data.laser_path_mode != "estimated":
        print(f"  Laser path mode: {config.data.laser_path_mode}")
    print(f"  {graph.num_nodes} nodes, {graph.num_steps} steps.")

    if args.graph_device == "auto":
        graph_device = device if device.type == "cuda" else torch.device("cpu")
    else:
        graph_device = torch.device(args.graph_device)
    if graph_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("graph_device=cuda was requested, but CUDA is not available.")
    print(f"  Moving graph tensors to {graph_device}.")
    graph.to(graph_device)

    _, _, test_idx = split_indices(
        graph.num_steps,
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
    )
    print(f"  Test steps: {test_idx[0]} – {test_idx[-1]} ({len(test_idx)} steps)")

    test_dataset = DEDTemporalDataset(
        graph,
        window_size=config.data.window_size,
        predict_steps=config.data.predict_steps,
        time_indices=test_idx,
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
    actual_node_feature_dim = test_dataset.input_feature_dim
    if int(config.model.node_feature_dim) != actual_node_feature_dim:
        print(
            "  Node feature dim: "
            f"config={config.model.node_feature_dim}, "
            f"actual={actual_node_feature_dim}; using actual dataset dim."
        )
        config.model.node_feature_dim = actual_node_feature_dim
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_temporal_batch,
    )
    print(f"  {len(test_dataset)} test windows.")
    print()

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    print("Loading model ...")
    model = DTSTPINN(config, config.material)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    epoch = ckpt.get("epoch", "?")
    print(f"  Checkpoint epoch: {epoch}")
    print()

    # ------------------------------------------------------------------
    # 4. Run inference
    # ------------------------------------------------------------------
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    records = collect_predictions(model, test_loader, device, amp_dtype)

    # Concatenate all predictions
    all_preds = torch.cat([r["pred"] for r in records])
    all_targets = torch.cat([r["target"] for r in records])
    hot_prob_records = [r["hotspot_prob"] for r in records if r.get("hotspot_prob") is not None]
    all_hot_probs = torch.cat(hot_prob_records) if hot_prob_records else None
    print(f"\nTotal valid predictions: {all_preds.shape[0]:,}")
    print(f"Prediction range: [{all_preds.min().item():.1f}, {all_preds.max().item():.1f}] °C")
    print(f"Target range:     [{all_targets.min().item():.1f}, {all_targets.max().item():.1f}] °C")

    # Organize by time step for per-timestep analysis
    preds_by_step: dict[int, list[np.ndarray]] = {}
    targets_by_step: dict[int, list[np.ndarray]] = {}
    times_by_step: dict[int, float] = {}
    for rec in records:
        step = rec["target_step"]
        if step not in preds_by_step:
            preds_by_step[step] = []
            targets_by_step[step] = []
        preds_by_step[step].append(rec["pred"].numpy())
        targets_by_step[step].append(rec["target"].numpy())
        times_by_step[step] = rec["target_time"]

    preds_by_step = {k: np.concatenate(v) for k, v in preds_by_step.items()}
    targets_by_step = {k: np.concatenate(v) for k, v in targets_by_step.items()}

    # ------------------------------------------------------------------
    # 5. Per-bin metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TEMPERATURE-BINNED METRICS")
    print("=" * 90)
    bin_results = compute_bin_metrics(
        all_preds.numpy(), all_targets.numpy(), solidus, liquidus
    )

    header = f"{'Bin':<30} {'Count':>8} {'Prop':>7} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'P95':>8} {'P99':>8} {'MaxErr':>8}"
    print(header)
    print("-" * len(header))
    for label, _lo, _hi in BIN_DEFS:
        b = bin_results.get(label, {})
        if b.get("count", 0) == 0:
            print(f"{label:<30} {'0':>8} {'0.0%':>7} {'-':>8}")
        else:
            print(
                f"{label:<30} {b['count']:>8,} "
                f"{b['proportion']:>6.1%} "
                f"{b['MAE']:>8.2f} {b['RMSE']:>8.2f} {b['Bias']:>+8.2f} "
                f"{b['AbsErrorP95']:>8.2f} {b['AbsErrorP99']:>8.2f} "
                f"{b['MaxError']:>8.2f}"
            )

    print()
    all_b = bin_results.get("all", {})
    print(f"Global  —  MAE={all_b.get('MAE', 0):.2f}  RMSE={all_b.get('RMSE', 0):.2f}  "
          f"P50={all_b.get('AbsErrorP50', 0):.2f}  P90={all_b.get('AbsErrorP90', 0):.2f}  "
          f"P95={all_b.get('AbsErrorP95', 0):.2f}  P99={all_b.get('AbsErrorP99', 0):.2f}  "
          f"MaxError={all_b.get('MaxError', 0):.2f}")

    # ------------------------------------------------------------------
    # 6. High-temperature detection
    # ------------------------------------------------------------------
    det = bin_results.get("_detection", {})
    liquidus_det = compute_score_detection_metrics(
        all_preds.numpy(), all_targets.numpy(), liquidus, liquidus
    )
    canonical_metrics = dict(all_b)
    canonical_metrics.update(flatten_detection_metrics(
        "Solidus", {
            "recall": det.get("recall_above_solidus", 0.0),
            "precision": det.get("precision_above_solidus", 0.0),
            "f1": det.get("f1_above_solidus", 0.0),
            "iou": det.get("iou_above_solidus", 0.0),
            "true_positive": det.get("true_positive", 0),
            "false_positive": det.get("false_positive", 0),
            "false_negative": det.get("false_negative", 0),
        }
    ))
    canonical_metrics.update(flatten_detection_metrics(
        "Liquidus", liquidus_det
    ))
    print(f"\n--- High-Temperature Detection (threshold = solidus {solidus} °C) ---")
    print(f"  Recall above solidus    : {det.get('recall_above_solidus', 0):.4f}")
    print(f"  Precision above solidus : {det.get('precision_above_solidus', 0):.4f}")
    print(f"  F1 above solidus        : {det.get('f1_above_solidus', 0):.4f}")
    print(f"  IoU above solidus       : {det.get('iou_above_solidus', 0):.4f}")
    print(f"  TP={det.get('true_positive', 0)}, FP={det.get('false_positive', 0)}, FN={det.get('false_negative', 0)}")

    print(f"\n--- High-Temperature Detection (threshold = liquidus {liquidus} C) ---")
    print(f"  Recall above liquidus    : {liquidus_det['recall']:.4f}")
    print(f"  Precision above liquidus : {liquidus_det['precision']:.4f}")
    print(f"  F1 above liquidus        : {liquidus_det['f1']:.4f}")
    print(f"  IoU above liquidus       : {liquidus_det['iou']:.4f}")
    print(f"  TP={liquidus_det['true_positive']}, FP={liquidus_det['false_positive']}, FN={liquidus_det['false_negative']}")

    cls_det = None
    if all_hot_probs is not None:
        cls_det = compute_score_detection_metrics(
            all_hot_probs.numpy(), all_targets.numpy(), solidus, 0.5
        )
        print("\n--- Hotspot Head Detection (probability threshold = 0.5) ---")
        print(f"  Recall above solidus    : {cls_det['recall']:.4f}")
        print(f"  Precision above solidus : {cls_det['precision']:.4f}")
        print(f"  F1 above solidus        : {cls_det['f1']:.4f}")
        print(f"  IoU above solidus       : {cls_det['iou']:.4f}")
        print(f"  TP={cls_det['true_positive']}, FP={cls_det['false_positive']}, FN={cls_det['false_negative']}")

    laser_region_metrics = compute_laser_region_metrics(records, config)
    print("\n--- Target Laser Influence Region ---")
    if laser_region_metrics.get("count", 0) > 0:
        canonical_metrics.update({
            f"LaserRegion{key}": value
            for key, value in laser_region_metrics.items()
            if key != "count"
        })
        canonical_metrics["LaserRegionCount"] = laser_region_metrics["count"]
        print(f"  Count    : {laser_region_metrics['count']:,}")
        print(f"  MAE      : {laser_region_metrics['MAE']:.2f}")
        print(f"  RMSE     : {laser_region_metrics['RMSE']:.2f}")
        print(f"  P95/P99  : {laser_region_metrics['AbsErrorP95']:.2f} / {laser_region_metrics['AbsErrorP99']:.2f}")
        print(f"  MaxError : {laser_region_metrics['MaxError']:.2f}")
    else:
        canonical_metrics["LaserRegionCount"] = 0
        print("  No valid nodes fell inside the configured laser influence region.")

    # ------------------------------------------------------------------
    # 7. Per-timestep max temperature
    # ------------------------------------------------------------------
    ts_rows = compute_per_timestep_max(preds_by_step, targets_by_step, times_by_step)
    true_maxes = [r["true_max"] for r in ts_rows]
    pred_maxes = [r["pred_max"] for r in ts_rows]
    max_errs = [r["max_abs_error"] for r in ts_rows]

    print(f"\n--- Per-Timestep Max Temperature ({len(ts_rows)} steps) ---")
    print(f"  True max range : [{min(true_maxes):.1f}, {max(true_maxes):.1f}] °C")
    print(f"  Pred max range : [{min(pred_maxes):.1f}, {max(pred_maxes):.1f}] °C")
    print(f"  Max abs error  : {max(max_errs):.1f} °C")
    print(f"  Mean abs error : {np.mean(max_errs):.1f} °C")

    # Identify the worst step for max-T prediction
    worst_max_idx = int(np.argmax(max_errs))
    worst_max_step = ts_rows[worst_max_idx]
    print(f"\n  Worst max-T step: {worst_max_step['step']} "
          f"(t={worst_max_step['time_s']:.2f} s)  "
          f"True={worst_max_step['true_max']:.1f} °C  "
          f"Pred={worst_max_step['pred_max']:.1f} °C  "
          f"Error={worst_max_step['max_error']:+.1f} °C")

    # ------------------------------------------------------------------
    # 8. Worst-case node diagnostic
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("WORST-CASE NODE DIAGNOSTIC")
    print("=" * 90)
    worst = analyze_worst_case(records, graph, config)
    worst_cases_top10 = collect_worst_cases(records, config, top_k=10)

    if worst:
        diag = worst.get("diagnostic", {})
        print(f"  Step        : {worst['target_step']}  (t ≈ {worst['target_time_s']:.2f} s)")
        print(f"  Node index  : {worst['node_index']}")
        print(f"  Coord (mm)  : {worst['coord_mm']}")
        print(f"  Prediction  : {worst['prediction']:.2f} °C")
        print(f"  Target      : {worst['target']:.2f} °C")
        print(f"  Abs error   : {worst['abs_error']:.2f} °C")
        if "hotspot_probability" in worst:
            print(f"  Hotspot prob: {worst['hotspot_probability']:.4f}")
        if "hotspot_specialist_temp" in worst:
            print(f"  Specialist T: {worst['hotspot_specialist_temp']:.2f} 掳C")
        if "hotspot_specialist_gate" in worst:
            print(f"  Specialist gate: {worst['hotspot_specialist_gate']:.4f}")
        if "process_gate" in worst:
            print(f"  Process gate   : {worst['process_gate']:.4f}")
        if "laser_target_heat_gate" in worst:
            print(f"  Target heat gate: {worst['laser_target_heat_gate']:.4f}")
        if "laser_body_heat_gate" in worst:
            print(f"  Body heat gate  : {worst['laser_body_heat_gate']:.4f}")
        if "laser_path_body_heat_gate" in worst:
            print(f"  Path body gate  : {worst['laser_path_body_heat_gate']:.4f}")
        if "laser_path_wake_gate" in worst:
            print(f"  Path wake gate  : {worst['laser_path_wake_gate']:.4f}")
        if "laser_sweep_heat_gate" in worst:
            print(f"  Sweep heat gate : {worst['laser_sweep_heat_gate']:.4f}")
        if "laser_arrival_gate" in worst:
            print(f"  Arrival gate   : {worst['laser_arrival_gate']:.4f}")
        if "laser_path_active_gate" in worst:
            print(f"  Path active gate: {worst['laser_path_active_gate']:.4f}")
        if "laser_path_pre_arrival_gate" in worst:
            print(f"  Path pre gate  : {worst['laser_path_pre_arrival_gate']:.4f}")
        if "laser_path_post_arrival_gate" in worst:
            print(f"  Path post gate : {worst['laser_path_post_arrival_gate']:.4f}")
        if "laser_endpoint_gate" in worst:
            print(f"  Endpoint gate  : {worst['laser_endpoint_gate']:.4f}")
        if "laser_path_program_track_gate" in worst:
            print(f"  Program-track gate: {worst['laser_path_program_track_gate']:.4f}")
        if "laser_path_elapsed_gate" in worst:
            print(f"  Path elapsed gate : {worst['laser_path_elapsed_gate']:.4f}")
        if "laser_path_time_until_gate" in worst:
            print(f"  Path time-until gate: {worst['laser_path_time_until_gate']:.4f}")
        if "laser_path_scanned_gate" in worst:
            print(f"  Path scanned gate : {worst['laser_path_scanned_gate']:.4f}")
        if "laser_path_cooling_tail_gate" in worst:
            print(f"  Path cooling tail : {worst['laser_path_cooling_tail_gate']:.4f}")
        if "neighbor_hot_gate" in worst:
            print(f"  Neighbor gate  : {worst['neighbor_hot_gate']:.4f}")
        if "hotspot_laser_prior_temp" in worst:
            print(f"  Laser prior T  : {worst['hotspot_laser_prior_temp']:.2f} C")
        if "hotspot_laser_prior_gate" in worst:
            print(f"  Laser prior gate: {worst['hotspot_laser_prior_gate']:.4f}")
        if "laser_residual_prior_gate" in worst:
            print(f"  Residual prior gate : {worst['laser_residual_prior_gate']:.4f}")
        if "laser_residual_learned_gate" in worst:
            print(f"  Residual learned gate: {worst['laser_residual_learned_gate']:.4f}")
        if "laser_residual_control_gate" in worst:
            print(f"  Residual control gate: {worst['laser_residual_control_gate']:.4f}")
        if "laser_residual_cold_start_gate" in worst:
            print(f"  Residual cold gate   : {worst['laser_residual_cold_start_gate']:.4f}")
        if "laser_residual_gate" in worst:
            print(f"  Residual gate       : {worst['laser_residual_gate']:.4f}")
        if "laser_residual_delta" in worst:
            print(f"  Residual delta      : {worst['laser_residual_delta']:.2f} C")
        if "laser_residual_boost" in worst:
            print(f"  Residual boost      : {worst['laser_residual_boost']:.2f} C")
        if "cold_to_hot_gate" in worst:
            print(f"  Cold-to-hot gate    : {worst['cold_to_hot_gate']:.4f}")
        print(f"  Input laser dist : {diag.get('input_laser_distance_mm', 'N/A')} mm")
        print(f"  Input laser pos  : {diag.get('input_laser_position_mm', 'N/A')}")
        print(f"  Target laser dist: {diag.get('target_laser_distance_mm', 'N/A')} mm")
        print(f"  Target laser pos : {diag.get('target_laser_position_mm', 'N/A')}")
        print(f"  Boundary    : {diag.get('boundary_label', 'N/A')}")
        print(f"  Became active: {diag.get('node_just_became_active', 'N/A')}")
        print(f"  1st-order neighbours: {diag.get('neighbor_count_1st_order', 0)}")
        print(f"  2nd-order neighbours: {diag.get('neighbor_count_2nd_order', 0)}")
        print(f"\n  Input window temperatures:")
        for k, v in diag.get("input_window_temperatures", {}).items():
            print(f"    {k}: {v:.2f} °C")
        print(f"\n  Activation status:")
        for k, v in diag.get("activation_status", {}).items():
            print(f"    {k}: {v}")
        if diag.get("neighbor_temperatures_1st_order_sample"):
            n1 = diag["neighbor_temperatures_1st_order_sample"]
            vals = list(n1.values())
            print(f"\n  1st-order neighbour temps (sample of {len(n1)}): "
                  f"min={min(vals):.1f}  max={max(vals):.1f}  mean={np.mean(vals):.1f}")
        if diag.get("neighbor_temperatures_2nd_order_sample"):
            n2 = diag["neighbor_temperatures_2nd_order_sample"]
            vals = list(n2.values())
            print(f"  2nd-order neighbour temps (sample of {len(n2)}): "
                  f"min={min(vals):.1f}  max={max(vals):.1f}  mean={np.mean(vals):.1f}")

        if worst_cases_top10:
            print("\n  Worst 10 point errors:")
            for rank, item in enumerate(worst_cases_top10, start=1):
                dist = item.get("target_laser_distance_mm")
                if dist is None:
                    dist = item.get("distance_to_laser_mm")
                tta = item.get("time_to_arrival_s")
                dist_text = "N/A" if dist is None else f"{dist:.3f} mm"
                tta_text = "N/A" if tta is None else f"{tta:.6f} s"
                print(
                    f"    #{rank:02d} step={item['target_step']} "
                    f"node={item['node_index']} err={item['abs_error']:.2f} "
                    f"pred/target={item['prediction']:.2f}/{item['target']:.2f} "
                    f"laser_dist={dist_text} arrival={tta_text}"
                )

        # ------------------------------------------------------------------
        # 9. Export VTU for the worst time step
        # ------------------------------------------------------------------
        print(f"\n--- Exporting VTU for worst step {worst['target_step']} ---")
        # Try to recover cells from the original VTU data
        if vtu_data is not None and len(vtu_data) > 0:
            last_vtu = vtu_data[-1]
            if hasattr(last_vtu, 'cells') and last_vtu.cells:
                # Temporarily attach cells for the VTU writer
                graph._cells = last_vtu.cells
        export_worst_vtu(records, worst, graph, args.output_dir)

    # ------------------------------------------------------------------
    # 10. Save JSON report
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "solidus": solidus,
        "liquidus": liquidus,
        "global_metrics": all_b,
        "canonical_metrics": canonical_metrics,
        "per_bin_metrics": {
            label: bin_results.get(label, {})
            for label, _, _ in BIN_DEFS
        },
        "detection_metrics": det,
        "liquidus_detection_metrics": liquidus_det,
        "hotspot_head_detection_metrics": cls_det,
        "laser_region_metrics": laser_region_metrics,
        "per_timestep_max_summary": {
            "num_steps": len(ts_rows),
            "true_max_range": [float(min(true_maxes)), float(max(true_maxes))],
            "pred_max_range": [float(min(pred_maxes)), float(max(pred_maxes))],
            "max_abs_error": float(max(max_errs)),
            "mean_abs_error": float(np.mean(max_errs)),
        },
        "per_timestep_max": ts_rows,
        "worst_case": worst,
        "worst_cases_top10": worst_cases_top10,
    }
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
