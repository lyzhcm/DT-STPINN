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
from src.utils.visualization import write_temperature_vtu

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

        # Flatten
        pred_flat = T_pred.reshape(-1).cpu()
        target_flat = target.detach().float().reshape(-1).cpu()
        coords = graph_seq[-1].coords.detach().float().cpu()

        # Build valid mask
        valid = torch.isfinite(pred_flat) & torch.isfinite(target_flat)
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

        # Collect input window temperatures (ground truth for context)
        window_temps = []
        for g in graph_seq:
            y = g.y.detach().float().reshape(-1).cpu()
            window_temps.append(y)

        records.append({
            "pred": pred_flat[valid],
            "target": target_flat[valid],
            "coords": coords,
            "valid_mask": valid,
            "target_step": target_step,
            "target_time": target_time,
            "window_temps": window_temps,
            "boundary": graph_seq[-1].boundary.detach().cpu() if hasattr(graph_seq[-1], "boundary") else None,
            "laser_pos": graph_seq[-1].laser_pos.detach().cpu() if hasattr(graph_seq[-1], "laser_pos") else None,
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
            activation[f"step_{target_step}_target"] = bool(live_vec[target_step].item() > 0.5)

    # Check if node just became active
    became_active = False
    if target_step > 0 and target_step < graph.num_steps:
        prev_live = graph.live[target_step - 1][node_idx].item() if node_idx < graph.live.shape[1] else 0
        curr_live = graph.live[target_step][node_idx].item() if node_idx < graph.live.shape[1] else 0
        became_active = prev_live < 0.5 and curr_live > 0.5

    # --- laser position and distance ---
    laser_pos = worst_rec["laser_pos"]
    laser_distance = None
    if laser_pos is not None:
        node_coord = coords_all[node_idx]
        laser_distance = float(torch.norm(node_coord - laser_pos).item())

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
        "laser_position_mm": laser_pos.tolist() if laser_pos is not None else None,
        "laser_distance_mm": laser_distance,
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
    print("Loading VTU data ...")
    loader = VTULoader(vtu_dir)
    vtu_data = loader.parse_sequence(verbose=True)
    print(f"  {len(vtu_data)} time steps loaded.")

    graph = DynamicGraph(
        vtu_data,
        material_props=config.material,
        k_neighbors=config.data.k_neighbors,
        use_mesh_edges=config.data.use_mesh_edges,
    )
    print(f"  {graph.num_nodes} nodes, {graph.num_steps} steps.")

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
    )
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
    print(f"\n--- High-Temperature Detection (threshold = solidus {solidus} °C) ---")
    print(f"  Recall above solidus    : {det.get('recall_above_solidus', 0):.4f}")
    print(f"  Precision above solidus : {det.get('precision_above_solidus', 0):.4f}")
    print(f"  F1 above solidus        : {det.get('f1_above_solidus', 0):.4f}")
    print(f"  IoU above solidus       : {det.get('iou_above_solidus', 0):.4f}")
    print(f"  TP={det.get('true_positive', 0)}, FP={det.get('false_positive', 0)}, FN={det.get('false_negative', 0)}")

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

    if worst:
        diag = worst.get("diagnostic", {})
        print(f"  Step        : {worst['target_step']}  (t ≈ {worst['target_time_s']:.2f} s)")
        print(f"  Node index  : {worst['node_index']}")
        print(f"  Coord (mm)  : {worst['coord_mm']}")
        print(f"  Prediction  : {worst['prediction']:.2f} °C")
        print(f"  Target      : {worst['target']:.2f} °C")
        print(f"  Abs error   : {worst['abs_error']:.2f} °C")
        print(f"  Laser dist  : {diag.get('laser_distance_mm', 'N/A')} mm")
        print(f"  Laser pos   : {diag.get('laser_position_mm', 'N/A')}")
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

        # ------------------------------------------------------------------
        # 9. Export VTU for the worst time step
        # ------------------------------------------------------------------
        print(f"\n--- Exporting VTU for worst step {worst['target_step']} ---")
        # Try to recover cells from the original VTU data
        if vtu_data and len(vtu_data) > 0:
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
        "per_bin_metrics": {
            label: bin_results.get(label, {})
            for label, _, _ in BIN_DEFS
        },
        "detection_metrics": det,
        "per_timestep_max_summary": {
            "num_steps": len(ts_rows),
            "true_max_range": [float(min(true_maxes)), float(max(true_maxes))],
            "pred_max_range": [float(min(pred_maxes)), float(max(pred_maxes))],
            "max_abs_error": float(max(max_errs)),
            "mean_abs_error": float(np.mean(max_errs)),
        },
        "worst_case": worst,
    }
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
