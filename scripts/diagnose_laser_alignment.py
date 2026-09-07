"""Diagnose alignment between prescribed laser path and VTU hotspots.

This script does not train a model.  It compares the configured
``additive_z_scan`` laser position against the hottest live node in each VTU
frame, then optionally sweeps the path time offset to find a better clock
alignment before launching another long training run.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import graph_cache_path, load_graph_cache, save_graph_cache
from src.config import Config, MaterialProps
from src.data.dataset import DEDTemporalDataset
from src.data.preprocessing import load_or_build_split_indices
from src.data.vtu_loader import VTULoader
from src.graph_builder.dynamic_graph import DynamicGraph
from src.utils.laser_path import AdditiveZScanPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare XML laser trajectory against VTU hotspot positions."
    )
    parser.add_argument("--config", type=str, default="configs/ablation_d_lookahead.yaml")
    parser.add_argument("--vtu_dir", type=str, default=None)
    parser.add_argument(
        "--laser_xml",
        "--xml",
        dest="laser_xml",
        type=str,
        default=None,
        help="Optional para.xml path; overrides YAML laser path geometry.",
    )
    parser.add_argument("--cache_dir", type=str, default="data/processed")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--hot_threshold", type=float, default=500.0)
    parser.add_argument("--solidus", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=1,
                        help="Use the centroid of the top-K hottest live nodes.")
    parser.add_argument("--offset_radius_s", type=float, default=0.5,
                        help="Sweep +/- this many seconds around the configured offset.")
    parser.add_argument("--offset_step_s", type=float, default=0.02)
    parser.add_argument("--scale_min", type=float, default=None,
                        help="Optional lower bound for time-scale sweep in seconds/raw-time.")
    parser.add_argument("--scale_max", type=float, default=None,
                        help="Optional upper bound for time-scale sweep in seconds/raw-time.")
    parser.add_argument("--scale_steps", type=int, default=1,
                        help="Number of time-scale values to test when scale_min/max are set.")
    parser.add_argument("--sweep_path_variants", action="store_true",
                        help=(
                            "Also compare XML-literal and calibrated layer/hatch "
                            "direction variants before choosing the best offset."
                        ))
    parser.add_argument("--focus_raw_time", type=float, nargs="*", default=[42560.0, 42620.0])
    parser.add_argument("--focus_step", type=int, nargs="*", default=[],
                        help="Specific VTU step indices to inspect in detail.")
    parser.add_argument("--focus_node", type=int, nargs="*", default=[],
                        help="Specific node indices to inspect against the laser path.")
    parser.add_argument("--focus_window_steps", type=int, default=8,
                        help="Inspect +/- this many steps around each focus step/node.")
    parser.add_argument("--focus_output_csv", type=str, default=None,
                        help="Optional CSV path for focus node/path window diagnostics.")
    parser.add_argument("--raw_time_min", type=float, default=None,
                        help="Only use frames at or after this raw VTU time for scoring.")
    parser.add_argument("--raw_time_max", type=float, default=None,
                        help="Only use frames at or before this raw VTU time for scoring.")
    parser.add_argument("--gate_coverage", action="store_true",
                        help="Report how well XML-derived process gates cover true solidus nodes.")
    parser.add_argument("--gate_split", choices=["train", "val", "test", "all"], default="test",
                        help="Temporal split used for --gate_coverage.")
    parser.add_argument(
        "--split_indices",
        type=str,
        default=None,
        help="Frozen split_indices.json used for gate diagnostics.",
    )
    parser.add_argument("--gate_thresholds", type=float, nargs="*",
                        default=[0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.5],
                        help="Gate thresholds to score.")
    parser.add_argument("--gate_chunk_nodes", type=int, default=65536,
                        help="Node chunk size for --gate_coverage.")
    parser.add_argument("--gate_device", type=str, default="auto",
                        help="Device for --gate_coverage: auto, cpu, or cuda.")
    parser.add_argument("--gate_confusion", action="store_true",
                        help=(
                            "Compare XML-derived process gates on true-hot nodes "
                            "against cold false-positive-risk nodes."
                        ))
    parser.add_argument("--cold_threshold", type=float, default=100.0,
                        help="Target temperature threshold for cold false-positive-risk nodes.")
    parser.add_argument("--gate_confusion_out", type=str, default=None,
                        help="Optional output directory for gate confusion CSV/markdown files.")
    parser.add_argument("--gate_top_k_examples", type=int, default=20,
                        help="Number of representative gate-confusion examples to save.")
    return parser.parse_args()


def load_graph(config: Config, vtu_dir: str, cache_dir: str,
               no_cache: bool, rebuild_cache: bool) -> DynamicGraph:
    loader = VTULoader(vtu_dir)
    print(f"VTU directory: {vtu_dir}")
    print(f"Found {loader.num_steps} VTU files.")
    if loader.num_steps == 0:
        raise FileNotFoundError(f"No Data-*.vtu files found in {vtu_dir}")

    cache_path = graph_cache_path(cache_dir, vtu_dir, loader, config)
    if not no_cache and cache_path.exists() and not rebuild_cache:
        print(f"Loading graph cache: {cache_path}")
        return load_graph_cache(cache_path, MaterialProps(**config.material.__dict__))

    print("Parsing VTU files and rebuilding graph cache...")
    vtu_data = loader.parse_sequence(verbose=True)
    graph = DynamicGraph(
        vtu_data,
        MaterialProps(**config.material.__dict__),
        k_neighbors=config.data.k_neighbors,
        use_mesh_edges=config.data.use_mesh_edges,
    )
    if not no_cache:
        save_graph_cache(cache_path, graph, {"vtu_dir": str(Path(vtu_dir).resolve())})
        print(f"Saved graph cache: {cache_path}")
    return graph


def hottest_live_positions(graph: DynamicGraph, top_k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = graph.coords.cpu()
    positions = []
    max_temps = []
    node_ids = []
    k = max(1, int(top_k))

    for step in range(graph.num_steps):
        temp = graph.temperatures[step].cpu()
        live = graph.live[step].cpu() > 0.5
        values = temp.clone()
        if live.any():
            values[~live] = -float("inf")
        count = min(k, values.numel())
        top_vals, top_idx = torch.topk(values, count)
        finite = torch.isfinite(top_vals)
        if finite.any():
            top_idx = top_idx[finite]
            top_vals = top_vals[finite]
        else:
            top_idx = torch.tensor([int(torch.argmax(temp))])
            top_vals = temp[top_idx]
        positions.append(coords[top_idx].mean(dim=0))
        max_temps.append(top_vals.max())
        node_ids.append(top_idx[torch.argmax(top_vals)])

    return torch.stack(positions), torch.stack(max_temps), torch.stack(node_ids).long()


def apply_offset(graph: DynamicGraph, config: Config, offset_s: float,
                 time_scale_to_s: float | None = None,
                 alternate_layer_scan_direction: bool | None = None,
                 reverse_hatch_order_parity: int | None = None) -> torch.Tensor:
    data = config.data
    graph.apply_additive_z_scan_path(
        start_point_mm=data.laser_start_point_mm,
        scan_direction=data.laser_scan_direction,
        scan_length_mm=data.laser_scan_length_mm,
        hatch_direction=data.laser_hatch_direction,
        hatch_count=data.laser_hatch_count,
        hatch_spacing_mm=data.laser_hatch_spacing_mm,
        layer_count=data.laser_layer_count,
        layer_thickness_mm=data.laser_layer_thickness_mm,
        velocity_mm_s=data.laser_velocity_mm_s,
        each_path_time_s=data.laser_each_path_time_s,
        each_layer_time_s=data.laser_each_layer_time_s,
        time_scale_to_s=(data.laser_path_time_scale_to_s if time_scale_to_s is None else time_scale_to_s),
        time_offset_s=offset_s,
        alternate_layer_scan_direction=(
            getattr(data, "laser_alternate_layer_scan_direction", False)
            if alternate_layer_scan_direction is None
            else bool(alternate_layer_scan_direction)
        ),
        reverse_hatch_order_parity=(
            getattr(data, "laser_reverse_hatch_order_parity", -1)
            if reverse_hatch_order_parity is None
            else int(reverse_hatch_order_parity)
        ),
    )
    return graph._laser_positions.cpu()


def path_variants(config: Config, enabled: bool) -> list[tuple[str, bool, int]]:
    """Return layer/hatch direction variants to compare.

    The XML description is literal serpentine scanning for every layer.  Some
    Flow-3D AM outputs appear better aligned when odd/even layers are mirrored,
    so the diagnostic can score both interpretations against the VTU hotspots.
    """
    data = config.data
    configured = (
        "configured",
        bool(getattr(data, "laser_alternate_layer_scan_direction", False)),
        int(getattr(data, "laser_reverse_hatch_order_parity", -1)),
    )
    if not enabled:
        return [configured]

    candidates = [
        configured,
        ("xml_literal", False, -1),
        ("alt_layers", True, -1),
        ("reverse_even", False, 0),
        ("reverse_odd", False, 1),
        ("alt_reverse_even", True, 0),
        ("alt_reverse_odd", True, 1),
    ]
    unique = []
    seen = set()
    for name, alternate, reverse in candidates:
        key = (alternate, reverse)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, alternate, reverse))
    return unique


def summarize(dist_xy: np.ndarray, dist_3d: np.ndarray) -> dict[str, float]:
    return {
        "mean_xy": float(np.mean(dist_xy)),
        "p50_xy": float(np.percentile(dist_xy, 50)),
        "p90_xy": float(np.percentile(dist_xy, 90)),
        "p95_xy": float(np.percentile(dist_xy, 95)),
        "mean_3d": float(np.mean(dist_3d)),
        "p95_3d": float(np.percentile(dist_3d, 95)),
    }


def node_laser_feature_preview(graph: DynamicGraph, config: Config,
                               node: int, target_step: int) -> dict[str, float]:
    """Preview the direct laser features used for process gating at one node."""
    data = config.data
    coord = graph.coords[node].detach().cpu().float()
    laser_positions = graph._laser_positions.detach().cpu().float()
    raw_times = graph.times.detach().cpu().float()

    radius = max(float(data.laser_feature_radius_mm), 1.0e-6)
    depth = max(float(data.laser_feature_depth_mm), 1.0e-6)
    time_scale = float(data.laser_feature_time_scale_to_s)
    decay_s = max(float(data.laser_feature_exposure_time_decay_s), 1.0e-6)

    def heat_from_delta(delta: torch.Tensor) -> torch.Tensor:
        dxy_sq = delta[..., :2].pow(2).sum(dim=-1)
        z_sq = delta[..., 2].pow(2)
        return torch.exp(-2.0 * dxy_sq / (radius ** 2) - z_sq / (depth ** 2))

    target_laser = laser_positions[target_step]
    target_delta = coord - target_laser
    target_heat = float(heat_from_delta(target_delta).item())
    target_dxy = float(torch.linalg.vector_norm(target_delta[:2]).item())
    target_d3 = float(torch.linalg.vector_norm(target_delta).item())

    sweep_heat = 0.0
    if bool(data.laser_feature_include_sweep):
        start_step = max(0, target_step - int(data.window_size))
        for src_step in range(start_step, target_step):
            source_laser = laser_positions[src_step]
            segment = target_laser - source_laser
            seg_len_sq = segment.dot(segment).clamp_min(1.0e-12)
            progress = ((coord - source_laser).dot(segment) / seg_len_sq).clamp(0.0, 1.0)
            closest = source_laser + progress * segment
            sweep_heat = max(sweep_heat, float(heat_from_delta(coord - closest).item()))

    exposure_integral = 0.0
    exposure_best_heat = 0.0
    exposure_best_dt = 0.0
    exposure_best_dxy = float("inf")
    if bool(data.laser_feature_include_exposure):
        start = max(0, target_step - int(data.laser_feature_exposure_past_steps))
        end = min(
            graph.num_steps - 1,
            target_step + int(data.laser_feature_exposure_future_steps),
        )
        window_pos = laser_positions[start:end + 1]
        window_times = raw_times[start:end + 1]
        target_time = raw_times[target_step]

        if bool(data.laser_feature_exposure_use_segments) and window_pos.shape[0] >= 2:
            segment_start = window_pos[:-1]
            segment_end = window_pos[1:]
            segment = segment_end - segment_start
            seg_len_sq = segment.pow(2).sum(dim=1).clamp_min(1.0e-12)
            rel = coord.view(1, 3) - segment_start
            progress = ((rel * segment).sum(dim=1) / seg_len_sq).clamp(0.0, 1.0)
            closest = segment_start + progress.view(-1, 1) * segment
            delta = coord.view(1, 3) - closest
            heat = heat_from_delta(delta)
            closest_times = window_times[:-1] + progress * (window_times[1:] - window_times[:-1])
            rel_dt = (target_time - closest_times) * time_scale
        else:
            delta = coord.view(1, 3) - window_pos
            heat = heat_from_delta(delta)
            rel_dt = (target_time - window_times) * time_scale

        decayed_heat = heat * torch.exp(-rel_dt.abs() / decay_s)
        best_idx = int(decayed_heat.argmax().item())
        exposure_integral = float(decayed_heat.sum().item())
        exposure_best_heat = float(heat[best_idx].item())
        exposure_best_dt = float(rel_dt[best_idx].item())
        exposure_best_dxy = float(torch.linalg.vector_norm(delta[best_idx, :2]).item())

    neighbor_gate = 0.0
    if bool(data.laser_feature_include_neighbor_temp):
        edge_index = graph.edge_index.detach().cpu().long()
        dst_mask = edge_index[1] == int(node)
        src_nodes = edge_index[0, dst_mask]
        if src_nodes.numel() > 0:
            start_step = max(0, target_step - int(data.window_size))
            if target_step > start_step:
                neigh_max = graph.temperatures[start_step:target_step, src_nodes].max()
                solidus_norm = max(float(config.material.solidus_temp) / 1000.0, 1.001)
                neighbor_gate = float(
                    ((neigh_max / 1000.0 - 1.0) / (solidus_norm - 1.0))
                    .clamp(0.0, 1.0)
                    .item()
                )

    process_gate = max(target_heat, sweep_heat, exposure_integral)
    return {
        "target_dxy": target_dxy,
        "target_d3": target_d3,
        "target_heat": target_heat,
        "sweep_heat": sweep_heat,
        "exposure_best_dxy": exposure_best_dxy,
        "exposure_best_heat": exposure_best_heat,
        "exposure_integral": exposure_integral,
        "exposure_best_dt": exposure_best_dt,
        "process_gate": min(max(process_gate, 0.0), 1.0),
        "neighbor_hot_gate": neighbor_gate,
    }


def path_states(config: Config, raw_times: torch.Tensor,
                offset_s: float, time_scale_to_s: float) -> list[dict[str, float]]:
    """Return additive_z_scan state for each raw VTU time."""
    data = config.data
    scan_time = float(data.laser_scan_length_mm) / max(float(data.laser_velocity_mm_s), 1.0e-12)
    path_period = scan_time + max(float(data.laser_each_path_time_s), 0.0)
    layer_period = path_period * int(data.laser_hatch_count) + max(float(data.laser_each_layer_time_s), 0.0)
    total_tracks = max(1, int(data.laser_hatch_count) * int(data.laser_layer_count))
    t0 = float(raw_times[0].item()) * float(time_scale_to_s)
    states = []

    for raw_time in raw_times.detach().cpu().tolist():
        elapsed = max(0.0, float(raw_time) * float(time_scale_to_s) - t0 - float(offset_s))
        if layer_period <= 0.0:
            layer_idx = 0
            track_in_layer = 0
            local_path_time = 0.0
        else:
            layer_idx = min(int(elapsed // layer_period), int(data.laser_layer_count) - 1)
            layer_time = elapsed - layer_idx * layer_period
            track_in_layer = min(int(layer_time // path_period), int(data.laser_hatch_count) - 1)
            global_track = min(layer_idx * int(data.laser_hatch_count) + track_in_layer,
                               total_tracks - 1)
            layer_idx = global_track // int(data.laser_hatch_count)
            track_in_layer = global_track % int(data.laser_hatch_count)
            local_path_time = layer_time - track_in_layer * path_period

        progress = min(max(local_path_time / scan_time, 0.0), 1.0) if scan_time > 0.0 else 1.0
        physical_track = track_in_layer
        parity = int(getattr(data, "laser_reverse_hatch_order_parity", -1))
        if parity in (0, 1) and layer_idx % 2 == parity:
            physical_track = int(data.laser_hatch_count) - 1 - track_in_layer
        forward = track_in_layer % 2 == 0
        if getattr(data, "laser_alternate_layer_scan_direction", False) and layer_idx % 2 == 1:
            forward = not forward
        states.append({
            "elapsed": elapsed,
            "layer": float(layer_idx),
            "track": float(track_in_layer),
            "physical_track": float(physical_track),
            "progress": progress,
            "forward": float(forward),
        })

    return states


def print_focus_node_windows(graph: DynamicGraph, config: Config, *,
                             focus_nodes: list[int], focus_steps: list[int],
                             window_steps: int, base_laser: np.ndarray,
                             best_laser: np.ndarray, best_offset: float,
                             best_scale: float,
                             focus_output_csv: str | None = None) -> None:
    if not focus_nodes and not focus_steps:
        return

    coords = graph.coords.cpu().numpy()
    temps = graph.temperatures.cpu().numpy()
    live = graph.live.cpu().numpy()
    times = graph.times.cpu()
    states = path_states(config, times, best_offset, best_scale)
    raw_origin = float(times[0].item()) if graph.num_steps else 0.0
    feature_path = AdditiveZScanPath.from_config(config)
    requested_step_set = set()
    focus_rows: list[dict[str, object]] = []

    def process_diag(node_xyz: np.ndarray, step_idx: int) -> dict[str, float]:
        features, columns = feature_path.node_process_features(
            node_xyz.reshape(1, 3),
            float(times[step_idx].item()),
            raw_origin=raw_origin,
        )
        return {name: float(value) for name, value in zip(columns, features[0].tolist())}

    def add_focus_row(kind: str, node: int, step_idx: int, node_xyz: np.ndarray,
                      state: dict[str, float], preview: dict[str, float],
                      diag: dict[str, float]) -> None:
        base_delta = node_xyz - base_laser[step_idx]
        best_delta = node_xyz - best_laser[step_idx]
        focus_rows.append({
            "kind": kind,
            "step": int(step_idx),
            "raw_time": float(times[step_idx].item()),
            "node_index": int(node),
            "temperature_c": float(temps[step_idx, node]),
            "live": float(live[step_idx, node]),
            "node_x_mm": float(node_xyz[0]),
            "node_y_mm": float(node_xyz[1]),
            "node_z_mm": float(node_xyz[2]),
            "laser_x_mm": float(best_laser[step_idx, 0]),
            "laser_y_mm": float(best_laser[step_idx, 1]),
            "laser_z_mm": float(best_laser[step_idx, 2]),
            "base_distance_xy_mm": float(np.linalg.norm(base_delta[:2])),
            "best_distance_xy_mm": float(np.linalg.norm(best_delta[:2])),
            "best_distance_3d_mm": float(np.linalg.norm(best_delta)),
            "layer": int(state["layer"]),
            "track_program_state": int(state["track"]),
            "track_physical_state": int(state["physical_track"]),
            "forward_state": int(state["forward"] > 0.5),
            "progress": float(state["progress"]),
            "line_along_mm": diag["line_along_mm"],
            "line_cross_mm": diag["line_cross_mm"],
            "time_to_arrival_s": diag["time_to_arrival_s"],
            "arrival_raw_time": diag["arrival_raw_time"],
            "track_physical_feature": int(round(diag["track_physical"])),
            "track_program_feature": int(round(diag["track_program"])),
            "direction_sign": int(round(diag["direction_sign"])),
            "in_laser_ellipsoid": int(round(diag["in_laser_ellipsoid"])),
            "in_track_neighborhood": int(round(diag["in_track_neighborhood"])),
            "process_gate": preview["process_gate"],
            "target_heat": preview["target_heat"],
            "sweep_heat": preview["sweep_heat"],
            "exposure_integral": preview["exposure_integral"],
            "exposure_best_dt_s": preview["exposure_best_dt"],
            "neighbor_hot_gate": preview["neighbor_hot_gate"],
        })

    for step in focus_steps:
        if step < 0 or step >= graph.num_steps:
            raise ValueError(f"--focus_step out of range: {step}")
        for idx in range(max(0, step - window_steps), min(graph.num_steps, step + window_steps + 1)):
            requested_step_set.add(idx)

    print("\nFocus node/path windows:")
    for node in focus_nodes:
        if node < 0 or node >= graph.num_nodes:
            raise ValueError(f"--focus_node out of range: {node}")
        node_xyz = coords[node]
        step_set = set(requested_step_set)
        if not step_set:
            hot_step = int(np.argmax(temps[:, node]))
            for idx in range(max(0, hot_step - window_steps), min(graph.num_steps, hot_step + window_steps + 1)):
                step_set.add(idx)

        node_delta = node_xyz[None, :] - best_laser
        node_dist_xy = np.linalg.norm(node_delta[:, :2], axis=1)
        node_dist_3d = np.linalg.norm(node_delta, axis=1)
        nearest_step = int(np.argmin(node_dist_3d))
        hottest_step = int(np.argmax(temps[:, node]))
        print(
            f"\n  Node {node} xyz=[{node_xyz[0]:.3f},{node_xyz[1]:.3f},{node_xyz[2]:.3f}] "
            f"hottest_step={hottest_step} raw={times[hottest_step].item():.0f} "
            f"maxT={temps[hottest_step, node]:.1f} "
            f"nearest_laser_step={nearest_step} raw={times[nearest_step].item():.0f} "
            f"nearest_d3d={node_dist_3d[nearest_step]:.3f}"
        )

        print(
            "\n  step raw_time node targetT live layer track phys dir prog "
            "base_xy best_xy best_3d arr_dt arr_raw line_cross proc_gate tgt_heat sweep exp_int exp_dt neigh best_laser_xyz"
        )
        rows = sorted(step_set)
        for idx in rows:
            state = states[idx]
            base_delta = node_xyz - base_laser[idx]
            best_delta = node_xyz - best_laser[idx]
            preview = node_laser_feature_preview(graph, config, node, idx)
            diag = process_diag(node_xyz, idx)
            add_focus_row("focus_node", node, idx, node_xyz, state, preview, diag)
            print(
                f"  {idx:4d} {times[idx].item():8.0f} {node:5d} "
                f"{temps[idx, node]:7.1f} {live[idx, node]:4.0f} "
                f"{int(state['layer']):5d} {int(state['track']):5d} {int(state['physical_track']):4d} "
                f"{'+' if state['forward'] > 0.5 else '-':>3s} {state['progress']:5.2f} "
                f"{np.linalg.norm(base_delta[:2]):7.3f} "
                f"{np.linalg.norm(best_delta[:2]):7.3f} "
                f"{np.linalg.norm(best_delta):7.3f} "
                f"{diag['time_to_arrival_s']:7.3f} "
                f"{diag['arrival_raw_time']:7.0f} "
                f"{diag['line_cross_mm']:7.3f} "
                f"{preview['process_gate']:7.3f} "
                f"{preview['target_heat']:7.3f} "
                f"{preview['sweep_heat']:7.3f} "
                f"{preview['exposure_integral']:7.3f} "
                f"{preview['exposure_best_dt']:7.3f} "
                f"{preview['neighbor_hot_gate']:7.3f} "
                f"[{best_laser[idx,0]:7.3f},{best_laser[idx,1]:7.3f},{best_laser[idx,2]:5.3f}]"
            )

    if not focus_nodes:
        rows = sorted(requested_step_set)
        print(
            "\n  step raw_time node targetT live layer track phys dir prog "
            "base_xy best_xy best_3d arr_dt arr_raw line_cross proc_gate tgt_heat sweep exp_int exp_dt neigh best_laser_xyz"
        )
        for idx in rows:
            hot_values = temps[idx].copy()
            live_mask = live[idx] > 0.5
            if live_mask.any():
                hot_values[~live_mask] = -np.inf
            node = int(np.argmax(hot_values))
            node_xyz = coords[node]
            target_t = temps[idx, node]
            is_live = live[idx, node]
            base_delta = node_xyz - base_laser[idx]
            best_delta = node_xyz - best_laser[idx]
            state = states[idx]
            preview = node_laser_feature_preview(graph, config, node, idx)
            diag = process_diag(node_xyz, idx)
            add_focus_row("focus_step_hottest", node, idx, node_xyz, state, preview, diag)
            print(
                f"  {idx:4d} {times[idx].item():8.0f} {node:5d} "
                f"{target_t:7.1f} {is_live:4.0f} "
                f"{int(state['layer']):5d} {int(state['track']):5d} {int(state['physical_track']):4d} "
                f"{'+' if state['forward'] > 0.5 else '-':>3s} {state['progress']:5.2f} "
                f"{np.linalg.norm(base_delta[:2]):7.3f} "
                f"{np.linalg.norm(best_delta[:2]):7.3f} "
                f"{np.linalg.norm(best_delta):7.3f} "
                f"{diag['time_to_arrival_s']:7.3f} "
                f"{diag['arrival_raw_time']:7.0f} "
                f"{diag['line_cross_mm']:7.3f} "
                f"{preview['process_gate']:7.3f} "
                f"{preview['target_heat']:7.3f} "
                f"{preview['sweep_heat']:7.3f} "
                f"{preview['exposure_integral']:7.3f} "
                f"{preview['exposure_best_dt']:7.3f} "
                f"{preview['neighbor_hot_gate']:7.3f} "
                f"[{best_laser[idx,0]:7.3f},{best_laser[idx,1]:7.3f},{best_laser[idx,2]:5.3f}]"
            )

    if focus_output_csv and focus_rows:
        out_path = Path(focus_output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(focus_rows[0].keys()))
            writer.writeheader()
            writer.writerows(focus_rows)
        print(f"\nFocus window CSV: {out_path}")


def gate_target_steps(
        graph: DynamicGraph,
        config: Config,
        split_name: str,
        split_indices_path: str | None = None,
        ) -> tuple[list[int], str]:
    if split_name == "all":
        source_indices = list(range(graph.num_steps))
        split_source = "all"
    else:
        train_idx, val_idx, test_idx, split_source = load_or_build_split_indices(
            graph.num_steps,
            train_ratio=config.data.train_split,
            val_ratio=config.data.val_split,
            split_indices_path=split_indices_path,
        )
        source_indices = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }[split_name]

    window_size = int(config.data.window_size)
    predict_steps = int(config.data.predict_steps)
    targets = []
    for start in source_indices:
        target = int(start) + window_size + predict_steps - 1
        if target < graph.num_steps:
            targets.append(target)
    return targets, split_source


def target_laser_heat(coords: torch.Tensor, laser: torch.Tensor, scan_angle: torch.Tensor,
                      radius: torch.Tensor, along_radius: torch.Tensor,
                      depth: torch.Tensor) -> torch.Tensor:
    delta = coords - laser.view(1, 3)
    scan_cos = torch.cos(scan_angle)
    scan_sin = torch.sin(scan_angle)
    along_xy = delta[:, 0] * scan_cos + delta[:, 1] * scan_sin
    cross_xy = -delta[:, 0] * scan_sin + delta[:, 1] * scan_cos
    return torch.exp(
        -2.0 * cross_xy.pow(2) / radius.pow(2)
        -2.0 * along_xy.pow(2) / along_radius.pow(2)
        - delta[:, 2].pow(2) / depth.pow(2)
    )


def timed_segment_heat(coords: torch.Tensor, segment_start: torch.Tensor,
                       segment_end: torch.Tensor, radius: torch.Tensor,
                       along_radius: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    segment = segment_end - segment_start
    seg_len_sq = segment.pow(2).sum(dim=1).clamp_min(1.0e-12)
    seg_len = torch.sqrt(seg_len_sq)
    rel = coords.unsqueeze(1) - segment_start.unsqueeze(0)
    progress = ((rel * segment.unsqueeze(0)).sum(dim=-1) / seg_len_sq.view(1, -1)).clamp(0.0, 1.0)
    closest = segment_start.unsqueeze(0) + progress.unsqueeze(-1) * segment.unsqueeze(0)
    delta = coords.unsqueeze(1) - closest

    segment_xy = segment[:, :2]
    seg_xy_len = torch.linalg.vector_norm(segment_xy, dim=1).clamp_min(1.0e-12)
    dir_xy = segment_xy / seg_xy_len.unsqueeze(-1)
    raw_along = (rel[..., :2] * dir_xy.unsqueeze(0)).sum(dim=-1)
    along_outside = raw_along.clamp_max(0.0) + (raw_along - seg_len.view(1, -1)).clamp_min(0.0)
    cross_delta_xy = delta[..., :2] - along_outside.unsqueeze(-1) * dir_xy.unsqueeze(0)
    cross_xy_sq = cross_delta_xy.pow(2).sum(dim=-1)
    heat = torch.exp(
        -2.0 * cross_xy_sq / radius.pow(2)
        -2.0 * along_outside.pow(2) / along_radius.pow(2)
        - delta[..., 2].pow(2) / depth.pow(2)
    )
    return heat, progress


def swept_heat(coords: torch.Tensor, source_lasers: torch.Tensor, target_laser: torch.Tensor,
               radius: torch.Tensor, along_radius: torch.Tensor,
               depth: torch.Tensor) -> torch.Tensor:
    if source_lasers.numel() == 0:
        return torch.zeros(coords.shape[0], device=coords.device, dtype=coords.dtype)
    heat, _ = timed_segment_heat(
        coords, source_lasers, target_laser.view(1, 3).expand_as(source_lasers),
        radius, along_radius, depth,
    )
    return heat.max(dim=1).values


def segment_exposure_integral(graph: DynamicGraph, coords: torch.Tensor, target_step: int,
                              target_time: torch.Tensor, radius: torch.Tensor,
                              along_radius: torch.Tensor, depth: torch.Tensor, decay_s: torch.Tensor,
                              time_scale_to_s: torch.Tensor) -> torch.Tensor:
    starts = graph._laser_path_segment_starts
    ends = graph._laser_path_segment_ends
    start_times = graph._laser_path_segment_start_times
    end_times = graph._laser_path_segment_end_times
    if starts is None or ends is None or start_times is None or end_times is None:
        return torch.zeros(coords.shape[0], device=coords.device, dtype=coords.dtype)

    past = int(config_data_value(graph, "laser_feature_exposure_past_steps", 0))
    future = int(config_data_value(graph, "laser_feature_exposure_future_steps", 0))
    start_idx = max(0, target_step - past)
    end_idx = min(graph.num_steps - 1, target_step + future)
    window_start = graph.times[start_idx].to(device=coords.device, dtype=coords.dtype)
    window_end = graph.times[end_idx].to(device=coords.device, dtype=coords.dtype)
    in_window = (end_times >= window_start) & (start_times <= window_end)
    if not bool(in_window.any().item()):
        mid_times = 0.5 * (start_times + end_times)
        nearest = torch.argmin((mid_times - target_time).abs())
        in_window = torch.zeros_like(mid_times, dtype=torch.bool)
        in_window[nearest] = True

    segment_start = starts[in_window]
    segment_end = ends[in_window]
    segment_start_times = start_times[in_window]
    segment_end_times = end_times[in_window]
    heat, progress = timed_segment_heat(
        coords, segment_start, segment_end, radius, along_radius, depth,
    )
    closest_times = segment_start_times.view(1, -1) + progress * (
        segment_end_times - segment_start_times
    ).view(1, -1)
    rel_dt = (target_time - closest_times) * time_scale_to_s
    return (heat * torch.exp(-rel_dt.abs() / decay_s)).sum(dim=1).clamp(0.0, 1.0)


def config_data_value(graph: DynamicGraph, name: str, default):
    # Values are attached in report_gate_coverage before feature scoring.
    return getattr(graph, f"_diag_{name}", default)


def neighbor_gate_scores(graph: DynamicGraph, source_steps: range, solidus: float) -> torch.Tensor:
    edge_index = graph.edge_index.long()
    src, dst = edge_index[0], edge_index[1]
    num_nodes = graph.num_nodes
    dtype = graph.temperatures.dtype
    device = graph.temperatures.device
    fill = torch.full((num_nodes,), -float("inf"), device=device, dtype=dtype)
    max_over_window = fill.clone()
    for source_step in source_steps:
        temp = graph.temperatures[source_step]
        if temp.ndim > 1:
            temp = temp.squeeze(-1)
        neigh_max = fill.clone()
        neigh_max.scatter_reduce_(0, dst, temp[src], reduce="amax", include_self=False)
        max_over_window = torch.maximum(max_over_window, neigh_max)
    max_over_window = torch.where(torch.isfinite(max_over_window), max_over_window, torch.zeros_like(max_over_window))
    solidus_norm = max(float(solidus) / 1000.0, 1.001)
    return ((max_over_window / 1000.0 - 1.0) / (solidus_norm - 1.0)).clamp(0.0, 1.0)


def make_diag_dataset(graph: DynamicGraph, config: Config) -> DEDTemporalDataset:
    data = config.data
    return DEDTemporalDataset(
        graph,
        window_size=data.window_size,
        predict_steps=data.predict_steps,
        time_indices=[0],
        use_target_laser_features=True,
        laser_feature_radius_mm=data.laser_feature_radius_mm,
        laser_feature_along_radius_mm=data.laser_feature_along_radius_mm,
        laser_feature_depth_mm=data.laser_feature_depth_mm,
        laser_feature_time_scale_to_s=data.laser_feature_time_scale_to_s,
        laser_feature_include_sweep=data.laser_feature_include_sweep,
        laser_feature_include_exposure=data.laser_feature_include_exposure,
        laser_feature_exposure_source=data.laser_feature_exposure_source,
        laser_feature_exposure_past_steps=data.laser_feature_exposure_past_steps,
        laser_feature_exposure_future_steps=data.laser_feature_exposure_future_steps,
        laser_feature_exposure_time_decay_s=data.laser_feature_exposure_time_decay_s,
        laser_feature_exposure_use_segments=data.laser_feature_exposure_use_segments,
        laser_feature_include_exposure_split=data.laser_feature_include_exposure_split,
        laser_feature_include_arrival_time=data.laser_feature_include_arrival_time,
        laser_feature_arrival_time_decay_s=data.laser_feature_arrival_time_decay_s,
        laser_feature_include_neighbor_temp=data.laser_feature_include_neighbor_temp,
        laser_feature_include_neighbor_hot_stats=data.laser_feature_include_neighbor_hot_stats,
        laser_feature_neighbor_hot_threshold=data.laser_feature_neighbor_hot_threshold,
        laser_feature_include_neighbor_warm_stats=data.laser_feature_include_neighbor_warm_stats,
        laser_feature_neighbor_warm_threshold=data.laser_feature_neighbor_warm_threshold,
        laser_feature_include_body_source=data.laser_feature_include_body_source,
        laser_body_radius_mm=data.laser_body_radius_mm,
        laser_body_height_mm=data.laser_body_height_mm,
        laser_body_radius_front_mm=data.laser_body_radius_front_mm,
        laser_body_radius_back_mm=data.laser_body_radius_back_mm,
        laser_body_coeff_front=data.laser_body_coeff_front,
        laser_body_coeff_back=data.laser_body_coeff_back,
        laser_feature_include_path_arrival=True,
        laser_feature_path_arrival_time_decay_s=data.laser_feature_path_arrival_time_decay_s,
        laser_feature_path_arrival_gate_mode=data.laser_feature_path_arrival_gate_mode,
        laser_feature_path_arrival_neighbor_tracks=data.laser_feature_path_arrival_neighbor_tracks,
        laser_feature_include_path_phase=True,
        laser_feature_include_path_coordinates=data.laser_feature_include_path_coordinates,
        laser_feature_include_path_timing=data.laser_feature_include_path_timing,
        laser_feature_include_path_endpoint=True,
        laser_feature_include_path_wake=data.laser_feature_include_path_wake,
        laser_feature_wake_cross_radius_mm=data.laser_feature_wake_cross_radius_mm,
        laser_feature_wake_tail_decay_mm=data.laser_feature_wake_tail_decay_mm,
        laser_feature_wake_lead_decay_mm=data.laser_feature_wake_lead_decay_mm,
        laser_feature_wake_time_decay_s=data.laser_feature_wake_time_decay_s,
        laser_feature_endpoint_radius_mm=data.laser_feature_endpoint_radius_mm,
        laser_feature_endpoint_time_decay_s=data.laser_feature_endpoint_time_decay_s,
    )


def update_hist(hist: torch.Tensor, scores: torch.Tensor, bins: int) -> None:
    if scores.numel() == 0:
        return
    clipped = scores.detach().float().clamp(0.0, 1.0).cpu()
    hist += torch.histc(clipped, bins=bins, min=0.0, max=1.0).to(hist.dtype)


def hist_quantile(hist: torch.Tensor, q: float) -> float:
    total = float(hist.sum().item())
    if total <= 0.0:
        return float("nan")
    target = max(min(float(q), 1.0), 0.0) * total
    cdf = torch.cumsum(hist, dim=0)
    idx = int(torch.searchsorted(cdf, torch.tensor(target, dtype=cdf.dtype)).item())
    idx = max(0, min(idx, hist.numel() - 1))
    return (idx + 0.5) / hist.numel()


def append_gate_examples(rows: list[dict], *, kind: str, gate_name: str,
                         graph: DynamicGraph, target_step: int,
                         chunk_start: int,
                         node_idx: torch.Tensor, scores: torch.Tensor,
                         target: torch.Tensor, gates: dict[str, torch.Tensor],
                         top_k: int, largest: bool) -> None:
    if top_k <= 0 or scores.numel() == 0:
        return
    k = min(top_k, scores.numel())
    ranked_scores, ranked_pos = torch.topk(scores, k=k, largest=largest)
    selected_nodes = node_idx[ranked_pos].detach().cpu().long()
    coords = graph.coords[selected_nodes.to(graph.coords.device)].detach().cpu().float()
    selected_target = target[selected_nodes.to(target.device)].detach().cpu().float()
    gate_columns = [
        "target_heat", "sweep", "exposure", "path_arrival", "path_active",
        "path_pre", "path_post", "path_body_heat", "path_body_gate",
        "endpoint", "neighbor", "process_max",
        "arrival_sweep_product", "arrival_track_support_product",
        "arrival_track_support_blend", "arrival_support_blend",
        "pre_arrival_track_support_blend", "pre_arrival_track_heat_support",
    ]
    for rank in range(k):
        node = int(selected_nodes[rank].item())
        row = {
            "kind": kind,
            "gate": gate_name,
            "step": int(target_step),
            "raw_time": float(graph.times[target_step].detach().cpu().item()),
            "node": node,
            "x_mm": float(coords[rank, 0].item()),
            "y_mm": float(coords[rank, 1].item()),
            "z_mm": float(coords[rank, 2].item()),
            "target_temp": float(selected_target[rank].item()),
            "gate_score": float(ranked_scores[rank].detach().cpu().item()),
        }
        for name in gate_columns:
            value = gates.get(name)
            if value is None:
                row[name] = float("nan")
            else:
                row[name] = float(value[node - int(chunk_start)].detach().cpu().item())
        rows.append(row)
    rows.sort(key=lambda r: r["gate_score"], reverse=largest)
    del rows[top_k:]


def report_gate_confusion(graph: DynamicGraph, config: Config, *, split_name: str,
                          thresholds: list[float], chunk_nodes: int,
                          device_name: str, cold_threshold: float,
                          output_dir: str | None, top_k: int,
                          raw_time_min: float | None = None,
                          raw_time_max: float | None = None,
                          split_indices_path: str | None = None) -> None:
    if config.data.laser_path_mode != "additive_z_scan":
        print("\nGate confusion skipped: laser_path_mode is not additive_z_scan.")
        return
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--gate_device cuda requested but CUDA is not available")

    graph.to(device)
    graph.apply_laser_path_config(config.data)
    diag_dataset = make_diag_dataset(graph, config)

    targets, split_source = gate_target_steps(
        graph, config, split_name, split_indices_path=split_indices_path
    )
    if raw_time_min is not None or raw_time_max is not None:
        filtered_targets = []
        for target in targets:
            raw_time = float(graph.times[target].detach().cpu().item())
            if raw_time_min is not None and raw_time < float(raw_time_min):
                continue
            if raw_time_max is not None and raw_time > float(raw_time_max):
                continue
            filtered_targets.append(target)
        targets = filtered_targets
    thresholds = sorted(float(t) for t in thresholds)
    bins = 1000
    radius = torch.as_tensor(float(config.data.laser_feature_radius_mm), device=device, dtype=graph.coords.dtype)
    along_radius = torch.as_tensor(
        max(float(config.data.laser_feature_along_radius_mm), float(config.data.laser_feature_radius_mm)),
        device=device,
        dtype=graph.coords.dtype,
    )
    depth = torch.as_tensor(float(config.data.laser_feature_depth_mm), device=device, dtype=graph.coords.dtype)
    decay_s = torch.as_tensor(
        float(config.data.laser_feature_exposure_time_decay_s), device=device, dtype=graph.coords.dtype,
    )
    feature_time_scale = torch.as_tensor(
        float(config.data.laser_feature_time_scale_to_s), device=device, dtype=graph.coords.dtype,
    )
    solidus = float(getattr(config.material, "solidus_temp", 1604.85))
    window_size = int(config.data.window_size)
    predict_steps = int(config.data.predict_steps)

    gate_names = [
        "target_heat", "sweep", "exposure", "path_arrival", "path_active",
        "path_pre", "path_post", "path_body_heat", "path_body_gate",
        "endpoint", "neighbor", "process_max",
        "arrival_sweep_product", "arrival_track_support_product",
        "arrival_track_support_blend", "arrival_support_blend",
        "pre_arrival_track_support_blend", "pre_arrival_track_heat_support",
        "pre_arrival_endpoint_sweep",
    ]
    stats = {
        name: {
            "hot_count": 0,
            "cold_count": 0,
            "hot_hist": torch.zeros(bins, dtype=torch.float64),
            "cold_hist": torch.zeros(bins, dtype=torch.float64),
            "thresholds": {
                threshold: {"hot_positive": 0, "cold_positive": 0}
                for threshold in thresholds
            },
        }
        for name in gate_names
    }
    hot_total = 0
    cold_total = 0
    valid_total = 0
    true_hot_low_examples: list[dict] = []
    cold_high_examples: list[dict] = []
    t_start = time.time()

    print(
        f"\nGate confusion ({split_name}): {len(targets)} windows, "
        f"solidus>={solidus:.1f} C, cold<={cold_threshold:.1f} C"
    )
    print(f"  split source: {split_source}")
    if raw_time_min is not None or raw_time_max is not None:
        print(f"  raw time filter: min={raw_time_min}, max={raw_time_max}")

    for idx, target_step in enumerate(targets, start=1):
        target = graph.temperatures[target_step]
        if target.ndim > 1:
            target = target.squeeze(-1)
        valid = graph.live[target_step] > 0.5
        hot = (target >= solidus) & valid
        cold = (target <= float(cold_threshold)) & valid
        hot_total += int(hot.sum().item())
        cold_total += int(cold.sum().item())
        valid_total += int(valid.sum().item())

        start_step = target_step - predict_steps - window_size + 1
        source_steps = range(max(0, start_step), target_step)
        target_laser = graph._laser_positions[target_step]
        scan_angle = graph._scan_directions[target_step]
        source_lasers = (
            graph._laser_positions[list(source_steps)]
            if target_step > start_step else graph._laser_positions[0:0]
        )
        neighbor_gate = neighbor_gate_scores(graph, source_steps, solidus)

        for start in range(0, graph.num_nodes, chunk_nodes):
            end = min(graph.num_nodes, start + chunk_nodes)
            coords = graph.coords[start:end]
            target_heat = target_laser_heat(coords, target_laser, scan_angle, radius, along_radius, depth).clamp(0.0, 1.0)
            sweep = swept_heat(coords, source_lasers, target_laser, radius, along_radius, depth).clamp(0.0, 1.0)
            exposure = segment_exposure_integral(
                graph,
                coords,
                target_step,
                graph.times[target_step].to(device=device, dtype=coords.dtype),
                radius,
                along_radius,
                depth,
                decay_s,
                feature_time_scale,
            ).clamp(0.0, 1.0)
            path_arrival = diag_dataset._analytic_path_arrival_features(
                coords=coords,
                target_time=graph.times[target_step],
                radius=radius,
                along_radius=along_radius,
                depth=depth,
                device=device,
                dtype=coords.dtype,
            )[7].squeeze(-1).clamp(0.0, 1.0)
            phase = diag_dataset._analytic_path_phase_features(
                coords=coords,
                target_time=graph.times[target_step],
                radius=radius,
                depth=depth,
                device=device,
                dtype=coords.dtype,
            )
            path_active = phase[7].squeeze(-1).clamp(0.0, 1.0)
            path_pre = phase[8].squeeze(-1).clamp(0.0, 1.0)
            path_post = phase[9].squeeze(-1).clamp(0.0, 1.0)
            endpoint = diag_dataset._analytic_path_endpoint_features(
                coords=coords,
                target_time=graph.times[target_step],
                depth=depth,
                device=device,
                dtype=coords.dtype,
            )[3].squeeze(-1).clamp(0.0, 1.0)
            path_body = diag_dataset._analytic_path_body_support_features(
                coords=coords,
                device=device,
                dtype=coords.dtype,
            )
            path_body_heat = path_body[0].squeeze(-1).clamp(0.0, 1.0)
            path_body_gate = path_body[1].squeeze(-1).clamp(0.0, 1.0)
            neighbor = neighbor_gate[start:end].to(device=device, dtype=coords.dtype).clamp(0.0, 1.0)
            support_track = torch.maximum(sweep, endpoint)
            support_with_neighbor = torch.maximum(support_track, neighbor)
            track_heat_support = torch.stack([
                target_heat, sweep, endpoint, neighbor,
            ], dim=0).max(dim=0).values
            gates = {
                "target_heat": target_heat,
                "sweep": sweep,
                "exposure": exposure,
                "path_arrival": path_arrival,
                "path_active": path_active,
                "path_pre": path_pre,
                "path_post": path_post,
                "path_body_heat": path_body_heat,
                "path_body_gate": path_body_gate,
                "endpoint": endpoint,
                "neighbor": neighbor,
                "process_max": torch.stack([
                    target_heat, sweep, exposure, path_arrival,
                    path_active, path_pre, path_post, endpoint,
                ], dim=0).max(dim=0).values,
                "arrival_sweep_product": path_arrival * sweep,
                "arrival_track_support_product": path_arrival * support_track,
                "arrival_track_support_blend": path_arrival * (0.45 + 0.55 * support_track).clamp(0.0, 1.0),
                "arrival_support_blend": path_arrival * (0.65 + 0.35 * support_with_neighbor).clamp(0.0, 1.0),
                "pre_arrival_track_support_blend": path_pre * (0.45 + 0.55 * support_with_neighbor).clamp(0.0, 1.0),
                "pre_arrival_track_heat_support": path_pre * (track_heat_support >= 0.003).to(dtype=coords.dtype),
                "pre_arrival_endpoint_sweep": torch.stack([
                    path_pre, path_active, endpoint, sweep,
                ], dim=0).max(dim=0).values,
            }

            hot_chunk = hot[start:end]
            cold_chunk = cold[start:end]
            for name, scores in gates.items():
                hot_scores = scores[hot_chunk]
                cold_scores = scores[cold_chunk]
                stats[name]["hot_count"] += int(hot_scores.numel())
                stats[name]["cold_count"] += int(cold_scores.numel())
                update_hist(stats[name]["hot_hist"], hot_scores, bins)
                update_hist(stats[name]["cold_hist"], cold_scores, bins)
                for threshold in thresholds:
                    stats[name]["thresholds"][threshold]["hot_positive"] += int((hot_scores >= threshold).sum().item())
                    stats[name]["thresholds"][threshold]["cold_positive"] += int((cold_scores >= threshold).sum().item())

            node_idx = torch.arange(start, end, device=device)
            process_scores = gates["process_max"]
            append_gate_examples(
                true_hot_low_examples,
                kind="true_hot_low_process_gate",
                gate_name="process_max",
                graph=graph,
                target_step=target_step,
                chunk_start=start,
                node_idx=node_idx[hot_chunk],
                scores=process_scores[hot_chunk],
                target=target,
                gates=gates,
                top_k=top_k,
                largest=False,
            )
            append_gate_examples(
                cold_high_examples,
                kind="cold_high_process_gate",
                gate_name="process_max",
                graph=graph,
                target_step=target_step,
                chunk_start=start,
                node_idx=node_idx[cold_chunk],
                scores=process_scores[cold_chunk],
                target=target,
                gates=gates,
                top_k=top_k,
                largest=True,
            )

        if idx % 25 == 0 or idx == len(targets):
            elapsed = time.time() - t_start
            print(f"  scored {idx}/{len(targets)} windows ({elapsed:.1f}s)")

    print(f"  valid nodes: {valid_total:,}, true-hot nodes: {hot_total:,}, cold nodes: {cold_total:,}")

    if output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path("results") / f"laser_gate_confusion_{split_name}_{stamp}"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    threshold_rows = []
    quantile_rows = []
    quantiles = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99, 0.999]
    for name in gate_names:
        gate_stats = stats[name]
        hot_count = gate_stats["hot_count"]
        cold_count = gate_stats["cold_count"]
        for threshold in thresholds:
            row_stats = gate_stats["thresholds"][threshold]
            hot_positive = row_stats["hot_positive"]
            cold_positive = row_stats["cold_positive"]
            threshold_rows.append({
                "gate": name,
                "threshold": threshold,
                "hot_recall": hot_positive / hot_count if hot_count else 0.0,
                "cold_positive_rate": cold_positive / cold_count if cold_count else 0.0,
                "precision_hot_vs_cold": hot_positive / (hot_positive + cold_positive) if (hot_positive + cold_positive) else 0.0,
                "hot_positive": hot_positive,
                "cold_positive": cold_positive,
                "hot_count": hot_count,
                "cold_count": cold_count,
            })
        for group_name, hist in (("true_hot", gate_stats["hot_hist"]), ("cold", gate_stats["cold_hist"])):
            row = {"gate": name, "group": group_name, "count": int(hist.sum().item())}
            for q in quantiles:
                row[f"p{q * 100:g}"] = hist_quantile(hist, q)
            quantile_rows.append(row)

    threshold_csv = output_path / "gate_thresholds.csv"
    quantile_csv = output_path / "gate_quantiles.csv"
    examples_csv = output_path / "gate_examples.csv"
    with threshold_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(threshold_rows)
    with quantile_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(quantile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(quantile_rows)
    examples = true_hot_low_examples + cold_high_examples
    if examples:
        with examples_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(examples[0].keys()))
            writer.writeheader()
            writer.writerows(examples)

    summary_md = output_path / "summary.md"
    candidates = [row for row in threshold_rows if row["hot_recall"] > 0.0]
    candidates.sort(key=lambda r: (-r["hot_recall"], r["cold_positive_rate"], -r["precision_hot_vs_cold"]))
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Laser Gate Confusion\n\n")
        f.write(f"split: {split_name}\n\n")
        f.write(f"solidus: {solidus:.6f} C\n\n")
        f.write(f"cold_threshold: {cold_threshold:.6f} C\n\n")
        f.write(f"valid_nodes: {valid_total}\n\n")
        f.write(f"true_hot_nodes: {hot_total}\n\n")
        f.write(f"cold_nodes: {cold_total}\n\n")
        f.write("## Best Threshold Rows\n\n")
        f.write("| gate | threshold | hot recall | cold positive rate | hot-vs-cold precision |\n")
        f.write("| --- | ---: | ---: | ---: | ---: |\n")
        for row in candidates[:20]:
            f.write(
                f"| {row['gate']} | {row['threshold']:.4g} | "
                f"{row['hot_recall']:.6f} | {row['cold_positive_rate']:.8f} | "
                f"{row['precision_hot_vs_cold']:.8f} |\n"
            )

    print(f"\nGate confusion files written to: {output_path}")
    print("  Best rows by hot recall, then cold false-positive rate:")
    for row in candidates[:10]:
        print(
            f"  {row['gate']:30s} th={row['threshold']:<8.4g} "
            f"recall={row['hot_recall']:.4f} "
            f"cold+={row['cold_positive_rate']:.8f} "
            f"prec={row['precision_hot_vs_cold']:.8f}"
        )


def report_gate_coverage(graph: DynamicGraph, config: Config, *, split_name: str,
                         thresholds: list[float], chunk_nodes: int,
                         device_name: str,
                         split_indices_path: str | None = None) -> None:
    if config.data.laser_path_mode != "additive_z_scan":
        print("\nGate coverage skipped: laser_path_mode is not additive_z_scan.")
        return
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--gate_device cuda requested but CUDA is not available")

    graph.to(device)
    graph.apply_laser_path_config(config.data)
    for name in (
        "laser_feature_exposure_past_steps",
        "laser_feature_exposure_future_steps",
    ):
        setattr(graph, f"_diag_{name}", getattr(config.data, name))

    targets, split_source = gate_target_steps(
        graph, config, split_name, split_indices_path=split_indices_path
    )
    thresholds = sorted(float(t) for t in thresholds)
    radius = torch.as_tensor(float(config.data.laser_feature_radius_mm), device=device, dtype=graph.coords.dtype)
    along_radius = torch.as_tensor(
        max(float(config.data.laser_feature_along_radius_mm), float(config.data.laser_feature_radius_mm)),
        device=device,
        dtype=graph.coords.dtype,
    )
    depth = torch.as_tensor(float(config.data.laser_feature_depth_mm), device=device, dtype=graph.coords.dtype)
    decay_s = torch.as_tensor(float(config.data.laser_feature_exposure_time_decay_s), device=device, dtype=graph.coords.dtype)
    feature_time_scale = torch.as_tensor(float(config.data.laser_feature_time_scale_to_s), device=device, dtype=graph.coords.dtype)
    window_size = int(config.data.window_size)
    predict_steps = int(config.data.predict_steps)
    solidus = float(config.material.solidus_temp)
    chunk_nodes = max(1, int(chunk_nodes))

    stats = {
        "process": {t: {"tp": 0, "positive": 0} for t in thresholds},
        "final": {t: {"tp": 0, "positive": 0} for t in thresholds},
    }
    hot_total = 0
    valid_total = 0
    process_hot_scores = []
    final_hot_scores = []

    print(f"\nXML gate coverage ({split_name}, {len(targets)} target windows, device={device}):")
    print(f"  split source: {split_source}")
    for idx, target_step in enumerate(targets, start=1):
        target = graph.temperatures[target_step]
        if target.ndim > 1:
            target = target.squeeze(-1)
        valid = graph.live[target_step] > 0.5
        hot = (target >= solidus) & valid
        hot_total += int(hot.sum().item())
        valid_total += int(valid.sum().item())

        start_step = target_step - predict_steps - window_size + 1
        source_steps = range(max(0, start_step), target_step)
        target_laser = graph._laser_positions[target_step]
        scan_angle = graph._scan_directions[target_step]
        source_lasers = graph._laser_positions[list(source_steps)] if target_step > start_step else graph._laser_positions[0:0]
        neighbor_gate = None
        if bool(getattr(config.data, "laser_feature_include_neighbor_temp", False)):
            neighbor_gate = neighbor_gate_scores(graph, source_steps, solidus)

        for start in range(0, graph.num_nodes, chunk_nodes):
            end = min(graph.num_nodes, start + chunk_nodes)
            coords = graph.coords[start:end]
            process_gate = target_laser_heat(coords, target_laser, scan_angle, radius, along_radius, depth)
            if bool(getattr(config.data, "laser_feature_include_sweep", False)):
                process_gate = torch.maximum(
                    process_gate,
                    swept_heat(coords, source_lasers, target_laser, radius, along_radius, depth),
                )
            if bool(getattr(config.data, "laser_feature_include_exposure", False)):
                process_gate = torch.maximum(
                    process_gate,
                    segment_exposure_integral(
                        graph,
                        coords,
                        target_step,
                        graph.times[target_step].to(device=device, dtype=coords.dtype),
                        radius,
                        along_radius,
                        depth,
                        decay_s,
                        feature_time_scale,
                    ),
                )
            process_gate = process_gate.clamp(0.0, 1.0)
            final_gate = process_gate
            if neighbor_gate is not None:
                final_gate = torch.maximum(final_gate, neighbor_gate[start:end].to(dtype=process_gate.dtype))

            hot_chunk = hot[start:end]
            valid_chunk = valid[start:end]
            if bool(hot_chunk.any().item()):
                process_hot_scores.append(process_gate[hot_chunk].detach().float().cpu())
                final_hot_scores.append(final_gate[hot_chunk].detach().float().cpu())
            for threshold in thresholds:
                process_positive = (process_gate >= threshold) & valid_chunk
                final_positive = (final_gate >= threshold) & valid_chunk
                stats["process"][threshold]["tp"] += int((process_positive & hot_chunk).sum().item())
                stats["process"][threshold]["positive"] += int(process_positive.sum().item())
                stats["final"][threshold]["tp"] += int((final_positive & hot_chunk).sum().item())
                stats["final"][threshold]["positive"] += int(final_positive.sum().item())

        if idx % 50 == 0 or idx == len(targets):
            print(f"  scored {idx}/{len(targets)} windows")

    print(f"  valid nodes: {valid_total:,}, true solidus nodes: {hot_total:,}")
    for name in ("process", "final"):
        print(f"\n  {name} gate thresholds:")
        print("    threshold     recall   precision        tp      positive")
        for threshold in thresholds:
            tp = stats[name][threshold]["tp"]
            positive = stats[name][threshold]["positive"]
            recall = tp / hot_total if hot_total else 0.0
            precision = tp / positive if positive else 0.0
            print(f"    {threshold:9.4g}   {recall:8.4f}   {precision:9.6f}  {tp:8d}  {positive:12d}")

    for name, values in (("process", process_hot_scores), ("final", final_hot_scores)):
        if values:
            scores = torch.cat(values)
            qs = torch.quantile(scores, torch.tensor([0.01, 0.05, 0.10, 0.50], dtype=scores.dtype))
            print(
                f"\n  true-hot {name} gate quantiles: "
                f"p01={qs[0].item():.6f}, p05={qs[1].item():.6f}, "
                f"p10={qs[2].item():.6f}, p50={qs[3].item():.6f}"
            )


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")
    if args.offset_step_s <= 0:
        raise ValueError("--offset_step_s must be positive")

    config = Config.from_yaml(args.config)
    if args.laser_xml is not None:
        config.data.laser_xml_path = args.laser_xml
        config.data.laser_path_mode = "additive_z_scan"
    vtu_dir = args.vtu_dir or config.data.vtu_dir
    threshold = args.solidus if args.solidus is not None else args.hot_threshold

    graph = load_graph(config, vtu_dir, args.cache_dir, args.no_cache, args.rebuild_cache)
    hotspot_pos, max_temps, hotspot_nodes = hottest_live_positions(graph, args.top_k)
    times = graph.times.cpu()
    hot_mask = max_temps >= float(threshold)
    if args.raw_time_min is not None:
        hot_mask &= times >= float(args.raw_time_min)
    if args.raw_time_max is not None:
        hot_mask &= times <= float(args.raw_time_max)
    if hot_mask.sum() == 0:
        hot_mask = max_temps >= float(args.hot_threshold)
        if args.raw_time_min is not None:
            hot_mask &= times >= float(args.raw_time_min)
        if args.raw_time_max is not None:
            hot_mask &= times <= float(args.raw_time_max)
    if hot_mask.sum() == 0:
        raise RuntimeError("No hot frames found; lower --hot_threshold.")

    print(f"Frames: {graph.num_steps}, nodes: {graph.num_nodes}")
    print(f"Hot frames used: {int(hot_mask.sum())} / {graph.num_steps} (threshold={threshold:.1f} C)")
    print(f"Configured offset: {config.data.laser_path_time_offset_s:.6f} s")
    print(f"Configured time scale: {config.data.laser_path_time_scale_to_s:.12g} s/raw")
    if config.data.laser_xml_path:
        print(f"Laser XML: {config.data.laser_xml_path}")
    print(
        "Configured path variant: "
        f"alternate_layer_scan_direction={config.data.laser_alternate_layer_scan_direction}, "
        f"reverse_hatch_order_parity={config.data.laser_reverse_hatch_order_parity}"
    )

    base_offset = float(config.data.laser_path_time_offset_s)
    base_alternate = bool(config.data.laser_alternate_layer_scan_direction)
    base_reverse = int(config.data.laser_reverse_hatch_order_parity)
    radius = float(args.offset_radius_s)
    step = float(args.offset_step_s)
    n = int(math.floor((2.0 * radius) / step)) + 1
    offsets = [base_offset - radius + i * step for i in range(n)]
    if not any(abs(o - base_offset) < 1.0e-12 for o in offsets):
        offsets.append(base_offset)
    offsets = sorted(offsets)

    base_scale = float(config.data.laser_path_time_scale_to_s)
    if args.scale_min is None or args.scale_max is None:
        scales = [base_scale]
    else:
        if args.scale_min <= 0 or args.scale_max <= 0:
            raise ValueError("--scale_min and --scale_max must be positive")
        if args.scale_min > args.scale_max:
            raise ValueError("--scale_min cannot exceed --scale_max")
        if args.scale_steps <= 1:
            scales = [float(args.scale_min)]
        else:
            scales = np.linspace(float(args.scale_min), float(args.scale_max), int(args.scale_steps)).tolist()
        if not any(abs(s - base_scale) < 1.0e-12 for s in scales):
            scales.append(base_scale)
        scales = sorted(scales)

    variants = path_variants(config, args.sweep_path_variants)
    if args.sweep_path_variants:
        print("\nPath variants being compared:")
        for name, alternate, reverse in variants:
            print(
                f"  {name:18s} alternate_layer={str(alternate):5s} "
                f"reverse_hatch_order_parity={reverse}"
            )

    rows = []
    hot_np = hot_mask.numpy().astype(bool)
    hp = hotspot_pos.numpy()
    for variant_name, alternate, reverse in variants:
        for scale in scales:
            for offset in offsets:
                laser = apply_offset(
                    graph,
                    config,
                    offset,
                    scale,
                    alternate_layer_scan_direction=alternate,
                    reverse_hatch_order_parity=reverse,
                ).numpy()
                delta = hp[hot_np] - laser[hot_np]
                dist_xy = np.linalg.norm(delta[:, :2], axis=1)
                dist_3d = np.linalg.norm(delta, axis=1)
                rows.append((variant_name, alternate, reverse, scale, offset, summarize(dist_xy, dist_3d)))

    rows.sort(key=lambda item: (item[5]["p95_xy"], item[5]["p50_xy"]))
    print("\nBest path/scale/offset pairs by hotspot P95 XY distance:")
    print("  variant             alt rev  scale_s/raw     offset_s    mean_xy  p50_xy  p90_xy  p95_xy  mean_3d  p95_3d")
    for variant_name, alternate, reverse, scale, offset, stats in rows[:10]:
        print(
            f"  {variant_name:18s} {int(alternate):3d} {reverse:3d}  "
            f"{scale:12.9f} {offset:9.4f}  "
            f"{stats['mean_xy']:7.3f} {stats['p50_xy']:7.3f} "
            f"{stats['p90_xy']:7.3f} {stats['p95_xy']:7.3f} "
            f"{stats['mean_3d']:8.3f} {stats['p95_3d']:7.3f}"
        )

    if len(scales) == 1 and not args.sweep_path_variants:
        print("\nBest offsets by hotspot P95 XY distance:")
        print("  offset_s    mean_xy  p50_xy  p90_xy  p95_xy  mean_3d  p95_3d")
        for _, _, _, _, offset, stats in rows[:10]:
            print(
                f"  {offset:8.4f}  {stats['mean_xy']:7.3f} {stats['p50_xy']:7.3f} "
                f"{stats['p90_xy']:7.3f} {stats['p95_xy']:7.3f} "
                f"{stats['mean_3d']:8.3f} {stats['p95_3d']:7.3f}"
            )

    best_variant, best_alternate, best_reverse, best_scale, best_offset, _ = rows[0]
    config.data.laser_alternate_layer_scan_direction = bool(best_alternate)
    config.data.laser_reverse_hatch_order_parity = int(best_reverse)
    config.data.laser_path_time_scale_to_s = float(best_scale)
    config.data.laser_path_time_offset_s = float(best_offset)
    laser = apply_offset(
        graph,
        config,
        best_offset,
        best_scale,
        alternate_layer_scan_direction=best_alternate,
        reverse_hatch_order_parity=best_reverse,
    ).numpy()
    base_laser = apply_offset(
        graph,
        config,
        base_offset,
        base_scale,
        alternate_layer_scan_direction=base_alternate,
        reverse_hatch_order_parity=base_reverse,
    ).numpy()

    print("\nFocus frames:")
    print("  raw_time step node maxT    hot_xyz(mm)                 base_laser(mm)              best_laser(mm)")
    for raw_time in args.focus_raw_time:
        idx = int(torch.argmin(torch.abs(times - float(raw_time))).item())
        hot_xyz = hp[idx]
        print(
            f"  {times[idx].item():7.0f} {idx:4d} {int(hotspot_nodes[idx]):5d} "
            f"{float(max_temps[idx]):7.1f}  "
            f"[{hot_xyz[0]:7.3f},{hot_xyz[1]:7.3f},{hot_xyz[2]:5.3f}]  "
            f"[{base_laser[idx,0]:7.3f},{base_laser[idx,1]:7.3f},{base_laser[idx,2]:5.3f}]  "
            f"[{laser[idx,0]:7.3f},{laser[idx,1]:7.3f},{laser[idx,2]:5.3f}]"
        )

    print_focus_node_windows(
        graph,
        config,
        focus_nodes=args.focus_node,
        focus_steps=args.focus_step,
        window_steps=args.focus_window_steps,
        base_laser=base_laser,
        best_laser=laser,
        best_offset=best_offset,
        best_scale=best_scale,
        focus_output_csv=args.focus_output_csv,
    )

    print("\nSuggested YAML override:")
    print(f"  # selected path variant: {best_variant}")
    print(f"  laser_path_time_scale_to_s: {best_scale:.12g}")
    print(f"  laser_path_time_offset_s: {best_offset:.12g}")
    print(f"  laser_alternate_layer_scan_direction: {str(bool(best_alternate)).lower()}")
    print(f"  laser_reverse_hatch_order_parity: {int(best_reverse)}")

    if args.gate_coverage:
        report_gate_coverage(
            graph,
            config,
            split_name=args.gate_split,
            thresholds=args.gate_thresholds,
            chunk_nodes=args.gate_chunk_nodes,
            device_name=args.gate_device,
            split_indices_path=args.split_indices,
        )

    if args.gate_confusion:
        report_gate_confusion(
            graph,
            config,
            split_name=args.gate_split,
            thresholds=args.gate_thresholds,
            chunk_nodes=args.gate_chunk_nodes,
            device_name=args.gate_device,
            cold_threshold=args.cold_threshold,
            output_dir=args.gate_confusion_out,
            top_k=args.gate_top_k_examples,
            raw_time_min=args.raw_time_min,
            raw_time_max=args.raw_time_max,
            split_indices_path=args.split_indices,
        )


if __name__ == "__main__":
    main()
