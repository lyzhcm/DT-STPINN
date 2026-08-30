"""Plot laser position against high-temperature VTU nodes for alignment checks."""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dtstpinn_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.vtu_loader import VTULoader
from src.utils.laser_path import AdditiveZScanPath


def raw_time_from_file(path: Path, fallback: int) -> float:
    match = re.match(r"Data-(\d+)\.vtu", path.name, re.IGNORECASE)
    return float(match.group(1)) if match else float(fallback)


def parse_steps(text: str) -> list[int]:
    steps = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not steps:
        raise argparse.ArgumentTypeError("Expected at least one step index")
    return steps


def make_path(args, config: Config) -> AdditiveZScanPath:
    data = config.data
    if args.xml:
        return AdditiveZScanPath.from_xml(
            args.xml,
            time_scale_to_s=args.time_scale_to_s or data.laser_path_time_scale_to_s,
            time_offset_s=args.time_offset_s
            if args.time_offset_s is not None
            else data.laser_path_time_offset_s,
            alternate_layer_scan_direction=data.laser_alternate_layer_scan_direction,
            reverse_hatch_order_parity=args.reverse_hatch_order_parity
            if args.reverse_hatch_order_parity is not None
            else data.laser_reverse_hatch_order_parity,
        )
    path = AdditiveZScanPath.from_config(config)
    if args.time_scale_to_s is None and args.time_offset_s is None and args.reverse_hatch_order_parity is None:
        return path
    return AdditiveZScanPath(
        start_point_mm=path.start_point_mm,
        scan_direction=path.scan_direction,
        scan_length_mm=path.scan_length_mm,
        hatch_direction=path.hatch_direction,
        hatch_count=path.hatch_count,
        hatch_spacing_mm=path.hatch_spacing_mm,
        layer_count=path.layer_count,
        layer_thickness_mm=path.layer_thickness_mm,
        velocity_mm_s=path.velocity_mm_s,
        each_path_time_s=path.each_path_time_s,
        each_layer_time_s=path.each_layer_time_s,
        time_scale_to_s=args.time_scale_to_s or path.time_scale_to_s,
        time_offset_s=args.time_offset_s if args.time_offset_s is not None else path.time_offset_s,
        alternate_layer_scan_direction=path.alternate_layer_scan_direction,
        reverse_hatch_order_parity=args.reverse_hatch_order_parity
        if args.reverse_hatch_order_parity is not None
        else path.reverse_hatch_order_parity,
        body_radius_mm=path.body_radius_mm,
        body_height_mm=path.body_height_mm,
        body_radius_front_mm=path.body_radius_front_mm,
        body_radius_back_mm=path.body_radius_back_mm,
    )


def local_to_xy(origin: np.ndarray, active_dir: np.ndarray, along: np.ndarray, cross: np.ndarray) -> np.ndarray:
    perp = np.asarray([-active_dir[1], active_dir[0], 0.0], dtype=np.float64)
    return origin.reshape(1, 3) + along.reshape(-1, 1) * active_dir.reshape(1, 3) + cross.reshape(-1, 1) * perp.reshape(1, 3)


def ellipsoid_outline_xy(path: AdditiveZScanPath, state: dict[str, object]) -> np.ndarray:
    active_dir = np.asarray(state["active_direction"], dtype=np.float64)
    laser = np.asarray(state["position"], dtype=np.float64)

    theta_front = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 96)
    front_along = path.body_radius_front_mm * np.cos(theta_front)
    front_cross = path.body_radius_mm * np.sin(theta_front)

    theta_back = np.linspace(0.5 * np.pi, 1.5 * np.pi, 96)
    back_along = path.body_radius_back_mm * np.cos(theta_back)
    back_cross = path.body_radius_mm * np.sin(theta_back)

    along = np.concatenate([front_along, back_along])
    cross = np.concatenate([front_cross, back_cross])
    return local_to_xy(laser, active_dir, along, cross)


def plot_step(path: AdditiveZScanPath, data, raw_time: float, raw_origin: float,
              sample_index: int, source_path: Path, out_dir: Path,
              threshold: float, top_k: int | None, show_all_nodes: bool,
              zoom_radius_mm: float | None) -> dict[str, object]:
    coords = data.coords.numpy().astype(np.float64)
    temps = data.temperature.numpy().astype(np.float64)
    state = path.state_at_raw_time(raw_time, raw_origin=raw_origin)
    laser = np.asarray(state["position"], dtype=np.float64)
    segment_start = np.asarray(state["start"], dtype=np.float64)
    segment_end = np.asarray(state["end"], dtype=np.float64)

    hot_mask = temps >= threshold
    hot_indices = np.flatnonzero(hot_mask)
    if hot_indices.size == 0:
        order = np.argsort(-temps)[: min(top_k or 20, temps.size)]
    else:
        order = hot_indices[np.argsort(-temps[hot_indices])]
        if top_k is not None:
            order = order[:top_k]

    selected_coords = coords[order]
    selected_temps = temps[order]
    features, columns = path.node_process_features(selected_coords, raw_time, raw_origin=raw_origin)
    column_index = {name: idx for idx, name in enumerate(columns)}
    distances = features[:, column_index["distance_to_laser_mm"]]
    ellipsoid_flags = features[:, column_index["in_laser_ellipsoid"]]
    track_flags = features[:, column_index["in_track_neighborhood"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    if show_all_nodes:
        ax.scatter(coords[:, 0], coords[:, 1], s=1.0, c="#d0d0d0", alpha=0.25, linewidths=0)
    scatter = ax.scatter(
        selected_coords[:, 0],
        selected_coords[:, 1],
        s=18,
        c=selected_temps,
        cmap="inferno",
        edgecolors="none",
        label="hot nodes" if hot_indices.size else "hottest nodes",
    )
    ax.plot(
        [segment_start[0], segment_end[0]],
        [segment_start[1], segment_end[1]],
        color="#3c3c3c",
        linewidth=1.0,
        linestyle="--",
        label="scan line",
    )
    outline = ellipsoid_outline_xy(path, state)
    ax.plot(outline[:, 0], outline[:, 1], color="#1f77b4", linewidth=1.4, label="ellipsoid XY")
    ax.scatter([laser[0]], [laser[1]], marker="*", s=160, color="#0072b2", label="laser")
    if zoom_radius_mm is not None and zoom_radius_mm > 0.0:
        focus = np.vstack([selected_coords[:, :2], laser[:2].reshape(1, 2)])
        center = 0.5 * (focus.min(axis=0) + focus.max(axis=0))
        radius = max(float(zoom_radius_mm), 0.55 * float(np.ptp(focus, axis=0).max()))
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(
        f"Laser / hotspot alignment: sample {sample_index}, raw {raw_time:.0f}\n"
        f"hot>={threshold:g}C: {hot_indices.size}, nearest selected: {float(distances.min()):.3f} mm"
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (C)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    plot_path = out_dir / f"laser_hotspots_step_{sample_index:05d}.png"
    fig.savefig(plot_path)
    plt.close(fig)

    hottest_local = int(np.argmax(selected_temps))
    return {
        "sample_index": sample_index,
        "vtu_file": str(source_path),
        "raw_time": raw_time,
        "threshold": threshold,
        "hot_node_count": int(hot_indices.size),
        "plotted_node_count": int(order.size),
        "laser_x_mm": float(laser[0]),
        "laser_y_mm": float(laser[1]),
        "laser_z_mm": float(laser[2]),
        "hottest_node_index": int(order[hottest_local]),
        "hottest_temperature": float(selected_temps[hottest_local]),
        "hottest_distance_to_laser_mm": float(distances[hottest_local]),
        "min_plotted_distance_to_laser_mm": float(distances.min()),
        "plotted_in_ellipsoid_count": int(ellipsoid_flags.sum()),
        "plotted_in_track_neighborhood_count": int(track_flags.sum()),
        "plot_path": str(plot_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot laser position and hot VTU nodes")
    parser.add_argument("--config", default="configs/feature_e3_path_phase.yaml")
    parser.add_argument(
        "--xml",
        "--laser_xml",
        dest="xml",
        default=None,
        help="Optional para.xml path",
    )
    parser.add_argument("--vtu_dir", required=True)
    parser.add_argument("--steps", type=parse_steps, required=True,
                        help="Comma-separated VTU sample indices, for example 2128,2131")
    parser.add_argument("--output_dir", default="results/laser_hotspot_alignment")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Hotspot threshold in C; defaults to material.solidus_temp")
    parser.add_argument("--top_k", type=int, default=500,
                        help="Maximum hot nodes to plot per step; use 0 for all hot nodes")
    parser.add_argument("--show_all_nodes", action="store_true")
    parser.add_argument("--zoom_radius_mm", type=float, default=None,
                        help="Zoom each plot around the selected hot nodes and laser")
    parser.add_argument("--time_scale_to_s", type=float, default=None)
    parser.add_argument("--time_offset_s", type=float, default=None)
    parser.add_argument("--reverse_hatch_order_parity", type=int, choices=[-1, 0, 1], default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    path = make_path(args, config)
    loader = VTULoader(args.vtu_dir)
    if loader.num_steps == 0:
        raise SystemExit(f"No Data-*.vtu files found in {args.vtu_dir}")
    raw_times = [raw_time_from_file(fp, i) for i, fp in enumerate(loader.files)]
    raw_origin = raw_times[0]
    threshold = float(args.threshold if args.threshold is not None else config.material.solidus_temp)
    top_k = None if args.top_k == 0 else max(1, int(args.top_k))
    out_dir = Path(args.output_dir)

    summaries = []
    for step in args.steps:
        if step < 0 or step >= loader.num_steps:
            raise IndexError(f"step {step} outside VTU range [0, {loader.num_steps - 1}]")
        source_path = loader.files[step]
        data = loader.parse_single(source_path)
        summary = plot_step(
            path,
            data,
            raw_times[step],
            raw_origin,
            step,
            source_path,
            out_dir,
            threshold,
            top_k,
            bool(args.show_all_nodes),
            args.zoom_radius_mm,
        )
        summaries.append(summary)
        print(
            f"step={step} hot={summary['hot_node_count']} "
            f"hottest={summary['hottest_temperature']:.2f}C "
            f"hottest_dist={summary['hottest_distance_to_laser_mm']:.4f}mm "
            f"plot={summary['plot_path']}"
        )

    summary_path = out_dir / "laser_hotspot_alignment_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
