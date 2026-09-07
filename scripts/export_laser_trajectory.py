"""Export and diagnose the prescribed additive_z_scan laser trajectory."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.vtu_loader import VTULoader
from src.utils.laser_path import AdditiveZScanPath


def raw_times_from_vtu(vtu_dir: str | Path) -> list[float]:
    loader = VTULoader(vtu_dir)
    pattern = re.compile(r"Data-(\d+)\.vtu", re.IGNORECASE)
    raw_times: list[float] = []
    for fp in loader.files:
        match = pattern.match(fp.name)
        raw_times.append(float(match.group(1)) if match else float(len(raw_times)))
    return raw_times


def parse_vec3(text: str) -> np.ndarray:
    values = [float(v.strip()) for v in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected x,y,z")
    return np.asarray(values, dtype=np.float64)


def load_vtu_by_sample_index(vtu_dir: str | Path, sample_index: int):
    loader = VTULoader(vtu_dir)
    if sample_index < 0 or sample_index >= loader.num_steps:
        raise IndexError(
            f"sample index {sample_index} outside VTU range [0, {loader.num_steps - 1}]"
        )
    file_path = loader.files[sample_index]
    return file_path, loader.parse_single(file_path)


def make_path(args, config: Config) -> AdditiveZScanPath:
    data = config.data
    if args.xml:
        return AdditiveZScanPath.from_xml(
            args.xml,
            time_scale_to_s=args.time_scale_to_s or data.laser_path_time_scale_to_s,
            time_offset_s=args.time_offset_s if args.time_offset_s is not None else data.laser_path_time_offset_s,
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


def write_segments(path: AdditiveZScanPath, out_path: Path, raw_origin: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer", "track_in_layer", "physical_track", "forward",
            "start_raw_time", "end_raw_time", "start_s", "end_s",
            "start_x_mm", "start_y_mm", "start_z_mm",
            "end_x_mm", "end_y_mm", "end_z_mm",
        ])
        for seg in path.iter_segments():
            start = np.asarray(seg["start"], dtype=np.float64)
            end = np.asarray(seg["end"], dtype=np.float64)
            writer.writerow([
                seg["layer"], seg["track_in_layer"], seg["physical_track"], int(bool(seg["forward"])),
                path.elapsed_to_raw(float(seg["elapsed_start_s"]), raw_origin=raw_origin),
                path.elapsed_to_raw(float(seg["elapsed_end_s"]), raw_origin=raw_origin),
                seg["elapsed_start_s"], seg["elapsed_end_s"],
                start[0], start[1], start[2], end[0], end[1], end[2],
            ])


def write_samples(path: AdditiveZScanPath, out_path: Path, raw_times: list[float], raw_origin: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_index", "raw_time", "time_s", "layer", "track_in_layer", "physical_track",
            "forward", "progress", "on_scan", "x_mm", "y_mm", "z_mm", "scan_angle_rad",
        ])
        for i, raw_time in enumerate(raw_times):
            state = path.state_at_raw_time(raw_time, raw_origin=raw_origin)
            pos = np.asarray(state["position"], dtype=np.float64)
            writer.writerow([
                i, raw_time, state["elapsed_s"], state["layer"], state["track_in_layer"],
                state["physical_track"], int(bool(state["forward"])), state["progress"],
                int(bool(state["on_scan"])), pos[0], pos[1], pos[2], state["scan_angle_rad"],
            ])


def write_hot_nodes(
    path: AdditiveZScanPath,
    args,
    raw_times: list[float],
    raw_origin: float,
    out_dir: Path,
) -> dict[str, Any] | None:
    if not args.hotspot_vtu and args.hotspot_step_index is None:
        return None

    if args.hotspot_step_index is not None:
        if not args.vtu_dir:
            raise ValueError("--hotspot_step_index requires --vtu_dir")
        source_path, data = load_vtu_by_sample_index(args.vtu_dir, args.hotspot_step_index)
        raw_time = raw_times[args.hotspot_step_index]
    else:
        source_path = Path(args.hotspot_vtu)
        loader = VTULoader(source_path.parent)
        data = loader.parse_single(source_path)
        raw_time = args.hotspot_raw_time if args.hotspot_raw_time is not None else data.time

    mask = data.temperature.numpy() >= args.hotspot_threshold
    node_indices = np.flatnonzero(mask)
    coords = data.coords.numpy()[node_indices]
    temps = data.temperature.numpy()[node_indices]
    state = path.state_at_raw_time(raw_time, raw_origin=raw_origin)
    laser = np.asarray(state["position"], dtype=np.float64)
    summary: dict[str, Any] = {
        "vtu_file": str(source_path),
        "sample_index": int(args.hotspot_step_index) if args.hotspot_step_index is not None else None,
        "raw_time": float(raw_time),
        "threshold": float(args.hotspot_threshold),
        "hot_node_count": int(node_indices.size),
        "laser_x_mm": float(laser[0]),
        "laser_y_mm": float(laser[1]),
        "laser_z_mm": float(laser[2]),
    }
    if coords.size == 0:
        print(f"No nodes above {args.hotspot_threshold} in {source_path}")
        return summary

    order = np.argsort(-temps)
    if args.hotspot_top_k is not None:
        order = order[:args.hotspot_top_k]
    features, columns = path.node_process_features(coords, raw_time, raw_origin=raw_origin)
    column_index = {name: idx for idx, name in enumerate(columns)}
    distances = features[:, column_index["distance_to_laser_mm"]]
    ellipsoid_flags = features[:, column_index["in_laser_ellipsoid"]]
    track_flags = features[:, column_index["in_track_neighborhood"]]
    hottest = int(np.argmax(temps))
    nearest = int(np.argmin(distances))
    summary.update({
        "hottest_node_index": int(node_indices[hottest]),
        "hottest_temperature": float(temps[hottest]),
        "hottest_distance_to_laser_mm": float(distances[hottest]),
        "nearest_hot_node_index": int(node_indices[nearest]),
        "nearest_hot_temperature": float(temps[nearest]),
        "nearest_hot_distance_to_laser_mm": float(distances[nearest]),
        "hot_in_laser_ellipsoid_count": int(ellipsoid_flags.sum()),
        "hot_in_track_neighborhood_count": int(track_flags.sum()),
    })
    out_path = out_dir / f"hot_nodes_{source_path.stem}.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "node_index", "temperature", "x_mm", "y_mm", "z_mm", *columns])
        for rank, i in enumerate(order, start=1):
            writer.writerow([
                rank,
                int(node_indices[i]),
                float(temps[i]),
                *coords[i].tolist(),
                *features[i].tolist(),
            ])
    print(f"Hot-node overlay CSV: {out_path}")
    summary["overlay_csv"] = str(out_path)
    return summary


def fmt_float(value: object, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_alignment_report(
    meta: dict[str, Any],
    segments_path: Path,
    samples_path: Path,
    manifest_path: Path,
    out_path: Path,
) -> None:
    node_diag = meta.get("node_index_diagnosis") or meta.get("diagnosis")
    hotspot = meta.get("hotspot_overlay")
    lines = [
        "# Laser Trajectory Alignment Report",
        "",
        "## Inputs",
        "",
        f"- Config: `{meta.get('config')}`",
        f"- XML: `{meta.get('xml')}`",
        f"- Raw origin: {fmt_float(meta.get('raw_origin'), 6)}",
        f"- Samples: {meta.get('num_samples')}",
        f"- Time scale: {fmt_float(meta.get('time_scale_to_s'), 12)} s/raw",
        f"- Time offset: {fmt_float(meta.get('time_offset_s'), 12)} s",
        f"- Reverse hatch order parity: {meta.get('reverse_hatch_order_parity')}",
        "",
        "## Path Timing",
        "",
        f"- Scan time per track: {fmt_float(meta.get('scan_time_s'), 6)} s",
        f"- Path period: {fmt_float(meta.get('path_period_s'), 6)} s",
        f"- Layer period: {fmt_float(meta.get('layer_period_s'), 6)} s",
        f"- Total tracks: {meta.get('total_tracks')}",
        "",
    ]

    if isinstance(node_diag, dict):
        coord = node_diag.get("coord_mm", [])
        lines.extend([
            "## Node Diagnosis",
            "",
            f"- Sample index: {node_diag.get('sample_index', 'n/a')}",
            f"- VTU file: `{node_diag.get('vtu_file', 'n/a')}`",
            f"- Raw time: {fmt_float(node_diag.get('raw_time'), 3)}",
            f"- Node index: {node_diag.get('node_index', 'n/a')}",
            f"- Coordinate mm: {json.dumps(coord)}",
            f"- Temperature C: {fmt_float(node_diag.get('temperature'), 4)}",
            f"- Laser xyz mm: [{fmt_float(node_diag.get('laser_x_mm'), 6)}, {fmt_float(node_diag.get('laser_y_mm'), 6)}, {fmt_float(node_diag.get('laser_z_mm'), 6)}]",
            f"- Distance to laser: {fmt_float(node_diag.get('distance_to_laser_mm'), 6)} mm",
            f"- Line along/cross: {fmt_float(node_diag.get('line_along_mm'), 6)} / {fmt_float(node_diag.get('line_cross_mm'), 6)} mm",
            f"- Time to arrival: {fmt_float(node_diag.get('time_to_arrival_s'), 6)} s",
            f"- Arrival raw time: {fmt_float(node_diag.get('arrival_raw_time'), 3)}",
            f"- Layer/track: {fmt_float(node_diag.get('layer_idx'), 0)} / {fmt_float(node_diag.get('track_physical'), 0)} physical / {fmt_float(node_diag.get('track_program'), 0)} program",
            f"- Direction sign: {fmt_float(node_diag.get('direction_sign'), 0)}",
            f"- In laser ellipsoid: {fmt_float(node_diag.get('in_laser_ellipsoid'), 0)}",
            f"- In track neighborhood: {fmt_float(node_diag.get('in_track_neighborhood'), 0)}",
            "",
        ])

    if isinstance(hotspot, dict):
        lines.extend([
            "## Hotspot Overlay",
            "",
            f"- VTU file: `{hotspot.get('vtu_file')}`",
            f"- Sample index: {hotspot.get('sample_index')}",
            f"- Raw time: {fmt_float(hotspot.get('raw_time'), 3)}",
            f"- Threshold C: {fmt_float(hotspot.get('threshold'), 2)}",
            f"- Hot nodes: {hotspot.get('hot_node_count')}",
            f"- Hottest node: {hotspot.get('hottest_node_index', 'n/a')} at {fmt_float(hotspot.get('hottest_temperature'), 4)} C, distance {fmt_float(hotspot.get('hottest_distance_to_laser_mm'), 6)} mm",
            f"- Nearest hot node: {hotspot.get('nearest_hot_node_index', 'n/a')} at {fmt_float(hotspot.get('nearest_hot_temperature'), 4)} C, distance {fmt_float(hotspot.get('nearest_hot_distance_to_laser_mm'), 6)} mm",
            f"- Hot nodes in laser ellipsoid: {hotspot.get('hot_in_laser_ellipsoid_count', 'n/a')}",
            f"- Hot nodes in track neighborhood: {hotspot.get('hot_in_track_neighborhood_count', 'n/a')}",
            f"- Overlay CSV: `{hotspot.get('overlay_csv', 'n/a')}`",
            "",
        ])

    lines.extend([
        "## Generated Files",
        "",
        f"- Segments CSV: `{segments_path}`",
        f"- Samples CSV: `{samples_path}`",
        f"- Manifest JSON: `{manifest_path}`",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export additive_z_scan trajectory CSV files")
    parser.add_argument("--config", default="configs/paper1_fast.yaml")
    parser.add_argument(
        "--xml",
        "--laser_xml",
        dest="xml",
        default=None,
        help="Optional para.xml path to read additive_z_scan geometry",
    )
    parser.add_argument("--vtu_dir", default=None, help="Use Data-*.vtu names as raw output times")
    parser.add_argument("--output_dir", default="results/laser_trajectory")
    parser.add_argument("--time_scale_to_s", type=float, default=None)
    parser.add_argument("--time_offset_s", type=float, default=None)
    parser.add_argument("--reverse_hatch_order_parity", type=int, choices=[-1, 0, 1], default=None)
    parser.add_argument("--raw_start", type=float, default=0.0)
    parser.add_argument("--raw_end", type=float, default=None)
    parser.add_argument("--raw_step", type=float, default=20.0)
    parser.add_argument("--diagnose_coord", type=parse_vec3, default=None, help="Coordinate x,y,z in mm")
    parser.add_argument("--diagnose_raw_time", type=float, default=None)
    parser.add_argument("--diagnose_step_index", type=int, default=None,
                        help="VTU sample index for node diagnosis; requires --vtu_dir and --diagnose_node_index")
    parser.add_argument("--diagnose_node_index", type=int, default=None,
                        help="Node index to diagnose at --diagnose_step_index")
    parser.add_argument("--hotspot_vtu", default=None, help="Optional single VTU for high-temperature overlay CSV")
    parser.add_argument("--hotspot_step_index", type=int, default=None,
                        help="VTU sample index for high-temperature overlay CSV; requires --vtu_dir")
    parser.add_argument("--hotspot_raw_time", type=float, default=None)
    parser.add_argument("--hotspot_threshold", type=float, default=1604.85)
    parser.add_argument("--hotspot_top_k", type=int, default=None,
                        help="Limit hot-node overlay CSV to the top-K hottest nodes above threshold")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    path = make_path(args, config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.vtu_dir:
        raw_times = raw_times_from_vtu(args.vtu_dir)
    else:
        raw_end = args.raw_end if args.raw_end is not None else path.elapsed_to_raw(
            path.layer_period_s * path.layer_count,
            raw_origin=args.raw_start,
        )
        raw_times = np.arange(args.raw_start, raw_end + 0.5 * args.raw_step, args.raw_step).tolist()
    raw_origin = raw_times[0] if raw_times else args.raw_start

    segments_path = out_dir / "laser_segments.csv"
    samples_path = out_dir / "laser_samples.csv"
    write_segments(path, segments_path, raw_origin)
    write_samples(path, samples_path, raw_times, raw_origin)

    meta = {
        "config": args.config,
        "xml": args.xml,
        "raw_origin": raw_origin,
        "num_samples": len(raw_times),
        "scan_time_s": path.scan_time_s,
        "path_period_s": path.path_period_s,
        "layer_period_s": path.layer_period_s,
        "total_tracks": path.total_tracks,
        "time_scale_to_s": path.time_scale_to_s,
        "time_offset_s": path.time_offset_s,
        "reverse_hatch_order_parity": path.reverse_hatch_order_parity,
    }

    if args.diagnose_coord is not None and args.diagnose_raw_time is not None:
        features, columns = path.node_process_features(
            args.diagnose_coord.reshape(1, 3), args.diagnose_raw_time, raw_origin=raw_origin
        )
        diag = {name: float(value) for name, value in zip(columns, features[0].tolist())}
        meta["diagnosis"] = {
            "coord_mm": args.diagnose_coord.tolist(),
            "raw_time": args.diagnose_raw_time,
            **diag,
        }
        print("Node diagnosis:")
        print(json.dumps(meta["diagnosis"], indent=2))

    if args.diagnose_node_index is not None or args.diagnose_step_index is not None:
        if args.diagnose_node_index is None or args.diagnose_step_index is None:
            raise ValueError("--diagnose_node_index and --diagnose_step_index must be used together")
        if not args.vtu_dir:
            raise ValueError("node-index diagnosis requires --vtu_dir")
        source_path, data = load_vtu_by_sample_index(args.vtu_dir, args.diagnose_step_index)
        node = int(args.diagnose_node_index)
        if node < 0 or node >= data.coords.shape[0]:
            raise IndexError(f"node index {node} outside node range [0, {data.coords.shape[0] - 1}]")
        raw_time = raw_times[args.diagnose_step_index]
        coord = data.coords[node].numpy().astype(np.float64)
        features, columns = path.node_process_features(coord.reshape(1, 3), raw_time, raw_origin=raw_origin)
        diag = {name: float(value) for name, value in zip(columns, features[0].tolist())}
        meta["node_index_diagnosis"] = {
            "sample_index": int(args.diagnose_step_index),
            "vtu_file": str(source_path),
            "raw_time": float(raw_time),
            "node_index": node,
            "coord_mm": coord.tolist(),
            "temperature": float(data.temperature[node].item()),
            "live": float(data.live[node].item()),
            "boundary": float(data.boundary[node].item()),
            **diag,
        }
        print("Node-index diagnosis:")
        print(json.dumps(meta["node_index_diagnosis"], indent=2))

    hotspot_summary = write_hot_nodes(path, args, raw_times, raw_origin, out_dir)
    if hotspot_summary is not None:
        meta["hotspot_overlay"] = hotspot_summary
    meta_path = out_dir / "laser_trajectory_manifest.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    report_path = out_dir / "laser_alignment_report.md"
    write_alignment_report(meta, segments_path, samples_path, meta_path, report_path)
    print(f"Segments CSV: {segments_path}")
    print(f"Samples CSV: {samples_path}")
    print(f"Manifest: {meta_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
