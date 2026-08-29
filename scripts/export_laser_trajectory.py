"""Export and diagnose the prescribed additive_z_scan laser trajectory."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

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


def write_hot_nodes(path: AdditiveZScanPath, args, raw_origin: float, out_dir: Path) -> None:
    if not args.hotspot_vtu:
        return
    from src.data.vtu_loader import VTULoader

    loader = VTULoader(Path(args.hotspot_vtu).parent)
    data = loader.parse_single(Path(args.hotspot_vtu))
    raw_time = args.hotspot_raw_time if args.hotspot_raw_time is not None else data.time
    mask = data.temperature.numpy() >= args.hotspot_threshold
    coords = data.coords.numpy()[mask]
    temps = data.temperature.numpy()[mask]
    if coords.size == 0:
        print(f"No nodes above {args.hotspot_threshold} in {args.hotspot_vtu}")
        return
    features, columns = path.node_process_features(coords, raw_time, raw_origin=raw_origin)
    out_path = out_dir / f"hot_nodes_{Path(args.hotspot_vtu).stem}.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_rank", "temperature", "x_mm", "y_mm", "z_mm", *columns])
        for i in np.argsort(-temps):
            writer.writerow([int(i), float(temps[i]), *coords[i].tolist(), *features[i].tolist()])
    print(f"Hot-node overlay CSV: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export additive_z_scan trajectory CSV files")
    parser.add_argument("--config", default="configs/paper1_fast.yaml")
    parser.add_argument("--xml", default=None, help="Optional para.xml path to read additive_z_scan geometry")
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
    parser.add_argument("--hotspot_vtu", default=None, help="Optional single VTU for high-temperature overlay CSV")
    parser.add_argument("--hotspot_raw_time", type=float, default=None)
    parser.add_argument("--hotspot_threshold", type=float, default=1604.85)
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

    write_hot_nodes(path, args, raw_origin, out_dir)
    meta_path = out_dir / "laser_trajectory_manifest.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Segments CSV: {segments_path}")
    print(f"Samples CSV: {samples_path}")
    print(f"Manifest: {meta_path}")


if __name__ == "__main__":
    main()

