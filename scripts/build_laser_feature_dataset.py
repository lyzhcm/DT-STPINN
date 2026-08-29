"""Build chunked additive-path feature tables for E1/E2/E3 experiments."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.preprocessing import split_indices
from src.data.vtu_loader import VTULoader
from src.utils.laser_path import AdditiveZScanPath


FEATURE_GROUPS = {
    "E1": [
        "laser_x_mm", "laser_y_mm", "laser_z_mm",
        "dx_mm", "dy_mm", "dz_mm", "distance_to_laser_mm",
    ],
    "E2": [
        "laser_x_mm", "laser_y_mm", "laser_z_mm",
        "dx_mm", "dy_mm", "dz_mm", "distance_to_laser_mm",
        "current_along_mm", "current_cross_mm",
        "line_along_mm", "line_cross_mm", "time_to_arrival_s", "arrival_raw_time",
    ],
    "E3": [
        "laser_x_mm", "laser_y_mm", "laser_z_mm",
        "dx_mm", "dy_mm", "dz_mm", "distance_to_laser_mm",
        "current_along_mm", "current_cross_mm",
        "line_along_mm", "line_cross_mm", "time_to_arrival_s", "arrival_raw_time",
        "layer_idx", "track_physical", "track_program", "direction_sign",
        "layer_norm", "track_physical_norm", "track_program_norm",
        "in_laser_ellipsoid", "in_track_neighborhood",
    ],
}


def raw_time_from_file(path: Path, fallback: int) -> float:
    match = re.match(r"Data-(\d+)\.vtu", path.name, re.IGNORECASE)
    return float(match.group(1)) if match else float(fallback)


def select_steps(config: Config, raw_times: list[float], selector: str, explicit: str | None) -> list[int]:
    eligible = list(range(config.data.window_size, max(config.data.window_size, len(raw_times))))
    if explicit:
        wanted = {int(v.strip()) for v in explicit.split(",") if v.strip()}
        return [i for i in eligible if i in wanted]
    train_idx, val_idx, test_idx = split_indices(
        len(raw_times),
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
    )
    split_map = {
        "all": eligible,
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
    if selector not in split_map:
        raise ValueError(f"Unknown step selector {selector!r}")
    selected = set(split_map[selector])
    return [i for i in eligible if i in selected]


def make_path(args, config: Config) -> AdditiveZScanPath:
    if args.xml:
        return AdditiveZScanPath.from_xml(
            args.xml,
            time_scale_to_s=config.data.laser_path_time_scale_to_s,
            time_offset_s=config.data.laser_path_time_offset_s,
            alternate_layer_scan_direction=config.data.laser_alternate_layer_scan_direction,
            reverse_hatch_order_parity=config.data.laser_reverse_hatch_order_parity,
        )
    return AdditiveZScanPath.from_config(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build laser trajectory feature NPZ chunks")
    parser.add_argument("--config", default="configs/paper1_fast.yaml")
    parser.add_argument("--xml", default=None)
    parser.add_argument("--vtu_dir", required=True)
    parser.add_argument("--output_dir", default="data/processed/laser_features")
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--steps", default=None, help="Comma-separated target step indices; overrides --split")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    loader = VTULoader(args.vtu_dir)
    if loader.num_steps == 0:
        raise SystemExit(f"No VTU files found in {args.vtu_dir}")

    first = loader.parse_single(loader.files[0])
    coords = first.coords.numpy().astype(np.float64)
    raw_times = [raw_time_from_file(fp, i) for i, fp in enumerate(loader.files)]
    raw_origin = raw_times[0]
    path = make_path(args, config)
    steps = select_steps(config, raw_times, args.split, args.steps)
    if args.max_steps is not None:
        steps = steps[:args.max_steps]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": args.config,
        "xml": args.xml,
        "vtu_dir": args.vtu_dir,
        "raw_origin": raw_origin,
        "num_nodes": int(coords.shape[0]),
        "num_vtu_steps": len(raw_times),
        "target_steps": steps,
        "dtype": args.dtype,
        "feature_groups": FEATURE_GROUPS,
        "ellipsoid_source": {
            "body_radius_mm": path.body_radius_mm,
            "body_height_mm": path.body_height_mm,
            "body_radius_front_mm": path.body_radius_front_mm,
            "body_radius_back_mm": path.body_radius_back_mm,
        },
        "chunks": [],
    }

    columns: list[str] | None = None
    for n, step_idx in enumerate(steps, start=1):
        raw_time = raw_times[step_idx]
        features, feature_columns = path.node_process_features(coords, raw_time, raw_origin=raw_origin)
        columns = feature_columns
        column_index = {name: idx for idx, name in enumerate(feature_columns)}
        ellipsoid_count = int(features[:, column_index["in_laser_ellipsoid"]].sum())
        track_neighborhood_count = int(features[:, column_index["in_track_neighborhood"]].sum())
        min_laser_distance_mm = float(features[:, column_index["distance_to_laser_mm"]].min())
        if args.dtype == "float64":
            features = features.astype(np.float64)
        chunk_name = f"laser_features_step_{step_idx:05d}.npz"
        chunk_path = out_dir / chunk_name
        np.savez_compressed(
            chunk_path,
            features=features,
            coords_mm=coords.astype(np.float32),
            columns=np.asarray(feature_columns),
            target_step=np.asarray([step_idx], dtype=np.int64),
            raw_time=np.asarray([raw_time], dtype=np.float64),
        )
        manifest["chunks"].append({
            "step": int(step_idx),
            "raw_time": float(raw_time),
            "path": str(chunk_path),
            "bytes": chunk_path.stat().st_size,
            "in_laser_ellipsoid_count": ellipsoid_count,
            "in_track_neighborhood_count": track_neighborhood_count,
            "min_laser_distance_mm": min_laser_distance_mm,
        })
        print(
            f"[{n}/{len(steps)}] wrote {chunk_path} "
            f"ellipsoid={ellipsoid_count} track={track_neighborhood_count} "
            f"min_dist={min_laser_distance_mm:.4f}mm"
        )

    manifest["columns"] = columns or []
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
