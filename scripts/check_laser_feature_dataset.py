"""Validate chunked laser trajectory feature datasets before ablation runs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_COLUMNS = [
    "laser_x_mm", "laser_y_mm", "laser_z_mm",
    "dx_mm", "dy_mm", "dz_mm",
    "distance_to_laser_mm",
    "current_along_mm", "current_cross_mm",
    "line_along_mm", "line_cross_mm", "time_to_arrival_s", "arrival_raw_time",
    "layer_idx", "track_physical", "track_program", "direction_sign",
    "in_laser_ellipsoid", "in_track_neighborhood",
]

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

BINARY_COLUMNS = ["in_laser_ellipsoid", "in_track_neighborhood"]
NORMALIZED_COLUMNS = ["layer_norm", "track_physical_norm", "track_program_norm"]


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return manifest


def resolve_chunk_path(manifest_path: Path, chunk: dict[str, Any]) -> Path:
    chunk_path = Path(str(chunk["path"]))
    if chunk_path.is_absolute():
        return chunk_path
    candidate = Path.cwd() / chunk_path
    if candidate.exists():
        return candidate
    return manifest_path.parent / chunk_path


def resolve_manifest_path(manifest_path: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidate = Path.cwd() / path
    if candidate.exists():
        return candidate
    return manifest_path.parent / path


def validate_columns(manifest: dict[str, Any], feature_groups: list[str]) -> tuple[list[str], list[str]]:
    columns = [str(c) for c in manifest.get("columns", [])]
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    group_errors: list[str] = []
    column_set = set(columns)
    manifest_groups = manifest.get("feature_groups") or {}
    if not isinstance(manifest_groups, dict):
        group_errors.append("feature_groups must be a JSON object")
        manifest_groups = {}

    for group in feature_groups:
        canonical_columns = FEATURE_GROUPS[group]
        manifest_columns = [str(c) for c in manifest_groups.get(group, canonical_columns)]
        canonical_absent = [name for name in canonical_columns if name not in column_set]
        if canonical_absent:
            group_errors.append(f"{group}: missing canonical columns {', '.join(canonical_absent)}")
        manifest_absent = [name for name in manifest_columns if name not in column_set]
        if manifest_absent:
            group_errors.append(f"{group}: missing manifest columns {', '.join(manifest_absent)}")
        if group in manifest_groups:
            manifest_set = set(manifest_columns)
            canonical_set = set(canonical_columns)
            if manifest_set != canonical_set:
                missing_from_manifest = [name for name in canonical_columns if name not in manifest_set]
                extra_in_manifest = [name for name in manifest_columns if name not in canonical_set]
                details: list[str] = []
                if missing_from_manifest:
                    details.append("missing " + ", ".join(missing_from_manifest))
                if extra_in_manifest:
                    details.append("extra " + ", ".join(extra_in_manifest))
                group_errors.append(f"{group}: manifest group definition drift ({'; '.join(details)})")

    for group, group_columns in manifest_groups.items():
        absent = [str(c) for c in group_columns if str(c) not in column_set]
        if absent:
            group_errors.append(f"{group}: missing {', '.join(absent)}")
    return missing, group_errors


def parse_feature_groups(value: str) -> list[str]:
    group = value.upper()
    if group == "ALL":
        return list(FEATURE_GROUPS)
    if group not in FEATURE_GROUPS:
        raise argparse.ArgumentTypeError(
            f"Expected one of {', '.join(['all', *FEATURE_GROUPS])}, got {value!r}"
        )
    return [group]


def validate_manifest_consistency(manifest: dict[str, Any], require_xml: bool) -> list[str]:
    errors: list[str] = []
    chunks = list(manifest.get("chunks", []))
    target_steps = list(manifest.get("target_steps", []))

    if target_steps and len(target_steps) != len(chunks):
        errors.append(
            f"target_steps/chunks count mismatch: {len(target_steps)} vs {len(chunks)}"
        )
    if target_steps and target_steps != sorted(target_steps):
        errors.append("target_steps must be sorted in ascending order")

    chunk_steps = [int(chunk.get("step", -1)) for chunk in chunks]
    if target_steps and chunk_steps != [int(step) for step in target_steps]:
        errors.append("chunk step list does not match manifest target_steps")
    if len(chunk_steps) != len(set(chunk_steps)):
        errors.append("duplicate chunk step entries in manifest")

    num_nodes = manifest.get("num_nodes")
    if num_nodes is not None and int(num_nodes) <= 0:
        errors.append(f"num_nodes must be positive, got {num_nodes}")

    if not manifest.get("coords_embedded_in_chunks", True) and not manifest.get("coords_path"):
        errors.append("coords_path is required when coords are not embedded in chunks")

    if require_xml and manifest.get("xml") is None:
        errors.append("manifest xml is null; fixed roadmap expects XML-derived trajectory features")

    return errors


def summarize_chunk(
    chunk_path: Path,
    expected_columns: list[str],
    node_index: int | None,
    coords_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, float] | None, list[str]]:
    errors: list[str] = []
    with np.load(chunk_path, allow_pickle=False) as data:
        features = data["features"]
        coords = data["coords_mm"] if "coords_mm" in data.files else None
        columns = [str(c) for c in data["columns"].tolist()]
        target_step = int(data["target_step"][0])
        raw_time = float(data["raw_time"][0])

    if coords is None and coords_path is not None and coords_path.exists():
        coords = np.load(coords_path, allow_pickle=False)
    if coords is None:
        coords = np.empty((0, 3), dtype=np.float32)
        errors.append("coords_mm missing from chunk and coords_path is unavailable")

    if columns != expected_columns:
        errors.append("chunk columns do not match manifest columns")
    if features.ndim != 2:
        errors.append(f"features must be 2D, got shape {features.shape}")
    if coords.ndim != 2 or coords.shape[1] != 3:
        errors.append(f"coords_mm must have shape [N, 3], got {coords.shape}")
    if features.shape[0] != coords.shape[0]:
        errors.append(f"features/coords node count mismatch: {features.shape[0]} vs {coords.shape[0]}")
    if features.shape[1] != len(expected_columns):
        errors.append(f"feature column count mismatch: {features.shape[1]} vs {len(expected_columns)}")
    if not np.isfinite(features).all():
        errors.append("features contain NaN or Inf")
    if not np.isfinite(coords).all():
        errors.append("coords_mm contains NaN or Inf")

    column_index = {name: idx for idx, name in enumerate(expected_columns)}

    def col(name: str) -> np.ndarray:
        return features[:, column_index[name]]

    if "target_step" in expected_columns:
        errors.append("target_step should be stored as chunk metadata, not as a feature column")

    rel_distance = np.linalg.norm(
        np.stack([col("dx_mm"), col("dy_mm"), col("dz_mm")], axis=1),
        axis=1,
    )
    if not np.allclose(rel_distance, col("distance_to_laser_mm"), atol=1.0e-3, rtol=1.0e-4):
        max_diff = float(np.max(np.abs(rel_distance - col("distance_to_laser_mm"))))
        errors.append(f"distance_to_laser_mm inconsistent with dx/dy/dz, max_diff={max_diff:.6g}")

    for name in BINARY_COLUMNS:
        values = col(name)
        if np.any((values < -1.0e-6) | (values > 1.0 + 1.0e-6)):
            errors.append(f"{name} must be in [0, 1]")
        if not np.allclose(values, np.round(values), atol=1.0e-6):
            errors.append(f"{name} must be binary 0/1")

    for name in NORMALIZED_COLUMNS:
        if name in column_index:
            values = col(name)
            if np.any((values < -1.0e-6) | (values > 1.0 + 1.0e-6)):
                errors.append(f"{name} must be normalized to [0, 1]")

    if np.any(~np.isin(col("direction_sign"), [-1.0, 1.0])):
        errors.append("direction_sign must be -1 or 1")

    summary = {
        "target_step": target_step,
        "raw_time": raw_time,
        "path": str(chunk_path),
        "num_nodes": int(features.shape[0]),
        "num_columns": int(features.shape[1]),
        "min_distance_to_laser_mm": float(np.min(col("distance_to_laser_mm"))),
        "p50_distance_to_laser_mm": float(np.percentile(col("distance_to_laser_mm"), 50)),
        "min_abs_time_to_arrival_s": float(np.min(np.abs(col("time_to_arrival_s")))),
        "in_laser_ellipsoid_count": int(np.sum(col("in_laser_ellipsoid") > 0.5)),
        "in_track_neighborhood_count": int(np.sum(col("in_track_neighborhood") > 0.5)),
    }

    node_features: dict[str, float] | None = None
    if node_index is not None:
        if node_index < 0 or node_index >= features.shape[0]:
            errors.append(f"node index {node_index} outside [0, {features.shape[0] - 1}]")
        else:
            node_features = {name: float(features[node_index, idx]) for name, idx in column_index.items()}
            node_features.update({
                "node_index": float(node_index),
                "coord_x_mm": float(coords[node_index, 0]),
                "coord_y_mm": float(coords[node_index, 1]),
                "coord_z_mm": float(coords[node_index, 2]),
            })

    return summary, node_features, errors


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to laser feature manifest.json")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--step", type=int, default=None, help="Only inspect this target step")
    parser.add_argument(
        "--feature_group",
        default="all",
        help="Validate canonical E1, E2, E3, or all group columns. Default: all.",
    )
    parser.add_argument("--node_index", type=int, default=None, help="Optional node row to include in the JSON report")
    parser.add_argument(
        "--require_xml",
        action="store_true",
        help="Fail if the manifest was not generated from an explicit XML path.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    try:
        requested_feature_groups = parse_feature_groups(args.feature_group)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    missing_columns, group_errors = validate_columns(manifest, requested_feature_groups)
    errors = [f"missing required column: {name}" for name in missing_columns]
    errors.extend(group_errors)
    errors.extend(validate_manifest_consistency(manifest, require_xml=args.require_xml))

    columns = [str(c) for c in manifest.get("columns", [])]
    coords_path = resolve_manifest_path(manifest_path, manifest.get("coords_path"))
    chunks = list(manifest.get("chunks", []))
    if args.step is not None:
        chunks = [chunk for chunk in chunks if int(chunk.get("step", -1)) == args.step]
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]
    if not chunks:
        errors.append("no chunks selected for validation")

    summaries: list[dict[str, Any]] = []
    node_reports: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_path = resolve_chunk_path(manifest_path, chunk)
        if not chunk_path.exists():
            errors.append(f"missing chunk: {chunk_path}")
            continue
        summary, node_features, chunk_errors = summarize_chunk(
            chunk_path,
            columns,
            args.node_index,
            coords_path=coords_path,
        )
        summaries.append(summary)
        errors.extend(f"{chunk_path}: {err}" for err in chunk_errors)
        if node_features is not None:
            node_reports.append({
                "target_step": summary["target_step"],
                "raw_time": summary["raw_time"],
                **node_features,
            })

    report = {
        "manifest": str(manifest_path),
        "ok": not errors,
        "config": manifest.get("config"),
        "xml": manifest.get("xml"),
        "num_manifest_chunks": len(manifest.get("chunks", [])),
        "num_checked_chunks": len(summaries),
        "num_columns": len(columns),
        "coords_path": manifest.get("coords_path"),
        "coords_embedded_in_chunks": manifest.get("coords_embedded_in_chunks", True),
        "required_columns": REQUIRED_COLUMNS,
        "requested_feature_groups": requested_feature_groups,
        "canonical_feature_groups": FEATURE_GROUPS,
        "errors": errors,
        "summaries": summaries,
        "node_reports": node_reports,
    }

    output_json = Path(args.output_json) if args.output_json else manifest_path.parent / "feature_check_report.json"
    output_csv = Path(args.output_csv) if args.output_csv else manifest_path.parent / "feature_check_summary.csv"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(output_csv, summaries)

    print(f"Checked chunks: {len(summaries)}")
    print(f"Required columns: {len(REQUIRED_COLUMNS)}, feature columns: {len(columns)}")
    print(f"Feature groups checked: {', '.join(requested_feature_groups)}")
    if summaries:
        min_dist = min(row["min_distance_to_laser_mm"] for row in summaries)
        ellipsoid_hits = sum(row["in_laser_ellipsoid_count"] for row in summaries)
        track_hits = sum(row["in_track_neighborhood_count"] for row in summaries)
        print(f"Min laser distance: {min_dist:.6f} mm")
        print(f"Ellipsoid hits: {ellipsoid_hits}")
        print(f"Track-neighborhood hits: {track_hits}")
    print(f"JSON report: {output_json}")
    print(f"CSV summary: {output_csv}")
    if errors:
        print("Validation failed:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    print("Validation OK")


if __name__ == "__main__":
    main()
