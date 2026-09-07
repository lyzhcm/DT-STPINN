"""Verify laser alignment protocol outputs.

The roadmap needs trajectory alignment evidence that can be trusted before long
feature ablations. This read-only guard checks the wrapper manifest and the key
CSV/Markdown artifacts, including the focus-window CSV around the known worst
node.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_COMMAND_PHASES = ("export", "focus", "plot")

FOCUS_COLUMNS = (
    "kind",
    "step",
    "raw_time",
    "node_index",
    "temperature_c",
    "laser_x_mm",
    "laser_y_mm",
    "laser_z_mm",
    "best_distance_xy_mm",
    "best_distance_3d_mm",
    "line_cross_mm",
    "time_to_arrival_s",
    "arrival_raw_time",
    "layer",
    "track_program_state",
    "track_physical_state",
    "direction_sign",
    "in_laser_ellipsoid",
    "in_track_neighborhood",
)


def add_check(checks: list[tuple[bool, str]], ok: bool, message: str) -> None:
    checks.append((ok, message))


def load_manifest(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "laser_alignment_protocol_manifest.json"
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def resolve_artifact(manifest_path: Path, artifact_value: Any) -> Path | None:
    if not artifact_value:
        return None
    path = Path(str(artifact_value))
    if path.is_absolute():
        return path
    root = manifest_path if manifest_path.is_dir() else manifest_path.parent
    candidate = root / path
    if candidate.exists():
        return candidate
    return Path(str(artifact_value))


def is_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def check_commands(checks: list[tuple[bool, str]], manifest: dict[str, Any]) -> None:
    commands = manifest.get("commands")
    add_check(checks, isinstance(commands, list), "commands: list")
    if not isinstance(commands, list):
        return
    phases = {item.get("phase") for item in commands if isinstance(item, dict)}
    for phase in REQUIRED_COMMAND_PHASES:
        add_check(checks, phase in phases, f"commands: phase {phase}")


def check_focus_csv(
    checks: list[tuple[bool, str]],
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    min_rows: int,
) -> None:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    csv_path = resolve_artifact(manifest_path, artifacts.get("focus_window_csv"))
    add_check(checks, csv_path is not None and csv_path.is_file(), f"focus_window_csv exists: {csv_path}")
    if csv_path is None or not csv_path.is_file():
        return

    try:
        columns, rows = read_csv_rows(csv_path)
        add_check(checks, True, "focus_window_csv: readable")
    except Exception as exc:  # noqa: BLE001
        add_check(checks, False, f"focus_window_csv: {exc}")
        return

    missing = [name for name in FOCUS_COLUMNS if name not in columns]
    add_check(checks, not missing, "focus_window_csv columns" + ("" if not missing else f" missing {missing}"))
    add_check(checks, len(rows) >= min_rows, f"focus_window_csv rows: {len(rows)}/{min_rows}")

    if not rows:
        return
    numeric_columns = (
        "step",
        "raw_time",
        "node_index",
        "temperature_c",
        "best_distance_3d_mm",
        "time_to_arrival_s",
        "arrival_raw_time",
    )
    for column in numeric_columns:
        if column in rows[0]:
            add_check(checks, is_number(rows[0][column]), f"focus_window_csv first row numeric: {column}")

    focus = manifest.get("focus") if isinstance(manifest.get("focus"), dict) else {}
    focus_step = focus.get("step")
    focus_node = focus.get("node")
    if focus_step is not None:
        add_check(checks, any(row.get("step") == str(focus_step) for row in rows), f"focus step present: {focus_step}")
    if focus_node is not None:
        add_check(checks, any(row.get("node_index") == str(focus_node) for row in rows), f"focus node present: {focus_node}")


def check_core_artifacts(checks: list[tuple[bool, str]], manifest_path: Path, manifest: dict[str, Any]) -> None:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    for key in ("trajectory_manifest", "alignment_report", "segments_csv", "samples_csv"):
        path = resolve_artifact(manifest_path, artifacts.get(key))
        add_check(checks, path is not None and path.is_file(), f"{key} exists: {path}")
    plot_status = next(
        (
            str(item.get("status", ""))
            for item in manifest.get("commands", [])
            if isinstance(item, dict) and item.get("phase") == "plot"
        ),
        "",
    )
    if plot_status != "skipped":
        path = resolve_artifact(manifest_path, artifacts.get("plot_summary_csv"))
        add_check(checks, path is not None and path.is_file(), f"plot_summary_csv exists: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="laser_alignment_protocol_manifest.json or output directory")
    parser.add_argument("--min_focus_rows", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if manifest_path.is_dir():
        manifest_file = manifest_path / "laser_alignment_protocol_manifest.json"
    else:
        manifest_file = manifest_path


    checks: list[tuple[bool, str]] = []
    add_check(checks, manifest_file.is_file(), f"manifest exists: {manifest_file}")
    if not manifest_file.is_file():
        for ok, message in checks:
            print(f"[{'OK' if ok else 'FAIL'}] {message}")
        raise SystemExit(1)

    try:
        manifest = load_manifest(manifest_file)
        add_check(checks, True, "manifest JSON: readable")
    except Exception as exc:  # noqa: BLE001
        add_check(checks, False, f"manifest JSON: {exc}")
        manifest = {}

    add_check(checks, manifest.get("protocol") == "laser_alignment", "protocol: laser_alignment")
    add_check(checks, bool(manifest.get("generated_at_utc")), "generated_at_utc recorded")
    check_commands(checks, manifest)
    check_core_artifacts(checks, manifest_file, manifest)
    check_focus_csv(checks, manifest_file, manifest, min_rows=args.min_focus_rows)

    if not args.quiet:
        print(f"Laser alignment protocol: {manifest_file}")
        for ok, message in checks:
            print(f"[{'OK' if ok else 'FAIL'}] {message}")

    if any(not ok for ok, _ in checks):
        raise SystemExit(1)
    if args.quiet:
        print(f"OK: {manifest_file}")


if __name__ == "__main__":
    main()
