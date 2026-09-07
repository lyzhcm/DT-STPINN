"""Verify that an evaluation report contains the fixed acceptance metrics.

The roadmap compares experiments by hotspot recall and worst-case behavior, not
only by global RMSE. This read-only guard checks that an
``evaluation_report.json`` contains every metric block needed for those
comparisons before the report is summarized or frozen.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


GLOBAL_METRIC_KEYS = (
    "RMSE",
    "MAE",
    "AbsErrorP95",
    "AbsErrorP99",
    "MaxError",
)

SOLIDUS_KEYS = (
    "TempPrecisionAboveSolidus",
    "TempRecallAboveSolidus",
    "TempF1AboveSolidus",
)

LIQUIDUS_KEYS = (
    "TempPrecisionAboveLiquidus",
    "TempRecallAboveLiquidus",
    "TempF1AboveLiquidus",
)

LASER_REGION_KEYS = (
    "LaserRegionMAE",
    "LaserRegionMaxError",
)

PER_TIMESTEP_SUMMARY_KEYS = (
    "num_steps",
    "true_max_range",
    "pred_max_range",
    "max_abs_error",
    "mean_abs_error",
)

WORST_CASE_KEYS = (
    "target_step",
    "node_index",
    "prediction",
    "target",
    "abs_error",
)

COORDINATE_KEYS = ("coordinate_mm", "coord_mm")

LASER_CONTEXT_KEYS = (
    "target_laser_distance_mm",
    "distance_to_laser_mm",
    "time_to_arrival_s",
    "target_time_to_arrival_s",
)


def load_report(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "evaluation_report.json"
    with path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    if not isinstance(report, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return report


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def add_check(checks: list[tuple[bool, str]], ok: bool, message: str) -> None:
    checks.append((ok, message))


def metric_source(report: dict[str, Any]) -> dict[str, Any]:
    canonical = report.get("canonical_metrics")
    if isinstance(canonical, dict):
        return canonical
    global_metrics = report.get("global_metrics")
    if isinstance(global_metrics, dict):
        return global_metrics
    return {}


def require_numeric_keys(
    checks: list[tuple[bool, str]],
    source: dict[str, Any],
    keys: tuple[str, ...],
    label: str,
) -> None:
    for key in keys:
        add_check(
            checks,
            key in source and is_finite_number(source[key]),
            f"{label}: {key}",
        )


def has_any_key(source: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(key in source and source[key] is not None for key in keys)


def check_worst_cases(
    checks: list[tuple[bool, str]],
    report: dict[str, Any],
    *,
    min_worst_cases: int,
    require_laser_context: bool,
) -> None:
    worst_cases = report.get("worst_cases_top10")
    add_check(
        checks,
        isinstance(worst_cases, list) and len(worst_cases) >= min_worst_cases,
        f"worst_cases_top10: at least {min_worst_cases} rows",
    )
    if not isinstance(worst_cases, list):
        return

    for i, item in enumerate(worst_cases[:min_worst_cases], start=1):
        if not isinstance(item, dict):
            add_check(checks, False, f"worst_cases_top10[{i}]: object")
            continue
        missing = [key for key in WORST_CASE_KEYS if key not in item]
        if not has_any_key(item, COORDINATE_KEYS):
            missing.append("coordinate_mm|coord_mm")
        add_check(
            checks,
            not missing,
            f"worst_cases_top10[{i}]: required fields"
            + ("" if not missing else f" missing {missing}"),
        )
        if require_laser_context:
            add_check(
                checks,
                has_any_key(item, LASER_CONTEXT_KEYS),
                f"worst_cases_top10[{i}]: laser distance or arrival context",
            )


def check_per_timestep(checks: list[tuple[bool, str]], report: dict[str, Any]) -> None:
    summary = report.get("per_timestep_max_summary")
    add_check(checks, isinstance(summary, dict), "per_timestep_max_summary: object")
    if isinstance(summary, dict):
        for key in PER_TIMESTEP_SUMMARY_KEYS:
            add_check(checks, key in summary, f"per_timestep_max_summary: {key}")
        num_steps = summary.get("num_steps")
    else:
        num_steps = None

    rows = report.get("per_timestep_max")
    add_check(checks, isinstance(rows, list) and len(rows) > 0, "per_timestep_max: non-empty")
    if isinstance(rows, list) and isinstance(num_steps, int):
        add_check(checks, len(rows) == num_steps, f"per_timestep_max length: {len(rows)}/{num_steps}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Path to evaluation_report.json or its output directory")
    parser.add_argument("--min_worst_cases", type=int, default=10)
    parser.add_argument(
        "--require_laser_context",
        action="store_true",
        help="Require worst-case rows to include laser distance or arrival timing.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report_path = Path(args.report)
    if report_path.is_dir():
        report_path = report_path / "evaluation_report.json"
    checks: list[tuple[bool, str]] = []
    add_check(checks, report_path.is_file(), f"report exists: {report_path}")
    if not report_path.is_file():
        for ok, message in checks:
            print(f"[{'OK' if ok else 'FAIL'}] {message}")
        raise SystemExit(1)

    try:
        report = load_report(report_path)
        add_check(checks, True, "report JSON: readable")
    except Exception as exc:  # noqa: BLE001 - verifier should print a clear failure.
        add_check(checks, False, f"report JSON: {exc}")
        report = {}

    metrics = metric_source(report)
    add_check(checks, bool(metrics), "canonical/global metrics: present")
    require_numeric_keys(checks, metrics, GLOBAL_METRIC_KEYS, "global metrics")
    require_numeric_keys(checks, metrics, SOLIDUS_KEYS, "solidus detection")
    require_numeric_keys(checks, metrics, LIQUIDUS_KEYS, "liquidus detection")

    laser_count = metrics.get("LaserRegionCount")
    add_check(
        checks,
        is_finite_number(laser_count) and int(laser_count) > 0,
        "laser region: non-empty count",
    )
    require_numeric_keys(checks, metrics, LASER_REGION_KEYS, "laser region")

    check_per_timestep(checks, report)
    check_worst_cases(
        checks,
        report,
        min_worst_cases=args.min_worst_cases,
        require_laser_context=args.require_laser_context,
    )

    if not args.quiet:
        print(f"Evaluation report: {report_path}")
        if metrics:
            print("Metrics:")
            for key in GLOBAL_METRIC_KEYS + SOLIDUS_KEYS + LIQUIDUS_KEYS + LASER_REGION_KEYS:
                if key in metrics:
                    print(f"  {key}: {metrics[key]}")
        print("Checks:")
        for ok, message in checks:
            print(f"[{'OK' if ok else 'FAIL'}] {message}")

    if any(not ok for ok, _ in checks):
        raise SystemExit(1)

    if args.quiet:
        print(f"OK: {report_path}")


if __name__ == "__main__":
    main()
