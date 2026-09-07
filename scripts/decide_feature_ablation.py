"""Apply the fixed continue/stop criteria to E0-E3 feature ablations.

This script reads canonical ``evaluation_report.json`` files, using
``acceptance_summary.json`` sidecars when available, and turns the roadmap rules
into a repeatable decision:

* If E1/E2 do not improve solidus recall, re-check laser trajectory alignment.
* If recall improves but RMSE regresses strongly, tune threshold/sampling/head
  weights before expanding the architecture.
* If trajectory features improve hotspot recall without unacceptable global
  regression, stop threshold tinkering and proceed to E4/E5.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RUNS = {
    "E0": "feature_e0_baseline",
    "E1": "feature_e1_laser_distance",
    "E2": "feature_e2_scan_arrival",
    "E3": "feature_e3_path_phase",
}

METRIC_KEYS = (
    "RMSE",
    "MAE",
    "AbsErrorP95",
    "AbsErrorP99",
    "MaxError",
    "TempRecallAboveSolidus",
    "TempPrecisionAboveSolidus",
    "TempF1AboveSolidus",
    "TempFPAboveSolidus",
    "TempFNAboveSolidus",
    "TempRecallAboveLiquidus",
    "TempPrecisionAboveLiquidus",
    "TempF1AboveLiquidus",
    "LaserRegionMAE",
    "LaserRegionMaxError",
)

ACCEPTANCE_METRIC_MAP = {
    "RMSE": "RMSE",
    "MAE": "MAE",
    "AbsErrorP95": "AbsErrorP95",
    "AbsErrorP99": "AbsErrorP99",
    "MaxError": "MaxError",
    "TempRecallAboveSolidus": "SolidusRecall",
    "TempPrecisionAboveSolidus": "SolidusPrecision",
    "TempF1AboveSolidus": "SolidusF1",
    "TempFPAboveSolidus": "SolidusFalseHot",
    "TempFNAboveSolidus": "SolidusMissedHot",
    "TempRecallAboveLiquidus": "LiquidusRecall",
    "TempPrecisionAboveLiquidus": "LiquidusPrecision",
    "TempF1AboveLiquidus": "LiquidusF1",
    "LaserRegionMAE": "LaserRegionMAE",
    "LaserRegionMaxError": "LaserRegionMaxError",
}

ACCEPTANCE_WORST_MAP = {
    "target_step": "WorstStep",
    "node_index": "WorstNode",
    "prediction": "WorstPrediction",
    "target": "WorstTarget",
    "abs_error": "WorstAbsError",
}


@dataclass(frozen=True)
class RunResult:
    key: str
    run_name: str
    report_path: Path
    acceptance_path: Path | None
    metrics: dict[str, float]
    worst: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def canonical_metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("canonical_metrics")
    if isinstance(metrics, dict):
        return metrics
    metrics = report.get("global_metrics")
    if isinstance(metrics, dict):
        return metrics
    return {}


def load_acceptance_summary(run_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    path = run_dir / "acceptance_summary.json"
    if not path.is_file():
        return None, {}
    payload = load_json(path)
    return path, payload


def metrics_from_report(report: dict[str, Any]) -> dict[str, float]:
    raw_metrics = canonical_metrics(report)
    return {
        name: value
        for name in METRIC_KEYS
        if (value := finite_float(raw_metrics.get(name))) is not None
    }


def merge_acceptance_metrics(metrics: dict[str, float], acceptance: dict[str, Any]) -> None:
    for metric_key, acceptance_key in ACCEPTANCE_METRIC_MAP.items():
        value = finite_float(acceptance.get(acceptance_key))
        if value is not None:
            metrics[metric_key] = value


def worst_from_report(report: dict[str, Any]) -> dict[str, Any]:
    worst = report.get("worst_case")
    if not isinstance(worst, dict) and isinstance(report.get("worst_cases_top10"), list):
        first = report["worst_cases_top10"][0] if report["worst_cases_top10"] else {}
        worst = first if isinstance(first, dict) else {}
    return worst if isinstance(worst, dict) else {}


def merge_acceptance_worst(worst: dict[str, Any], acceptance: dict[str, Any]) -> None:
    for worst_key, acceptance_key in ACCEPTANCE_WORST_MAP.items():
        value = acceptance.get(acceptance_key)
        if value is not None:
            worst[worst_key] = value


def load_run(logs_dir: Path, key: str, run_name: str) -> RunResult | None:
    run_dir = logs_dir / run_name
    report_path = run_dir / "evaluation_report.json"
    acceptance_path, acceptance = load_acceptance_summary(run_dir)
    if not report_path.is_file() and acceptance_path is None:
        return None
    report = load_json(report_path) if report_path.is_file() else {}
    metrics = metrics_from_report(report)
    worst = worst_from_report(report)
    merge_acceptance_metrics(metrics, acceptance)
    merge_acceptance_worst(worst, acceptance)
    source_path = report_path if report_path.is_file() else acceptance_path
    assert source_path is not None
    return RunResult(
        key=key,
        run_name=run_name,
        report_path=source_path,
        acceptance_path=acceptance_path,
        metrics=metrics,
        worst=worst,
    )


def pct_change(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or base == 0:
        return None
    return (value - base) / abs(base)


def get_metric(run: RunResult, key: str) -> float | None:
    return run.metrics.get(key)


def format_number(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        if abs(value) >= 1000 or (abs(value) < 0.001 and value != 0):
            return f"{value:.4e}"
        return f"{value:.6g}"
    return str(value)


def row_for(run: RunResult, baseline: RunResult | None) -> dict[str, Any]:
    rmse = get_metric(run, "RMSE")
    recall = get_metric(run, "TempRecallAboveSolidus")
    max_error = get_metric(run, "MaxError")
    row: dict[str, Any] = {
        "experiment": run.key,
        "run": run.run_name,
        "RMSE": rmse,
        "MAE": get_metric(run, "MAE"),
        "P95": get_metric(run, "AbsErrorP95"),
        "P99": get_metric(run, "AbsErrorP99"),
        "MaxError": max_error,
        "SolidusRecall": recall,
        "SolidusPrecision": get_metric(run, "TempPrecisionAboveSolidus"),
        "SolidusF1": get_metric(run, "TempF1AboveSolidus"),
        "FalseHotSolidus": get_metric(run, "TempFPAboveSolidus"),
        "MissedHotSolidus": get_metric(run, "TempFNAboveSolidus"),
        "LaserRegionMAE": get_metric(run, "LaserRegionMAE"),
        "LaserRegionMaxError": get_metric(run, "LaserRegionMaxError"),
        "WorstStep": run.worst.get("target_step"),
        "WorstNode": run.worst.get("node_index"),
        "WorstPred": run.worst.get("prediction"),
        "WorstTarget": run.worst.get("target"),
        "WorstAbsError": run.worst.get("abs_error"),
        "report": str(run.report_path),
        "acceptance_summary": str(run.acceptance_path) if run.acceptance_path else "",
    }
    if baseline is not None and run.key != baseline.key:
        row["RecallGainVsE0"] = None
        base_recall = get_metric(baseline, "TempRecallAboveSolidus")
        if recall is not None and base_recall is not None:
            row["RecallGainVsE0"] = recall - base_recall
        row["RMSEChangeVsE0Pct"] = pct_change(rmse, get_metric(baseline, "RMSE"))
        row["MaxErrorChangeVsE0Pct"] = pct_change(max_error, get_metric(baseline, "MaxError"))
    else:
        row["RecallGainVsE0"] = ""
        row["RMSEChangeVsE0Pct"] = ""
        row["MaxErrorChangeVsE0Pct"] = ""
    return row


def choose_decision(
    runs: dict[str, RunResult],
    *,
    min_recall_gain: float,
    max_rmse_regression_pct: float,
) -> tuple[str, list[str]]:
    baseline = runs["E0"]
    base_recall = get_metric(baseline, "TempRecallAboveSolidus")
    base_rmse = get_metric(baseline, "RMSE")
    notes: list[str] = []

    trajectory_candidates = [runs[key] for key in ("E1", "E2") if key in runs]
    recall_gains: list[tuple[str, float]] = []
    rmse_changes: list[tuple[str, float]] = []
    for run in trajectory_candidates:
        recall = get_metric(run, "TempRecallAboveSolidus")
        rmse = get_metric(run, "RMSE")
        gain = None if recall is None or base_recall is None else recall - base_recall
        rmse_change = pct_change(rmse, base_rmse)
        if gain is not None:
            recall_gains.append((run.key, gain))
            notes.append(f"{run.key} solidus recall gain vs E0: {gain:+.4f}")
        if rmse_change is not None:
            rmse_changes.append((run.key, rmse_change))
            notes.append(f"{run.key} RMSE change vs E0: {rmse_change:+.1%}")

    if not recall_gains:
        return "Missing comparable E1/E2 recall; complete the evaluation reports first.", notes

    best_key, best_gain = max(recall_gains, key=lambda item: item[1])
    worst_rmse_change = max((change for _, change in rmse_changes), default=0.0)

    if best_gain < min_recall_gain:
        return (
            "E1/E2 did not materially improve hotspot recall; re-check laser time, "
            "coordinates, layer height, and scan-direction alignment next.",
            notes,
        )

    if worst_rmse_change > max_rmse_regression_pct:
        return (
            "Hotspot recall improved but global RMSE regressed too much; tune "
            "classification threshold, sampling, and residual/aux weights next.",
            notes,
        )

    return (
        f"Trajectory features are effective; use {best_key} as the signal to "
        "proceed to E4/E5 hotspot heads and conditional residuals, and stop "
        "expanding threshold/gate tuning.",
        notes,
    )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], decision: str, notes: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "experiment",
        "run",
        "RMSE",
        "P99",
        "MaxError",
        "SolidusRecall",
        "SolidusF1",
        "FalseHotSolidus",
        "RecallGainVsE0",
        "RMSEChangeVsE0Pct",
        "WorstStep",
        "WorstNode",
    ]
    lines = ["# Feature Ablation Decision", "", f"Decision: {decision}", ""]
    if notes:
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(format_number(row.get(col, "")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_run_names(values: list[str]) -> dict[str, str]:
    runs = dict(DEFAULT_RUNS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=run_name, got {value!r}")
        key, run_name = value.split("=", 1)
        key = key.strip().upper()
        if key not in DEFAULT_RUNS:
            raise ValueError(f"Unknown experiment key {key!r}")
        runs[key] = run_name.strip()
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs_dir", type=Path, default=Path("logs"))
    parser.add_argument("--experiments", nargs="+", default=["E0", "E1", "E2", "E3"])
    parser.add_argument(
        "--run_name",
        action="append",
        default=[],
        help="Override a default run directory, e.g. --run_name E1=my_run.",
    )
    parser.add_argument("--min_recall_gain", type=float, default=0.02)
    parser.add_argument("--max_rmse_regression_pct", type=float, default=0.25)
    parser.add_argument("--output_csv", type=Path, default=Path("results/feature_ablation_decision.csv"))
    parser.add_argument("--output_md", type=Path, default=Path("results/feature_ablation_decision.md"))
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    run_names = parse_run_names(args.run_name)
    selected = [key.upper() for key in args.experiments]
    unknown = [key for key in selected if key not in run_names]
    if unknown:
        raise SystemExit(f"Unknown experiments: {', '.join(unknown)}")

    runs: dict[str, RunResult] = {}
    missing: list[str] = []
    for key in selected:
        run = load_run(args.logs_dir, key, run_names[key])
        if run is None:
            missing.append(f"{key}:{args.logs_dir / run_names[key] / 'evaluation_report.json'}")
        else:
            runs[key] = run

    if missing and not args.allow_missing:
        print("Missing evaluation reports:")
        for item in missing:
            print(f"  {item}")
        raise SystemExit(2)
    if "E0" not in runs:
        raise SystemExit("E0 baseline report is required for decision comparison.")

    baseline = runs["E0"]
    rows = [row_for(runs[key], baseline) for key in selected if key in runs]
    decision, notes = choose_decision(
        runs,
        min_recall_gain=args.min_recall_gain,
        max_rmse_regression_pct=args.max_rmse_regression_pct,
    )

    write_csv(rows, args.output_csv)
    write_markdown(rows, decision, notes, args.output_md)

    print(f"Decision: {decision}")
    for note in notes:
        print(f"  {note}")
    if missing:
        print("Missing reports ignored:")
        for item in missing:
            print(f"  {item}")
    print(f"CSV: {args.output_csv}")
    print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
