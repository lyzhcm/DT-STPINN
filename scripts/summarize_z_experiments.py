"""Summarize z-series ablation runs from TensorBoard event files.

Example:
    python scripts/summarize_z_experiments.py --logs_dir logs --pattern "ablation_z*_probe*"
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:  # pragma: no cover - dependency is environment-specific.
    raise SystemExit(
        "tensorboard is required to read event files. Install it or run inside the "
        "dtstpinn environment."
    ) from exc


DEFAULT_TAGS = [
    "val/val_loss",
    "val/RMSE",
    "val/MAE",
    "val/R2",
    "val/MaxError",
    "val/AbsErrorP50",
    "val/AbsErrorP90",
    "val/AbsErrorP95",
    "val/AbsErrorP99",
    "val/TempRecallAboveSolidus",
    "val/TempPrecisionAboveSolidus",
    "val/TempF1AboveSolidus",
    "val/TempRecallAboveLiquidus",
    "val/TempPrecisionAboveLiquidus",
    "val/TempF1AboveLiquidus",
    "val/LaserRegionMAE",
    "val/LaserRegionMaxError",
    "val/LaserRegionAbsErrorP95",
    "val/LaserRegionAbsErrorP99",
    "val/SpecGateRecallAt0.1",
    "val/SpecGatePrecisionAt0.1",
    "val/SpecGateNormalP99",
    "val/ArrivalGateRecallAt0.1",
    "val/ArrivalGatePrecisionAt0.1",
    "val/ArrivalGateNormalP99",
    "train/loss",
    "train/seconds_per_batch",
    "train/peak_vram_gb",
]

DEFAULT_TEST_KEYS = [
    "test/RMSE",
    "test/MAE",
    "test/R2",
    "test/MaxError",
    "test/AbsErrorP50",
    "test/AbsErrorP90",
    "test/AbsErrorP95",
    "test/AbsErrorP99",
    "test/TempRecallAboveSolidus",
    "test/TempPrecisionAboveSolidus",
    "test/TempF1AboveSolidus",
    "test/TempTPAboveSolidus",
    "test/TempFPAboveSolidus",
    "test/TempFNAboveSolidus",
    "test/FalseHotAboveSolidus",
    "test/MissedHotAboveSolidus",
    "test/TempRecallAboveLiquidus",
    "test/TempPrecisionAboveLiquidus",
    "test/TempF1AboveLiquidus",
    "test/TempTPAboveLiquidus",
    "test/TempFPAboveLiquidus",
    "test/TempFNAboveLiquidus",
    "test/FalseHotAboveLiquidus",
    "test/MissedHotAboveLiquidus",
    "test/LaserRegionMAE",
    "test/LaserRegionCount",
    "test/LaserRegionMaxError",
    "test/LaserRegionAbsErrorP95",
    "test/LaserRegionAbsErrorP99",
    "test/PeakTempNumSteps",
    "test/PeakTempMaxAbsError",
    "test/PeakTempMeanAbsError",
    "test/worst_step",
    "test/worst_raw_time",
    "test/worst_node",
    "test/worst_pred",
    "test/worst_target",
    "test/worst_abs_error",
    "test/worst_laser_distance",
    "test/worst_time_to_arrival",
    "test/worst_path_pre_arrival_gate",
    "test/worst_endpoint_gate",
    "test/worst_program_track_gate",
    "test/worst_time_until_gate",
    "test/worst_residual_gate",
    "test/worst_residual_boost",
]


ACCEPTANCE_KEY_MAP = {
    "test/RMSE": "RMSE",
    "test/MAE": "MAE",
    "test/MaxError": "MaxError",
    "test/AbsErrorP95": "AbsErrorP95",
    "test/AbsErrorP99": "AbsErrorP99",
    "test/TempRecallAboveSolidus": "SolidusRecall",
    "test/TempPrecisionAboveSolidus": "SolidusPrecision",
    "test/TempF1AboveSolidus": "SolidusF1",
    "test/TempFPAboveSolidus": "SolidusFalseHot",
    "test/FalseHotAboveSolidus": "SolidusFalseHot",
    "test/TempFNAboveSolidus": "SolidusMissedHot",
    "test/MissedHotAboveSolidus": "SolidusMissedHot",
    "test/TempRecallAboveLiquidus": "LiquidusRecall",
    "test/TempPrecisionAboveLiquidus": "LiquidusPrecision",
    "test/TempF1AboveLiquidus": "LiquidusF1",
    "test/TempFPAboveLiquidus": "LiquidusFalseHot",
    "test/FalseHotAboveLiquidus": "LiquidusFalseHot",
    "test/TempFNAboveLiquidus": "LiquidusMissedHot",
    "test/MissedHotAboveLiquidus": "LiquidusMissedHot",
    "test/LaserRegionCount": "LaserRegionCount",
    "test/LaserRegionMAE": "LaserRegionMAE",
    "test/LaserRegionMaxError": "LaserRegionMaxError",
    "test/PeakTempNumSteps": "PeakTempNumSteps",
    "test/PeakTempMaxAbsError": "PeakTempMaxAbsError",
    "test/PeakTempMeanAbsError": "PeakTempMeanAbsError",
    "test/worst_step": "WorstStep",
    "test/worst_node": "WorstNode",
    "test/worst_pred": "WorstPrediction",
    "test/worst_target": "WorstTarget",
    "test/worst_abs_error": "WorstAbsError",
    "test/worst_laser_distance": "WorstLaserDistance",
    "test/worst_time_to_arrival": "WorstTimeToArrival",
}


def read_latest_scalars(run_dir: Path, tags: list[str]) -> dict[str, float | int | str]:
    event_files = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    row: dict[str, float | int | str] = {"run": run_dir.name, "event_file": ""}
    if not event_files:
        return row

    event_file = event_files[-1]
    row["event_file"] = event_file.name
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))

    last_step = None
    for tag in tags:
        key = tag.replace("val/", "").replace("train/", "train_")
        if tag not in available:
            row[key] = ""
            continue
        events = accumulator.Scalars(tag)
        if not events:
            row[key] = ""
            continue
        last = events[-1]
        row[key] = float(last.value)
        if tag.startswith("val/"):
            last_step = last.step

    if last_step is not None:
        row["last_val_step"] = int(last_step)
    return row


def read_test_report(run_dir: Path, test_keys: list[str]) -> dict[str, float | int | str]:
    acceptance_path = run_dir / "acceptance_summary.json"
    acceptance = {}
    if acceptance_path.is_file():
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        if not isinstance(acceptance, dict):
            acceptance = {}

    report_path = run_dir / "test_metrics.json"
    report_type = "train_test_metrics"
    if not report_path.is_file():
        report_path = run_dir / "evaluation_report.json"
        report_type = "checkpoint_evaluation"
    row: dict[str, float | int | str] = {key.replace("test/", "test_"): "" for key in test_keys}
    if not report_path.is_file():
        if acceptance:
            for key in test_keys:
                acceptance_key = ACCEPTANCE_KEY_MAP.get(key)
                value = acceptance.get(acceptance_key) if acceptance_key else None
                if value is not None:
                    row[key.replace("test/", "test_")] = value
            row["test_report"] = ""
            row["test_report_type"] = "acceptance_summary"
            row["acceptance_summary"] = acceptance_path.name
            row["split_source"] = acceptance.get("split_source", "")
            row["split_path"] = acceptance.get("split_path", "")
            row["split_test_count"] = acceptance.get("split_test_count", "")
        return row

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        return row

    if report_type == "checkpoint_evaluation":
        nested_metrics = report.get("canonical_metrics", {})
        worst = report.get("worst_case", {})
        if not worst and report.get("worst_cases_top10"):
            worst = report["worst_cases_top10"][0]
    else:
        metrics = report.get("test_metrics", {})
        nested_metrics = metrics.get("metrics", {}) if isinstance(metrics, dict) else {}
        worst = metrics.get("worst_case", {}) if isinstance(metrics, dict) else {}
    split_protocol = report.get("split_protocol", {}) if isinstance(report, dict) else {}
    split_test = split_protocol.get("test", {}) if isinstance(split_protocol, dict) else {}
    worst_diag = worst.get("diagnostic", {}) if isinstance(worst, dict) else {}
    peak_summary = report.get("per_timestep_max_summary", {})
    if not isinstance(peak_summary, dict):
        peak_summary = {}

    values = {
        "test/RMSE": nested_metrics.get("RMSE"),
        "test/MAE": nested_metrics.get("MAE"),
        "test/R2": nested_metrics.get("R2"),
        "test/MaxError": nested_metrics.get("MaxError"),
        "test/AbsErrorP50": nested_metrics.get("AbsErrorP50"),
        "test/AbsErrorP90": nested_metrics.get("AbsErrorP90"),
        "test/AbsErrorP95": nested_metrics.get("AbsErrorP95"),
        "test/AbsErrorP99": nested_metrics.get("AbsErrorP99"),
        "test/TempRecallAboveSolidus": nested_metrics.get("TempRecallAboveSolidus"),
        "test/TempPrecisionAboveSolidus": nested_metrics.get("TempPrecisionAboveSolidus"),
        "test/TempF1AboveSolidus": nested_metrics.get("TempF1AboveSolidus"),
        "test/TempTPAboveSolidus": nested_metrics.get("TempTPAboveSolidus"),
        "test/TempFPAboveSolidus": nested_metrics.get("TempFPAboveSolidus"),
        "test/TempFNAboveSolidus": nested_metrics.get("TempFNAboveSolidus"),
        "test/FalseHotAboveSolidus": nested_metrics.get(
            "FalseHotAboveSolidus", nested_metrics.get("TempFPAboveSolidus")
        ),
        "test/MissedHotAboveSolidus": nested_metrics.get(
            "MissedHotAboveSolidus", nested_metrics.get("TempFNAboveSolidus")
        ),
        "test/TempRecallAboveLiquidus": nested_metrics.get("TempRecallAboveLiquidus"),
        "test/TempPrecisionAboveLiquidus": nested_metrics.get("TempPrecisionAboveLiquidus"),
        "test/TempF1AboveLiquidus": nested_metrics.get("TempF1AboveLiquidus"),
        "test/TempTPAboveLiquidus": nested_metrics.get("TempTPAboveLiquidus"),
        "test/TempFPAboveLiquidus": nested_metrics.get("TempFPAboveLiquidus"),
        "test/TempFNAboveLiquidus": nested_metrics.get("TempFNAboveLiquidus"),
        "test/FalseHotAboveLiquidus": nested_metrics.get(
            "FalseHotAboveLiquidus", nested_metrics.get("TempFPAboveLiquidus")
        ),
        "test/MissedHotAboveLiquidus": nested_metrics.get(
            "MissedHotAboveLiquidus", nested_metrics.get("TempFNAboveLiquidus")
        ),
        "test/LaserRegionMAE": nested_metrics.get("LaserRegionMAE"),
        "test/LaserRegionCount": nested_metrics.get("LaserRegionCount"),
        "test/LaserRegionMaxError": nested_metrics.get("LaserRegionMaxError"),
        "test/LaserRegionAbsErrorP95": nested_metrics.get("LaserRegionAbsErrorP95"),
        "test/LaserRegionAbsErrorP99": nested_metrics.get("LaserRegionAbsErrorP99"),
        "test/PeakTempNumSteps": peak_summary.get("num_steps"),
        "test/PeakTempMaxAbsError": peak_summary.get("max_abs_error"),
        "test/PeakTempMeanAbsError": peak_summary.get("mean_abs_error"),
        "test/worst_step": worst.get("target_step"),
        "test/worst_raw_time": worst.get("target_time_raw", worst.get("target_time")),
        "test/worst_node": worst.get("node_index"),
        "test/worst_pred": worst.get("prediction"),
        "test/worst_target": worst.get("target"),
        "test/worst_abs_error": worst.get("abs_error"),
        "test/worst_laser_distance": worst.get(
            "target_laser_distance_mm",
            worst_diag.get(
                "target_laser_distance_mm",
                worst.get("distance_to_laser_mm"),
            ),
        ),
        "test/worst_time_to_arrival": worst.get(
            "time_to_arrival_s",
            worst.get("laser_path_time_until_gate"),
        ),
        "test/worst_path_pre_arrival_gate": worst.get("laser_path_pre_arrival_gate"),
        "test/worst_endpoint_gate": worst.get("laser_endpoint_gate"),
        "test/worst_program_track_gate": worst.get("laser_path_program_track_gate"),
        "test/worst_time_until_gate": worst.get("laser_path_time_until_gate"),
        "test/worst_residual_gate": worst.get("laser_residual_gate"),
        "test/worst_residual_boost": worst.get("laser_residual_boost"),
    }
    for key in test_keys:
        value = values.get(key)
        acceptance_key = ACCEPTANCE_KEY_MAP.get(key)
        if acceptance_key and acceptance.get(acceptance_key) is not None:
            value = acceptance[acceptance_key]
        if value is not None:
            row[key.replace("test/", "test_")] = value
    row["test_report"] = report_path.name
    row["test_report_type"] = report_type
    row["acceptance_summary"] = acceptance_path.name if acceptance else ""
    row["split_source"] = split_protocol.get("source", "") if isinstance(split_protocol, dict) else ""
    row["split_path"] = split_protocol.get("path", "") if isinstance(split_protocol, dict) else ""
    row["split_test_count"] = split_test.get("count", "") if isinstance(split_test, dict) else ""
    if acceptance:
        for row_key, acceptance_key in [
            ("split_source", "split_source"),
            ("split_path", "split_path"),
            ("split_test_count", "split_test_count"),
        ]:
            value = acceptance.get(acceptance_key)
            if value is not None:
                row[row_key] = value
    row["split_test_first"] = split_test.get("first", "") if isinstance(split_test, dict) else ""
    row["split_test_last"] = split_test.get("last", "") if isinstance(split_test, dict) else ""
    return row


def format_value(value: object) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1000 or (abs(value) < 0.001 and value != 0):
            return f"{value:.3e}"
        return f"{value:.6g}"
    return str(value)


def write_markdown(rows: list[dict[str, object]], columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(col, "")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs_dir", type=Path, default=Path("logs"))
    parser.add_argument("--pattern", type=str, default="ablation_z*_probe*")
    parser.add_argument("--output_csv", type=Path, default=Path("results/z_experiment_summary.csv"))
    parser.add_argument("--output_md", type=Path, default=Path("results/z_experiment_summary.md"))
    parser.add_argument("--tags", nargs="*", default=DEFAULT_TAGS)
    parser.add_argument("--test_keys", nargs="*", default=DEFAULT_TEST_KEYS)
    args = parser.parse_args()

    run_dirs = sorted(p for p in args.logs_dir.glob(args.pattern) if p.is_dir())
    rows = []
    for run_dir in run_dirs:
        row = read_latest_scalars(run_dir, args.tags)
        row.update(read_test_report(run_dir, args.test_keys))
        rows.append(row)
    if not rows:
        raise SystemExit(f"No runs matched {args.logs_dir / args.pattern}")

    columns = ["run", "last_val_step"]
    for tag in args.tags:
        columns.append(tag.replace("val/", "").replace("train/", "train_"))
    for key in args.test_keys:
        columns.append(key.replace("test/", "test_"))
    columns.append("event_file")
    columns.append("test_report")
    columns.append("test_report_type")
    columns.append("acceptance_summary")
    columns.append("split_source")
    columns.append("split_test_count")
    columns.append("split_test_first")
    columns.append("split_test_last")
    columns.append("split_path")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    markdown_columns = [col for col in columns if col != "event_file"]
    write_markdown(rows, markdown_columns, args.output_md)
    print(f"Summarized {len(rows)} runs")
    print(f"CSV: {args.output_csv}")
    print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
