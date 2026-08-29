"""Decide between the z236 and z237 additive-path probe runs.

The z236/z237 pair isolates one question: whether the solver output is better
aligned by the empirically calibrated odd-layer hatch reversal or by the literal
para.xml/additive_z_scan path.  This script reads the z-group summary CSV and
writes a short Markdown decision report.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


RUN_KEYS = {
    "z236": "ablation_z236_",
    "z237": "ablation_z237_",
}

REQUIRED_FLOATS = [
    "test_RMSE",
    "test_MAE",
    "test_MaxError",
    "test_AbsErrorP99",
    "test_TempRecallAboveSolidus",
    "test_TempPrecisionAboveSolidus",
    "test_TempF1AboveSolidus",
    "test_worst_pred",
    "test_worst_target",
    "test_worst_abs_error",
]

OPTIONAL_FLOATS = [
    "test_worst_path_pre_arrival_gate",
    "test_worst_endpoint_gate",
    "test_worst_program_track_gate",
    "test_worst_time_until_gate",
    "test_worst_residual_gate",
    "test_worst_residual_boost",
]

DISPLAY_COLUMNS = [
    "run",
    "test_RMSE",
    "test_MAE",
    "test_AbsErrorP99",
    "test_MaxError",
    "test_TempRecallAboveSolidus",
    "test_TempPrecisionAboveSolidus",
    "test_TempF1AboveSolidus",
    "test_worst_step",
    "test_worst_node",
    "test_worst_pred",
    "test_worst_target",
    "test_worst_abs_error",
    "test_worst_path_pre_arrival_gate",
    "test_worst_endpoint_gate",
    "test_worst_program_track_gate",
    "test_worst_time_until_gate",
    "test_worst_residual_gate",
    "test_worst_residual_boost",
]


def parse_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_value(value: object) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer() and abs(number) < 100000:
        return str(int(number))
    if abs(number) >= 1000.0 or (abs(number) < 0.001 and number != 0.0):
        return f"{number:.3e}"
    return f"{number:.6g}"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def select_latest(rows: list[dict[str, str]], token: str) -> dict[str, str] | None:
    matches = [row for row in rows if token in row.get("run", "")]
    if not matches:
        return None

    def sort_key(row: dict[str, str]) -> tuple[float, str]:
        step = parse_float(row, "last_val_step")
        return ((-1.0 if step is None else step), row.get("run", ""))

    return sorted(matches, key=sort_key)[-1]


def validate_row(label: str, row: dict[str, str] | None) -> list[str]:
    if row is None:
        return [f"Missing {label} run in the summary CSV."]
    missing = [key for key in REQUIRED_FLOATS if parse_float(row, key) is None]
    if missing:
        return [f"{label} is missing required metrics: {', '.join(missing)}."]
    return []


def collect_metrics(row: dict[str, str]) -> dict[str, float | None]:
    keys = REQUIRED_FLOATS + OPTIONAL_FLOATS
    return {key: parse_float(row, key) for key in keys}


def pass_criteria(metrics: dict[str, float | None]) -> list[str]:
    reasons = []
    max_error = metrics["test_MaxError"]
    p99 = metrics["test_AbsErrorP99"]
    recall = metrics["test_TempRecallAboveSolidus"]
    precision = metrics["test_TempPrecisionAboveSolidus"]
    f1 = metrics["test_TempF1AboveSolidus"]
    worst_pred = metrics["test_worst_pred"]
    worst_target = metrics["test_worst_target"]

    if worst_pred is not None and worst_target is not None:
        if worst_pred >= 1200.0 and worst_target < 1200.0:
            reasons.append("worst case is a cold false-hot above 1200 C")
    if recall is not None and recall < 0.960:
        reasons.append(f"solidus recall {recall:.4f} < 0.960")
    if f1 is not None and f1 < 0.458:
        reasons.append(f"solidus F1 {f1:.4f} < 0.458")
    if precision is not None and precision < 0.295:
        reasons.append(f"solidus precision {precision:.4f} < 0.295")
    if p99 is not None and p99 > 21.2:
        reasons.append(f"AbsErrorP99 {p99:.2f} > 21.2 C")
    if max_error is not None and max_error >= 2000.0:
        reasons.append(f"MaxError {max_error:.1f} C is not below 2000 C")
    return reasons


def score(metrics: dict[str, float | None]) -> float:
    """Lower is better; emphasize worst miss while preserving detection quality."""
    max_error = metrics["test_MaxError"] or 1.0e9
    p99 = metrics["test_AbsErrorP99"] or 1.0e6
    rmse = metrics["test_RMSE"] or 1.0e6
    recall = metrics["test_TempRecallAboveSolidus"] or 0.0
    f1 = metrics["test_TempF1AboveSolidus"] or 0.0
    precision = metrics["test_TempPrecisionAboveSolidus"] or 0.0
    return max_error + 10.0 * p99 + rmse - 250.0 * recall - 120.0 * f1 - 50.0 * precision


def markdown_table(rows: list[dict[str, str]]) -> list[str]:
    lines = []
    lines.append("| " + " | ".join(DISPLAY_COLUMNS) + " |")
    lines.append("| " + " | ".join(["---"] * len(DISPLAY_COLUMNS)) + " |")
    for row in rows:
        lines.append(
            "| " + " | ".join(format_value(row.get(col, "")) for col in DISPLAY_COLUMNS) + " |"
        )
    return lines


def build_report(summary_path: Path) -> tuple[str, int]:
    rows = load_rows(summary_path)
    selected = {label: select_latest(rows, token) for label, token in RUN_KEYS.items()}

    errors = []
    for label, row in selected.items():
        errors.extend(validate_row(label, row))
    if errors:
        text = "# z236/z237 Decision\n\n" + "\n".join(f"- {err}" for err in errors) + "\n"
        return text, 2

    assert selected["z236"] is not None
    assert selected["z237"] is not None
    z236 = selected["z236"]
    z237 = selected["z237"]
    metrics = {"z236": collect_metrics(z236), "z237": collect_metrics(z237)}
    failures = {label: pass_criteria(vals) for label, vals in metrics.items()}
    passing = [label for label, reasons in failures.items() if not reasons]

    if passing:
        winner = min(passing, key=lambda label: score(metrics[label]))
        status = f"Use {winner} as the next base."
    else:
        winner = min(metrics, key=lambda label: score(metrics[label]))
        status = f"No run fully passes the probe gates; keep {winner} only as the next diagnostic base."

    z236_time = metrics["z236"].get("test_worst_time_until_gate")
    z237_time = metrics["z237"].get("test_worst_time_until_gate")
    z236_res = metrics["z236"].get("test_worst_residual_gate")
    z237_res = metrics["z237"].get("test_worst_residual_gate")

    guidance = []
    if winner == "z237":
        guidance.append("z237 winning means the literal para.xml hatch order is better aligned with the VTU sequence.")
    else:
        guidance.append("z236 winning means the odd-layer hatch-order reversal remains the better empirical solver alignment.")
    for label, vals in (("z236", metrics["z236"]), ("z237", metrics["z237"])):
        time_gate = vals.get("test_worst_time_until_gate")
        residual_gate = vals.get("test_worst_residual_gate")
        if time_gate is not None and residual_gate is not None:
            if time_gate <= 0.020 and residual_gate < 0.1:
                guidance.append(
                    f"{label}: ETA says near-future arrival but residual gate stayed closed; relax/trace support logic."
                )
            elif time_gate > 0.020 and vals["test_worst_abs_error"] and vals["test_worst_abs_error"] > 1200.0:
                guidance.append(
                    f"{label}: worst miss has weak ETA support; revisit path time offset/scale or hatch ordering."
                )

    lines = [
        "# z236/z237 Decision",
        "",
        f"Summary CSV: `{summary_path}`",
        "",
        f"Decision: **{status}**",
        "",
        "## Selected Runs",
        "",
    ]
    lines.extend(markdown_table([z236, z237]))
    lines.extend(["", "## Probe Gate Failures", ""])
    for label in ("z236", "z237"):
        if failures[label]:
            lines.append(f"- {label}: " + "; ".join(failures[label]))
        else:
            lines.append(f"- {label}: passes all probe gates")
    lines.extend(["", "## Diagnostic Notes", ""])
    lines.append(f"- z236 worst time-until/residual gate: {format_value(z236_time)} / {format_value(z236_res)}")
    lines.append(f"- z237 worst time-until/residual gate: {format_value(z237_time)} / {format_value(z237_res)}")
    for item in guidance:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n", 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_csv", type=Path, default=Path("results/z_experiment_summary.csv"))
    parser.add_argument("--output_md", type=Path, default=Path("results/z236_z237_decision.md"))
    args = parser.parse_args()

    report, exit_code = build_report(args.summary_csv)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(report)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
