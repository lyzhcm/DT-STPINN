"""Compare original FEM laser XML against the tracked reconstructed XML.

This is a lightweight guard for the fixed experiment protocol: the trajectory
features should come from the same additive_z_scan process definition used by
the simulator.  The comparison intentionally covers only process fields that are
defined in XML.  Runtime calibration knobs such as time_offset_s are configured
outside the XML and are not part of this check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.laser_path import AdditiveZScanPath


DEFAULT_REFERENCE = "F:\\datas\\5-block-fem\\para.xml"
DEFAULT_CANDIDATE = "configs\\laser_paths\\5_block_fem_additive_z_scan.xml"

PROCESS_FIELDS = [
    "start_point_mm",
    "scan_direction",
    "scan_length_mm",
    "hatch_direction",
    "hatch_count",
    "hatch_spacing_mm",
    "layer_count",
    "layer_thickness_mm",
    "velocity_mm_s",
    "each_path_time_s",
    "each_layer_time_s",
    "body_radius_mm",
    "body_height_mm",
    "body_radius_front_mm",
    "body_radius_back_mm",
]


def as_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.reshape(-1)]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def process_dict(path: AdditiveZScanPath) -> dict[str, Any]:
    return {field: as_builtin(getattr(path, field)) for field in PROCESS_FIELDS}


def values_match(left: Any, right: Any, *, rtol: float, atol: float) -> bool:
    if isinstance(left, list) or isinstance(right, list):
        return bool(np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), rtol=rtol, atol=atol))
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        return bool(np.isclose(float(left), float(right), rtol=rtol, atol=atol))
    return left == right


def compare_processes(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    rtol: float,
    atol: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in PROCESS_FIELDS:
        ref_value = reference[field]
        cand_value = candidate[field]
        ok = values_match(ref_value, cand_value, rtol=rtol, atol=atol)
        rows.append(
            {
                "field": field,
                "match": ok,
                "reference": ref_value,
                "candidate": cand_value,
            }
        )
    return rows


def load_xml(path: Path) -> AdditiveZScanPath:
    try:
        return AdditiveZScanPath.from_xml(path)
    except Exception as exc:  # noqa: BLE001 - CLI should surface malformed XML clearly.
        raise SystemExit(f"ERROR: could not parse {path}: {exc}") from exc


def format_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(f"{float(v):.10g}" for v in value) + "]"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def print_candidate_summary(candidate_path: Path, candidate: dict[str, Any]) -> None:
    print(f"Candidate XML parsed OK: {candidate_path}")
    print("Process fields:")
    for field in PROCESS_FIELDS:
        print(f"  {field}: {format_value(candidate[field])}")


def write_report(output_json: Path, payload: dict[str, Any]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote report: {output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare additive_z_scan process fields between two XML files."
    )
    parser.add_argument("--reference", default=DEFAULT_REFERENCE, help="Original solver para.xml path.")
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE, help="Tracked reconstructed XML path.")
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument(
        "--allow_missing_reference",
        action="store_true",
        help="Exit successfully after parsing the candidate if the original XML is unavailable.",
    )
    parser.add_argument("--output_json", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_path = Path(args.reference)
    candidate_path = Path(args.candidate)

    if not candidate_path.exists():
        raise SystemExit(f"ERROR: candidate XML missing: {candidate_path}")
    candidate = process_dict(load_xml(candidate_path))

    if not reference_path.exists():
        print(f"Reference XML missing: {reference_path}")
        print_candidate_summary(candidate_path, candidate)
        payload = {
            "status": "reference_missing",
            "reference_xml": str(reference_path),
            "candidate_xml": str(candidate_path),
            "candidate_process": candidate,
        }
        if args.output_json:
            write_report(Path(args.output_json), payload)
        if args.allow_missing_reference:
            return
        raise SystemExit(2)

    reference = process_dict(load_xml(reference_path))
    rows = compare_processes(reference, candidate, rtol=args.rtol, atol=args.atol)
    mismatches = [row for row in rows if not row["match"]]

    print("Laser XML process comparison")
    print(f"  Reference: {reference_path}")
    print(f"  Candidate: {candidate_path}")
    for row in rows:
        mark = "OK" if row["match"] else "DIFF"
        print(
            f"  {mark:4s} {row['field']}: "
            f"ref={format_value(row['reference'])} cand={format_value(row['candidate'])}"
        )

    payload = {
        "status": "match" if not mismatches else "mismatch",
        "reference_xml": str(reference_path),
        "candidate_xml": str(candidate_path),
        "rtol": args.rtol,
        "atol": args.atol,
        "rows": rows,
        "mismatch_count": len(mismatches),
    }
    if args.output_json:
        write_report(Path(args.output_json), payload)

    if mismatches:
        fields = ", ".join(row["field"] for row in mismatches)
        raise SystemExit(f"ERROR: XML process mismatch in: {fields}")

    print("XML process fields match.")


if __name__ == "__main__":
    main()
