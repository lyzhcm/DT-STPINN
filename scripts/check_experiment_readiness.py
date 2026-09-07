"""Preflight checks before launching fixed baseline or feature ablations.

The script is intentionally read-only. It checks paths, fixed split metadata,
experiment configs, and prints the long-run commands that should be launched
only after the checks pass.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.preprocessing import load_split_indices
from src.utils.laser_path import AdditiveZScanPath
from scripts.compare_laser_xml import compare_processes, process_dict


EXPERIMENT_CONFIGS = {
    "E0": Path("configs/feature_e0_baseline.yaml"),
    "E1": Path("configs/feature_e1_laser_distance.yaml"),
    "E2": Path("configs/feature_e2_scan_arrival.yaml"),
    "E3": Path("configs/feature_e3_path_phase.yaml"),
    "E4": Path("configs/feature_e4_hotspot_aux.yaml"),
    "E5": Path("configs/feature_e5_hotspot_residual.yaml"),
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def print_result(result: CheckResult) -> None:
    status = "OK" if result.ok else "FAIL"
    print(f"[{status}] {result.name}: {result.detail}")


def display_command(cmd: list[str]) -> str:
    if sys.platform == "win32":
        return subprocess.list2cmdline([str(part) for part in cmd])
    return shlex.join(str(part) for part in cmd)


def count_vtu_files(vtu_dir: Path) -> int:
    pattern = re.compile(r"Data-(\d+)\.vtu$", re.IGNORECASE)
    if not vtu_dir.exists():
        return 0
    return sum(1 for path in vtu_dir.glob("*.vtu") if pattern.match(path.name))


def check_vtu_dir(vtu_dir: Path, min_vtu_steps: int) -> CheckResult:
    count = count_vtu_files(vtu_dir)
    if count < min_vtu_steps:
        return CheckResult(
            "VTU sequence",
            False,
            f"found {count} Data-*.vtu files in {vtu_dir}; expected at least {min_vtu_steps}",
        )
    return CheckResult("VTU sequence", True, f"found {count} Data-*.vtu files in {vtu_dir}")


def find_xml_candidates(search_roots: list[str], max_candidates: int) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for root_value in search_roots:
        if not root_value:
            continue
        root = Path(root_value)
        if not root.exists():
            continue
        patterns = ["para.xml", "*.xml"]
        for pattern in patterns:
            try:
                matches = root.rglob(pattern) if root.is_dir() else [root]
                for path in matches:
                    if not path.is_file():
                        continue
                    key = str(path.resolve()).lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(path)
                    if len(candidates) >= max_candidates:
                        return candidates
            except OSError:
                continue
    return candidates


def check_xml(xml_path: Path | None, search_roots: list[str], max_candidates: int) -> CheckResult:
    if xml_path is None:
        return CheckResult("laser XML", False, "--laser_xml is required for the fixed roadmap protocol")
    if not xml_path.exists():
        candidates = find_xml_candidates(search_roots, max_candidates)
        if candidates:
            suggestion = "; candidates: " + ", ".join(str(path) for path in candidates)
        else:
            suggestion = "; no XML candidates found under " + ", ".join(search_roots)
        return CheckResult("laser XML", False, f"missing {xml_path}{suggestion}")
    try:
        path = AdditiveZScanPath.from_xml(xml_path)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("laser XML", False, f"could not parse {xml_path}: {exc}")
    return CheckResult(
        "laser XML",
        True,
        f"{xml_path}, layers={path.layer_count}, tracks/layer={path.hatch_count}, velocity={path.velocity_mm_s:g}",
    )


def check_xml_reference(
    candidate_path: Path | None,
    reference_path: Path | None,
    *,
    strict: bool,
    rtol: float,
    atol: float,
) -> CheckResult:
    if reference_path is None:
        return CheckResult("laser XML reference", True, "not requested")
    if candidate_path is None or not candidate_path.exists():
        return CheckResult("laser XML reference", True, "skipped until --laser_xml exists")
    if not reference_path.exists():
        return CheckResult(
            "laser XML reference",
            not strict,
            f"reference missing {reference_path}; candidate parsed only",
        )

    try:
        reference = process_dict(AdditiveZScanPath.from_xml(reference_path))
        candidate = process_dict(AdditiveZScanPath.from_xml(candidate_path))
    except Exception as exc:  # noqa: BLE001
        return CheckResult("laser XML reference", False, f"could not parse XML for comparison: {exc}")

    rows = compare_processes(reference, candidate, rtol=rtol, atol=atol)
    mismatches = [row["field"] for row in rows if not row["match"]]
    if mismatches:
        return CheckResult(
            "laser XML reference",
            False,
            "process mismatch in " + ", ".join(mismatches),
        )
    return CheckResult(
        "laser XML reference",
        True,
        f"matches {reference_path}",
    )


def check_split(split_path: Path | None, total_steps: int | None) -> CheckResult:
    if split_path is None:
        return CheckResult("frozen split", False, "--split_indices is required for comparable long runs")
    if not split_path.exists():
        return CheckResult("frozen split", False, f"missing {split_path}")
    try:
        train, val, test = load_split_indices(split_path, total_steps=total_steps)
    except Exception as exc:  # noqa: BLE001 - preflight should report any malformed split.
        return CheckResult("frozen split", False, f"could not load {split_path}: {exc}")
    return CheckResult(
        "frozen split",
        True,
        f"train={len(train)}, val={len(val)}, test={len(test)} from {split_path}",
    )


def check_config(exp: str, path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult(f"{exp} config", False, f"missing {path}")
    try:
        config = Config.from_yaml(path)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(f"{exp} config", False, f"could not load {path}: {exc}")

    expected_group = exp.lower() if exp in {"E0", "E1", "E2", "E3"} else "e3"
    group = str(config.data.laser_feature_group).lower()
    if group != expected_group:
        return CheckResult(
            f"{exp} config",
            False,
            f"data.laser_feature_group={group!r}; expected {expected_group!r}",
        )

    if exp == "E4" and not config.model.enable_hotspot_head:
        return CheckResult("E4 config", False, "enable_hotspot_head must be true")
    if exp == "E5" and not config.model.enable_laser_residual_head:
        return CheckResult("E5 config", False, "enable_laser_residual_head must be true")

    return CheckResult(
        f"{exp} config",
        True,
        f"{path} feature_group={group}, hidden_dim={config.model.hidden_dim}",
    )


def check_baseline_artifact(path: Path | None) -> CheckResult:
    if path is None:
        return CheckResult("historical baseline artifact", True, "not requested")
    if not path.exists():
        return CheckResult("historical baseline artifact", False, f"missing {path}")
    required = ["baseline_manifest.json", "best_model.pt", "evaluation_report.json", "split_indices.json"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        return CheckResult("historical baseline artifact", False, f"missing {', '.join(missing)} in {path}")
    try:
        manifest = json.loads((path / "baseline_manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return CheckResult("historical baseline artifact", False, f"manifest unreadable: {exc}")
    return CheckResult(
        "historical baseline artifact",
        True,
        f"{path.name}, commit={manifest.get('git', {}).get('commit', 'unknown')}",
    )


def build_baseline_dry_run(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_baseline_protocol.py",
        "--config",
        str(EXPERIMENT_CONFIGS["E0"]),
        "--vtu_dir",
        args.vtu_dir,
        "--laser_xml",
        args.laser_xml or "<missing-laser-xml>",
        "--epochs",
        str(args.epochs),
        "--graph_device",
        args.graph_device,
        "--split_indices",
        args.split_indices or "<missing-split-indices>",
        "--resume_from_last",
        "--dry_run",
    ]
    return cmd


def build_ablation_dry_run(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_feature_ablation.py",
        "--experiments",
        *args.experiments,
        "--vtu_dir",
        args.vtu_dir,
        "--laser_xml",
        args.laser_xml or "<missing-laser-xml>",
        "--epochs",
        str(args.epochs),
        "--graph_device",
        args.graph_device,
        "--split_indices",
        args.split_indices or "<missing-split-indices>",
        "--resume_from_last",
        "--dry_run",
        "--skip_summary",
    ]
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only preflight for fixed DT-STPINN experiments")
    parser.add_argument("--vtu_dir", default="F:\\VTU")
    parser.add_argument("--laser_xml", default="configs\\laser_paths\\5_block_fem_additive_z_scan.xml")
    parser.add_argument(
        "--reference_laser_xml",
        default="F:\\datas\\5-block-fem\\para.xml",
        help="Original solver para.xml to compare against when available.",
    )
    parser.add_argument(
        "--strict_reference_laser_xml",
        action="store_true",
        help="Fail readiness if --reference_laser_xml is missing instead of warning.",
    )
    parser.add_argument("--xml_compare_rtol", type=float, default=1.0e-6)
    parser.add_argument("--xml_compare_atol", type=float, default=1.0e-8)
    parser.add_argument(
        "--split_indices",
        default="artifacts\\baselines\\paper1_fast_50epoch_canonical_eval_20260829T185514Z\\split_indices.json",
    )
    parser.add_argument(
        "--baseline_artifact",
        default="artifacts\\baselines\\paper1_fast_50epoch_canonical_eval_20260829T185514Z",
    )
    parser.add_argument("--experiments", nargs="+", default=["E0", "E1", "E2", "E3"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--graph_device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--min_vtu_steps", type=int, default=2000)
    parser.add_argument(
        "--xml_search_roots",
        nargs="*",
        default=["F:\\datas", "F:\\DT-STPINN"],
        help="Roots to scan for XML candidates when --laser_xml is missing.",
    )
    parser.add_argument("--xml_search_max", type=int, default=10)
    parser.add_argument("--no_command_preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = [exp for exp in args.experiments if exp not in EXPERIMENT_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown experiments: {', '.join(unknown)}")

    vtu_dir = Path(args.vtu_dir)
    total_steps = count_vtu_files(vtu_dir)
    results = [
        check_vtu_dir(vtu_dir, args.min_vtu_steps),
        check_xml(
            Path(args.laser_xml) if args.laser_xml else None,
            args.xml_search_roots,
            args.xml_search_max,
        ),
        check_xml_reference(
            Path(args.laser_xml) if args.laser_xml else None,
            Path(args.reference_laser_xml) if args.reference_laser_xml else None,
            strict=args.strict_reference_laser_xml,
            rtol=args.xml_compare_rtol,
            atol=args.xml_compare_atol,
        ),
        check_split(Path(args.split_indices) if args.split_indices else None, total_steps or None),
        check_baseline_artifact(Path(args.baseline_artifact) if args.baseline_artifact else None),
    ]
    results.extend(check_config(exp, EXPERIMENT_CONFIGS[exp]) for exp in args.experiments)

    print("Fixed experiment readiness")
    for result in results:
        print_result(result)

    if not args.no_command_preview:
        print("\nCommand preview")
        print("$ " + display_command(build_baseline_dry_run(args)))
        print("$ " + display_command(build_ablation_dry_run(args)))

    if any(not result.ok for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
