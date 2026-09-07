"""Build and validate fixed laser trajectory feature chunks for E1-E3.

The model can generate target-step laser features online, but the roadmap also
expects a reproducible trajectory feature dataset artifact before long ablation
runs.  This wrapper builds train/val/test manifests with the approved XML path
and immediately validates the canonical E1/E2/E3 column groups.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_XML = "configs\\laser_paths\\5_block_fem_additive_z_scan.xml"
DEFAULT_SPLIT = (
    "artifacts\\baselines\\paper1_fast_50epoch_canonical_eval_20260829T185514Z"
    "\\split_indices.json"
)


def display_command(cmd: list[str]) -> str:
    if sys.platform == "win32":
        return subprocess.list2cmdline([str(part) for part in cmd])
    return shlex.join(str(part) for part in cmd)


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print("\n$ " + display_command(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def split_output_dir(args: argparse.Namespace, split: str) -> Path:
    return Path(args.output_root) / split


def build_preflight_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        "scripts/check_experiment_readiness.py",
        "--vtu_dir",
        args.vtu_dir,
        "--laser_xml",
        args.laser_xml,
        "--reference_laser_xml",
        args.reference_laser_xml,
        "--split_indices",
        args.split_indices,
        "--experiments",
        "E1",
        "E2",
        "E3",
        "--no_command_preview",
    ]
    if args.strict_reference_laser_xml:
        cmd.append("--strict_reference_laser_xml")
    return cmd


def build_feature_command(args: argparse.Namespace, split: str) -> list[str]:
    cmd = [
        args.python,
        "scripts/build_laser_feature_dataset.py",
        "--config",
        args.config,
        "--xml",
        args.laser_xml,
        "--vtu_dir",
        args.vtu_dir,
        "--output_dir",
        str(split_output_dir(args, split)),
        "--split",
        split,
        "--split_indices",
        args.split_indices,
        "--dtype",
        args.dtype,
    ]
    if args.max_steps is not None:
        cmd.extend(["--max_steps", str(args.max_steps)])
    return cmd


def build_check_command(args: argparse.Namespace, split: str) -> list[str]:
    cmd = [
        args.python,
        "scripts/check_laser_feature_dataset.py",
        "--manifest",
        str(split_output_dir(args, split) / "manifest.json"),
        "--feature_group",
        args.feature_group,
        "--require_xml",
    ]
    if args.check_max_chunks is not None:
        cmd.extend(["--max_chunks", str(args.check_max_chunks)])
    if args.node_index is not None:
        cmd.extend(["--node_index", str(args.node_index)])
    return cmd


def should_build(args: argparse.Namespace, split: str) -> bool:
    manifest = split_output_dir(args, split) / "manifest.json"
    return args.force or not manifest.exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/feature_e3_path_phase.yaml")
    parser.add_argument("--vtu_dir", required=True)
    parser.add_argument("--laser_xml", default=DEFAULT_XML)
    parser.add_argument(
        "--reference_laser_xml",
        default="F:\\datas\\5-block-fem\\para.xml",
        help="Original solver para.xml compared during preflight when available.",
    )
    parser.add_argument(
        "--strict_reference_laser_xml",
        action="store_true",
        help="Fail preflight unless --reference_laser_xml exists and matches --laser_xml.",
    )
    parser.add_argument("--split_indices", default=DEFAULT_SPLIT)
    parser.add_argument("--output_root", default="data/processed/laser_features_fixed")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test", "all"],
        help="Splits to build and validate.",
    )
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--feature_group",
        default="all",
        help="Feature group passed to check_laser_feature_dataset.py: E1, E2, E3, or all.",
    )
    parser.add_argument("--max_steps", type=int, default=None, help="Smoke-test limit per split.")
    parser.add_argument("--check_max_chunks", type=int, default=None)
    parser.add_argument("--node_index", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true", help="Rebuild a split even when manifest.json exists.")
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--skip_check", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max_steps must be greater than zero.")
    if args.check_max_chunks is not None and args.check_max_chunks <= 0:
        parser.error("--check_max_chunks must be greater than zero.")
    return args


def main() -> None:
    args = parse_args()
    print("Laser feature dataset protocol")
    print(f"  Config      : {args.config}")
    print(f"  VTU dir     : {args.vtu_dir}")
    print(f"  Laser XML   : {args.laser_xml}")
    print(f"  Split file  : {args.split_indices}")
    print(f"  Output root : {args.output_root}")
    print(f"  Splits      : {', '.join(args.splits)}")
    print(f"  Feature grp : {args.feature_group}")
    print(f"  Dry run     : {args.dry_run}")

    if not args.skip_preflight:
        print("\nPreflight checks")
        run_command(build_preflight_command(args), dry_run=False)

    for split in args.splits:
        print(f"\n=== {split} ===")
        if not args.skip_build:
            if should_build(args, split):
                run_command(build_feature_command(args, split), dry_run=args.dry_run)
            else:
                print(f"Skipping build; manifest exists: {split_output_dir(args, split) / 'manifest.json'}")
        if not args.skip_check:
            run_command(build_check_command(args, split), dry_run=args.dry_run)

    print("\nLaser feature dataset protocol complete.")


if __name__ == "__main__":
    main()
