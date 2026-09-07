"""Run the fixed baseline protocol: train, canonical evaluation, freeze.

This is the one-command wrapper for the baseline control run.  It keeps the
steps explicit so a long 50-epoch run can be dry-run first, resumed after an
interruption, and frozen with a manifest after canonical evaluation.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config


def display_command(cmd: list[str]) -> str:
    if sys.platform == "win32":
        return subprocess.list2cmdline([str(part) for part in cmd])
    return shlex.join(str(part) for part in cmd)


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print("\n$ " + display_command(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def build_preflight_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        "scripts/check_experiment_readiness.py",
        "--vtu_dir",
        args.vtu_dir,
        "--laser_xml",
        args.laser_xml,
        "--split_indices",
        args.split_indices,
        "--baseline_artifact",
        args.baseline_artifact,
        "--experiments",
        "E0",
        "--epochs",
        str(args.epochs),
        "--graph_device",
        args.graph_device,
        "--no_command_preview",
    ]
    return cmd

def build_train_command(args: argparse.Namespace, run_name: str) -> list[str]:
    cmd = [
        args.python,
        "scripts/train.py",
        "--config",
        args.config,
        "--vtu_dir",
        args.vtu_dir,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--experiment_name",
        run_name,
        "--graph_device",
        args.graph_device,
        "--skip_test",
    ]
    if args.device != "auto":
        cmd.extend(["--device", args.device])
    if args.cache_dir != "data/processed":
        cmd.extend(["--cache_dir", args.cache_dir])
    if args.laser_xml:
        cmd.extend(["--laser_xml", args.laser_xml])
    if args.split_indices:
        cmd.extend(["--split_indices", args.split_indices])

    last_checkpoint = Path(args.log_dir) / run_name / "last_model.pt"
    if args.resume_from_last and last_checkpoint.is_file():
        cmd.extend(["--resume", str(last_checkpoint)])
    return cmd


def build_eval_command(args: argparse.Namespace, run_name: str) -> list[str]:
    checkpoint = Path(args.log_dir) / run_name / args.checkpoint_name
    output_dir = Path(args.log_dir) / run_name
    if not args.dry_run and not checkpoint.is_file():
        raise FileNotFoundError(
            f"Expected baseline checkpoint at {checkpoint}. Run training first or "
            "use --checkpoint_name for an existing checkpoint."
        )
    cmd = [
        args.python,
        "scripts/evaluate_checkpoint.py",
        "--config",
        args.config,
        "--checkpoint",
        str(checkpoint),
        "--vtu_dir",
        args.vtu_dir,
        "--output_dir",
        str(output_dir),
        "--graph_device",
        args.graph_device,
    ]
    if args.device != "auto":
        cmd.extend(["--device", args.device])
    if args.cache_dir != "data/processed":
        cmd.extend(["--cache_dir", args.cache_dir])
    if args.laser_xml:
        cmd.extend(["--laser_xml", args.laser_xml])
    if args.split_indices:
        cmd.extend(["--split_indices", args.split_indices])
    return cmd


def build_freeze_command(
        args: argparse.Namespace,
        run_name: str,
        train_cmd: list[str],
        eval_cmd: list[str],
        ) -> list[str]:
    checkpoint = Path(args.log_dir) / run_name / args.checkpoint_name
    report = Path(args.log_dir) / run_name / "evaluation_report.json"
    if not args.dry_run:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Cannot freeze missing checkpoint: {checkpoint}")
        if not report.is_file():
            raise FileNotFoundError(f"Cannot freeze missing evaluation report: {report}")
    cmd = [
        args.python,
        "scripts/freeze_baseline.py",
        "--name",
        args.baseline_name,
        "--config",
        args.config,
        "--checkpoint",
        str(checkpoint),
        "--test_report",
        str(report),
        "--vtu_dir",
        args.vtu_dir,
        "--seed",
        str(args.seed),
        "--train_command",
        display_command(train_cmd),
        "--eval_command",
        display_command(eval_cmd),
        "--notes",
        args.notes,
        "--output_dir",
        args.output_dir,
    ]
    if args.laser_xml:
        cmd.extend(["--laser_xml", args.laser_xml])
    if args.split_indices:
        cmd.extend(["--split_indices", args.split_indices])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/feature_e0_baseline.yaml")
    parser.add_argument("--vtu_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--graph_device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cache_dir", default="data/processed")
    parser.add_argument(
        "--laser_xml",
        default="configs\\laser_paths\\5_block_fem_additive_z_scan.xml",
        help="Optional para.xml path passed through training/evaluation/freeze.",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Run log directory. Defaults to config.logging.log_dir and must match it.",
    )
    parser.add_argument("--output_dir", default="artifacts/baselines")
    parser.add_argument("--checkpoint_name", default="best_model.pt")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--baseline_name", default="feature_e0_baseline_50epoch")
    parser.add_argument("--baseline_artifact", default="artifacts\\baselines\\paper1_fast_50epoch_canonical_eval_20260829T185514Z")
    parser.add_argument(
        "--split_indices",
        default="artifacts\\baselines\\paper1_fast_50epoch_canonical_eval_20260829T185514Z\\split_indices.json",
        help="Frozen split_indices.json. Omit only when creating the first frozen split.",
    )
    parser.add_argument("--resume_from_last", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_freeze", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_preflight", action="store_true", help="Skip read-only readiness checks before running.")
    parser.add_argument("--notes", default="Fixed 50-epoch baseline protocol run.")
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")
    if args.resume_from_last and args.skip_train:
        parser.error("--resume_from_last has no effect with --skip_train.")
    return args


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    config_log_dir = str(config.logging.log_dir)
    if args.log_dir is None:
        args.log_dir = config_log_dir
    elif str(args.log_dir) != config_log_dir:
        raise ValueError(
            "--log_dir must match config.logging.log_dir because scripts/train.py "
            f"does not override it from CLI: {args.log_dir!r} != {config_log_dir!r}"
        )
    run_name = args.experiment_name or config.logging.experiment_name

    train_cmd = build_train_command(args, run_name)
    eval_cmd = build_eval_command(args, run_name)

    print("Fixed baseline protocol")
    print(f"  Config      : {args.config}")
    print(f"  Run name    : {run_name}")
    print(f"  Baseline    : {args.baseline_name}")
    print(f"  VTU dir     : {args.vtu_dir}")
    print(f"  Laser XML   : {args.laser_xml or 'from config'}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Seed        : {args.seed}")
    print(f"  Split       : {args.split_indices or 'ratio from config'}")
    print(f"  Checkpoint  : {args.checkpoint_name}")
    print(f"  Dry run     : {args.dry_run}")

    if not args.no_preflight:
        print("\nPreflight checks")
        run_command(build_preflight_command(args), dry_run=False)

    if not args.skip_train:
        run_command(train_cmd, dry_run=args.dry_run)
    if not args.skip_eval:
        run_command(eval_cmd, dry_run=args.dry_run)
    if not args.skip_freeze:
        freeze_cmd = build_freeze_command(args, run_name, train_cmd, eval_cmd)
        run_command(freeze_cmd, dry_run=args.dry_run)

    print("\nFixed baseline protocol complete.")


if __name__ == "__main__":
    main()
