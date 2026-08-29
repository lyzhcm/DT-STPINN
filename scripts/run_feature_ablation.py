"""Run the fixed laser-feature and minimal model-head ablation protocol.

The default protocol runs E0-E3, which isolates trajectory-feature changes.
E4/E5 are opt-in model-head ablations for the next stage: hotspot auxiliary
classification and hotspot-gated high-temperature residual prediction.

Example:
    python scripts/run_feature_ablation.py --vtu_dir F:\VTU --epochs 50 --graph_device cuda

For a safety check before a long run:
    python scripts/run_feature_ablation.py --vtu_dir F:\VTU --dry_run
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config


@dataclass(frozen=True)
class Experiment:
    key: str
    config_path: Path
    eval_checkpoint: str = "best_model.pt"


EXPERIMENTS = {
    "E0": Experiment("E0", Path("configs/feature_e0_baseline.yaml")),
    "E1": Experiment("E1", Path("configs/feature_e1_laser_distance.yaml")),
    "E2": Experiment("E2", Path("configs/feature_e2_scan_arrival.yaml")),
    "E3": Experiment("E3", Path("configs/feature_e3_path_phase.yaml")),
    "E4": Experiment(
        "E4",
        Path("configs/feature_e4_hotspot_aux.yaml"),
        eval_checkpoint="best_hot_model.pt",
    ),
    "E5": Experiment(
        "E5",
        Path("configs/feature_e5_hotspot_residual.yaml"),
        eval_checkpoint="best_hot_model.pt",
    ),
}


def display_command(cmd: list[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print("\n$ " + display_command(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def experiment_name(config_path: Path) -> str:
    config = Config.from_yaml(config_path)
    return config.logging.experiment_name


def optional_limit_args(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    for cli_name, value in [
        ("--max_train_samples", args.max_train_samples),
        ("--max_val_samples", args.max_val_samples),
        ("--max_test_samples", args.max_test_samples),
    ]:
        if value is not None:
            result.extend([cli_name, str(value)])
    return result


def build_train_command(args: argparse.Namespace, exp: Experiment, run_name: str) -> list[str]:
    cmd = [
        args.python,
        "scripts/train.py",
        "--config",
        str(exp.config_path),
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
    if args.split_indices:
        cmd.extend(["--split_indices", args.split_indices])
    cmd.extend(optional_limit_args(args))

    last_checkpoint = Path("logs") / run_name / "last_model.pt"
    if args.resume_from_last and last_checkpoint.is_file():
        cmd.extend(["--resume", str(last_checkpoint)])
    return cmd


def build_eval_command(args: argparse.Namespace, exp: Experiment, run_name: str) -> list[str]:
    checkpoint_name = args.checkpoint_name or exp.eval_checkpoint
    checkpoint = Path("logs") / run_name / checkpoint_name
    if not args.dry_run and not checkpoint.is_file():
        raise FileNotFoundError(
            f"Expected checkpoint for {run_name}: {checkpoint}. "
            "Use --skip_train only after the checkpoint exists."
        )
    cmd = [
        args.python,
        "scripts/evaluate_checkpoint.py",
        "--config",
        str(exp.config_path),
        "--checkpoint",
        str(checkpoint),
        "--vtu_dir",
        args.vtu_dir,
        "--output_dir",
        str(Path("logs") / run_name),
        "--graph_device",
        args.graph_device,
    ]
    if args.split_indices:
        cmd.extend(["--split_indices", args.split_indices])
    return cmd


def build_summary_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        "scripts/summarize_z_experiments.py",
        "--logs_dir",
        "logs",
        "--pattern",
        args.summary_pattern,
        "--output_csv",
        args.output_csv,
        "--output_md",
        args.output_md,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["E0", "E1", "E2", "E3"],
        choices=sorted(EXPERIMENTS),
        help="Experiment keys to run in order.",
    )
    parser.add_argument("--vtu_dir", required=True, help="Directory containing Data-*.vtu files.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--graph_device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cache_dir", default="data/processed")
    parser.add_argument(
        "--split_indices",
        default=None,
        help="Frozen split_indices.json to reuse across all experiments.",
    )
    parser.add_argument(
        "--checkpoint_name",
        default=None,
        help="Checkpoint filename to evaluate for every experiment. Defaults to each experiment's protocol checkpoint.",
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument(
        "--resume_from_last",
        action="store_true",
        help="Resume an experiment from logs/<experiment>/last_model.pt when present.",
    )
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary_pattern", default="feature_e*")
    parser.add_argument("--output_csv", default="results/feature_ablation_summary.csv")
    parser.add_argument("--output_md", default="results/feature_ablation_summary.md")
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")
    for name in ("max_train_samples", "max_val_samples", "max_test_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name} must be greater than zero.")
    return args


def main() -> None:
    args = parse_args()
    selected = [EXPERIMENTS[key] for key in args.experiments]

    print("Feature ablation protocol")
    print(f"  Experiments : {', '.join(exp.key for exp in selected)}")
    print(f"  VTU dir     : {args.vtu_dir}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Seed        : {args.seed}")
    print(f"  Split       : {args.split_indices or 'ratio from config'}")
    checkpoint_label = args.checkpoint_name or "protocol default"
    print(f"  Checkpoint  : {checkpoint_label}")
    print(f"  Dry run     : {args.dry_run}")

    for exp in selected:
        if not exp.config_path.is_file():
            raise FileNotFoundError(exp.config_path)
        run_name = experiment_name(exp.config_path)
        print(f"\n=== {exp.key}: {run_name} ===")
        if not args.skip_train:
            run_command(build_train_command(args, exp, run_name), dry_run=args.dry_run)
        if not args.skip_eval:
            run_command(build_eval_command(args, exp, run_name), dry_run=args.dry_run)

    if not args.skip_summary:
        run_command(build_summary_command(args), dry_run=args.dry_run)

    print("\nFeature ablation protocol complete.")


if __name__ == "__main__":
    main()
