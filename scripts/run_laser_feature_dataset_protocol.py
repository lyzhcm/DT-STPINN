"""Build and validate fixed laser trajectory feature chunks for E1-E3.

The model can generate target-step laser features online, but the roadmap also
expects a reproducible trajectory feature dataset artifact before long ablation
runs.  This wrapper builds train/val/test manifests with the approved XML path
and immediately validates the canonical E1/E2/E3 column groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_input_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        return None
    return value


def git_info() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root(),
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        return result.stdout.strip()

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "status_short": status.splitlines() if status else [],
        "is_dirty": bool(status),
    }


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


def command_record(phase: str, cmd: list[str], *, split: str | None = None, status: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "phase": phase,
        "command": display_command(cmd),
        "status": status,
    }
    if split is not None:
        record["split"] = split
    return record


def split_manifest_summary(args: argparse.Namespace, split: str) -> dict[str, Any]:
    manifest_path = split_output_dir(args, split) / "manifest.json"
    check_report_path = split_output_dir(args, split) / "feature_check_report.json"
    manifest = read_json(manifest_path)
    check_report = read_json(check_report_path)
    chunks = manifest.get("chunks", []) if manifest else []
    target_steps = manifest.get("target_steps", []) if manifest else []
    columns = manifest.get("columns", []) if manifest else []
    summary: dict[str, Any] = {
        "split": split,
        "manifest": str(manifest_path),
        "manifest_exists": manifest is not None,
        "check_report": str(check_report_path),
        "check_report_exists": check_report is not None,
        "check_ok": check_report.get("ok") if check_report else None,
        "coords_path": manifest.get("coords_path") if manifest else None,
        "coords_embedded_in_chunks": manifest.get("coords_embedded_in_chunks") if manifest else None,
        "coords_bytes": manifest.get("coords_bytes") if manifest else None,
        "num_chunks": len(chunks) if isinstance(chunks, list) else None,
        "num_target_steps": len(target_steps) if isinstance(target_steps, list) else None,
        "num_columns": len(columns) if isinstance(columns, list) else None,
    }
    if chunks:
        summary["first_step"] = int(chunks[0].get("step"))
        summary["last_step"] = int(chunks[-1].get("step"))
    return summary


def write_protocol_manifest(
    args: argparse.Namespace,
    splits: list[str],
    commands: list[dict[str, Any]],
) -> Path:
    config_path = resolve_input_path(args.config)
    laser_xml_path = resolve_input_path(args.laser_xml)
    reference_xml_path = resolve_input_path(args.reference_laser_xml)
    split_indices_path = resolve_input_path(args.split_indices)
    manifest = {
        "protocol": "laser_feature_dataset",
        "protocol_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info(),
        "inputs": {
            "config": args.config,
            "config_sha256": sha256_file(config_path),
            "laser_xml": args.laser_xml,
            "laser_xml_sha256": sha256_file(laser_xml_path),
            "reference_laser_xml": args.reference_laser_xml,
            "reference_laser_xml_sha256": sha256_file(reference_xml_path),
            "split_indices": args.split_indices,
            "split_indices_sha256": sha256_file(split_indices_path),
            "vtu_dir": args.vtu_dir,
        },
        "options": {
            "output_root": args.output_root,
            "splits": splits,
            "feature_group": args.feature_group,
            "dtype": args.dtype,
            "max_steps": args.max_steps,
            "check_max_chunks": args.check_max_chunks,
            "node_index": args.node_index,
            "strict_reference_laser_xml": args.strict_reference_laser_xml,
            "skip_preflight": args.skip_preflight,
            "skip_build": args.skip_build,
            "skip_check": args.skip_check,
        },
        "commands": commands,
        "split_manifests": [split_manifest_summary(args, split) for split in splits],
    }
    output_path = Path(args.output_root) / "laser_feature_dataset_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_path


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
    commands: list[dict[str, Any]] = []
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
        preflight_cmd = build_preflight_command(args)
        run_command(preflight_cmd, dry_run=False)
        commands.append(command_record("preflight", preflight_cmd, status="ok"))
    else:
        commands.append({"phase": "preflight", "status": "skipped"})

    for split in args.splits:
        print(f"\n=== {split} ===")
        if not args.skip_build:
            if should_build(args, split):
                build_cmd = build_feature_command(args, split)
                run_command(build_cmd, dry_run=args.dry_run)
                commands.append(command_record(
                    "build",
                    build_cmd,
                    split=split,
                    status="dry_run" if args.dry_run else "ok",
                ))
            else:
                print(f"Skipping build; manifest exists: {split_output_dir(args, split) / 'manifest.json'}")
                commands.append({"phase": "build", "split": split, "status": "skipped_existing"})
        else:
            commands.append({"phase": "build", "split": split, "status": "skipped"})
        if not args.skip_check:
            check_cmd = build_check_command(args, split)
            run_command(check_cmd, dry_run=args.dry_run)
            commands.append(command_record(
                "check",
                check_cmd,
                split=split,
                status="dry_run" if args.dry_run else "ok",
            ))
        else:
            commands.append({"phase": "check", "split": split, "status": "skipped"})

    if args.dry_run:
        print("\nDry run complete; protocol manifest was not written.")
    else:
        protocol_manifest = write_protocol_manifest(args, args.splits, commands)
        print(f"\nProtocol manifest: {protocol_manifest}")

    print("\nLaser feature dataset protocol complete.")


if __name__ == "__main__":
    main()
