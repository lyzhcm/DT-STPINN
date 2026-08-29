"""Freeze a reproducible baseline manifest and local artifact bundle.

This script intentionally stores heavyweight checkpoints locally.  The repo's
``.gitignore`` keeps ``*.pt`` out of Git, while the JSON manifest records the
exact paths and SHA256 hashes needed to reproduce the baseline later.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.preprocessing import load_split_indices, split_indices
from src.data.vtu_loader import VTULoader


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return f"<unavailable: {exc}>"


def copy_artifact(src: Path, dst_dir: Path) -> dict[str, object]:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = dst_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return {
        "source": str(src),
        "frozen_path": str(dst),
        "bytes": dst.stat().st_size,
        "sha256": sha256_file(dst),
    }


def copy_optional_artifact(
    src: Path, dst_dir: Path, *, dst_name: str | None = None
) -> dict[str, object]:
    """Copy an optional artifact and keep source/hash metadata."""
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = dst_dir / (dst_name or src.name)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return {
        "source": str(src),
        "frozen_path": str(dst),
        "bytes": dst.stat().st_size,
        "sha256": sha256_file(dst),
    }


def summarize_split(indices: list[int]) -> dict[str, object]:
    if not indices:
        return {"count": 0, "first": None, "last": None}
    return {"count": len(indices), "first": indices[0], "last": indices[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze DT-STPINN baseline artifacts")
    parser.add_argument("--name", required=True, help="Baseline name, e.g. paper1_fast_50epoch")
    parser.add_argument("--config", default="configs/paper1_fast.yaml")
    parser.add_argument("--checkpoint", required=True, help="Best checkpoint path to freeze")
    parser.add_argument("--test_report", required=True, help="test_metrics.json or evaluation report")
    parser.add_argument("--vtu_dir", default=None, help="VTU directory used by the run")
    parser.add_argument(
        "--split_indices",
        default=None,
        help=(
            "Frozen split_indices.json used by training/evaluation. "
            "If omitted, the split is rebuilt from config ratios."
        ),
    )
    parser.add_argument(
        "--laser_xml",
        default=None,
        help="Optional para.xml path used to override YAML laser path geometry.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_command", default="", help="Original training command, if known")
    parser.add_argument("--eval_command", default="", help="Original evaluation command, if known")
    parser.add_argument("--notes", default="")
    parser.add_argument("--output_dir", default="artifacts/baselines")
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    report_path = Path(args.test_report)
    config = Config.from_yaml(config_path)
    vtu_dir = Path(args.vtu_dir or config.data.vtu_dir)

    loader = VTULoader(vtu_dir)
    if loader.num_steps == 0:
        raise SystemExit(f"No VTU files found in {vtu_dir}")

    if args.split_indices:
        split_source = f"frozen:{args.split_indices}"
        train_idx, val_idx, test_idx = load_split_indices(
            args.split_indices, total_steps=loader.num_steps
        )
    else:
        split_source = "ratio"
        train_idx, val_idx, test_idx = split_indices(
            loader.num_steps,
            train_ratio=config.data.train_split,
            val_ratio=config.data.val_split,
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"{args.name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = {
        "config": copy_artifact(config_path, out_dir),
        "checkpoint": copy_artifact(checkpoint_path, out_dir),
        "test_report": copy_artifact(report_path, out_dir),
    }

    split_payload = {
        "protocol": "contiguous chronological split over VTU sequence",
        "source": split_source,
        "source_path": args.split_indices,
        "num_steps": loader.num_steps,
        "train_split": config.data.train_split,
        "val_split": config.data.val_split,
        "test_split": config.data.test_split,
        "window_size": config.data.window_size,
        "predict_steps": config.data.predict_steps,
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "summary": {
            "train": summarize_split(train_idx),
            "val": summarize_split(val_idx),
            "test": summarize_split(test_idx),
        },
    }
    split_path = out_dir / "split_indices.json"
    split_path.write_text(json.dumps(split_payload, indent=2) + "\n", encoding="utf-8")

    source_split = None
    if args.split_indices:
        source_split = copy_optional_artifact(
            Path(args.split_indices), out_dir, dst_name="source_split_indices.json"
        )

    laser_xml_artifact = None
    if args.laser_xml:
        laser_xml_artifact = copy_optional_artifact(
            Path(args.laser_xml), out_dir, dst_name="laser_path.xml"
        )

    manifest = {
        "name": args.name,
        "created_utc": stamp,
        "seed": args.seed,
        "git": {
            "commit": git_output(["rev-parse", "HEAD"]),
            "branch": git_output(["branch", "--show-current"]),
            "status_short": git_output(["status", "--short"]),
        },
        "commands": {
            "train": args.train_command,
            "evaluate": args.eval_command,
        },
        "vtu_dir": str(vtu_dir),
        "laser_xml": args.laser_xml,
        "laser_xml_artifact": laser_xml_artifact,
        "artifacts": copied,
        "split_indices": {
            "path": str(split_path),
            "sha256": sha256_file(split_path),
            "source": split_source,
            "source_artifact": source_split,
            "summary": split_payload["summary"],
        },
        "notes": args.notes,
    }
    manifest_path = out_dir / "baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Frozen baseline: {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Git commit: {manifest['git']['commit']}")
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")


if __name__ == "__main__":
    main()
