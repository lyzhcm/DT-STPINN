"""Verify that a frozen baseline artifact is complete and internally consistent.

This is a read-only guard for the fixed experiment roadmap. It checks that the
manifest, checkpoint, config, evaluation report, split file, optional source
split, and optional laser XML artifact exist and still match their recorded
SHA256 hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACT_KEYS = ("config", "checkpoint", "test_report")
ACCEPTANCE_ARTIFACT_KEYS = (
    "acceptance_summary.json",
    "acceptance_summary.csv",
    "worst_cases_top10.csv",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_check(checks: list[tuple[bool, str]], ok: bool, message: str) -> None:
    checks.append((ok, message))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def resolve_recorded_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = root / path
    if candidate.exists():
        return candidate
    return path


def check_hash_record(
    checks: list[tuple[bool, str]],
    root: Path,
    label: str,
    record: dict[str, Any] | None,
    *,
    required: bool,
) -> Path | None:
    if record is None:
        add_check(checks, not required, f"{label}: {'missing' if required else 'not recorded'}")
        return None
    if not isinstance(record, dict):
        add_check(checks, False, f"{label}: manifest record is not an object")
        return None

    frozen_value = record.get("frozen_path") or record.get("path")
    path = resolve_recorded_path(root, str(frozen_value) if frozen_value else None)
    if path is None:
        add_check(checks, False, f"{label}: no frozen_path/path in manifest")
        return None

    if not path.is_file():
        fallback = root / path.name
        if fallback.is_file():
            path = fallback
        else:
            add_check(checks, False, f"{label}: missing file {path}")
            return None

    recorded_hash = record.get("sha256")
    if recorded_hash:
        actual_hash = sha256_file(path)
        add_check(
            checks,
            actual_hash == recorded_hash,
            f"{label}: sha256 {'OK' if actual_hash == recorded_hash else 'mismatch'} ({path})",
        )
    else:
        add_check(checks, False, f"{label}: missing sha256 in manifest")

    recorded_bytes = record.get("bytes")
    if recorded_bytes is not None:
        actual_bytes = path.stat().st_size
        add_check(
            checks,
            int(recorded_bytes) == actual_bytes,
            f"{label}: bytes {'OK' if int(recorded_bytes) == actual_bytes else 'mismatch'}",
        )
    return path


def split_count(values: Any) -> int:
    return len(values) if isinstance(values, list) else -1


def summarize_report(report_path: Path | None) -> dict[str, Any]:
    if report_path is None or not report_path.is_file():
        return {}
    try:
        report = load_json(report_path)
    except Exception:
        return {}

    if isinstance(report.get("canonical_metrics"), dict):
        candidate = report["canonical_metrics"]
    elif isinstance(report.get("test_metrics"), dict):
        candidate = report["test_metrics"]
    elif isinstance(report.get("global_metrics"), dict):
        candidate = report["global_metrics"]
    else:
        candidate = report
    keys = [
        "RMSE",
        "MAE",
        "AbsErrorP95",
        "AbsErrorP99",
        "MaxError",
        "TempRecallAboveSolidus",
        "TempF1AboveSolidus",
    ]
    return {key: candidate[key] for key in keys if key in candidate}


def command_contains_value(command: str, value: str) -> bool:
    if not value:
        return True
    candidates = {value, value.replace("/", "\\"), value.replace("\\", "/")}
    return any(candidate and candidate in command for candidate in candidates)


def require_command_option(
    checks: list[tuple[bool, str]],
    command: str,
    command_label: str,
    option: str,
    value: str = "",
) -> None:
    has_option = option in command
    has_value = command_contains_value(command, value)
    detail = option if not value else f"{option} {value}"
    add_check(checks, has_option and has_value, f"{command_label}: contains {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", help="Frozen baseline artifact directory")
    parser.add_argument("--require_laser_xml", action="store_true")
    parser.add_argument("--require_acceptance_artifacts", action="store_true")
    parser.add_argument("--require_commands", action="store_true")
    parser.add_argument(
        "--require_protocol_args",
        action="store_true",
        help="Require recorded train/evaluate commands to contain fixed protocol arguments.",
    )
    parser.add_argument("--require_clean_git", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = Path(args.artifact_dir)
    checks: list[tuple[bool, str]] = []

    add_check(checks, root.is_dir(), f"artifact directory: {root}")
    manifest_path = root / "baseline_manifest.json"
    add_check(checks, manifest_path.is_file(), f"manifest exists: {manifest_path}")
    if not root.is_dir() or not manifest_path.is_file():
        for ok, message in checks:
            print(f"[{'OK' if ok else 'FAIL'}] {message}")
        raise SystemExit(1)

    try:
        manifest = load_json(manifest_path)
        add_check(checks, True, "manifest JSON: readable")
    except Exception as exc:  # noqa: BLE001 - verification should report all failures.
        add_check(checks, False, f"manifest JSON: {exc}")
        manifest = {}

    artifacts = manifest.get("artifacts", {}) if isinstance(manifest.get("artifacts"), dict) else {}
    paths: dict[str, Path | None] = {}
    for key in REQUIRED_ARTIFACT_KEYS:
        paths[key] = check_hash_record(
            checks,
            root,
            key,
            artifacts.get(key) if isinstance(artifacts, dict) else None,
            required=True,
        )

    acceptance_artifacts = manifest.get("acceptance_artifacts")
    if not isinstance(acceptance_artifacts, dict):
        acceptance_artifacts = {}
    for key in ACCEPTANCE_ARTIFACT_KEYS:
        check_hash_record(
            checks,
            root,
            key,
            acceptance_artifacts.get(key),
            required=args.require_acceptance_artifacts,
        )

    split_record = manifest.get("split_indices") if isinstance(manifest.get("split_indices"), dict) else None
    split_path = check_hash_record(checks, root, "split_indices", split_record, required=True)
    split_payload: dict[str, Any] = {}
    if split_path and split_path.is_file():
        try:
            split_payload = load_json(split_path)
            add_check(checks, True, "split_indices JSON: readable")
        except Exception as exc:  # noqa: BLE001
            add_check(checks, False, f"split_indices JSON: {exc}")

    if split_payload:
        counts = {
            "train": split_count(split_payload.get("train")),
            "val": split_count(split_payload.get("val")),
            "test": split_count(split_payload.get("test")),
        }
        add_check(
            checks,
            all(value > 0 for value in counts.values()),
            f"split counts: train={counts['train']}, val={counts['val']}, test={counts['test']}",
        )
        num_steps = split_payload.get("num_steps")
        if isinstance(num_steps, int):
            add_check(
                checks,
                sum(counts.values()) == num_steps,
                f"split total: {sum(counts.values())}/{num_steps}",
            )

    source_split = None
    if split_record and isinstance(split_record.get("source_artifact"), dict):
        source_split = split_record.get("source_artifact")
    if source_split:
        check_hash_record(checks, root, "source_split_indices", source_split, required=False)

    laser_record = manifest.get("laser_xml_artifact")
    check_hash_record(
        checks,
        root,
        "laser_xml_artifact",
        laser_record if isinstance(laser_record, dict) else None,
        required=args.require_laser_xml,
    )

    commands = manifest.get("commands") if isinstance(manifest.get("commands"), dict) else {}
    train_cmd = str(commands.get("train", "")) if commands else ""
    eval_cmd = str(commands.get("evaluate", "")) if commands else ""
    if args.require_commands:
        add_check(checks, bool(train_cmd), "train command recorded")
        add_check(checks, bool(eval_cmd), "evaluate command recorded")
    if args.require_protocol_args:
        config_record = artifacts.get("config") if isinstance(artifacts, dict) else {}
        checkpoint_record = artifacts.get("checkpoint") if isinstance(artifacts, dict) else {}
        config_source = str(config_record.get("source", "")) if isinstance(config_record, dict) else ""
        checkpoint_source = str(checkpoint_record.get("source", "")) if isinstance(checkpoint_record, dict) else ""
        vtu_dir = str(manifest.get("vtu_dir", ""))
        seed = manifest.get("seed")
        epochs = manifest.get("epochs")
        experiment_name = str(manifest.get("experiment_name", ""))
        split_source = str(split_record.get("source", "")) if split_record else ""
        split_source_path = split_source.removeprefix("frozen:") if split_source.startswith("frozen:") else ""
        laser_xml = str(manifest.get("laser_xml", ""))

        require_command_option(checks, train_cmd, "train command", "--config", config_source)
        require_command_option(checks, eval_cmd, "evaluate command", "--config", config_source)
        require_command_option(checks, train_cmd, "train command", "--vtu_dir", vtu_dir)
        require_command_option(checks, eval_cmd, "evaluate command", "--vtu_dir", vtu_dir)
        require_command_option(checks, train_cmd, "train command", "--seed", str(seed) if seed is not None else "")
        if epochs is not None:
            require_command_option(checks, train_cmd, "train command", "--epochs", str(epochs))
        if experiment_name:
            require_command_option(checks, train_cmd, "train command", "--experiment_name", experiment_name)
        require_command_option(checks, eval_cmd, "evaluate command", "--checkpoint", checkpoint_source)
        if split_source_path:
            require_command_option(checks, train_cmd, "train command", "--split_indices", split_source_path)
            require_command_option(checks, eval_cmd, "evaluate command", "--split_indices", split_source_path)
        if laser_xml:
            require_command_option(checks, train_cmd, "train command", "--laser_xml", laser_xml)
            require_command_option(checks, eval_cmd, "evaluate command", "--laser_xml", laser_xml)

    git = manifest.get("git") if isinstance(manifest.get("git"), dict) else {}
    commit = str(git.get("commit", "")) if git else ""
    add_check(checks, bool(commit), f"git commit recorded: {commit or '<missing>'}")
    if args.require_clean_git:
        status_short = str(git.get("status_short", "")) if git else ""
        add_check(checks, status_short == "", f"git status clean in manifest: {status_short!r}")

    if not args.quiet:
        name = manifest.get("name", root.name)
        print(f"Baseline artifact: {name}")
        print(f"Directory: {root}")
        if commit:
            print(f"Commit: {commit}")
        report_metrics = summarize_report(paths.get("test_report"))
        if report_metrics:
            print("Metrics:")
            for key, value in report_metrics.items():
                print(f"  {key}: {value}")
        print("Checks:")
        for ok, message in checks:
            print(f"[{'OK' if ok else 'FAIL'}] {message}")

    if any(not ok for ok, _ in checks):
        raise SystemExit(1)

    if args.quiet:
        print(f"OK: {root}")


if __name__ == "__main__":
    main()
