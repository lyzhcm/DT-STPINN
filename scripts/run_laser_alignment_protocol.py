"""Run the fixed laser trajectory alignment evidence protocol.

This wrapper keeps the roadmap's trajectory check reproducible: export the
laser CSVs, diagnose the known worst node, draw hotspot overlays for selected
VTU steps, and write a small manifest with the exact commands and generated
files. It does not train a model.
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_XML = "configs\\laser_paths\\5_block_fem_additive_z_scan.xml"


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


def parse_steps(values: list[str] | None) -> list[int]:
    if not values:
        return []
    steps: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                steps.append(int(part))
    return steps


def unique_steps(steps: list[int]) -> list[int]:
    seen: set[int] = set()
    unique: list[int] = []
    for step in steps:
        if step in seen:
            continue
        seen.add(step)
        unique.append(step)
    return unique


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    return value if isinstance(value, dict) else None


def read_plot_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_export_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    cmd = [
        args.python,
        "scripts/export_laser_trajectory.py",
        "--config",
        args.config,
        "--laser_xml",
        args.laser_xml,
        "--vtu_dir",
        args.vtu_dir,
        "--output_dir",
        str(output_dir),
        "--diagnose_step_index",
        str(args.focus_step),
        "--diagnose_node_index",
        str(args.focus_node),
        "--hotspot_step_index",
        str(args.focus_step),
        "--hotspot_top_k",
        str(args.hotspot_top_k),
    ]
    if args.time_scale_to_s is not None:
        cmd.extend(["--time_scale_to_s", str(args.time_scale_to_s)])
    if args.time_offset_s is not None:
        cmd.extend(["--time_offset_s", str(args.time_offset_s)])
    if args.reverse_hatch_order_parity is not None:
        cmd.extend(["--reverse_hatch_order_parity", str(args.reverse_hatch_order_parity)])
    if args.threshold is not None:
        cmd.extend(["--hotspot_threshold", str(args.threshold)])
    return cmd


def build_plot_command(args: argparse.Namespace, output_dir: Path, steps: list[int]) -> list[str]:
    cmd = [
        args.python,
        "scripts/plot_laser_hotspots.py",
        "--config",
        args.config,
        "--laser_xml",
        args.laser_xml,
        "--vtu_dir",
        args.vtu_dir,
        "--steps",
        ",".join(str(step) for step in steps),
        "--output_dir",
        str(output_dir),
        "--top_k",
        str(args.plot_top_k),
        "--zoom_radius_mm",
        str(args.zoom_radius_mm),
    ]
    if args.show_all_nodes:
        cmd.append("--show_all_nodes")
    if args.threshold is not None:
        cmd.extend(["--threshold", str(args.threshold)])
    if args.time_scale_to_s is not None:
        cmd.extend(["--time_scale_to_s", str(args.time_scale_to_s)])
    if args.time_offset_s is not None:
        cmd.extend(["--time_offset_s", str(args.time_offset_s)])
    if args.reverse_hatch_order_parity is not None:
        cmd.extend(["--reverse_hatch_order_parity", str(args.reverse_hatch_order_parity)])
    return cmd


def command_record(phase: str, cmd: list[str], status: str) -> dict[str, str]:
    return {"phase": phase, "command": display_command(cmd), "status": status}


def write_manifest(
    args: argparse.Namespace,
    steps: list[int],
    commands: list[dict[str, str]],
    export_dir: Path,
    plot_dir: Path,
) -> Path:
    trajectory_manifest = export_dir / "laser_trajectory_manifest.json"
    alignment_report = export_dir / "laser_alignment_report.md"
    plot_summary = plot_dir / "laser_hotspot_alignment_summary.csv"
    trajectory_summary = read_json(trajectory_manifest)
    hotspot_overlay = trajectory_summary.get("hotspot_overlay", {}) if trajectory_summary else {}
    manifest = {
        "protocol": "laser_alignment",
        "protocol_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info(),
        "inputs": {
            "config": args.config,
            "laser_xml": args.laser_xml,
            "vtu_dir": args.vtu_dir,
        },
        "focus": {
            "step": args.focus_step,
            "node": args.focus_node,
        },
        "plot_steps": steps,
        "options": {
            "threshold": args.threshold,
            "hotspot_top_k": args.hotspot_top_k,
            "plot_top_k": args.plot_top_k,
            "zoom_radius_mm": args.zoom_radius_mm,
            "show_all_nodes": args.show_all_nodes,
            "time_scale_to_s": args.time_scale_to_s,
            "time_offset_s": args.time_offset_s,
            "reverse_hatch_order_parity": args.reverse_hatch_order_parity,
        },
        "commands": commands,
        "artifacts": {
            "trajectory_manifest": str(trajectory_manifest),
            "trajectory_manifest_exists": trajectory_manifest.exists(),
            "alignment_report": str(alignment_report),
            "alignment_report_exists": alignment_report.exists(),
            "segments_csv": str(export_dir / "laser_segments.csv"),
            "samples_csv": str(export_dir / "laser_samples.csv"),
            "hotspot_overlay_csv": hotspot_overlay.get("overlay_csv"),
            "plot_summary_csv": str(plot_summary),
            "plot_summary_exists": plot_summary.exists(),
        },
        "trajectory_summary": trajectory_summary,
        "plot_summary": read_plot_summary(plot_summary),
    }
    output_path = Path(args.output_root) / "laser_alignment_protocol_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/feature_e3_path_phase.yaml")
    parser.add_argument("--laser_xml", default=DEFAULT_XML)
    parser.add_argument("--vtu_dir", required=True)
    parser.add_argument("--output_root", default="results/laser_alignment_protocol")
    parser.add_argument("--focus_step", type=int, default=2128)
    parser.add_argument("--focus_node", type=int, default=24437)
    parser.add_argument("--plot_steps", nargs="*", default=None,
                        help="Additional comma- or space-separated VTU sample indices to plot.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Hotspot threshold passed to plot_laser_hotspots.py.")
    parser.add_argument("--hotspot_top_k", type=int, default=20)
    parser.add_argument("--plot_top_k", type=int, default=200)
    parser.add_argument("--zoom_radius_mm", type=float, default=2.0)
    parser.add_argument("--show_all_nodes", action="store_true")
    parser.add_argument("--time_scale_to_s", type=float, default=None)
    parser.add_argument("--time_offset_s", type=float, default=None)
    parser.add_argument("--reverse_hatch_order_parity", type=int, choices=[-1, 0, 1], default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument("--skip_plot", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.focus_step < 0:
        parser.error("--focus_step must be non-negative.")
    if args.focus_node < 0:
        parser.error("--focus_node must be non-negative.")
    if args.hotspot_top_k <= 0:
        parser.error("--hotspot_top_k must be greater than zero.")
    if args.plot_top_k < 0:
        parser.error("--plot_top_k must be non-negative.")
    if args.zoom_radius_mm <= 0:
        parser.error("--zoom_radius_mm must be greater than zero.")
    return args


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    export_dir = output_root / "trajectory"
    plot_dir = output_root / "plots"
    plot_steps = unique_steps([args.focus_step, *parse_steps(args.plot_steps)])
    commands: list[dict[str, str]] = []

    print("Laser alignment protocol")
    print(f"  Config     : {args.config}")
    print(f"  Laser XML  : {args.laser_xml}")
    print(f"  VTU dir    : {args.vtu_dir}")
    print(f"  Focus      : step {args.focus_step}, node {args.focus_node}")
    print(f"  Plot steps : {', '.join(str(step) for step in plot_steps)}")
    print(f"  Output root: {output_root}")

    if not args.skip_export:
        cmd = build_export_command(args, export_dir)
        run_command(cmd, dry_run=args.dry_run)
        commands.append(command_record("export", cmd, "dry_run" if args.dry_run else "ok"))
    else:
        commands.append({"phase": "export", "command": "", "status": "skipped"})

    if not args.skip_plot:
        cmd = build_plot_command(args, plot_dir, plot_steps)
        run_command(cmd, dry_run=args.dry_run)
        commands.append(command_record("plot", cmd, "dry_run" if args.dry_run else "ok"))
    else:
        commands.append({"phase": "plot", "command": "", "status": "skipped"})

    if args.dry_run:
        print("\nDry run complete; protocol manifest was not written.")
    else:
        manifest_path = write_manifest(args, plot_steps, commands, export_dir, plot_dir)
        print(f"\nProtocol manifest: {manifest_path}")
    print("Laser alignment protocol complete.")


if __name__ == "__main__":
    main()
