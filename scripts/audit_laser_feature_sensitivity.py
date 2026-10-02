"""Audit sensitivity of E1/E2 predictions to target laser lookahead features.

This is a no-training mechanism diagnostic.  For each validation window it keeps
history/base node features unchanged and replaces only the appended target-laser
feature block (x[:, base_dim:]) with the corresponding block from another
validation window.  The target temperature label is not changed.
"""
from __future__ import annotations

import argparse, csv, json, math, sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.training_diagnostics import EXPERIMENTS, load_graph, make_dataset, write_csv
from scripts.training_diagnostics_multwindow import configure, build_model, calc_loss, window_metrics, aggregate


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fnum(x: Any, default: float = math.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def inum(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def unique_keep_order(vals: list[int]) -> list[int]:
    seen: set[int] = set(); out: list[int] = []
    for v in vals:
        if v not in seen:
            seen.add(v); out.append(v)
    return out


def updates_for(run_dir: Path, exp: str, mode: str, requested: list[str]) -> list[int]:
    out: list[int] = []
    sel_rows = read_csv(run_dir / "fp_weight_audit_alpha0p1" / "selection_audit.csv")
    sel = next((r for r in sel_rows if r.get("experiment") == exp and r.get("mode") == mode), None)
    for item in requested:
        if item == "final":
            out.append(2000)
        elif item == "best_f1":
            if sel:
                out.append(inum(sel.get("best_f1_update")))
        elif item == "best_rmse":
            if sel:
                out.append(inum(sel.get("best_rmse_update")))
        else:
            out.append(int(item))
    return [u for u in unique_keep_order(out) if u >= 0]


def load_model_for(cfg, ckpt_path: Path, device: torch.device):
    model, _opt = build_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def replace_target_laser_features(sample: dict[str, Any], alt: dict[str, Any], base_dim: int) -> dict[str, Any]:
    # sample was freshly built by the dataset, so in-place x replacement is safe.
    for data, alt_data in zip(sample["graph_sequence"], alt["graph_sequence"]):
        if data.x.shape[1] <= base_dim:
            raise ValueError(f"sample has no target-laser feature block: x dim={data.x.shape[1]}, base_dim={base_dim}")
        if data.x.shape[1] != alt_data.x.shape[1]:
            raise ValueError(f"feature dim mismatch: {data.x.shape[1]} vs {alt_data.x.shape[1]}")
        data.x = torch.cat([data.x[:, :base_dim], alt_data.x[:, base_dim:]], dim=1)
        if hasattr(alt_data, "target_laser_pos"):
            data.target_laser_pos = alt_data.target_laser_pos
    return sample


def bool_jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    if union == 0:
        return 1.0
    return float(inter / union)


def dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a.detach().float() - b.detach().float()).cpu())


def peak_and_region_metrics(base_pred: torch.Tensor, shift_pred: torch.Tensor, sample: dict[str, Any], alt: dict[str, Any], cfg) -> dict[str, Any]:
    solidus = float(cfg.material.solidus_temp)
    valid = sample["target_mask"].detach().bool().flatten()
    coords = sample["coords"].detach().float()
    bp = base_pred.detach().float().reshape_as(sample["target"]).flatten()
    sp = shift_pred.detach().float().reshape_as(sample["target"]).flatten()
    bv = bp[valid]; sv = sp[valid]
    delta = sv - bv
    vidx = torch.nonzero(valid, as_tuple=False).flatten()
    b_peak = vidx[int(torch.argmax(bv).cpu())]
    s_peak = vidx[int(torch.argmax(sv).cpu())]
    orig_laser = sample["target_laser_pos"].detach().float().view(-1)
    alt_laser = alt["target_laser_pos"].detach().float().view(-1)
    # Region around original / substituted target laser.  Same scale as existing laser features.
    radius = float(getattr(cfg.data, "laser_feature_radius_mm", 0.4))
    depth = float(getattr(cfg.data, "laser_feature_depth_mm", 0.1))
    d_orig = coords - orig_laser.view(1, 3).to(coords.device)
    d_alt = coords - alt_laser.view(1, 3).to(coords.device)
    near_orig = valid & (torch.linalg.vector_norm(d_orig[:, :2], dim=1) <= radius) & (d_orig[:, 2].abs() <= depth)
    near_alt = valid & (torch.linalg.vector_norm(d_alt[:, :2], dim=1) <= radius) & (d_alt[:, 2].abs() <= depth)
    true_hot = valid & (sample["target"].detach().float().flatten() >= solidus)
    base_hot = valid & (bp >= solidus)
    shift_hot = valid & (sp >= solidus)
    def mean_or_nan(x: torch.Tensor) -> float:
        return float(x.mean().cpu()) if x.numel() else math.nan
    return {
        "pred_delta_mean": float(delta.mean().cpu()),
        "pred_abs_delta_mean": float(delta.abs().mean().cpu()),
        "pred_delta_rmse": float(torch.sqrt(delta.pow(2).mean()).cpu()),
        "pred_abs_delta_max": float(delta.abs().max().cpu()),
        "pred_hot_jaccard": bool_jaccard(base_hot, shift_hot),
        "base_pred_hot_nodes": int((base_hot & valid).sum().cpu()),
        "shift_pred_hot_nodes": int((shift_hot & valid).sum().cpu()),
        "pred_hot_symmetric_diff": int((base_hot ^ shift_hot).sum().cpu()),
        "base_peak_node": int(b_peak.cpu()),
        "shift_peak_node": int(s_peak.cpu()),
        "peak_displacement": dist(coords[b_peak], coords[s_peak]),
        "target_laser_displacement": dist(orig_laser, alt_laser),
        "base_peak_to_orig_laser": dist(coords[b_peak], orig_laser.to(coords.device)),
        "shift_peak_to_alt_laser": dist(coords[s_peak], alt_laser.to(coords.device)),
        "base_peak_to_alt_laser": dist(coords[b_peak], alt_laser.to(coords.device)),
        "shift_peak_to_orig_laser": dist(coords[s_peak], orig_laser.to(coords.device)),
        "true_hot_pred_mean_base": mean_or_nan(bp[true_hot]),
        "true_hot_pred_mean_shift": mean_or_nan(sp[true_hot]),
        "true_hot_pred_delta_mean": mean_or_nan((sp - bp)[true_hot]),
        "near_orig_nodes": int(near_orig.sum().cpu()),
        "near_alt_nodes": int(near_alt.sum().cpu()),
        "near_orig_pred_mean_base": mean_or_nan(bp[near_orig]),
        "near_orig_pred_mean_shift": mean_or_nan(sp[near_orig]),
        "near_alt_pred_mean_base": mean_or_nan(bp[near_alt]),
        "near_alt_pred_mean_shift": mean_or_nan(sp[near_alt]),
    }


def feature_delta(sample: dict[str, Any], alt: dict[str, Any], base_dim: int) -> dict[str, float]:
    vals=[]
    for data, alt_data in zip(sample["graph_sequence"], alt["graph_sequence"]):
        d=(alt_data.x[:, base_dim:].detach().float()-data.x[:, base_dim:].detach().float()).flatten()
        vals.append(d)
    x=torch.cat(vals)
    return {
        "target_feature_delta_mean": float(x.mean().cpu()),
        "target_feature_abs_delta_mean": float(x.abs().mean().cpu()),
        "target_feature_delta_rmse": float(torch.sqrt(x.pow(2).mean()).cpu()),
        "target_feature_abs_delta_max": float(x.abs().max().cpu()),
    }


def run_checkpoint(exp: str, mode: str, update: int, run_dir: Path, graph, split: dict[str, Any], args, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = configure(exp, mode, args)
    starts = [int(x) for x in split[args.split + "_starts"]]
    ds = make_dataset(graph, cfg, starts)
    cfg.model.node_feature_dim = ds.input_feature_dim
    base_dim = int(graph.node_feature_builder.feature_dim)
    target_dim = int(ds.target_laser_feature_dim)
    if target_dim <= 0:
        raise ValueError(f"{exp} has no target laser features; sensitivity audit is for E1/E2+")
    ckpt_path = run_dir / f"{exp}_{mode}" / f"checkpoint_update_{update:04d}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    model, ckpt = load_model_for(cfg, ckpt_path, device)
    per_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for offset in args.alt_offsets:
        base_metric_rows=[]; shifted_metric_rows=[]; diff_acc=[]
        with torch.no_grad():
            for i, start in enumerate(starts):
                j=(i+int(offset)) % len(ds)
                sample=ds[i]
                alt=ds[j]
                _loss_b, comps_b, pred_b = calc_loss(model, sample, cfg, mode, None, args.alpha)
                base_row=window_metrics(pred_b, sample, cfg, int(start), comps_b)
                fd=feature_delta(sample, alt, base_dim)
                shifted=replace_target_laser_features(sample, alt, base_dim)
                _loss_s, comps_s, pred_s = calc_loss(model, shifted, cfg, mode, None, args.alpha)
                shift_row=window_metrics(pred_s, shifted, cfg, int(start), comps_s)
                dm=peak_and_region_metrics(pred_b, pred_s, shifted, alt, cfg)
                row={
                    "experiment": exp, "mode": mode, "update": update, "split": args.split,
                    "start": int(start), "target_step": int(shifted.get("target_step", int(start)+cfg.data.window_size+cfg.data.predict_steps-1)),
                    "alt_offset": int(offset), "alt_start": int(starts[j]), "alt_target_step": int(alt.get("target_step", starts[j]+cfg.data.window_size+cfg.data.predict_steps-1)),
                    "input_feature_dim": int(ds.input_feature_dim), "base_feature_dim": base_dim, "target_laser_feature_dim": target_dim,
                    **fd,
                    "base_rmse": base_row["rmse"], "shift_rmse": shift_row["rmse"], "delta_rmse": shift_row["rmse"]-base_row["rmse"],
                    "base_mae": base_row["mae"], "shift_mae": shift_row["mae"], "base_hot_mae": base_row["hot_mae"], "shift_hot_mae": shift_row["hot_mae"],
                    "base_bg_mae": base_row["background_mae"], "shift_bg_mae": shift_row["background_mae"],
                    "base_tp": base_row["solidus_tp"], "base_fp": base_row["solidus_fp"], "base_fn": base_row["solidus_fn"],
                    "shift_tp": shift_row["solidus_tp"], "shift_fp": shift_row["solidus_fp"], "shift_fn": shift_row["solidus_fn"],
                    "base_pred_max": base_row["pred_max"], "shift_pred_max": shift_row["pred_max"], "delta_pred_max": shift_row["pred_max"]-base_row["pred_max"],
                    "target_max": base_row["target_max"], "true_hot_nodes": base_row["true_hot_nodes"],
                    **dm,
                }
                per_rows.append(row)
                # Prefix metric rows for aggregate.
                base_metric_rows.append(base_row); shifted_metric_rows.append(shift_row); diff_acc.append(dm)
        b_agg=aggregate(base_metric_rows); s_agg=aggregate(shifted_metric_rows)
        def mean_metric(name: str) -> float:
            vals=[fnum(d.get(name)) for d in diff_acc]
            vals=[v for v in vals if not math.isnan(v)]
            return float(np.mean(vals)) if vals else math.nan
        summary_rows.append({
            "experiment": exp, "mode": mode, "update": update, "split": args.split, "alt_offset": int(offset),
            "checkpoint": str(ckpt_path), "checkpoint_recorded_update": ckpt.get("update", ""),
            "input_feature_dim": int(ds.input_feature_dim), "target_laser_feature_dim": target_dim,
            "windows": len(starts),
            "base_rmse": b_agg["rmse"], "shift_rmse": s_agg["rmse"], "delta_rmse": s_agg["rmse"]-b_agg["rmse"],
            "base_hot_mae": b_agg["hot_mae"], "shift_hot_mae": s_agg["hot_mae"], "delta_hot_mae": s_agg["hot_mae"]-b_agg["hot_mae"],
            "base_bg_mae": b_agg["background_mae"], "shift_bg_mae": s_agg["background_mae"], "delta_bg_mae": s_agg["background_mae"]-b_agg["background_mae"],
            "base_precision": b_agg["solidus_precision"], "base_recall": b_agg["solidus_recall"], "base_f1": b_agg["solidus_f1"],
            "shift_precision": s_agg["solidus_precision"], "shift_recall": s_agg["solidus_recall"], "shift_f1": s_agg["solidus_f1"],
            "base_tp": b_agg["solidus_tp"], "base_fp": b_agg["solidus_fp"], "base_fn": b_agg["solidus_fn"],
            "shift_tp": s_agg["solidus_tp"], "shift_fp": s_agg["solidus_fp"], "shift_fn": s_agg["solidus_fn"],
            "mean_pred_abs_delta": mean_metric("pred_abs_delta_mean"),
            "mean_pred_delta_rmse": mean_metric("pred_delta_rmse"),
            "mean_pred_abs_delta_max": mean_metric("pred_abs_delta_max"),
            "mean_hot_jaccard": mean_metric("pred_hot_jaccard"),
            "mean_peak_displacement": mean_metric("peak_displacement"),
            "mean_target_laser_displacement": mean_metric("target_laser_displacement"),
            "windows_pred_hot_changed": sum(1 for d in diff_acc if inum(d.get("pred_hot_symmetric_diff")) > 0),
            "mean_true_hot_pred_delta": mean_metric("true_hot_pred_delta_mean"),
            "mean_near_orig_pred_delta": mean_metric("near_orig_pred_mean_shift") - mean_metric("near_orig_pred_mean_base"),
            "mean_near_alt_pred_delta": mean_metric("near_alt_pred_mean_shift") - mean_metric("near_alt_pred_mean_base"),
        })
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary_rows, per_rows


def fmt(x: Any, n: int = 3) -> str:
    try:
        y=float(x)
        return "nan" if math.isnan(y) else f"{y:.{n}f}"
    except Exception:
        return str(x)


def write_report(outdir: Path, summaries: list[dict[str, Any]], args) -> None:
    lines=[
        "# E1/E2 target-laser feature sensitivity audit\n\n",
        "No training is performed. For each validation window, the historical/base feature block is kept unchanged and only the appended target-laser feature block is replaced by another validation window's target-laser feature block. The target temperature remains the original one.\n\n",
        f"Run dir: `{args.run_dir}`; split=`{args.split}`; alt_offsets={args.alt_offsets}; alpha={args.alpha}.\n\n",
        "## Aggregate sensitivity\n\n",
        "| Exp | Mode | Update | Alt offset | feature dim | dRMSE | dHotMAE | dBgMAE | base F1 P/R TP/FP/FN | shifted F1 P/R TP/FP/FN | mean abs dPred | mean dPred RMSE | hot Jaccard | peak disp. | target laser disp. | windows hot-set changed | dTrueHotPred | dNearOrigPred | dNearAltPred |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for r in summaries:
        lines.append(
            f"| {r['experiment']} | {r['mode']} | {r['update']} | {r['alt_offset']} | {r['input_feature_dim']} | "
            f"{fmt(r['delta_rmse'])} | {fmt(r['delta_hot_mae'])} | {fmt(r['delta_bg_mae'])} | "
            f"{fmt(r['base_f1'])} {fmt(r['base_precision'])}/{fmt(r['base_recall'])} {r['base_tp']}/{r['base_fp']}/{r['base_fn']} | "
            f"{fmt(r['shift_f1'])} {fmt(r['shift_precision'])}/{fmt(r['shift_recall'])} {r['shift_tp']}/{r['shift_fp']}/{r['shift_fn']} | "
            f"{fmt(r['mean_pred_abs_delta'])} | {fmt(r['mean_pred_delta_rmse'])} | {fmt(r['mean_hot_jaccard'])} | {fmt(r['mean_peak_displacement'])} | {fmt(r['mean_target_laser_displacement'])} | "
            f"{r['windows_pred_hot_changed']} | {fmt(r['mean_true_hot_pred_delta'])} | {fmt(r['mean_near_orig_pred_delta'])} | {fmt(r['mean_near_alt_pred_delta'])} |\n"
        )
    lines.append("\nInterpretation: small prediction deltas and unchanged hot sets suggest weak use of target trajectory features; large deltas/peak shifts show sensitivity, but not necessarily correct use.\n")
    (outdir/"laser_feature_sensitivity_report.md").write_text("".join(lines), encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run_dir", type=Path, default=Path("results/training_diagnostics/multwindow_3x3_seed42"))
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--vtu_dir", default="F:/VTU")
    p.add_argument("--cache_dir", default="data/processed")
    p.add_argument("--laser_xml", default="configs/laser_paths/5_block_fem_additive_z_scan.xml")
    p.add_argument("--experiments", nargs="+", default=["E1","E2"], choices=["E1","E2"])
    p.add_argument("--modes", nargs="+", default=["mse","c_hot"], choices=["mse","c_hot","c_hot_pde"])
    p.add_argument("--updates", nargs="+", default=["best_f1","final"], help="Integers or aliases: best_f1, best_rmse, final")
    p.add_argument("--alt_offsets", nargs="+", type=int, default=[7], help="Cyclic offset in the diagnostic split used as substituted target trajectory features.")
    p.add_argument("--split", default="val", choices=["train","val"])
    p.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    p.add_argument("--graph_device", default="auto", choices=["auto","cpu","cuda"])
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--lambda_pde", type=float, default=None)
    p.add_argument("--lr_start", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_cache", action="store_true")
    args=p.parse_args()
    args.output_dir = args.output_dir or (args.run_dir / "laser_feature_sensitivity")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else (args.device if args.device!="auto" else "cpu"))
    graph_device=device if args.graph_device=="auto" else torch.device(args.graph_device)
    ref=configure("E1", "mse", args)
    graph=load_graph(args.vtu_dir,args.cache_dir,ref,graph_device,args.no_cache)
    split=json.loads((args.run_dir/"small_split.json").read_text(encoding="utf-8"))
    all_sum=[]; all_per=[]
    for exp in args.experiments:
        for mode in args.modes:
            ups=updates_for(args.run_dir, exp, mode, args.updates)
            # MSE groups are not in selection_audit; aliases collapse to final only.
            if mode == "mse":
                ups=[2000 if u in (0,) else u for u in ups]
                ups=unique_keep_order([u for u in ups if (args.run_dir/f"{exp}_{mode}"/f"checkpoint_update_{u:04d}.pt").exists()]) or [2000]
            for upd in ups:
                ckpt=args.run_dir/f"{exp}_{mode}"/f"checkpoint_update_{upd:04d}.pt"
                if not ckpt.exists():
                    print(f"skip missing {ckpt}", flush=True)
                    continue
                print(f"audit {exp}/{mode} update {upd}", flush=True)
                s,pers=run_checkpoint(exp,mode,upd,args.run_dir,graph,split,args,device)
                all_sum.extend(s); all_per.extend(pers)
    write_csv(args.output_dir/"laser_feature_sensitivity_summary.csv", all_sum)
    write_csv(args.output_dir/"laser_feature_sensitivity_per_window.csv", all_per)
    manifest={**vars(args), "device": str(device), "graph_device": str(graph_device), "note": "no training; replace x[:, base_dim:] with another window's target-laser feature block"}
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str)+"\n", encoding="utf-8")
    write_report(args.output_dir, all_sum, args)
    print(args.output_dir/"laser_feature_sensitivity_report.md", flush=True)

if __name__ == "__main__":
    main()
