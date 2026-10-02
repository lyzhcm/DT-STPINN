"""Stage 4C diagnostic branch: full-graph MSE + alpha hot MSE.

Continues from update-500 checkpoints produced by training_diagnostics_stage4.py.
Optionally replays B-balanced branch to add near/far and unsampled-background FP statistics.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.training_diagnostics import (  # noqa: E402
    EXPERIMENTS,
    Config,
    DTSTPINN,
    disable_dropout,
    forward_loss,
    grad_norm,
    load_graph,
    make_dataset,
    metrics_from_pred,
    param_delta,
    set_seed,
    write_csv,
)
from scripts.training_diagnostics_stage4 import (  # noqa: E402
    balanced_loss,
    build_groups,
    extra_position_metrics,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def groups_from_json(data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "hot_idx": torch.tensor(data["hot_indices"], dtype=torch.long, device=device),
        "neighbor_idx": torch.tensor(data.get("neighbor_indices", []), dtype=torch.long, device=device),
        "background_idx": torch.tensor(data["background_indices"], dtype=torch.long, device=device),
        "bucket_counts": data["bucket_counts"],
        "background_counts": data["background_counts"],
    }


def clone_params_cpu(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().float().cpu().clone() for p in model.parameters()]


def build_model_optimizer(cfg: Config, device: torch.device) -> tuple[DTSTPINN, torch.optim.Optimizer]:
    model = DTSTPINN(cfg, cfg.material).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    return model, opt


def full_hot_loss(model: DTSTPINN, sample: dict[str, Any], groups: dict[str, Any], alpha: float) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    graph_seq = sample["graph_sequence"]
    dt = sample.get("dt", 1.0)
    if isinstance(dt, torch.Tensor):
        dt = float(dt.detach().cpu().item())
    out = model(graph_seq, dt=dt)
    pred = out["T_pred"].float()
    target = sample["target"].float()
    if pred.numel() == target.numel():
        pred = pred.reshape_as(target)
    mask = sample["target_mask"].bool()
    full_mse = ((pred[mask] - target[mask]) ** 2).mean()
    p_flat = pred.flatten()
    t_flat = target.flatten()
    hot_idx = groups["hot_idx"]
    hot_mse = ((p_flat[hot_idx] - t_flat[hot_idx]) ** 2).mean()
    loss = full_mse + alpha * hot_mse
    return loss, {
        "loss_c_total": float(loss.detach().cpu()),
        "loss_full_mse": float(full_mse.detach().cpu()),
        "loss_hot_mse": float(hot_mse.detach().cpu()),
        "alpha_hot": alpha,
    }, pred


def fp_breakdown(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, coords: torch.Tensor,
                 groups: dict[str, Any], solidus: float, near_dist_mm: float) -> dict[str, Any]:
    p = pred.detach().float()
    t = target.detach().float()
    if p.numel() == t.numel():
        p = p.reshape_as(t)
    p = p.flatten(); t = t.flatten()
    valid = mask.detach().bool().flatten()
    n = valid.numel()
    hot_idx = groups["hot_idx"]
    neighbor_idx = groups["neighbor_idx"]
    bg_idx = groups["background_idx"]
    hot_mask = torch.zeros(n, dtype=torch.bool, device=valid.device); hot_mask[hot_idx] = True
    neigh_mask = torch.zeros(n, dtype=torch.bool, device=valid.device); neigh_mask[neighbor_idx] = True
    bg_sample_mask = torch.zeros(n, dtype=torch.bool, device=valid.device); bg_sample_mask[bg_idx] = True
    supervised_mask = hot_mask | neigh_mask | bg_sample_mask
    unsampled_background_mask = valid & (~supervised_mask)
    fp_mask = valid & (p >= solidus) & (t < solidus)
    # Distance from every FP to nearest true-hot coordinate.
    fp_idx = torch.nonzero(fp_mask, as_tuple=False).flatten()
    near_dist_count = 0
    far_dist_count = 0
    if fp_idx.numel() > 0 and hot_idx.numel() > 0:
        d = torch.cdist(coords[fp_idx].float(), coords[hot_idx].float())
        min_d = d.min(dim=1).values
        near_dist_count = int((min_d <= near_dist_mm).sum().detach().cpu())
        far_dist_count = int((min_d > near_dist_mm).sum().detach().cpu())
    return {
        "fp_total": int(fp_mask.sum().detach().cpu()),
        "fp_near_onehop": int((fp_mask & neigh_mask).sum().detach().cpu()),
        "fp_far_not_onehop": int((fp_mask & (~neigh_mask)).sum().detach().cpu()),
        "fp_near_dist_le_threshold": near_dist_count,
        "fp_far_dist_gt_threshold": far_dist_count,
        "fp_near_dist_threshold_mm": near_dist_mm,
        "fp_in_sampled_background": int((fp_mask & bg_sample_mask).sum().detach().cpu()),
        "fp_in_unsampled_background": int((fp_mask & unsampled_background_mask).sum().detach().cpu()),
        "fp_in_neighbor_group": int((fp_mask & neigh_mask).sum().detach().cpu()),
        "fp_in_hot_group_should_be_zero": int((fp_mask & hot_mask).sum().detach().cpu()),
    }


def eval_row(model: DTSTPINN, sample: dict[str, Any], cfg: Config, groups: dict[str, Any],
             update: int, branch: str, loss_mode: str, loss_value: float | None,
             comps: dict[str, float] | None, last_grad_norm: float | None,
             init_params: list[torch.Tensor], near_dist_mm: float) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        _mse_loss, _out, pred = forward_loss(model, sample, False, None)
    model.train()
    row: dict[str, Any] = {
        "update": update,
        "branch": branch,
        "loss_mode": loss_mode,
        "train_loss_current_mode": loss_value,
        "grad_norm_pre_clip": last_grad_norm,
        "hot_group_n": int(groups["hot_idx"].numel()),
        "neighbor_group_n": int(groups["neighbor_idx"].numel()),
        "background_group_n": int(groups["background_idx"].numel()),
    }
    if comps:
        row.update(comps)
    row.update(metrics_from_pred(pred, sample["target"], sample["target_mask"], cfg.material.solidus_temp, cfg.material.liquidus_temp, None, prefix=""))
    row.update(extra_position_metrics(pred, sample["target"], sample["target_mask"], sample["coords"], groups, cfg.material.solidus_temp))
    row.update(fp_breakdown(pred, sample["target"], sample["target_mask"], sample["coords"], groups, cfg.material.solidus_temp, near_dist_mm))
    dl2, dmax = param_delta(model, init_params)
    row["param_delta_l2_from_init"] = dl2
    row["param_delta_max_from_init"] = dmax
    return row


def train_branch(model: DTSTPINN, opt: torch.optim.Optimizer, sample: dict[str, Any], cfg: Config,
                 groups: dict[str, Any], init_params: list[torch.Tensor], branch: str,
                 loss_mode: str, start_update: int, end_update: int, log_every: int,
                 alpha: float, near_dist_mm: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for update in range(start_update + 1, end_update + 1):
        opt.zero_grad(set_to_none=True)
        if loss_mode == "c_full_plus_hot":
            loss, comps, _pred = full_hot_loss(model, sample, groups, alpha)
        elif loss_mode == "balanced":
            graph_seq = sample["graph_sequence"]
            dt = sample.get("dt", 1.0)
            if isinstance(dt, torch.Tensor):
                dt = float(dt.detach().cpu().item())
            out = model(graph_seq, dt=dt)
            loss, comps = balanced_loss(out["T_pred"], sample["target"], groups)
        else:
            raise ValueError(loss_mode)
        loss.backward()
        gnorm = grad_norm(model)
        if cfg.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        opt.step()
        if update % log_every == 0 or update == end_update:
            row = eval_row(model, sample, cfg, groups, update, branch, loss_mode,
                           float(loss.detach().cpu()), comps, gnorm, init_params, near_dist_mm)
            row["lr"] = opt.param_groups[0]["lr"]
            rows.append(row)
            print(
                f"{branch} update {update}/{end_update} rmse={row['rmse']:.3f} "
                f"hot_mae={row['true_hot_mae_exact']:.3f} recall={row['solidus_recall']:.3f} "
                f"precision={row['solidus_precision']:.3f} fp={row['fp_total']} "
                f"unsampled_fp={row['fp_in_unsampled_background']} pred_max={row['pred_max']:.1f}",
                flush=True,
            )
    return rows


def setup_exp(exp: str, graph: Any, args: argparse.Namespace, device: torch.device) -> tuple[Config, dict[str, Any], DTSTPINN, torch.optim.Optimizer, list[torch.Tensor]]:
    cfg = Config.from_yaml(EXPERIMENTS[exp])
    if args.laser_xml:
        cfg.data.laser_xml_path = args.laser_xml
        cfg.data.laser_path_mode = "additive_z_scan"
    disable_dropout(cfg)
    cfg.training.use_amp = False
    ds = make_dataset(graph, cfg, [args.window_start])
    cfg.model.node_feature_dim = ds.input_feature_dim
    sample = ds[0]
    model, opt = build_model_optimizer(cfg, device)
    # init params are from the original seeded initialization before loading branch state.
    init_params = clone_params_cpu(model)
    ckpt = torch.load(args.stage4_dir / f"stage4_{exp}_branch_update{args.branch_update}.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    opt.load_state_dict(ckpt["optimizer_state_dict"])
    return cfg, sample, model, opt, init_params


def run_exp(exp: str, graph: Any, group_json: dict[str, Any], args: argparse.Namespace, device: torch.device, outdir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # C branch from checkpoint.
    set_seed(args.seed)
    cfg, sample, model, opt, init_params = setup_exp(exp, graph, args, device)
    groups = groups_from_json(group_json, sample["target"].device)
    c_rows = train_branch(model, opt, sample, cfg, groups, init_params, "C_full_plus_hot", "c_full_plus_hot",
                          args.branch_update, args.total_updates, args.log_every, args.alpha, args.near_dist_mm)
    for r in c_rows:
        r["experiment"] = exp
        r["input_feature_dim"] = cfg.model.node_feature_dim
        r["target_laser_feature_dim"] = int(cfg.model.node_feature_dim) - 12
    write_csv(outdir / f"stage4c_curve_{exp}.csv", c_rows)
    torch.save({
        "experiment": exp,
        "update": args.total_updates,
        "branch": "C_full_plus_hot",
        "alpha": args.alpha,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
    }, outdir / f"stage4c_{exp}_update{args.total_updates}.pt")
    del model, opt
    if device.type == "cuda": torch.cuda.empty_cache()

    b_rows: list[dict[str, Any]] = []
    if args.replay_b:
        set_seed(args.seed)
        cfg, sample, model, opt, init_params = setup_exp(exp, graph, args, device)
        groups = groups_from_json(group_json, sample["target"].device)
        b_rows = train_branch(model, opt, sample, cfg, groups, init_params, "B_balanced_replay", "balanced",
                              args.branch_update, args.total_updates, args.total_updates - args.branch_update,
                              args.alpha, args.near_dist_mm)
        for r in b_rows:
            r["experiment"] = exp
            r["input_feature_dim"] = cfg.model.node_feature_dim
            r["target_laser_feature_dim"] = int(cfg.model.node_feature_dim) - 12
        write_csv(outdir / f"stage4c_B_replay_final_{exp}.csv", b_rows)
        del model, opt
        if device.type == "cuda": torch.cuda.empty_cache()
    return c_rows, b_rows


def summarize(outdir: Path, stage4_dir: Path, exps: list[str], c_all: dict[str, list[dict[str, Any]]], b_all: dict[str, list[dict[str, Any]]], group_json: dict[str, Any], args: argparse.Namespace) -> None:
    old_summary = {r["experiment"]: r for r in csv.DictReader((stage4_dir / "stage4_summary.csv").open(encoding="utf-8-sig"))}
    rows: list[dict[str, Any]] = []
    for exp in exps:
        r: dict[str, Any] = {"experiment": exp, "alpha": args.alpha}
        old = old_summary.get(exp, {})
        for p in ["A2000", "B2000"]:
            for k in ["rmse", "mae", "background_mae_excluding_hot", "true_hot_mae_exact", "solidus_recall", "solidus_precision", "solidus_f1", "solidus_tp", "solidus_fp", "solidus_fn", "pred_max", "full_fp_count", "pred_peak_nearest_true_hot_dist_mm", "pred_peak_is_true_hot_node"]:
                r[f"{p}_{k}"] = old.get(f"{p}_{k}")
        c = c_all[exp][-1]
        for k, v in c.items():
            if k in {"experiment"}:
                continue
            r[f"C2000_{k}"] = v
        if b_all.get(exp):
            b = b_all[exp][-1]
            for k in ["fp_total", "fp_near_onehop", "fp_far_not_onehop", "fp_near_dist_le_threshold", "fp_far_dist_gt_threshold", "fp_in_sampled_background", "fp_in_unsampled_background", "fp_in_neighbor_group", "pred_peak_nearest_true_hot_dist_mm", "pred_peak_is_true_hot_node"]:
                r[f"B2000_replay_{k}"] = b.get(k)
        rows.append(r)
    write_csv(outdir / "stage4c_summary.csv", rows)

    def f(x: Any, n: int = 2) -> str:
        try:
            if x is None or x == "": return ""
            return f"{float(x):.{n}f}"
        except Exception:
            return str(x)
    lines: list[str] = []
    lines.append("# Stage 4C 全图约束 + 热点附加损失诊断\n\n")
    lines.append(f"C 分支损失：`L = full_graph_MSE + alpha * hot_MSE`，alpha = **{args.alpha}**。从已有 update-{args.branch_update} checkpoint 与 optimizer state 分叉，继续到 update-{args.total_updates}。\n\n")
    lines.append(f"近邻 FP 统计：one-hop 邻域；另给出距离阈值 `{args.near_dist_mm} mm` 的近/远划分。\n\n")
    bc = group_json["bucket_counts"]
    lines.append(f"有效节点总数 {bc['valid_total']}；热点 {group_json['hot_group_n']}；一跳邻域 {group_json['neighbor_group_n']}；固定背景 {group_json['background_group_n']}。\n\n")
    lines.append("## A/B/C @2000 对照\n\n")
    lines.append("| Exp | Branch | RMSE | MAE | 背景MAE | 真热点MAE | Recall | Precision | F1 | TP/FP/FN | Pred max | FP未采样背景 | FP一跳邻域 | 峰值距真热点(mm) |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|\n")
    for r in rows:
        exp = r['experiment']
        for p,label in [("A2000","A MSE"),("B2000","B Balanced"),("C2000","C Full+Hot")]:
            if p == "C2000":
                vals = {
                    'rmse': r.get('C2000_rmse'), 'mae': r.get('C2000_mae'), 'bg': r.get('C2000_background_mae_excluding_hot'),
                    'hot': r.get('C2000_true_hot_mae_exact'), 'rec': r.get('C2000_solidus_recall'), 'prec': r.get('C2000_solidus_precision'),
                    'f1': r.get('C2000_solidus_f1'), 'tp': r.get('C2000_solidus_tp'), 'fp': r.get('C2000_solidus_fp'), 'fn': r.get('C2000_solidus_fn'),
                    'predmax': r.get('C2000_pred_max'), 'unsamp': r.get('C2000_fp_in_unsampled_background'), 'onehop': r.get('C2000_fp_near_onehop'), 'dist': r.get('C2000_pred_peak_nearest_true_hot_dist_mm')}
            else:
                vals = {'rmse': r.get(p+'_rmse'), 'mae': r.get(p+'_mae'), 'bg': r.get(p+'_background_mae_excluding_hot'), 'hot': r.get(p+'_true_hot_mae_exact'), 'rec': r.get(p+'_solidus_recall'), 'prec': r.get(p+'_solidus_precision'), 'f1': r.get(p+'_solidus_f1'), 'tp': r.get(p+'_solidus_tp'), 'fp': r.get(p+'_solidus_fp'), 'fn': r.get(p+'_solidus_fn'), 'predmax': r.get(p+'_pred_max'), 'unsamp': r.get(p+'_full_fp_count') if p=='B2000' else 0, 'onehop': '', 'dist': r.get(p+'_pred_peak_nearest_true_hot_dist_mm')}
                if p == 'B2000' and r.get('B2000_replay_fp_in_unsampled_background') != '':
                    vals['unsamp'] = r.get('B2000_replay_fp_in_unsampled_background')
                    vals['onehop'] = r.get('B2000_replay_fp_near_onehop')
            lines.append(f"| {exp} | {label} | {f(vals['rmse'])} | {f(vals['mae'])} | {f(vals['bg'])} | {f(vals['hot'],1)} | {f(vals['rec'],3)} | {f(vals['prec'],3)} | {f(vals['f1'],3)} | {vals['tp']}/{vals['fp']}/{vals['fn']} | {f(vals['predmax'],1)} | {vals['unsamp']} | {vals['onehop']} | {f(vals['dist'],3)} |\n")
    lines.append("\n## C 分支 FP 拆分\n\n")
    lines.append("| Exp | FP total | one-hop near | not one-hop | dist-near | dist-far | sampled bg | unsampled bg | neighbor group |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        lines.append(f"| {r['experiment']} | {r.get('C2000_fp_total')} | {r.get('C2000_fp_near_onehop')} | {r.get('C2000_fp_far_not_onehop')} | {r.get('C2000_fp_near_dist_le_threshold')} | {r.get('C2000_fp_far_dist_gt_threshold')} | {r.get('C2000_fp_in_sampled_background')} | {r.get('C2000_fp_in_unsampled_background')} | {r.get('C2000_fp_in_neighbor_group')} |\n")
    if args.replay_b:
        lines.append("\n## B replay FP 拆分（用于解释已有 B 误报）\n\n")
        lines.append("| Exp | FP total | one-hop near | not one-hop | dist-near | dist-far | sampled bg | unsampled bg | neighbor group |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            lines.append(f"| {r['experiment']} | {r.get('B2000_replay_fp_total')} | {r.get('B2000_replay_fp_near_onehop')} | {r.get('B2000_replay_fp_far_not_onehop')} | {r.get('B2000_replay_fp_near_dist_le_threshold')} | {r.get('B2000_replay_fp_far_dist_gt_threshold')} | {r.get('B2000_replay_fp_in_sampled_background')} | {r.get('B2000_replay_fp_in_unsampled_background')} | {r.get('B2000_replay_fp_in_neighbor_group')} |\n")
    lines.append("\n## 初步判断\n\n")
    lines.append("- C 保留了全图背景约束，因此可以区分 B 的误报是否主要来自背景监督缩减。\n")
    lines.append("- 若 C 的 FP/背景 MAE 明显小于 B，但热点仍改善，说明正式方案应优先考虑 full-MSE + 热点附加项，而不是只监督抽样背景。\n")
    lines.append("- 若 C 仍不能越过 solidus，则 alpha=0.1 可能不足，下一步只在预设 `{0.03, 0.1, 0.3}` 中补有限候选，不做无限扫描。\n\n")
    lines.append("## 产物\n\n")
    for name in ["stage4c_summary.csv", "stage4c_curve_E0.csv", "stage4c_curve_E1.csv", "stage4c_curve_E2.csv", "stage4c_paired_curves.png"]:
        lines.append(f"- `{outdir/name}`\n")
    (outdir / "stage4c_report.md").write_text("".join(lines), encoding="utf-8")


def plot(outdir: Path, exps: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped: {exc}")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [("rmse", "RMSE"), ("true_hot_mae_exact", "True-hot MAE"), ("solidus_recall", "Recall"), ("fp_total", "FP total")]
    for exp in exps:
        p = outdir / f"stage4c_curve_{exp}.csv"
        if not p.exists(): continue
        rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
        xs = [int(r["update"]) for r in rows]
        for ax, (key, title) in zip(axes.ravel(), specs):
            ys = [float(r[key]) for r in rows]
            ax.plot(xs, ys, label=exp)
            ax.set_title(title); ax.set_xlabel("optimizer updates"); ax.grid(True, alpha=.3)
    for ax in axes.ravel(): ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "stage4c_paired_curves.png", dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue C branch from stage4 update-500 checkpoints")
    parser.add_argument("--stage4_dir", type=Path, default=Path("results/training_diagnostics/stage4_paired_supervision"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/training_diagnostics/stage4c_full_hot"))
    parser.add_argument("--vtu_dir", default="F:/VTU")
    parser.add_argument("--cache_dir", default="data/processed")
    parser.add_argument("--laser_xml", default="configs/laser_paths/5_block_fem_additive_z_scan.xml")
    parser.add_argument("--experiments", nargs="+", default=["E0", "E1", "E2"], choices=list(EXPERIMENTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--graph_device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--window_start", type=int, default=421)
    parser.add_argument("--branch_update", type=int, default=500)
    parser.add_argument("--total_updates", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--near_dist_mm", type=float, default=0.5)
    parser.add_argument("--replay_b", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu"))
    graph_device = device if args.graph_device == "auto" else torch.device(args.graph_device)
    ref_cfg = Config.from_yaml(EXPERIMENTS[args.experiments[0]])
    if args.laser_xml:
        ref_cfg.data.laser_xml_path = args.laser_xml; ref_cfg.data.laser_path_mode = "additive_z_scan"
    graph = load_graph(args.vtu_dir, args.cache_dir, ref_cfg, graph_device, args.no_cache)
    group_json = load_json(args.stage4_dir / "stage4_node_groups.json")
    (args.output_dir / "stage4c_manifest.json").write_text(json.dumps({**vars(args), "stage4_dir": str(args.stage4_dir), "output_dir": str(args.output_dir), "device": str(device), "graph_device": str(graph_device)}, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    c_all: dict[str, list[dict[str, Any]]] = {}
    b_all: dict[str, list[dict[str, Any]]] = {}
    for exp in args.experiments:
        print(f"=== Stage4C {exp} ===", flush=True)
        c_rows, b_rows = run_exp(exp, graph, group_json, args, device, args.output_dir)
        c_all[exp] = c_rows; b_all[exp] = b_rows
    summarize(args.output_dir, args.stage4_dir, args.experiments, c_all, b_all, group_json, args)
    plot(args.output_dir, args.experiments)
    print(args.output_dir / "stage4c_report.md")


if __name__ == "__main__":
    main()
