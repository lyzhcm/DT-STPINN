"""Stage 5 diagnostic: restore physics/regularization terms one at a time.

Starts from E1 Stage-4C update-2000 checkpoint (full MSE + alpha hot MSE)
and continues single-window training for a small number of optimizer updates.
This is a mechanism/compatibility diagnostic, not a generalization experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.training_diagnostics import (  # noqa: E402
    EXPERIMENTS,
    Config,
    DTSTPINN,
    clone_params_cpu,
    disable_dropout,
    grad_norm,
    load_graph,
    make_dataset,
    metrics_from_pred,
    param_delta,
    set_seed,
    write_csv,
)
from scripts.training_diagnostics_stage4 import extra_position_metrics  # noqa: E402
from scripts.training_diagnostics_stage4c import fp_breakdown, groups_from_json  # noqa: E402
from src.loss import DTSTPINNLoss  # noqa: E402


BRANCHES = ["Control", "PDE", "BC", "IC", "Smooth", "Dropout"]
PHYS_KEYS = ["PDE", "BC", "IC", "Smooth"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float_tensor_value(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float("nan")
        return float(x.detach().float().cpu().item())
    return float(x)


def set_only_branch_term(cfg: Config, branch: str) -> None:
    """Configure formal physics flags so only the requested branch term is active."""
    # Keep formal weights; only switch flags / smooth lambda.
    cfg.physics.heat_conduction = branch == "PDE"
    cfg.physics.boundary_convection = branch == "BC"
    cfg.physics.initial_condition = branch == "IC"
    if branch != "Smooth":
        cfg.loss.lambda_smooth = 0.0
    # These diagnostics should not activate unrelated auxiliary losses.
    for name in list(vars(cfg.loss).keys()):
        if name.startswith("lambda_") and name not in {
            "lambda_T", "lambda_PDE", "lambda_BC", "lambda_IC", "lambda_smooth"
        }:
            try:
                setattr(cfg.loss, name, 0.0)
            except Exception:
                pass


def set_probe_terms(cfg: Config) -> None:
    """Enable all formal PDE/BC/IC/Smooth probes; auxiliary learned losses off."""
    cfg.physics.heat_conduction = True
    cfg.physics.boundary_convection = True
    cfg.physics.initial_condition = True
    # Keep formal lambda_smooth from file.
    for name in list(vars(cfg.loss).keys()):
        if name.startswith("lambda_") and name not in {
            "lambda_T", "lambda_PDE", "lambda_BC", "lambda_IC", "lambda_smooth"
        }:
            try:
                setattr(cfg.loss, name, 0.0)
            except Exception:
                pass


def make_cfg(args: argparse.Namespace, *, dropout: bool, branch: str | None = None, probe: bool = False) -> Config:
    cfg = Config.from_yaml(EXPERIMENTS[args.experiment])
    if args.laser_xml:
        cfg.data.laser_xml_path = args.laser_xml
        cfg.data.laser_path_mode = "additive_z_scan"
    if not dropout:
        disable_dropout(cfg)
    cfg.training.use_amp = False
    cfg.training.lr = args.lr
    if probe:
        set_probe_terms(cfg)
    elif branch is not None:
        set_only_branch_term(cfg, branch)
    return cfg


def build_model_optimizer(cfg: Config, device: torch.device) -> tuple[DTSTPINN, torch.optim.Optimizer]:
    model = DTSTPINN(cfg, cfg.material).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    return model, opt


def load_stage4c_state(model: DTSTPINN, opt: torch.optim.Optimizer, ckpt_path: Path, lr: float, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    opt.load_state_dict(ckpt["optimizer_state_dict"])
    for group in opt.param_groups:
        group["lr"] = lr
    return ckpt


def common_forward(model: DTSTPINN, sample: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    dt = sample.get("dt", 1.0)
    if isinstance(dt, torch.Tensor):
        dt = float(dt.detach().cpu().item())
    out = model(sample["graph_sequence"], dt=dt)
    pred = out["T_pred"].float()
    target = sample["target"].float()
    if pred.numel() == target.numel():
        pred = pred.reshape_as(target)
    return out, pred


def c_loss_from_pred(pred: torch.Tensor, sample: dict[str, Any], groups: dict[str, Any], alpha: float) -> tuple[torch.Tensor, dict[str, float]]:
    target = sample["target"].float()
    if pred.numel() == target.numel():
        pred = pred.reshape_as(target)
    mask = sample["target_mask"].bool()
    p = pred.flatten()
    t = target.flatten()
    full_mse = ((pred[mask] - target[mask]) ** 2).mean()
    hot_idx = groups["hot_idx"]
    hot_mse = ((p[hot_idx] - t[hot_idx]) ** 2).mean()
    hot_weighted = alpha * hot_mse
    loss_c = full_mse + hot_weighted
    return loss_c, {
        "loss_full_mse": float(full_mse.detach().cpu()),
        "loss_hot_mse": float(hot_mse.detach().cpu()),
        "loss_0p1_hot_mse": float(hot_weighted.detach().cpu()),
        "loss_C": float(loss_c.detach().cpu()),
    }


def get_boundary(sample: dict[str, Any]) -> torch.Tensor:
    seq = sample["graph_sequence"]
    return getattr(seq[-1], "boundary", torch.zeros(sample["coords"].shape[0], device=sample["coords"].device))


def get_is_initial(sample: dict[str, Any]) -> torch.Tensor:
    mask = sample.get("target_mask")
    n = sample["coords"].shape[0]
    target_step = int(sample.get("target_step", -1))
    if target_step == 0:
        return mask.bool() if mask is not None else torch.ones(n, dtype=torch.bool, device=sample["coords"].device)
    return torch.zeros(n, dtype=torch.bool, device=sample["coords"].device)


def loss_forward_args(output: dict[str, Any], pred: torch.Tensor, sample: dict[str, Any]) -> dict[str, Any]:
    dt = sample.get("dt", 1.0)
    if isinstance(dt, torch.Tensor):
        dt = float(dt.detach().cpu().item())
    return dict(
        pred=pred,
        target=sample["target"],
        prev_temp=sample.get("prev_temp"),
        coords=sample["coords"],
        edge_index=sample["edge_index"],
        boundary=get_boundary(sample),
        dt=dt,
        laser_pos=sample.get("target_laser_pos"),
        mask=sample.get("target_mask"),
        is_initial=get_is_initial(sample),
        hotspot_logit=output.get("hotspot_logit"),
        hotspot_delta=output.get("hotspot_delta"),
        hotspot_specialist_temp=output.get("hotspot_specialist_temp"),
        hotspot_specialist_gate=output.get("hotspot_specialist_gate"),
        process_gate=output.get("process_gate"),
        laser_arrival_gate=output.get("laser_arrival_gate"),
        laser_path_post_arrival_gate=output.get("laser_path_post_arrival_gate"),
        laser_endpoint_gate=output.get("laser_endpoint_gate"),
        neighbor_hot_gate=output.get("neighbor_hot_gate"),
        final_pred=output.get("T_pred"),
        laser_residual_prior_gate=output.get("laser_residual_prior_gate"),
        laser_residual_learned_gate=output.get("laser_residual_learned_gate"),
        laser_residual_gate=output.get("laser_residual_gate"),
        laser_residual_delta=output.get("laser_residual_delta"),
        laser_residual_boost=output.get("laser_residual_boost"),
        cold_to_hot_logit=output.get("cold_to_hot_logit"),
        cold_to_hot_gate=output.get("cold_to_hot_gate"),
    )


def compute_losses(model: DTSTPINN, sample: dict[str, Any], groups: dict[str, Any], alpha: float,
                   branch: str, branch_loss_fn: DTSTPINNLoss | None) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    output, pred = common_forward(model, sample)
    loss_c, comps = c_loss_from_pred(pred, sample, groups, alpha)
    term_loss = pred.new_tensor(0.0)
    term_key = branch if branch in PHYS_KEYS else None
    if term_key and branch_loss_fn is not None:
        _total_formal, formal = branch_loss_fn.forward(**loss_forward_args(output, pred, sample))
        maybe = formal.get(term_key)
        if maybe is not None:
            term_loss = maybe
    comps["loss_added_term_weighted"] = float(term_loss.detach().cpu())
    comps["loss_total"] = float((loss_c + term_loss).detach().cpu())
    if term_key:
        comps[f"loss_{term_key}_weighted"] = float(term_loss.detach().cpu())
    return loss_c + term_loss, comps, pred


def grad_vector(model: DTSTPINN) -> torch.Tensor:
    pieces = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            pieces.append(torch.zeros(p.numel(), dtype=torch.float32))
        else:
            pieces.append(p.grad.detach().float().reshape(-1).cpu())
    if not pieces:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(pieces)


def vec_norm(v: torch.Tensor) -> float:
    if v.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(v).item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = vec_norm(a)
    nb = vec_norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / (na * nb))


def gradient_diagnostics(model: DTSTPINN, sample: dict[str, Any], groups: dict[str, Any], alpha: float,
                         branch: str, branch_loss_fn: DTSTPINNLoss | None, grad_clip: float) -> dict[str, Any]:
    model.train()
    output, pred = common_forward(model, sample)
    loss_c, comps = c_loss_from_pred(pred, sample, groups, alpha)
    term_key = branch if branch in PHYS_KEYS else None
    term_loss = pred.new_tensor(0.0)
    if term_key and branch_loss_fn is not None:
        _total_formal, formal = branch_loss_fn.forward(**loss_forward_args(output, pred, sample))
        maybe = formal.get(term_key)
        if maybe is not None:
            term_loss = maybe
    # C supervision gradient.
    model.zero_grad(set_to_none=True)
    loss_c.backward(retain_graph=True)
    c_vec = grad_vector(model)
    # Added term gradient.
    model.zero_grad(set_to_none=True)
    if term_loss.requires_grad and float(term_loss.detach().abs().cpu()) != 0.0:
        term_loss.backward(retain_graph=True)
        term_vec = grad_vector(model)
    else:
        term_vec = torch.zeros_like(c_vec)
    # Total gradient.
    model.zero_grad(set_to_none=True)
    total_loss = loss_c + term_loss
    total_loss.backward()
    total_vec = grad_vector(model)
    model.zero_grad(set_to_none=True)
    row: dict[str, Any] = dict(comps)
    row.update({
        "grad_C_norm": vec_norm(c_vec),
        "grad_added_term_norm": vec_norm(term_vec),
        "grad_added_vs_C_cosine": cosine(term_vec, c_vec),
        "grad_total_norm_pre_clip_diag": vec_norm(total_vec),
        "grad_clip_threshold": grad_clip,
        "grad_would_clip_diag": bool(grad_clip > 0 and vec_norm(total_vec) > grad_clip),
        "loss_added_term_weighted": float(term_loss.detach().cpu()),
        "loss_total": float(total_loss.detach().cpu()),
    })
    if term_key:
        row[f"loss_{term_key}_weighted"] = float(term_loss.detach().cpu())
        formal_weight = {
            "PDE": getattr(branch_loss_fn, "lambda_PDE", float("nan")),
            "BC": getattr(branch_loss_fn, "lambda_BC", float("nan")),
            "IC": getattr(branch_loss_fn, "lambda_IC", float("nan")),
            "Smooth": getattr(branch_loss_fn, "lambda_smooth", float("nan")),
        }.get(term_key, float("nan"))
        row[f"loss_{term_key}_unweighted"] = float(term_loss.detach().cpu()) / formal_weight if formal_weight else float("nan")
    return row


def probe_physics_losses(model: DTSTPINN, sample: dict[str, Any], probe_loss_fn: DTSTPINNLoss) -> dict[str, float]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output, pred = common_forward(model, sample)
        _total, losses = probe_loss_fn.forward(**loss_forward_args(output, pred, sample))
    if was_training:
        model.train()
    out: dict[str, float] = {}
    weights = {
        "PDE": getattr(probe_loss_fn, "lambda_PDE", 0.0),
        "BC": getattr(probe_loss_fn, "lambda_BC", 0.0),
        "IC": getattr(probe_loss_fn, "lambda_IC", 0.0),
        "Smooth": getattr(probe_loss_fn, "lambda_smooth", 0.0),
    }
    for k in PHYS_KEYS:
        val = losses.get(k)
        weighted = float(val.detach().cpu()) if val is not None else 0.0
        out[f"probe_{k}_weighted"] = weighted
        out[f"probe_{k}_unweighted"] = weighted / weights[k] if weights[k] else float("nan")
    return out


def eval_row(model: DTSTPINN, sample: dict[str, Any], cfg: Config, groups: dict[str, Any], update: int,
             branch: str, alpha: float, train_comps: dict[str, Any] | None, grad_diag: dict[str, Any] | None,
             last_grad_norm: float | None, clip_applied: bool | None, start_params: list[torch.Tensor],
             near_dist_mm: float, probe_loss_fn: DTSTPINNLoss) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _out, pred = common_forward(model, sample)
        _lc, c_eval = c_loss_from_pred(pred, sample, groups, alpha)
    if was_training:
        model.train()
    row: dict[str, Any] = {
        "update": update,
        "branch": branch,
        "alpha": alpha,
        "lr": None,
        "grad_norm_pre_clip_train_step": last_grad_norm,
        "grad_clip_applied_train_step": clip_applied,
        "hot_group_n": int(groups["hot_idx"].numel()),
        "neighbor_group_n": int(groups["neighbor_idx"].numel()),
        "background_group_n": int(groups["background_idx"].numel()),
    }
    row.update(c_eval)
    if train_comps:
        for k, v in train_comps.items():
            row[f"train_{k}"] = v
    if grad_diag:
        row.update(grad_diag)
    row.update(metrics_from_pred(pred, sample["target"], sample.get("target_mask"), cfg.material.solidus_temp, cfg.material.liquidus_temp, None, prefix=""))
    row.update(extra_position_metrics(pred, sample["target"], sample["target_mask"], sample["coords"], groups, cfg.material.solidus_temp))
    row.update(fp_breakdown(pred, sample["target"], sample["target_mask"], sample["coords"], groups, cfg.material.solidus_temp, near_dist_mm))
    row.update(probe_physics_losses(model, sample, probe_loss_fn))
    dl2, dmax = param_delta(model, start_params)
    row["param_delta_l2_from_stage5_start"] = dl2
    row["param_delta_max_from_stage5_start"] = dmax
    return row


def branch_applicability(branch: str, sample: dict[str, Any]) -> tuple[bool, str]:
    if branch == "IC":
        n = int(get_is_initial(sample).sum().detach().cpu())
        if n == 0:
            return False, "no effective initial-condition nodes at this target step"
    return True, "applicable"


def run_branch(branch: str, graph: Any, group_json: dict[str, Any], args: argparse.Namespace,
               device: torch.device, outdir: Path) -> list[dict[str, Any]]:
    dropout = branch == "Dropout"
    cfg = make_cfg(args, dropout=dropout, branch=branch)
    ds = make_dataset(graph, cfg, [args.window_start])
    cfg.model.node_feature_dim = ds.input_feature_dim
    sample = ds[0]
    groups = groups_from_json(group_json, sample["target"].device)
    applicable, reason = branch_applicability(branch, sample)

    model, opt = build_model_optimizer(cfg, device)
    ckpt = load_stage4c_state(model, opt, args.start_ckpt, args.lr, device)
    start_update = int(ckpt.get("update", args.start_update))
    start_params = clone_params_cpu(model)
    branch_loss_fn = DTSTPINNLoss(cfg, cfg.material)
    probe_cfg = make_cfg(args, dropout=False, probe=True)
    probe_loss_fn = DTSTPINNLoss(probe_cfg, probe_cfg.material)

    rows: list[dict[str, Any]] = []
    grad_diag0 = gradient_diagnostics(model, sample, groups, args.alpha, branch, branch_loss_fn, cfg.training.grad_clip) if applicable else None
    row0 = eval_row(model, sample, cfg, groups, start_update, branch, args.alpha, None, grad_diag0, None, None, start_params, args.near_dist_mm, probe_loss_fn)
    row0.update({
        "status": "running" if applicable else "not_applicable",
        "status_reason": reason,
        "input_feature_dim": cfg.model.node_feature_dim,
        "target_laser_feature_dim": int(cfg.model.node_feature_dim) - 12,
        "dropout_spatial": getattr(cfg.model.spatial, "dropout", None),
        "dropout_temporal": getattr(cfg.model.temporal, "dropout", None),
    })
    rows.append(row0)

    if not applicable:
        write_csv(outdir / f"stage5_curve_{branch}.csv", rows)
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return rows

    end_update = start_update + args.extra_updates
    last_comps: dict[str, Any] | None = None
    last_gnorm: float | None = None
    last_clipped: bool | None = None
    for update in range(start_update + 1, end_update + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        total_loss, comps, _pred = compute_losses(model, sample, groups, args.alpha, branch, branch_loss_fn)
        total_loss.backward()
        last_gnorm = grad_norm(model)
        last_clipped = bool(cfg.training.grad_clip > 0 and last_gnorm > cfg.training.grad_clip)
        if cfg.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        opt.step()
        last_comps = comps
        if update % args.log_every == 0 or update == end_update:
            grad_diag = gradient_diagnostics(model, sample, groups, args.alpha, branch, branch_loss_fn, cfg.training.grad_clip)
            row = eval_row(model, sample, cfg, groups, update, branch, args.alpha, last_comps, grad_diag, last_gnorm, last_clipped, start_params, args.near_dist_mm, probe_loss_fn)
            row.update({
                "status": "running",
                "status_reason": reason,
                "input_feature_dim": cfg.model.node_feature_dim,
                "target_laser_feature_dim": int(cfg.model.node_feature_dim) - 12,
                "dropout_spatial": getattr(cfg.model.spatial, "dropout", None),
                "dropout_temporal": getattr(cfg.model.temporal, "dropout", None),
                "lr": opt.param_groups[0]["lr"],
            })
            rows.append(row)
            print(
                f"{branch} update {update}/{end_update} rmse={row['rmse']:.3f} "
                f"hot_mae={row['true_hot_mae_exact']:.3f} rec={row['solidus_recall']:.3f} "
                f"prec={row['solidus_precision']:.3f} fp={row['fp_total']} "
                f"bg_mae={row['background_mae_excluding_hot']:.3f} "
                f"term={row.get('loss_added_term_weighted', 0.0):.3g} "
                f"cos={row.get('grad_added_vs_C_cosine', float('nan')):.3g}",
                flush=True,
            )
    write_csv(outdir / f"stage5_curve_{branch}.csv", rows)
    torch.save({
        "experiment": args.experiment,
        "branch": branch,
        "start_checkpoint": str(args.start_ckpt),
        "start_update": start_update,
        "update": end_update,
        "alpha": args.alpha,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
    }, outdir / f"stage5_{branch}_update{end_update}.pt")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def write_audit(outdir: Path, sample: dict[str, Any], cfg: Config, group_json: dict[str, Any], args: argparse.Namespace) -> None:
    boundary = get_boundary(sample)
    mask = sample["target_mask"].bool()
    is_initial = get_is_initial(sample)
    edge_index = sample["edge_index"]
    active_edges = int((mask[edge_index[0]] & mask[edge_index[1]]).sum().detach().cpu())
    audit = {
        "experiment": args.experiment,
        "start_checkpoint": str(args.start_ckpt),
        "window_start": args.window_start,
        "target_step": int(sample["target_step"]),
        "target_time": as_float_tensor_value(sample["target_time"]),
        "dt_index_units": as_float_tensor_value(sample["dt"]),
        "physics_time_scale_to_s": cfg.physics.time_scale_to_s,
        "dt_physics_s": as_float_tensor_value(sample["dt"]) * cfg.physics.time_scale_to_s,
        "coordinate_scale_to_m": cfg.physics.coordinate_scale_to_m,
        "laser_pos_mm": [float(x) for x in sample["target_laser_pos"].detach().cpu().tolist()],
        "boundary_label_mode": cfg.physics.boundary_label_mode,
        "formal_boundary_convection_flag": cfg.physics.boundary_convection,
        "formal_boundary_radiation_flag": cfg.physics.boundary_radiation,
        "boundary_nonzero_nodes": int((boundary.abs() > 1e-6).sum().detach().cpu()),
        "boundary_positive_nodes": int((boundary > 0).sum().detach().cpu()),
        "boundary_negative_nodes": int((boundary < 0).sum().detach().cpu()),
        "boundary_positive_valid_nodes": int(((boundary > 0) & mask).sum().detach().cpu()),
        "ic_effective_nodes": int(is_initial.sum().detach().cpu()),
        "valid_nodes": int(mask.sum().detach().cpu()),
        "edge_count": int(edge_index.shape[1]),
        "active_active_edges_for_smooth": active_edges,
        "hot_group_n": group_json.get("hot_group_n"),
        "neighbor_group_n": group_json.get("neighbor_group_n"),
        "background_group_n": group_json.get("background_group_n"),
        "bucket_counts": group_json.get("bucket_counts"),
    }
    (outdir / "stage5_protocol_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize(outdir: Path, all_rows: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> None:
    summary: list[dict[str, Any]] = []
    for branch, rows in all_rows.items():
        start = rows[0]
        final = rows[-1]
        r: dict[str, Any] = {
            "branch": branch,
            "status": final.get("status"),
            "status_reason": final.get("status_reason"),
            "start_update": start.get("update"),
            "final_update": final.get("update"),
        }
        for prefix, row in [("start", start), ("final", final)]:
            for k in [
                "rmse", "mae", "background_mae_excluding_hot", "true_hot_mae_exact",
                "solidus_recall", "solidus_precision", "solidus_f1", "solidus_tp",
                "solidus_fp", "solidus_fn", "pred_max", "pred_peak_nearest_true_hot_dist_mm",
                "pred_peak_is_true_hot_node", "fp_total", "fp_in_unsampled_background",
                "fp_in_neighbor_group", "loss_C", "loss_added_term_weighted",
                "grad_C_norm", "grad_added_term_norm", "grad_added_vs_C_cosine",
                "grad_total_norm_pre_clip_diag", "probe_PDE_weighted", "probe_BC_weighted",
                "probe_IC_weighted", "probe_Smooth_weighted",
            ]:
                r[f"{prefix}_{k}"] = row.get(k)
        try:
            r["delta_true_hot_mae_vs_start"] = float(final.get("true_hot_mae_exact")) - float(start.get("true_hot_mae_exact"))
            r["delta_background_mae_vs_start"] = float(final.get("background_mae_excluding_hot")) - float(start.get("background_mae_excluding_hot"))
            r["delta_rmse_vs_start"] = float(final.get("rmse")) - float(start.get("rmse"))
            r["delta_fp_vs_start"] = float(final.get("fp_total")) - float(start.get("fp_total"))
        except Exception:
            pass
        summary.append(r)
    write_csv(outdir / "stage5_summary.csv", summary)

    def f(x: Any, n: int = 2) -> str:
        try:
            if x is None or x == "":
                return ""
            return f"{float(x):.{n}f}"
        except Exception:
            return str(x)

    lines: list[str] = []
    lines.append("# Stage 5 物理项逐项恢复诊断\n\n")
    lines.append("本轮是单窗口约束兼容性诊断，不用于证明泛化收益。起点为 E1-C(alpha=0.1) update-2000。每个适用分支继续 500 次 optimizer update，固定 LR，每 50 次记录。\n\n")
    lines.append("## Final vs common start\n\n")
    lines.append("| Branch | Status | RMSE | 背景MAE | 真热点MAE | Recall | Precision | F1 | FP | Pred max | 峰值距真热点(mm) | added loss | grad cos(term,C) | ΔRMSE | Δ背景MAE | Δ热点MAE | ΔFP |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in summary:
        lines.append(
            f"| {r['branch']} | {r.get('status')} | {f(r.get('final_rmse'))} | {f(r.get('final_background_mae_excluding_hot'))} | "
            f"{f(r.get('final_true_hot_mae_exact'))} | {f(r.get('final_solidus_recall'),3)} | {f(r.get('final_solidus_precision'),3)} | {f(r.get('final_solidus_f1'),3)} | "
            f"{f(r.get('final_fp_total'),0)} | {f(r.get('final_pred_max'))} | {f(r.get('final_pred_peak_nearest_true_hot_dist_mm'),3)} | "
            f"{f(r.get('final_loss_added_term_weighted'),4)} | {f(r.get('final_grad_added_vs_C_cosine'),3)} | {f(r.get('delta_rmse_vs_start'))} | "
            f"{f(r.get('delta_background_mae_vs_start'))} | {f(r.get('delta_true_hot_mae_vs_start'))} | {f(r.get('delta_fp_vs_start'),0)} |\n"
        )
    lines.append("\n## 产物\n\n")
    for name in ["stage5_protocol_audit.json", "stage5_summary.csv"]:
        lines.append(f"- `{outdir/name}`\n")
    for branch in BRANCHES:
        lines.append(f"- `{outdir / ('stage5_curve_' + branch + '.csv')}`\n")
    (outdir / "stage5_report.md").write_text("".join(lines), encoding="utf-8-sig")


def plot(outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped: {exc}")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [("rmse", "RMSE"), ("true_hot_mae_exact", "True-hot MAE"), ("background_mae_excluding_hot", "Background MAE"), ("fp_total", "FP total")]
    for branch in BRANCHES:
        p = outdir / f"stage5_curve_{branch}.csv"
        if not p.exists():
            continue
        rows = [r for r in csv.DictReader(p.open(encoding="utf-8-sig")) if r.get("status") != "not_applicable"]
        if not rows:
            continue
        xs = [int(float(r["update"])) for r in rows]
        for ax, (key, title) in zip(axes.ravel(), specs):
            ys = [float(r[key]) for r in rows]
            ax.plot(xs, ys, label=branch)
            ax.set_title(title)
            ax.set_xlabel("optimizer updates")
            ax.grid(True, alpha=.3)
    for ax in axes.ravel():
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "stage5_curves.png", dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore physics terms one at a time from E1-C update2000")
    parser.add_argument("--experiment", default="E1", choices=list(EXPERIMENTS))
    parser.add_argument("--start_ckpt", type=Path, default=Path("results/training_diagnostics/stage4c_full_hot/stage4c_E1_update2000.pt"))
    parser.add_argument("--stage4_dir", type=Path, default=Path("results/training_diagnostics/stage4_paired_supervision"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/training_diagnostics/stage5_physics_restore_e1c"))
    parser.add_argument("--vtu_dir", default="F:/VTU")
    parser.add_argument("--cache_dir", default="data/processed")
    parser.add_argument("--laser_xml", default="configs/laser_paths/5_block_fem_additive_z_scan.xml")
    parser.add_argument("--branches", nargs="+", default=BRANCHES, choices=BRANCHES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--graph_device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--window_start", type=int, default=421)
    parser.add_argument("--start_update", type=int, default=2000)
    parser.add_argument("--extra_updates", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--near_dist_mm", type=float, default=0.5)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu"))
    graph_device = device if args.graph_device == "auto" else torch.device(args.graph_device)

    ref_cfg = make_cfg(args, dropout=False, branch="Control")
    graph = load_graph(args.vtu_dir, args.cache_dir, ref_cfg, graph_device, args.no_cache)
    group_json = load_json(args.stage4_dir / "stage4_node_groups.json")
    ds = make_dataset(graph, ref_cfg, [args.window_start])
    ref_cfg.model.node_feature_dim = ds.input_feature_dim
    sample = ds[0]
    write_audit(args.output_dir, sample, ref_cfg, group_json, args)

    manifest = {**vars(args), "device": str(device), "graph_device": str(graph_device)}
    (args.output_dir / "stage5_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    all_rows: dict[str, list[dict[str, Any]]] = {}
    for branch in args.branches:
        print(f"=== Stage5 {branch} ===", flush=True)
        set_seed(args.seed)
        rows = run_branch(branch, graph, group_json, args, device, args.output_dir)
        all_rows[branch] = rows
    summarize(args.output_dir, all_rows, args)
    plot(args.output_dir)
    print(args.output_dir / "stage5_report.md")


if __name__ == "__main__":
    main()
