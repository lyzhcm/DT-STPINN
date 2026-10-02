"""No-training audit for multi-window C/C+PDE checkpoints."""
from __future__ import annotations

import argparse, csv, json, math, sys
from pathlib import Path
from typing import Any
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.training_diagnostics import EXPERIMENTS, load_graph, make_dataset, write_csv
from scripts.training_diagnostics_multwindow import configure, calc_loss, window_metrics, aggregate, MODE_LABEL
from scripts.training_diagnostics_stage5 import DTSTPINNLoss
from src.model import DTSTPINN

AUDIT_MODES=("c_hot","c_hot_pde")
TEMP_BINS=(("fp_true_lt500",None,500.0),("fp_true_500_1000",500.0,1000.0),("fp_true_1000_solidus",1000.0,"solidus"))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_split(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def update_from_name(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def checkpoints(exp_dir: Path) -> dict[int, Path]:
    return {update_from_name(p): p for p in exp_dir.glob("checkpoint_update_*.pt")}


def key_train_updates(curve_rows: list[dict[str, Any]], total_updates: int) -> set[int]:
    rows=[r for r in curve_rows if int(r.get("update",-1))>0]
    out={int(total_updates)}
    if rows:
        out.add(int(max(rows,key=lambda r: float(r.get("solidus_f1",0.0)))["update"]))
        out.add(int(min(rows,key=lambda r: float(r.get("rmse",float("inf")) ))["update"]))
    return out


def fp_bins(pred: torch.Tensor, sample: dict[str, Any], solidus: float) -> dict[str, Any]:
    target=sample["target"].detach().float()
    if pred.numel()==target.numel(): pred=pred.reshape_as(target)
    p=pred.detach().float().flatten(); t=target.flatten(); valid=sample["target_mask"].detach().bool().flatten()
    fp=valid & (p>=solidus) & (t<solidus)
    out={"fp_total_check": int(fp.sum().cpu())}
    for name,lo,hi in TEMP_BINS:
        m=fp.clone()
        if lo is not None: m &= t>=float(lo)
        h=solidus if hi=="solidus" else hi
        if h is not None: m &= t<float(h)
        out[name]=int(m.sum().cpu())
    return out


def eval_split(model, ds, starts, cfg, mode, loss_fn, alpha, update, split_name):
    model.eval(); rows=[]
    with torch.no_grad():
        for i,s in enumerate(starts):
            sample=ds[i]
            loss, comps, pred=calc_loss(model,sample,cfg,mode,loss_fn,alpha)
            row=window_metrics(pred,sample,cfg,int(s),comps)
            row.update(fp_bins(pred,sample,float(cfg.material.solidus_temp)))
            n=int(row["valid_nodes"]); nh=int(row["true_hot_nodes"])
            full=float(comps.get("loss_full_mse",float("nan")))
            hot=float(comps.get("loss_hot_mse",0.0))
            row.update(split=split_name, update=int(update), no_hot_window=(nh==0), pred_hot_nodes=int(row["solidus_tp"])+int(row["solidus_fp"]), eval_loss=float(loss.detach().cpu()),
                       effective_hot_node_coeff=(1.0+alpha*n/nh) if nh>0 else float("nan"),
                       hot_to_full_loss_ratio=(alpha*hot/full) if nh>0 and full!=0 else float("nan"),
                       unweighted_hot_to_full_mse_ratio=(hot/full) if nh>0 and full!=0 else float("nan"))
            rows.append(row)
    agg=aggregate(rows)
    coeffs=[float(r["effective_hot_node_coeff"]) for r in rows if not math.isnan(float(r["effective_hot_node_coeff"]))]
    ratios=[float(r["hot_to_full_loss_ratio"]) for r in rows if not math.isnan(float(r["hot_to_full_loss_ratio"]))]
    agg.update(split=split_name, update=int(update), no_hot_windows=sum(1 for r in rows if r["no_hot_window"]),
               no_hot_fp_total=sum(int(r["solidus_fp"]) for r in rows if r["no_hot_window"]), pred_hot_nodes=sum(int(r["pred_hot_nodes"]) for r in rows),
               fp_true_lt500=sum(int(r["fp_true_lt500"]) for r in rows), fp_true_500_1000=sum(int(r["fp_true_500_1000"]) for r in rows), fp_true_1000_solidus=sum(int(r["fp_true_1000_solidus"]) for r in rows),
               mean_effective_hot_node_coeff=sum(coeffs)/len(coeffs) if coeffs else float("nan"), max_effective_hot_node_coeff=max(coeffs) if coeffs else float("nan"),
               mean_hot_to_full_loss_ratio=sum(ratios)/len(ratios) if ratios else float("nan"), max_hot_to_full_loss_ratio=max(ratios) if ratios else float("nan"))
    return agg, rows


def load_model_for_ckpt(ckpt_path, cfg, device):
    ckpt=torch.load(ckpt_path,map_location=device,weights_only=False)
    model=DTSTPINN(cfg,cfg.material).to(device); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    return model


def selection_audit(curve_rows, budget):
    evals=[r for r in curve_rows if int(r.get("update",-1))>0]
    pass_rows=[r for r in evals if float(r.get("rmse",float("inf")))<=budget]
    best_f1=max(evals,key=lambda r: float(r.get("solidus_f1",0.0))) if evals else None
    best_rmse=min(evals,key=lambda r: float(r.get("rmse",float("inf")))) if evals else None
    return dict(num_eval_checkpoints=len(evals), num_rmse_budget_pass=len(pass_rows), all_checkpoints_participated=True,
                final_update=max(int(r["update"]) for r in evals) if evals else None,
                best_f1_update=best_f1.get("update") if best_f1 else None, best_f1=best_f1.get("solidus_f1") if best_f1 else None, best_f1_rmse=best_f1.get("rmse") if best_f1 else None,
                best_rmse_update=best_rmse.get("update") if best_rmse else None, best_rmse=best_rmse.get("rmse") if best_rmse else None, best_rmse_f1=best_rmse.get("solidus_f1") if best_rmse else None)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run_dir",type=Path,default=Path("results/training_diagnostics/multwindow_3x3_seed42")); ap.add_argument("--output_dir",type=Path,default=None)
    ap.add_argument("--vtu_dir",default="F:/VTU"); ap.add_argument("--cache_dir",default="data/processed"); ap.add_argument("--laser_xml",default="configs/laser_paths/5_block_fem_additive_z_scan.xml")
    ap.add_argument("--experiments",nargs="+",default=["E0","E1","E2"],choices=list(EXPERIMENTS)); ap.add_argument("--modes",nargs="+",default=list(AUDIT_MODES),choices=list(AUDIT_MODES))
    ap.add_argument("--alpha",type=float,default=0.1); ap.add_argument("--rmse_budget",type=float,default=19.373114702713575); ap.add_argument("--total_updates",type=int,default=2000)
    ap.add_argument("--device",default="auto",choices=["auto","cpu","cuda"]); ap.add_argument("--graph_device",default="auto",choices=["auto","cpu","cuda"]); ap.add_argument("--train_scope",choices=["key","all","none"],default="key")
    args=ap.parse_args(); out=args.output_dir or (args.run_dir/"fp_weight_audit_alpha0p1"); out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else (args.device if args.device!="auto" else "cpu")); graph_device=device if args.graph_device=="auto" else torch.device(args.graph_device)
    split=load_split(args.run_dir/"small_split.json")
    cfg_args=argparse.Namespace(laser_xml=args.laser_xml,lr_start=1e-3,weight_decay=1e-5,grad_clip=1.0,lambda_pde=None)
    ref=configure("E0","mse",cfg_args); graph=load_graph(args.vtu_dir,args.cache_dir,ref,graph_device,False)
    all_agg=[]; all_win=[]; sel_rows=[]; weight_rows=[]
    for exp in args.experiments:
      for mode in args.modes:
        expdir=args.run_dir/f"{exp}_{mode}"; curve_path=expdir/"validation_curve.csv"
        if not curve_path.exists(): print("missing",curve_path); continue
        curve=read_csv(curve_path); sel=selection_audit(curve,args.rmse_budget); sel.update(experiment=exp,mode=mode,rmse_budget=args.rmse_budget); sel_rows.append(sel)
        cfg=configure(exp,mode,cfg_args); tr=split["train_starts"]; va=split["val_starts"]; trds=make_dataset(graph,cfg,tr); vads=make_dataset(graph,cfg,va); cfg.model.node_feature_dim=trds.input_feature_dim
        loss_fn=DTSTPINNLoss(cfg,cfg.material) if mode=="c_hot_pde" else None; ckpts=checkpoints(expdir)
        train_updates=set()
        if args.train_scope=="all": train_updates=set(ckpts.keys())-{0}
        elif args.train_scope=="key": train_updates=key_train_updates(curve,args.total_updates)
        eval_plan=[("val",u) for u in sorted(k for k in ckpts if k>0)] + [("train",u) for u in sorted(train_updates) if u in ckpts]
        for split_name,u in eval_plan:
            model=load_model_for_ckpt(ckpts[u],cfg,device); ds,starts=(vads,va) if split_name=="val" else (trds,tr)
            agg,rows=eval_split(model,ds,starts,cfg,mode,loss_fn,args.alpha,u,split_name); agg.update(experiment=exp,mode=mode,mode_label=MODE_LABEL[mode]); all_agg.append(agg)
            for r in rows: r.update(experiment=exp,mode=mode); all_win.append(r)
            del model
            if device.type=="cuda": torch.cuda.empty_cache()
        for split_name, ds, starts in [("train",trds,tr),("val",vads,va)]:
            for i,s in enumerate(starts):
                sample=ds[i]; n=int(sample["target_mask"].bool().sum().cpu())
                target=sample["target"].detach().float().flatten(); mask=sample["target_mask"].detach().bool().flatten(); nh=int(((target>=cfg.material.solidus_temp)&mask).sum().cpu())
                weight_rows.append(dict(experiment=exp,mode=mode,split=split_name,start=int(s),valid_nodes=n,hot_nodes=nh,effective_hot_node_coeff_alpha_0p1=((1+0.1*n/nh) if nh>0 else float("nan")),effective_hot_node_coeff_alpha_0p03=((1+0.03*n/nh) if nh>0 else float("nan"))))
    write_csv(out/"selection_audit.csv",sel_rows); write_csv(out/"fp_bins_by_checkpoint.csv",all_agg); write_csv(out/"fp_by_window.csv",all_win); write_csv(out/"hot_weight_by_window.csv",weight_rows)
    final=[r for r in all_agg if r["split"]=="val" and int(r["update"])==args.total_updates]
    lines=["# alpha=0.1 C/C+PDE false-positive and weight audit\n\n","This audit only loads existing checkpoints and runs forward evaluation; no training is performed.\n\n","## final@2000 validation FP bins\n\n","| Exp | Mode | RMSE | hot MAE | F1 | TP/FP/FN | no-hot-window FP | FP<500 | FP 500-1000 | FP 1000-solidus |\n","|---|---|---:|---:|---:|---|---:|---:|---:|---:|\n"]
    for r in sorted(final,key=lambda x:(x["experiment"],x["mode"])):
        lines.append(f"| {r['experiment']} | {r['mode']} | {float(r['rmse']):.3f} | {float(r['hot_mae']):.3f} | {float(r['solidus_f1']):.3f} | {r['solidus_tp']}/{r['solidus_fp']}/{r['solidus_fn']} | {r['no_hot_fp_total']} | {r['fp_true_lt500']} | {r['fp_true_500_1000']} | {r['fp_true_1000_solidus']} |\n")
    lines.append("\nDetailed tables: fp_bins_by_checkpoint.csv, fp_by_window.csv, hot_weight_by_window.csv, selection_audit.csv.\n")
    (out/"fp_weight_audit_report.md").write_text("".join(lines),encoding="utf-8"); print(out/"fp_weight_audit_report.md")

if __name__=="__main__": main()
