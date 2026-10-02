"""Multi-window E0/E1/E2 x MSE/C/C+PDE diagnostic validation."""
from __future__ import annotations

import argparse, json, math, sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.training_diagnostics import EXPERIMENTS, disable_dropout, grad_norm, load_graph, make_dataset, set_seed, write_csv
from scripts.training_diagnostics_stage5 import DTSTPINNLoss, common_forward, loss_forward_args
from src.config import Config
from src.model import DTSTPINN

MODES = ("mse", "c_hot", "c_hot_pde")
MODE_LABEL = {"mse": "full MSE", "c_hot": "MSE+0.1 hot MSE", "c_hot_pde": "MSE+0.1 hot MSE+PDE"}
DEFAULT_TRAIN_BLOCKS = [[118,121],[160,162],[208,211],[421,424],[634,637],[847,850],[1061,1064],[1274,1277]]
DEFAULT_VAL_BLOCKS = [[40,41],[80,81],[370,371],[477,478],[615,616],[930,931],[1487,1488]]


def expand_blocks(blocks: list[list[int]]) -> list[int]:
    out=[]
    for a,b in blocks: out += list(range(int(a), int(b)+1))
    return out


def frame_set(starts: Iterable[int], window: int, pred: int) -> set[int]:
    fs=set(); span=window+pred
    for s in starts: fs.update(range(int(s), int(s)+span))
    return fs


def load_json(path: Path) -> Any | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def original_train_starts(path: Path) -> list[int] | None:
    d=load_json(path)
    if not d: return None
    v=d.get("train_indices") or d.get("train") or d.get("train_starts")
    return [int(x) for x in v] if v is not None else None


def parse_blocks(text: str | None, default: list[list[int]]) -> list[list[int]]:
    return deepcopy(default) if not text else [[int(a), int(b)] for a,b in json.loads(text)]


def sample_stats(graph: Any, cfg: Config, start: int) -> dict[str, Any]:
    tgt=int(start)+cfg.data.window_size+cfg.data.predict_steps-1
    g=graph.get_graph_at(tgt); y=g.y.detach().float().flatten(); m=g.mask.detach().bool().flatten(); v=y[m]
    hot=v>=cfg.material.solidus_temp; liq=v>=cfg.material.liquidus_temp
    return dict(start=int(start), target_step=tgt, frame_start=int(start), frame_end=tgt,
                valid_nodes=int(m.sum().cpu()), hot_nodes=int(hot.sum().cpu()), liquid_nodes=int(liq.sum().cpu()),
                target_max=float(v.max().cpu()), target_mean=float(v.mean().cpu()))


def split_summary(rows):
    hs=[r["hot_nodes"] for r in rows]
    return dict(num_windows=len(rows), start_min=min(r["start"] for r in rows), start_max=max(r["start"] for r in rows),
                target_step_min=min(r["target_step"] for r in rows), target_step_max=max(r["target_step"] for r in rows),
                hot_windows=sum(h>0 for h in hs), total_hot_nodes=sum(hs), max_hot_nodes_per_window=max(hs),
                max_target_temp=max(r["target_max"] for r in rows))


def audit_old(path: Path, cfg: Config) -> dict[str, Any]:
    d=load_json(path)
    if not d: return dict(path=str(path), exists=False, reusable=False, reason="not found")
    tr=[int(x) for x in (d.get("train_starts") or d.get("train") or d.get("train_indices") or [])]
    va=[int(x) for x in (d.get("val_starts") or d.get("val") or d.get("val_indices") or [])]
    ov=sorted(frame_set(tr,cfg.data.window_size,cfg.data.predict_steps) & frame_set(va,cfg.data.window_size,cfg.data.predict_steps))
    tr_cont=(tr==list(range(min(tr),max(tr)+1))) if tr else False; va_cont=(va==list(range(min(va),max(va)+1))) if va else False
    reusable=bool(tr and va and not ov and tr_cont and va_cont)
    return dict(path=str(path), exists=True, train_windows=len(tr), val_windows=len(va), train_contiguous_single_block=tr_cont,
                val_contiguous_single_block=va_cont, frame_overlap_count=len(ov), frame_overlap_first20=ov[:20],
                reusable=reusable, reason="ok" if reusable else "not continuous blocks and/or train-val frame overlap")


def make_split(graph, cfg, train_blocks, val_blocks, orig_train, old_path):
    tr=expand_blocks(train_blocks); va=expand_blocks(val_blocks)
    ov=sorted(frame_set(tr,cfg.data.window_size,cfg.data.predict_steps) & frame_set(va,cfg.data.window_size,cfg.data.predict_steps))
    outside=[] if orig_train is None else sorted([s for s in tr+va if s not in set(orig_train)])
    if ov: raise RuntimeError(f"new split has train/val frame overlap: {ov[:20]}")
    if outside: raise RuntimeError(f"starts outside original train split: {outside[:20]}")
    tr_rows=[sample_stats(graph,cfg,s) for s in tr]; va_rows=[sample_stats(graph,cfg,s) for s in va]
    audit=dict(protocol="multiwindow_3x3_seed42_diagnostic", window_size=cfg.data.window_size, predict_steps=cfg.data.predict_steps,
               sample_frame_span=cfg.data.window_size+cfg.data.predict_steps, train_blocks=train_blocks, val_blocks=val_blocks,
               train_starts=tr, val_starts=va, train_val_frame_overlap_count=len(ov), train_val_frame_overlap_first20=ov,
               all_starts_inside_original_train=(len(outside)==0) if orig_train is not None else None,
               outside_original_train_starts=outside, formal_val_test_sealed=True, old_small_split_audit=audit_old(old_path,cfg),
               train_summary=split_summary(tr_rows), val_summary=split_summary(va_rows))
    return audit,tr_rows,va_rows


def write_split_report(outdir: Path, audit, tr_rows, va_rows):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir/"small_split.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    write_csv(outdir/"small_split_train_windows.csv", tr_rows); write_csv(outdir/"small_split_val_windows.csv", va_rows)
    lines=["# 多窗口诊断 split 审计\n\n",
           f"- 新 split: train {audit['train_summary']['num_windows']} 窗口, val {audit['val_summary']['num_windows']} 窗口。\n",
           f"- train/val 完整帧重叠数: {audit['train_val_frame_overlap_count']}。\n",
           f"- 所有 start 均来自原正式训练 split: {audit['all_starts_inside_original_train']}。\n"]
    old=audit["old_small_split_audit"]
    lines.append(f"- 旧小 split 可复用: {old.get('reusable')}; 原因: {old.get('reason')}; 帧重叠数: {old.get('frame_overlap_count')}。\n\n")
    lines.append("| split | windows | hot windows | hot nodes | max target C | start range | target range |\n|---|---:|---:|---:|---:|---|---|\n")
    for name in ["train","val"]:
        s=audit[f"{name}_summary"]
        lines.append(f"| {name} | {s['num_windows']} | {s['hot_windows']} | {s['total_hot_nodes']} | {s['max_target_temp']:.2f} | {s['start_min']}-{s['start_max']} | {s['target_step_min']}-{s['target_step_max']} |\n")
    (outdir/"small_split_audit.md").write_text("".join(lines),encoding="utf-8")


def configure(exp: str, mode: str, args) -> Config:
    cfg=Config.from_yaml(EXPERIMENTS[exp])
    if args.laser_xml:
        cfg.data.laser_xml_path=args.laser_xml; cfg.data.laser_path_mode="additive_z_scan"
    disable_dropout(cfg); cfg.training.use_amp=False; cfg.training.lr=args.lr_start; cfg.training.weight_decay=args.weight_decay; cfg.training.grad_clip=args.grad_clip
    cfg.physics.heat_conduction=(mode=="c_hot_pde"); cfg.physics.boundary_convection=False; cfg.physics.initial_condition=False; cfg.loss.lambda_smooth=0.0
    if args.lambda_pde is not None: cfg.loss.lambda_PDE=args.lambda_pde
    keep={"lambda_T","lambda_PDE","lambda_BC","lambda_IC","lambda_smooth"}
    for n in list(vars(cfg.loss).keys()):
        if n.startswith("lambda_") and n not in keep:
            try: setattr(cfg.loss,n,0.0)
            except Exception: pass
    return cfg


def build_model(cfg, device):
    model=DTSTPINN(cfg,cfg.material).to(device); model.train()
    return model, torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)


def lr_at(update, total, constant, lr0, lr1):
    if update<=constant: return float(lr0)
    f=min(max((update-constant)/max(1,total-constant),0.0),1.0)
    return float(lr0+(lr1-lr0)*f)


def set_lr(opt, lr):
    for g in opt.param_groups: g["lr"]=float(lr)


def hot_idx(sample, solidus):
    t=sample["target"].detach().float().flatten(); m=sample["target_mask"].detach().bool().flatten()
    return torch.nonzero(m & (t>=solidus), as_tuple=False).flatten()


def calc_loss(model, sample, cfg, mode, loss_fn, alpha):
    out,pred=common_forward(model,sample); target=sample["target"].float()
    if pred.numel()==target.numel(): pred=pred.reshape_as(target)
    mask=sample["target_mask"].bool(); full=((pred[mask]-target[mask])**2).mean()
    hi=hot_idx(sample,cfg.material.solidus_temp); pf=pred.flatten(); tf=target.flatten()
    hm=((pf[hi]-tf[hi])**2).mean() if hi.numel()>0 else full.new_tensor(0.0)
    pde=full.new_tensor(0.0); loss=full
    if mode in {"c_hot","c_hot_pde"}: loss=loss+alpha*hm
    if mode=="c_hot_pde" and loss_fn is not None:
        _tot, formal=loss_fn.forward(**loss_forward_args(out,pred,sample)); pde=formal.get("PDE",pde); loss=loss+pde
    comps={"loss_full_mse":float(full.detach().cpu()),"loss_hot_mse":float(hm.detach().cpu()) if hi.numel()>0 else 0.0,
           "loss_alpha_hot_mse":float((alpha*hm).detach().cpu()) if hi.numel()>0 else 0.0,
           "loss_pde_weighted":float(pde.detach().cpu()),"loss_total":float(loss.detach().cpu()),"hot_nodes_in_sample":int(hi.numel())}
    return loss, comps, pred


def window_metrics(pred, sample, cfg, start, comps=None):
    solidus=float(cfg.material.solidus_temp); liquidus=float(cfg.material.liquidus_temp); target=sample["target"].detach().float()
    if pred.numel()==target.numel(): pred=pred.reshape_as(target)
    p=pred.detach().float().flatten(); t=target.flatten(); valid=sample["target_mask"].detach().bool().flatten(); coords=sample["coords"].detach().float()
    pv=p[valid]; tv=t[valid]; err=pv-tv; abs_e=err.abs(); sq=err.pow(2); th=tv>=solidus; ph=pv>=solidus
    tp=int((th&ph).sum().cpu()); fp=int((~th&ph).sum().cpu()); fn=int((th&~ph).sum().cpu())
    prec=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0; f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
    hot_full=valid & (t>=solidus); bg_full=valid & (~hot_full); hot_abs=(p[hot_full]-t[hot_full]).abs(); bg_abs=(p[bg_full]-t[bg_full]).abs()
    vidx=torch.nonzero(valid,as_tuple=False).flatten(); plocal=int(torch.argmax(pv).cpu()) if pv.numel() else -1; tlocal=int(torch.argmax(tv).cpu()) if tv.numel() else -1
    pidx=vidx[plocal] if plocal>=0 else torch.tensor(-1,device=valid.device); tidx=vidx[tlocal] if tlocal>=0 else torch.tensor(-1,device=valid.device)
    nd=float("nan"); pis=False
    if hot_full.any() and int(pidx)>=0:
        d=torch.linalg.norm(coords[torch.nonzero(hot_full,as_tuple=False).flatten()]-coords[int(pidx)].view(1,-1),dim=1); nd=float(d.min().cpu()); pis=bool(hot_full[int(pidx)].cpu())
    row=dict(start=int(start), target_step=int(sample.get("target_step", int(start)+cfg.data.window_size+cfg.data.predict_steps-1)),
             valid_nodes=int(valid.sum().cpu()), true_hot_nodes=int(th.sum().cpu()), true_liquid_nodes=int((tv>=liquidus).sum().cpu()),
             mae=float(abs_e.mean().cpu()), rmse=float(torch.sqrt(sq.mean()).cpu()), sq_sum=float(sq.sum().cpu()), abs_sum=float(abs_e.sum().cpu()),
             hot_abs_sum=float(hot_abs.sum().cpu()) if hot_abs.numel() else 0.0, hot_sq_sum=float(((p[hot_full]-t[hot_full])**2).sum().cpu()) if hot_full.any() else 0.0,
             hot_mae=float(hot_abs.mean().cpu()) if hot_abs.numel() else float("nan"), background_mae=float(bg_abs.mean().cpu()), background_abs_sum=float(bg_abs.sum().cpu()), background_count=int(bg_full.sum().cpu()),
             solidus_tp=tp, solidus_fp=fp, solidus_fn=fn, solidus_precision=prec, solidus_recall=rec, solidus_f1=f1,
             target_max=float(tv.max().cpu()), pred_max=float(pv.max().cpu()), peak_abs_error=float((pv.max()-tv.max()).abs().cpu()),
             pred_peak_node=int(pidx.cpu()), true_peak_node=int(tidx.cpu()), pred_peak_nearest_true_hot_dist_mm=nd, pred_peak_is_true_hot_node=pis)
    if comps: row.update(comps)
    return row


def aggregate(rows):
    vn=sum(r["valid_nodes"] for r in rows); ss=sum(r["sq_sum"] for r in rows); aa=sum(r["abs_sum"] for r in rows)
    hn=sum(r["true_hot_nodes"] for r in rows); ha=sum(r["hot_abs_sum"] for r in rows); hs=sum(r["hot_sq_sum"] for r in rows)
    bc=sum(r["background_count"] for r in rows); ba=sum(r["background_abs_sum"] for r in rows)
    tp=sum(r["solidus_tp"] for r in rows); fp=sum(r["solidus_fp"] for r in rows); fn=sum(r["solidus_fn"] for r in rows)
    prec=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0; f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
    peaks=[r["peak_abs_error"] for r in rows if not math.isnan(r["peak_abs_error"])]
    return dict(val_windows=len(rows), valid_nodes=vn, hot_windows=sum(r["true_hot_nodes"]>0 for r in rows), true_hot_nodes=hn,
                mae=aa/vn, rmse=math.sqrt(ss/vn), hot_mae=ha/hn if hn else float("nan"), hot_rmse=math.sqrt(hs/hn) if hn else float("nan"),
                background_mae=ba/bc if bc else float("nan"), solidus_tp=tp, solidus_fp=fp, solidus_fn=fn,
                solidus_precision=prec, solidus_recall=rec, solidus_f1=f1, mean_peak_abs_error=float(np.mean(peaks)) if peaks else float("nan"),
                worst_peak_abs_error=float(np.max(peaks)) if peaks else float("nan"))


def evaluate(model, ds, starts, cfg, mode, loss_fn, alpha, update):
    was=model.training; model.eval(); per=[]; sums={}
    with torch.no_grad():
        for i,s in enumerate(starts):
            sample=ds[i]; loss, comps, pred=calc_loss(model,sample,cfg,mode,loss_fn,alpha)
            row=window_metrics(pred,sample,cfg,int(s),comps); row.update(update=int(update), eval_loss=float(loss.detach().cpu()))
            per.append(row)
            for k,v in comps.items():
                if isinstance(v,(int,float)): sums[k]=sums.get(k,0.0)+float(v)
    if was: model.train()
    agg=aggregate(per); agg["update"]=int(update)
    for k,v in sums.items(): agg[f"mean_{k}"]=v/max(1,len(per))
    return agg, per


def save_ckpt(path, exp, mode, update, model, opt, args, split, lr):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(experiment=exp, mode=mode, update=int(update), model_state_dict=model.state_dict(), optimizer_state_dict=opt.state_dict(),
                    scheduler_state=dict(type="manual_linear_after_constant", lr_start=args.lr_start, lr_final=args.lr_final, constant_updates=args.lr_constant_updates, total_updates=args.total_updates, current_lr=lr),
                    args=vars(args), split=split), path)


def run_one(exp, mode, graph, split, args, device, outdir):
    print(f"=== {exp}/{mode} ===", flush=True); set_seed(args.seed); cfg=configure(exp,mode,args)
    tr=[int(x) for x in split["train_starts"]]; va=[int(x) for x in split["val_starts"]]
    trds=make_dataset(graph,cfg,tr); vads=make_dataset(graph,cfg,va); cfg.model.node_feature_dim=trds.input_feature_dim
    model,opt=build_model(cfg,device); loss_fn=DTSTPINNLoss(cfg,cfg.material) if mode=="c_hot_pde" else None; edir=outdir/f"{exp}_{mode}"; edir.mkdir(parents=True,exist_ok=True)
    eval_rows=[]; per_all=[]
    agg,per=evaluate(model,vads,va,cfg,mode,loss_fn,args.alpha,0); agg.update(experiment=exp,mode=mode,mode_label=MODE_LABEL[mode],lr=args.lr_start,input_feature_dim=int(cfg.model.node_feature_dim),target_laser_feature_dim=int(cfg.model.node_feature_dim)-12,checkpoint="initial")
    eval_rows.append(agg); [r.update(experiment=exp,mode=mode) for r in per]; per_all+=per; save_ckpt(edir/"checkpoint_update_0000.pt",exp,mode,0,model,opt,args,split,args.lr_start)
    for update in range(1,args.total_updates+1):
        lr=lr_at(update,args.total_updates,args.lr_constant_updates,args.lr_start,args.lr_final); set_lr(opt,lr)
        idx=(update-1)%len(trds); sample=trds[idx]; model.train(); opt.zero_grad(set_to_none=True)
        loss, comps, _=calc_loss(model,sample,cfg,mode,loss_fn,args.alpha); loss.backward(); gnorm=grad_norm(model); clipped=bool(cfg.training.grad_clip>0 and gnorm>cfg.training.grad_clip)
        if cfg.training.grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        opt.step()
        if update%args.eval_every==0 or update==args.total_updates:
            agg,per=evaluate(model,vads,va,cfg,mode,loss_fn,args.alpha,update); ck=f"checkpoint_update_{update:04d}.pt"; save_ckpt(edir/ck,exp,mode,update,model,opt,args,split,lr)
            agg.update(experiment=exp,mode=mode,mode_label=MODE_LABEL[mode],lr=lr,input_feature_dim=int(cfg.model.node_feature_dim),target_laser_feature_dim=int(cfg.model.node_feature_dim)-12,
                       train_start_current=int(tr[idx]), train_target_current=int(tr[idx]+cfg.data.window_size+cfg.data.predict_steps-1), grad_norm_pre_clip_last=gnorm, grad_clip_applied_last=clipped,
                       amp_enabled=False, amp_step_skipped=False, checkpoint=ck, **{f"train_{k}_last":v for k,v in comps.items()})
            eval_rows.append(agg); [r.update(experiment=exp,mode=mode) for r in per]; per_all+=per
            print(f"{exp}/{mode} upd {update}/{args.total_updates} rmse={agg['rmse']:.3f} hot_mae={agg['hot_mae']:.3f} F1={agg['solidus_f1']:.3f} P/R={agg['solidus_precision']:.3f}/{agg['solidus_recall']:.3f} FP/FN={agg['solidus_fp']}/{agg['solidus_fn']} bg={agg['background_mae']:.3f} lr={lr:.2e}", flush=True)
    write_csv(edir/"validation_curve.csv", eval_rows); write_csv(edir/"validation_per_window.csv", per_all)
    del model,opt
    if device.type=="cuda": torch.cuda.empty_cache()
    return eval_rows, per_all


def choose(rows, budget):
    c=[r for r in rows if int(r.get("update",-1))>0 and float(r.get("rmse",float("inf")))<=budget]
    def h(r):
        v=float(r.get("hot_mae",float("inf"))); return v if not math.isnan(v) else float("inf")
    if not c: return dict(selection_passed=False, rmse_budget=budget)
    s=sorted(c,key=lambda r:(-float(r.get("solidus_f1",0.0)),h(r),float(r.get("rmse",float("inf")))))[0]
    keep=["update","rmse","mae","hot_mae","background_mae","solidus_f1","solidus_precision","solidus_recall","solidus_tp","solidus_fp","solidus_fn","mean_peak_abs_error","worst_peak_abs_error","checkpoint"]
    return {"selection_passed":True,"rmse_budget":budget, **{f"selected_{k}":s.get(k) for k in keep}}


def summarize(outdir, all_eval, args):
    base_rows=all_eval.get(("E0","mse"),[]); final0=[r for r in base_rows if int(r.get("update",-1))==args.total_updates]
    base_rmse=float(final0[-1]["rmse"]) if final0 else float("nan"); budget=base_rmse*1.25 if not math.isnan(base_rmse) else float("inf")
    rows=[]
    order={"mse":0,"c_hot":1,"c_hot_pde":2}
    for (exp,mode),rs in all_eval.items():
        fin=([r for r in rs if int(r.get("update",-1))==args.total_updates] or [rs[-1]])[-1]; sel=choose(rs,budget)
        row=dict(experiment=exp,mode=mode,mode_label=MODE_LABEL[mode],rmse_budget_source=f"1.25*E0_mse_final_update{args.total_updates}",e0_mse_final_rmse_reference=base_rmse,rmse_budget=budget,
                 final_update=fin.get("update"),final_rmse=fin.get("rmse"),final_mae=fin.get("mae"),final_hot_mae=fin.get("hot_mae"),final_background_mae=fin.get("background_mae"),
                 final_f1=fin.get("solidus_f1"),final_precision=fin.get("solidus_precision"),final_recall=fin.get("solidus_recall"),final_tp=fin.get("solidus_tp"),final_fp=fin.get("solidus_fp"),final_fn=fin.get("solidus_fn"),
                 final_mean_peak_abs_error=fin.get("mean_peak_abs_error"),final_worst_peak_abs_error=fin.get("worst_peak_abs_error"))
        row.update(sel); rows.append(row)
    rows.sort(key=lambda r:(r["experiment"],order[r["mode"]])); write_csv(outdir/"summary_3x3.csv", rows)
    def fmt(x,n=3):
        try:
            y=float(x); return "nan" if math.isnan(y) else f"{y:.{n}f}"
        except Exception: return ""
    lines=["# 多窗口 3x3 诊断结果\n\n", "本轮是训练集内部小 split 诊断，不替代正式 val/test 泛化结论。\n\n",
           f"协议: seed={args.seed}, 从头初始化, {args.total_updates} optimizer updates, 每 {args.eval_every} updates 评估; LR 前 {args.lr_constant_updates} 为 {args.lr_start:g}, 后续线性到 {args.lr_final:g}; C=full MSE+{args.alpha:g} hot MSE; BC/IC/Smooth/Dropout/AMP 关闭, 仅 c_hot_pde 开 PDE。\n\n",
           f"选模预算: 1.25 x E0-MSE final RMSE = {budget:.4f}。\n\n",
           "| Exp | Mode | final RMSE | final hot MAE | final bg MAE | final F1 | final P/R | final FP/FN | pass | selected upd | selected RMSE | selected hot MAE | selected bg MAE | selected F1 | selected P/R | selected FP/FN |\n",
           "|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|---|\n"]
    for r in rows:
        lines.append(f"| {r['experiment']} | {r['mode']} | {fmt(r.get('final_rmse'))} | {fmt(r.get('final_hot_mae'))} | {fmt(r.get('final_background_mae'))} | {fmt(r.get('final_f1'))} | {fmt(r.get('final_precision'))}/{fmt(r.get('final_recall'))} | {r.get('final_fp')}/{r.get('final_fn')} | {r.get('selection_passed')} | {r.get('selected_update','')} | {fmt(r.get('selected_rmse'))} | {fmt(r.get('selected_hot_mae'))} | {fmt(r.get('selected_background_mae'))} | {fmt(r.get('selected_solidus_f1'))} | {fmt(r.get('selected_solidus_precision'))}/{fmt(r.get('selected_solidus_recall'))} | {r.get('selected_solidus_fp','')}/{r.get('selected_solidus_fn','')} |\n")
    lines.append("\n逐 checkpoint 与逐窗口数据见各实验目录的 validation_curve.csv 和 validation_per_window.csv。\n")
    (outdir/"multwindow_3x3_report.md").write_text("".join(lines),encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output_dir",type=Path,default=Path("results/training_diagnostics/multwindow_3x3_seed42")); p.add_argument("--vtu_dir",default="F:/VTU"); p.add_argument("--cache_dir",default="data/processed")
    p.add_argument("--laser_xml",default="configs/laser_paths/5_block_fem_additive_z_scan.xml"); p.add_argument("--split_artifact",type=Path,default=Path("artifacts/baselines/feature_e0_baseline_50epoch_20260908T090218Z/split_indices.json")); p.add_argument("--old_small_split",type=Path,default=Path("tmp/protocol_checks/small_learning_train_only_split.json"))
    p.add_argument("--train_blocks",default=None); p.add_argument("--val_blocks",default=None); p.add_argument("--experiments",nargs="+",default=["E0","E1","E2"],choices=list(EXPERIMENTS)); p.add_argument("--modes",nargs="+",default=list(MODES),choices=list(MODES))
    p.add_argument("--seed",type=int,default=42); p.add_argument("--device",default="auto",choices=["auto","cpu","cuda"]); p.add_argument("--graph_device",default="auto",choices=["auto","cpu","cuda"])
    p.add_argument("--total_updates",type=int,default=2000); p.add_argument("--eval_every",type=int,default=100); p.add_argument("--lr_start",type=float,default=1e-3); p.add_argument("--lr_final",type=float,default=1e-4); p.add_argument("--lr_constant_updates",type=int,default=500)
    p.add_argument("--alpha",type=float,default=0.1); p.add_argument("--lambda_pde",type=float,default=None); p.add_argument("--weight_decay",type=float,default=1e-5); p.add_argument("--grad_clip",type=float,default=1.0); p.add_argument("--no_cache",action="store_true")
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else (args.device if args.device!="auto" else "cpu")); graph_device=device if args.graph_device=="auto" else torch.device(args.graph_device)
    ref=configure("E0","mse",args); graph=load_graph(args.vtu_dir,args.cache_dir,ref,graph_device,args.no_cache)
    split,tr_rows,va_rows=make_split(graph,ref,parse_blocks(args.train_blocks,DEFAULT_TRAIN_BLOCKS),parse_blocks(args.val_blocks,DEFAULT_VAL_BLOCKS),original_train_starts(args.split_artifact),args.old_small_split)
    write_split_report(args.output_dir,split,tr_rows,va_rows)
    manifest={**vars(args),"device":str(device),"graph_device":str(graph_device),"rmse_budget_rule":f"1.25*E0_mse_final_update{args.total_updates}","bc_ic_smooth_dropout":"disabled","amp":"disabled"}
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str)+"\n",encoding="utf-8")
    all_eval={}; all_per=[]
    for exp in args.experiments:
        for mode in args.modes:
            er,pr=run_one(exp,mode,graph,split,args,device,args.output_dir); all_eval[(exp,mode)]=er; all_per+=pr
    write_csv(args.output_dir/"validation_curve_all.csv",[r for rs in all_eval.values() for r in rs]); write_csv(args.output_dir/"validation_per_window_all.csv",all_per)
    summarize(args.output_dir,all_eval,args); print(args.output_dir/"multwindow_3x3_report.md",flush=True)

if __name__=="__main__": main()
