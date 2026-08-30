# DT-STPINN

Dynamic Twin Spatio-Temporal Physics-Informed Neural Network for temperature-field prediction in directed-energy-deposition / additive-manufacturing simulations.

This repository keeps source code, fixed experiment configs, lightweight summaries, and reproducible run scripts under version control. Raw VTU files, preprocessed tensors, checkpoints, logs, and bulky per-run artifacts are intentionally local-only.

## Project Layout

- `src/` - model, dataset, graph construction, physics losses, metrics, and laser path utilities.
- `scripts/` - training, preprocessing, checkpoint evaluation, laser trajectory diagnostics, plotting, and experiment runners.
- `configs/` - canonical configs for the paper baseline and the fixed E0-E3 trajectory-feature ablations.
- `docs/` - architecture notes and experiment plans.
- `results/*.md`, `results/*.csv` - small tracked summaries only.
- `data/`, `logs/`, `artifacts/` - generated local outputs ignored by git.

Current roadmap completion evidence and next commands are tracked in
`docs/ROADMAP_STATUS.md`.

## Data Policy

Keep the full Flow-3D / FEM VTU sequence outside git. On the current workstation the VTU directory is expected at:

```powershell
F:\VTU
```

The repository `.gitignore` excludes raw data, preprocessed caches, model weights, run logs, and generated result folders:

```text
data/raw/
data/processed/
logs/
artifacts/
*.pt
*.pth
*.ckpt
*.vtu
*.vtk
*.npy
*.npz
results/*
!results/*.md
!results/*.csv
configs/ablation_*.yaml
```

If a generated file becomes part of the documented protocol, promote a small summary file instead of committing the heavy artifact.

## Smoke Test

Use the pipeline test before launching long training jobs:

```powershell
python scripts\test_pipeline.py --config configs\paper1_fast.yaml --vtu_dir F:\VTU
```

For a short full-stack training check:

```powershell
python scripts\train.py `
  --config configs\paper1_fast.yaml `
  --vtu_dir F:\VTU `
  --epochs 5 `
  --max_train_samples 200 `
  --max_val_samples 30 `
  --max_test_samples 30 `
  --graph_device cuda
```

## Canonical Evaluation

Use `scripts/evaluate_checkpoint.py` as the single checkpoint evaluation protocol. It reports global regression metrics, solidus/liquidus detection metrics, laser-region errors, per-step peak-temperature errors, and worst-point diagnostics.

For long comparisons, reuse the frozen baseline split file in both training and evaluation. The baseline freeze script writes it as `artifacts/baselines/<baseline_name>/split_indices.json`; the artifact directory stays local, but the path should be recorded in experiment notes.

To rerun and freeze the single 50-epoch baseline control with one command, use:

```powershell
python scripts\run_baseline_protocol.py `
  --config configs\feature_e0_baseline.yaml `
  --vtu_dir F:\VTU `
  --laser_xml F:\datas\5-block-fem\para.xml `
  --epochs 50 `
  --graph_device cuda `
  --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json `
  --resume_from_last `
  --dry_run
```

Remove `--dry_run` to launch the long run. The wrapper trains with `scripts/train.py`, evaluates `best_model.pt` with `scripts/evaluate_checkpoint.py`, then freezes the checkpoint, config, evaluation report, split, seed, git commit, commands, and optional XML laser path with `scripts/freeze_baseline.py`. When `--split_indices` is provided, the freeze manifest records that exact frozen split as the source protocol.

```powershell
python scripts\evaluate_checkpoint.py `
  --config configs\feature_e0_baseline.yaml `
  --checkpoint logs\feature_e0_baseline\best_model.pt `
  --vtu_dir F:\VTU `
  --laser_xml F:\datas\5-block-fem\para.xml `
  --output_dir logs\feature_e0_baseline `
  --graph_device cuda `
  --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json
```

Required acceptance metrics for every long run:

- Global `RMSE`, `MAE`, `AbsErrorP95`, `AbsErrorP99`, `MaxError`.
- Solidus and liquidus `Precision`, `Recall`, `F1`, `FalseHot`, and `MissedHot`.
- Laser-region `MAE` and `MaxError`.
- Per-timestep peak-temperature prediction error.
- Worst 10 point diagnostics with coordinates, predicted/target temperature, laser distance, and arrival timing.

## Fixed E0-E3 Feature Ablation

The fixed trajectory-feature ablation family is:

- `E0` - baseline features only, config `configs/feature_e0_baseline.yaml`.
- `E1` - E0 plus laser coordinates and node-laser distance.
- `E2` - E1 plus along/cross scan distances and `time_to_arrival`.
- `E3` - E2 plus layer, track, direction, ellipsoid, and track-neighborhood indicators.

Before starting a long run, use the read-only preflight check:

```powershell
python scripts\check_experiment_readiness.py `
  --vtu_dir F:\VTU `
  --laser_xml F:\datas\5-block-fem\para.xml `
  --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json `
  --experiments E0 E1 E2 E3
```

Dry-run the protocol:

```powershell
python scripts\run_feature_ablation.py `
  --vtu_dir F:\VTU `
  --laser_xml F:\datas\5-block-fem\para.xml `
  --epochs 50 `
  --graph_device cuda `
  --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json `
  --resume_from_last `
  --dry_run
```

Run only the fixed baseline first:

```powershell
python scripts\run_feature_ablation.py `
  --experiments E0 `
  --vtu_dir F:\VTU `
  --laser_xml F:\datas\5-block-fem\para.xml `
  --epochs 50 `
  --graph_device cuda `
  --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json `
  --resume_from_last
```

Run the complete E0-E3 protocol:

```powershell
python scripts\run_feature_ablation.py `
  --vtu_dir F:\VTU `
  --laser_xml F:\datas\5-block-fem\para.xml `
  --epochs 50 `
  --graph_device cuda `
  --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json `
  --resume_from_last
```

## Minimal Head Ablations

After E0-E3 confirm whether real trajectory features improve hotspot recall, run the opt-in model-head ablations:

- `E4` - E3 plus a supervised solidus hotspot classification head.
- `E5` - E4 plus a minimal hotspot-gated high-temperature residual head, using `T_pred = T_base + P_hot * delta_hot`.

```powershell
python scripts\run_feature_ablation.py `
  --experiments E4 E5 `
  --vtu_dir F:\VTU `
  --laser_xml F:\datas\5-block-fem\para.xml `
  --epochs 50 `
  --graph_device cuda `
  --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json `
  --resume_from_last
```

E4/E5 evaluate their protocol `best_hot_model.pt` by default. E4 selects that checkpoint with `HotClsF1AboveSolidus`; E5 selects it with `TempF1AboveSolidus`.

## Laser Trajectory Diagnostics

Before feature ablations, diagnose path timing and scan-direction alignment:

```powershell
python scripts\diagnose_laser_alignment.py `
  --config configs\feature_e3_path_phase.yaml `
  --vtu_dir F:\VTU `
  --laser_xml F:\datas\5-block-fem\para.xml `
  --focus_step 2128 `
  --focus_node 24437 `
  --gate_coverage `
  --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json
```

Export and diagnose the laser path around the known worst hotspot:

```powershell
python scripts\export_laser_trajectory.py `
  --config configs\feature_e3_path_phase.yaml `
  --laser_xml F:\datas\5-block-fem\para.xml `
  --vtu_dir F:\VTU `
  --output_dir results\laser_trajectory_step2128_smoke `
  --diagnose_step_index 2128 `
  --diagnose_node_index 24437 `
  --hotspot_step_index 2128 `
  --hotspot_top_k 20
```

Plot laser position against high-temperature nodes:

```powershell
python scripts\plot_laser_hotspots.py `
  --config configs\feature_e3_path_phase.yaml `
  --laser_xml F:\datas\5-block-fem\para.xml `
  --vtu_dir F:\VTU `
  --steps 2128 `
  --output_dir results\laser_hotspot_alignment_smoke `
  --top_k 200 `
  --show_all_nodes `
  --zoom_radius_mm 2.0
```

Validate generated trajectory-feature chunks before using them for ablations:

```powershell
python scripts\check_laser_feature_dataset.py `
  --manifest data\processed\laser_features_smoke\manifest.json `
  --step 2128 `
  --node_index 24437
```

## Experiment Decisions

- If E1/E2 do not improve solidus recall, recheck laser time, coordinate, layer, and hatch-direction alignment before changing model structure.
- If recall improves but RMSE degrades sharply, tune classification threshold, sampling, and residual weights.
- If real trajectory features improve hotspot recall, stop threshold-only experiments and prioritize the hotspot auxiliary head plus conditional high-temperature residual head.
- The optimization target is not lowest global RMSE alone. The target is acceptable global RMSE with much lower hotspot miss rate and lower worst-case error.
