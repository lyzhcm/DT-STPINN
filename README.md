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

```powershell
python scripts\evaluate_checkpoint.py `
  --config configs\feature_e0_baseline.yaml `
  --checkpoint logs\feature_e0_baseline\best_model.pt `
  --vtu_dir F:\VTU `
  --output_dir logs\feature_e0_baseline `
  --graph_device cuda
```

Required acceptance metrics for every long run:

- Global `RMSE`, `MAE`, `AbsErrorP95`, `AbsErrorP99`, `MaxError`.
- Solidus and liquidus `Precision`, `Recall`, `F1`.
- Laser-region `MAE` and `MaxError`.
- Per-timestep peak-temperature prediction error.
- Worst 10 point diagnostics with coordinates, predicted/target temperature, laser distance, and arrival timing.

## Fixed E0-E3 Feature Ablation

The fixed trajectory-feature ablation family is:

- `E0` - baseline features only, config `configs/feature_e0_baseline.yaml`.
- `E1` - E0 plus laser coordinates and node-laser distance.
- `E2` - E1 plus along/cross scan distances and `time_to_arrival`.
- `E3` - E2 plus layer, track, direction, ellipsoid, and track-neighborhood indicators.

Dry-run the protocol:

```powershell
python scripts\run_feature_ablation.py `
  --vtu_dir F:\VTU `
  --epochs 50 `
  --graph_device cuda `
  --resume_from_last `
  --dry_run
```

Run only the fixed baseline first:

```powershell
python scripts\run_feature_ablation.py `
  --experiments E0 `
  --vtu_dir F:\VTU `
  --epochs 50 `
  --graph_device cuda `
  --resume_from_last
```

Run the complete E0-E3 protocol:

```powershell
python scripts\run_feature_ablation.py `
  --vtu_dir F:\VTU `
  --epochs 50 `
  --graph_device cuda `
  --resume_from_last
```

## Laser Trajectory Diagnostics

Export and diagnose the laser path around the known worst hotspot:

```powershell
python scripts\export_laser_trajectory.py `
  --config configs\feature_e3_path_phase.yaml `
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

