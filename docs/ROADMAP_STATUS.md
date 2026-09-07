# Roadmap Status

Last updated: 2026-09-07

This file tracks the current evidence for the fixed DT-STPINN experiment roadmap.
It is intentionally lightweight and should stay in git. Large checkpoints, VTU
files, logs, preprocessed tensors, and per-run artifacts remain local-only.

## Current Repository State

- Remote branch: `master` on `origin`.
- Protocol support in git includes XML path overrides, fixed split handling, feature ablation configs, checkpoint evaluation metrics, and XML candidate reporting in readiness checks.
- Important local-only data paths:
  - VTU sequence: `F:\VTU`.
  - Process XML expected by protocol: `F:\datas\5-block-fem\para.xml`.
- Current session note: `F:\VTU` is visible, frozen split/config checks pass, but `F:\datas\5-block-fem\para.xml` is still not visible. The readiness preflight now scans XML search roots and reports candidate XML paths when the configured file is missing.

## Objective Checklist

| Priority | Item | Status | Evidence | Remaining work |
| --- | --- | --- | --- | --- |
| 1 | Freeze current baseline | Partially done | Local artifact `artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z` contains checkpoint, config, split, seed, git commit, and evaluation report. | Rerun the single control with the new fixed protocol, `feature_e0_baseline.yaml`, frozen split, and XML path once `para.xml` is available. |
| 1 | Fixed split and evaluation script | Done in code | `src/data/preprocessing.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/evaluate_checkpoint.py`, and protocol wrappers accept `--split_indices`. | Use the same `split_indices.json` in every long experiment. |
| 2 | Close z24x | Done | `results/z241_z245_summary.md`, `results/z241_z245_summary.csv`, and `results/z241_z245_decision.md`. | Do not add z246+ threshold/gate tuning unless the roadmap changes. |
| 3 | Verify XML laser trajectory alignment | Blocked by missing XML in this session | `scripts/export_laser_trajectory.py`, `scripts/plot_laser_hotspots.py`, and `--laser_xml` protocol support exist. | Restore or point to the real `para.xml`, then run the XML alignment commands below. |
| 4 | Build trajectory feature dataset | Done in code, not fully generated | `scripts/build_laser_feature_dataset.py` and `scripts/check_laser_feature_dataset.py` cover E1/E2/E3 columns and split-aware generation. | Generate full train/val/test chunks from XML after trajectory alignment passes. |
| 5 | Minimal feature ablations E0-E3 | Ready to run | `configs/feature_e0_baseline.yaml` through `configs/feature_e3_path_phase.yaml`; `scripts/run_feature_ablation.py` supports fixed split, seed, XML, and checkpoint protocol. | Run E0 first as the only control, then E1-E3 under the same protocol. |
| 6 | Hotspot auxiliary task | Done in code | `configs/feature_e4_hotspot_aux.yaml`, model hot head support, trainer/evaluator solidus and liquidus metrics. | Run E4 only after E1-E3 confirm whether trajectory features help. |
| 7 | Conditional high-temperature residual head | Done in code | `configs/feature_e5_hotspot_residual.yaml`; `src/model.py` supports `T_pred = T_base + P_hot * delta_hot` via hotspot gate mix. | Run E5 after E4 under the same protocol. |
| 8 | Acceptance metrics | Done in code | `scripts/evaluate_checkpoint.py` reports global metrics, threshold metrics, laser-region metrics, per-step peak errors, and worst-point diagnostics. | Require these metrics in every reported comparison. |
| 9 | Continue/stop criteria | Documented | `README.md` experiment decisions and this status file. | Apply the criteria after fixed E0-E3 results exist. |

## Current Baseline Evidence

Frozen historical baseline:

```text
artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z
```

This artifact is useful as a historical reference, but it is not the final fixed
control for the new roadmap because its manifest records:

```text
config: configs\paper1_fast.yaml
train: python scripts/train.py --config configs/paper1_fast.yaml --vtu_dir F:\VTU --epochs 50 --graph_device cuda
evaluate: python scripts/evaluate_checkpoint.py --config configs/paper1_fast.yaml --checkpoint logs/paper1_temperature_fast/best_model.pt --vtu_dir F:\VTU --output_dir logs/paper1_temperature_fast --graph_device cuda
```

The final control should use `configs\feature_e0_baseline.yaml`, the frozen split,
and the XML laser path override.

## z241-z245 Decision

The z24x threshold/gate family should be closed. The best z245 result improved
hotspot recall but still relied on handcrafted rescue/floor behavior and retained
large worst-case errors. The next experiments should be real trajectory features
and then learned hotspot/residual heads, not more z246+ threshold tuning.

Key z245 test values from `results/z241_z245_summary.md`:

```text
test_RMSE: 13.3884
test_MAE: 3.2797
test_AbsErrorP99: 22.4071
test_MaxError: 2046
test_TempRecallAboveSolidus: 0.945504
test_TempPrecisionAboveSolidus: 0.363732
test_TempF1AboveSolidus: 0.52536
test_TempTPAboveSolidus: 347
test_TempFPAboveSolidus: 607
test_TempFNAboveSolidus: 20
```

## Next Commands

### 1. Run the read-only preflight

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\check_experiment_readiness.py `
  --vtu_dir F:\VTU `
  --laser_xml F:\datas\5-block-fem\para.xml `
  --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json `
  --experiments E0 E1 E2 E3
```

This check should pass before launching any 50-epoch control or feature
ablation. It is read-only and prints the dry-run baseline and ablation commands.

### 2. Verify XML path visibility

```powershell
Test-Path F:\datas\5-block-fem\para.xml
```

If this prints `False`, either restore the file at that path or pass the correct
XML path to `--laser_xml` / `--xml`.

### 3. Export and check trajectory alignment at the known worst point

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\export_laser_trajectory.py `
  --config configs\feature_e3_path_phase.yaml `
  --laser_xml F:\datas\5-block-fem\para.xml `
  --vtu_dir F:\VTU `
  --output_dir results\laser_trajectory_step2128_xml `
  --diagnose_step_index 2128 `
  --diagnose_node_index 24437 `
  --hotspot_step_index 2128 `
  --hotspot_top_k 20
```

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\plot_laser_hotspots.py `
  --config configs\feature_e3_path_phase.yaml `
  --laser_xml F:\datas\5-block-fem\para.xml `
  --vtu_dir F:\VTU `
  --steps 2128 `
  --output_dir results\laser_hotspot_alignment_xml `
  --top_k 200 `
  --show_all_nodes `
  --zoom_radius_mm 2.0
```

Pass criteria: at hotspot steps, the laser position should be close to the high-temperature region, with sensible layer index, track index, scan direction, and `time_to_arrival_s` around the failure node.

### 4. Rerun and freeze the single fixed E0 baseline

Start with a dry run:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\run_baseline_protocol.py `
  --config configs\feature_e0_baseline.yaml `
  --vtu_dir F:\VTU `
  --laser_xml F:\datas\5-block-fem\para.xml `
  --epochs 50 `
  --graph_device cuda `
  --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json `
  --resume_from_last `
  --dry_run
```

Remove `--dry_run` only when ready for the long run.

### 5. Run fixed E0-E3 feature ablations

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\run_feature_ablation.py `
  --experiments E0 E1 E2 E3 `
  --vtu_dir F:\VTU `
  --laser_xml F:\datas\5-block-fem\para.xml `
  --epochs 50 `
  --graph_device cuda `
  --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json `
  --resume_from_last
```

Only proceed to E4/E5 after E1/E2/E3 show whether trajectory features improve hotspot recall and worst-case errors.
