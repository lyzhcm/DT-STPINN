# Roadmap Status

Last updated: 2026-09-07

This file tracks the current evidence for the fixed DT-STPINN experiment roadmap.
It is intentionally lightweight and should stay in git. Large checkpoints, VTU
files, logs, preprocessed tensors, and per-run artifacts remain local-only.

## Current Repository State

- Remote branch: `master` on `origin`.
- Protocol support in git includes XML path overrides, fixed split handling, feature ablation configs, checkpoint evaluation metrics, XML candidate reporting/comparison tools, and default preflight guards in the long-run baseline/ablation wrappers.
- Important local-only data paths:
  - VTU sequence: `F:\VTU`.
  - Approved process XML for current smoke/protocol checks: `configs\laser_paths\5_block_fem_additive_z_scan.xml` (`F:\datas\5-block-fem\para.xml` remains the preferred original if restored).
- Current session note: `F:\VTU` is visible and frozen split/config checks pass. The original `F:\datas\5-block-fem\para.xml` is still not visible, but `configs\laser_paths\5_block_fem_additive_z_scan.xml` reconstructs the provided `additive_z_scan` process fields, parses successfully with `scripts\compare_laser_xml.py`, and passes readiness for E0-E3.

## Objective Checklist

| Priority | Item | Status | Evidence | Remaining work |
| --- | --- | --- | --- | --- |
| 1 | Freeze current baseline | Partially done | Local artifact `artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z` contains checkpoint, config, split, seed, git commit, and evaluation report; `scripts\verify_baseline_artifact.py` verifies hashes, split counts, commands, commit, and metrics visibility; `scripts\freeze_baseline.py` now runs artifact verification and evaluation-report acceptance verification by default after writing a baseline artifact. | Rerun the single control with the new fixed protocol, `feature_e0_baseline.yaml`, frozen split, and XML path, then let `scripts\freeze_baseline.py` verify the frozen artifact automatically. |
| 1 | Fixed split and evaluation script | Done in code | `src/data/preprocessing.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/evaluate_checkpoint.py`, and protocol wrappers accept `--split_indices`; `scripts/run_baseline_protocol.py` and `scripts/run_feature_ablation.py` now default to the frozen split and run readiness checks before training. | Use the same `split_indices.json` in every long experiment. |
| 2 | Close z24x | Done | `results/z241_z245_summary.md`, `results/z241_z245_summary.csv`, and `results/z241_z245_decision.md`. | Do not add z246+ threshold/gate tuning unless the roadmap changes. |
| 3 | Verify XML laser trajectory alignment | Partially done with reconstructed XML | `configs\laser_paths\5_block_fem_additive_z_scan.xml` passes readiness; `scripts\compare_laser_xml.py` compares XML-defined process fields against the original `para.xml`; `results\laser_alignment_reconstructed_xml_smoke.md` records step 2128 diagnosis and hotspot overlay smoke. | Restore or point to the original full `para.xml`, run `scripts\compare_laser_xml.py`, then repeat or approve this reconstruction as the fixed process XML for long runs. |
| 4 | Build trajectory feature dataset | Done in code, smoke-tested | `scripts/build_laser_feature_dataset.py` and `scripts/check_laser_feature_dataset.py` cover E1/E2/E3 columns and split-aware generation; `scripts/run_laser_feature_dataset_protocol.py` builds and validates train/val/test manifests; reconstructed-XML smoke for steps 2128/2129 validates OK; `--feature_group E1/E2/E3` validates canonical group columns before ablation runs. | Generate full train/val/test chunks from the approved XML path before long E1-E3 runs. |
| 5 | Minimal feature ablations E0-E3 | Ready to run, E0-E3 smoke-tested | `configs/feature_e0_baseline.yaml` through `configs/feature_e3_path_phase.yaml`; `scripts/run_feature_ablation.py` supports fixed split, seed, XML, checkpoint protocol, resume, default preflight, and post-evaluation report verification; `results\e0_e3_training_smoke.md` verifies E0-E3 training entrypoints with fixed split and reconstructed XML. | Run E0 first as the only long-run control, then E1-E3 under the same protocol. |
| 6 | Hotspot auxiliary task | Done in code | `configs/feature_e4_hotspot_aux.yaml`, model hot head support, trainer/evaluator solidus and liquidus metrics. | Run E4 only after E1-E3 confirm whether trajectory features help. |
| 7 | Conditional high-temperature residual head | Done in code | `configs/feature_e5_hotspot_residual.yaml`; `src/model.py` supports `T_pred = T_base + P_hot * delta_hot` via hotspot gate mix. | Run E5 after E4 under the same protocol. |
| 8 | Acceptance metrics | Done in code and guarded | `scripts/evaluate_checkpoint.py` reports global metrics, threshold metrics, laser-region metrics, per-step peak errors, and worst-point diagnostics; `scripts/verify_evaluation_report.py` checks that each report contains the fixed comparison fields and worst-point laser context; fixed baseline and feature-ablation wrappers run this verifier after evaluation. | Run the verifier before manually summarizing any externally generated report. |
| 9 | Continue/stop criteria | Documented and scripted | `README.md` experiment decisions, this status file, and `scripts/decide_feature_ablation.py`; `scripts/run_feature_ablation.py` runs the decision gate after E0-E3 by default. | Apply the generated decision after fixed E0-E3 results exist. |

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
  --laser_xml configs\laser_paths\5_block_fem_additive_z_scan.xml `
  --reference_laser_xml F:\datas\5-block-fem\para.xml `
  --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json `
  --experiments E0 E1 E2 E3
```

This check should pass before launching any 50-epoch control or feature
ablation. It is read-only, parses the candidate laser XML, compares it against
the original `para.xml` when available, and prints the dry-run baseline and
ablation commands.

### 2. Verify XML path visibility

```powershell
Test-Path F:\datas\5-block-fem\para.xml
```

If this prints `False`, either restore the file at that path or pass the correct
XML path to `--laser_xml` / `--xml`.

To require a hard original-vs-reconstructed XML match before long runs, add
`--strict_reference_laser_xml` to the readiness command.

### 3. Export and check trajectory alignment at the known worst point

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\export_laser_trajectory.py `
  --config configs\feature_e3_path_phase.yaml `
  --laser_xml configs\laser_paths\5_block_fem_additive_z_scan.xml `
  --vtu_dir F:\VTU `
  --output_dir results\laser_trajectory_step2128_xml `
  --diagnose_step_index 2128 `
  --diagnose_node_index 24437 `
  --hotspot_step_index 2128 `
  --hotspot_top_k 20
```

This writes `laser_alignment_report.md` next to `laser_segments.csv`,
`laser_samples.csv`, `laser_trajectory_manifest.json`, and the optional
`hot_nodes_*.csv` overlay. Treat the Markdown report as the first human-readable
record for the step/node alignment decision.

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\plot_laser_hotspots.py `
  --config configs\feature_e3_path_phase.yaml `
  --laser_xml configs\laser_paths\5_block_fem_additive_z_scan.xml `
  --vtu_dir F:\VTU `
  --steps 2128 `
  --output_dir results\laser_hotspot_alignment_xml `
  --top_k 200 `
  --show_all_nodes `
  --zoom_radius_mm 2.0
```

Pass criteria: at hotspot steps, the laser position should be close to the high-temperature region, with sensible layer index, track index, scan direction, and `time_to_arrival_s` around the failure node.

### 4. Verify the currently frozen historical baseline artifact

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\verify_baseline_artifact.py `
  artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z `
  --require_commands
```

The new fixed E0 artifact should pass the same check, with `--require_laser_xml`
added because the current protocol records the reconstructed process XML.

### 5. Verify each evaluation report before comparison

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\verify_evaluation_report.py `
  logs\<run_name> `
  --require_laser_context
```

Pass criteria: the report must contain global RMSE/MAE/P95/P99/MaxError,
solidus and liquidus Precision/Recall/F1, laser-region MAE/MaxError,
per-timestep peak-temperature errors, and worst-case rows with laser distance or
arrival context.

### 6. Rerun and freeze the single fixed E0 baseline

Start with a dry run:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\run_baseline_protocol.py `
  --config configs\feature_e0_baseline.yaml `
  --vtu_dir F:\VTU `
  --laser_xml configs\laser_paths\5_block_fem_additive_z_scan.xml `
  --epochs 50 `
  --graph_device cuda `
  --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json `
  --resume_from_last `
  --dry_run
```

Remove `--dry_run` only when ready for the long run.

### 7. Run fixed E0-E3 feature ablations

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\run_feature_ablation.py `
  --experiments E0 E1 E2 E3 `
  --vtu_dir F:\VTU `
  --laser_xml configs\laser_paths\5_block_fem_additive_z_scan.xml `
  --epochs 50 `
  --graph_device cuda `
  --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json `
  --resume_from_last
```

Only proceed to E4/E5 after E1/E2/E3 show whether trajectory features improve hotspot recall and worst-case errors.

The wrapper runs the decision gate automatically when E0, E1, and E2 are part
of the selected experiments. To run it manually:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\decide_feature_ablation.py `
  --logs_dir logs `
  --experiments E0 E1 E2 E3 `
  --output_csv results\feature_ablation_decision.csv `
  --output_md results\feature_ablation_decision.md
```

Default decision thresholds are `--min_recall_gain 0.02` and
`--max_rmse_regression_pct 0.25`; adjust them only when the acceptance policy
changes, not per-run.
