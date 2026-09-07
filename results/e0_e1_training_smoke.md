# E0/E1 Protocol Training Smoke

Date: 2026-09-07

This smoke check verifies that the fixed protocol can start real training with the reconstructed additive-z-scan XML, frozen split indices, CUDA graph tensors, and bounded sample counts before launching any 50-epoch runs.

## Common Settings

- VTU directory: `F:\VTU`
- Laser XML: `configs\laser_paths\5_block_fem_additive_z_scan.xml`
- Split indices: `artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json`
- Seed: 42
- Epochs: 1
- Train samples: 2 / 1652
- Val samples: 1 / 354
- Test: skipped
- Device: cuda
- Graph device: cuda

## E0 Baseline Smoke

Command:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\train.py --config configs\feature_e0_baseline.yaml --vtu_dir F:\VTU --seed 42 --epochs 1 --experiment_name smoke_feature_e0_baseline --graph_device cuda --skip_test --laser_xml configs\laser_paths\5_block_fem_additive_z_scan.xml --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json --max_train_samples 2 --max_val_samples 1 --max_test_samples 1
```

Result: passed.

- Node feature dim: 12
- Laser feature dim: 0
- Parameters: 201,281
- Train loss: 3.9246e+03
- Val loss: 2.2975e+05
- Time: 2.5 s
- Seconds per batch: 1.2
- Peak VRAM: 9.2 GB

## E1 Laser-Distance Smoke

Command:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\train.py --config configs\feature_e1_laser_distance.yaml --vtu_dir F:\VTU --seed 42 --epochs 1 --experiment_name smoke_feature_e1_laser_distance --graph_device cuda --skip_test --laser_xml configs\laser_paths\5_block_fem_additive_z_scan.xml --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json --max_train_samples 2 --max_val_samples 1 --max_test_samples 1
```

Result: passed.

- Node feature dim: config=12, actual=27; using actual dataset dim
- Laser feature dim: 15 (base 12)
- Parameters: 202,241
- Train loss: 3.9308e+03
- Val loss: 2.2991e+05
- Time: 2.6 s
- Seconds per batch: 1.2
- Peak VRAM: 9.3 GB

## Interpretation

The smoke runs are intentionally too small to compare model quality. Their purpose is only to verify protocol wiring. E0 uses no laser features, while E1 increases the actual node feature dimension and reaches the model without shape or device errors.
