# Reconstructed XML Laser Alignment Smoke

Date: 2026-09-07

This smoke check uses `configs/laser_paths/5_block_fem_additive_z_scan.xml`, a lightweight XML reconstructed from the user-provided `additive_z_scan` process parameters. It is not the full solver `para.xml`, but it contains the fields consumed by `src/utils/laser_path.py`.

## Readiness

Command:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\check_experiment_readiness.py --vtu_dir F:\VTU --laser_xml configs\laser_paths\5_block_fem_additive_z_scan.xml --split_indices artifacts\baselines\paper1_fast_50epoch_canonical_eval_20260829T185514Z\split_indices.json --experiments E0 E1 E2 E3 --no_command_preview
```

Result: passed.

- VTU files: 2361
- Frozen split: train=1652, val=354, test=355
- E0/E1/E2/E3 configs: passed

## Step 2128 Worst-Point Diagnosis

- VTU file: `F:\VTU\Data-42560.vtu`
- Raw time: 42560
- Node index: 24437
- Coordinate mm: `[10.21634578704834, -9.375, 1.0]`
- Target temperature: 2370.953613 C
- Laser position mm: `[9.91569995880127, -10.156200408935547, 1.0]`
- Distance to current laser center: 0.837055 mm
- Line cross distance: 0.100154 mm
- Time to arrival: 0.068363 s
- Arrival raw time: 42640.449219
- Layer index: 9
- Physical track: 0
- Program track: 103
- Direction sign: -1
- In laser ellipsoid: 0
- In track neighborhood: 1

Interpretation: the hottest failure node is close to the active scan track but not inside the instantaneous ellipsoid at the current frame. This supports using scan-arrival and path-phase features, not only current laser distance.

## Hotspot Overlay Smoke

- Step: 2128
- Hot nodes above solidus: 3
- Hottest node: 24437
- Hottest temperature: 2370.953613 C
- Hottest distance to laser: 0.837055 mm
- Plotted nodes inside ellipsoid: 0
- Plotted nodes inside track neighborhood: 3
- Local plot: `results\laser_hotspot_alignment_reconstructed_xml\laser_hotspots_step_02128.png`

## Feature Dataset Smoke

Command generated two target steps, 2128 and 2129, under `data\processed\laser_features_reconstructed_xml_smoke`.

Checker result: validation OK.

- Checked chunks: 2
- Feature columns: 24
- Required columns: 19
- Minimum laser distance: 0.100800 mm
- Ellipsoid hits: 0
- Track-neighborhood hits: 91394

## Remaining Caveat

The original full XML at `F:\datas\5-block-fem\para.xml` is still missing. Before long E0-E3 training, prefer confirming this reconstructed XML against the original solver input or treating it explicitly as the process-path reconstruction.
