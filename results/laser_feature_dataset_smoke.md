# Laser Feature Dataset Smoke Check

Date: 2026-08-30

Command:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\build_laser_feature_dataset.py --config configs\feature_e3_path_phase.yaml --vtu_dir F:\VTU --steps 2128 --output_dir data\processed\laser_features_smoke --max_steps 1
```

Result:

| Field | Value |
|---|---:|
| VTU steps | 2361 |
| Nodes | 95986 |
| Target step | 2128 |
| Raw time | 42560 |
| Feature columns | 24 |
| In laser ellipsoid count | 0 |
| In track neighborhood count | 45697 |
| Minimum laser distance | 0.2789 mm |

The generated chunk and manifest are under `data/processed/laser_features_smoke`, which is intentionally ignored by git.

Note: the XML path `F:\datas\5-block-fem\para.xml` was not present on this machine during the check, so this smoke run used the additive scan parameters already fixed in `configs/feature_e3_path_phase.yaml`.

Validation command:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\check_laser_feature_dataset.py --manifest data\processed\laser_features_smoke\manifest.json --step 2128 --node_index 24437
```

Validation result:

| Field | Value |
|---|---:|
| Status | OK |
| Required core columns | 19 |
| Feature columns | 24 |
| Checked chunks | 1 |
| P50 laser distance | 19.6515 mm |
| Minimum absolute arrival-time gap | 4.334e-05 s |
| Node 24437 distance to laser | 0.8371 mm |
| Node 24437 time to arrival | 0.06836 s |
| Node 24437 in track neighborhood | 1 |
| Node 24437 in laser ellipsoid | 0 |

The validation report and CSV are also generated under `data/processed/laser_features_smoke`, and remain ignored by git.
