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
