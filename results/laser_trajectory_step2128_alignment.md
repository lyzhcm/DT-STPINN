# Laser Trajectory Alignment Check: Step 2128

Date: 2026-08-30

Command:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\export_laser_trajectory.py --config configs\feature_e3_path_phase.yaml --vtu_dir F:\VTU --output_dir results\laser_trajectory_step2128_smoke --diagnose_step_index 2128 --diagnose_node_index 24437 --hotspot_step_index 2128 --hotspot_top_k 20
```

Trajectory settings:

| Field | Value |
|---|---:|
| Config | `configs\feature_e3_path_phase.yaml` |
| XML | not available during this run |
| VTU sample count | 2361 |
| Time scale | 8.497742224741091e-4 s/raw |
| Time offset | -0.0111044328766 s |
| Reverse hatch order parity | 1 |
| Scan time per track | 0.033854 s |
| Path period | 0.034854 s |
| Layer period | 3.624816 s |

Node-index diagnosis:

| Field | Value |
|---|---:|
| Sample index | 2128 |
| VTU file | `F:\VTU\Data-42560.vtu` |
| Raw time | 42560 |
| Node index | 24437 |
| Node coordinate | [10.216346, -9.375000, 1.000000] mm |
| Target temperature | 2370.9536 C |
| Laser coordinate | [9.915700, -10.156200, 1.000000] mm |
| dx, dy, dz | [0.300646, 0.781200, 0.000000] mm |
| Laser distance | 0.837055 mm |
| Current along / cross | -0.781200 / 0.300646 mm |
| Line along / cross | 0.781200 / 0.100154 mm |
| Time to arrival | 0.068363 s |
| Arrival raw time | 42640.4492 |
| Layer index | 9 |
| Physical / program track | 0 / 103 |
| Direction sign | -1 |
| In track neighborhood | 1 |
| Ellipsoid body score | 12.8530 |
| In laser ellipsoid | 0 |

Interpretation:

- The diagnosed worst node is a true high-temperature point and lies in the scan-track neighborhood.
- It is outside the strict ellipsoid body at this exact sampled time because the current laser center is 0.837 mm away and the ellipsoid score is much larger than 1.
- This makes the trajectory timing and physical/program track convention a high-priority check before judging E1/E2/E3 failures. If feature ablations do not improve solidus recall, inspect whether the intended path convention should place this node closer to physical track 1/program track 102 near this raw time.

Generated CSV/JSON files were written under `results/laser_trajectory_step2128_smoke`, which is ignored by git.

Visualization smoke:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\plot_laser_hotspots.py --config configs\feature_e3_path_phase.yaml --vtu_dir F:\VTU --steps 2128 --output_dir results\laser_hotspot_alignment_smoke --top_k 200 --show_all_nodes --zoom_radius_mm 2.0
```

Plot summary:

| Field | Value |
|---|---:|
| Hot nodes above solidus | 3 |
| Hottest node temperature | 2370.95 C |
| Hottest node distance to laser | 0.8371 mm |
| Nearest plotted hot-node distance to laser | 0.788 mm |

The local zoom plot confirms the numerical diagnosis: the high-temperature nodes are close to the scan line but ahead of the current laser ellipsoid at sample 2128. The PNG and per-step CSV are generated under `results/laser_hotspot_alignment_smoke`, which is ignored by git.
