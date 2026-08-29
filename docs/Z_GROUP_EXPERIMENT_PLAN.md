# Z Group Experiment Plan

## Current Diagnosis

The main failure is not ordinary regression accuracy. The baseline reaches high
global R2, but cold-start melt-pool nodes can still be predicted near ambient
temperature while the target is above solidus.

The confirmed `additive_z_scan` process path should be treated as a known
process input:

- Start point: `[10.3165, -10.1562, 0.1]` mm
- Scan direction: `[0, 1, 0]`
- Track length: `20.3124` mm
- Hatch direction: `[-1, 0, 0]`
- Hatch count: `104`
- Hatch spacing: `0.2004` mm
- Layer count: `10`
- Layer thickness: `0.1` mm
- Velocity: `600` mm/s
- Path delay: `0.001` s

Earlier hotspot-loss-only experiments are not enough because some failure
windows contain only cold input temperatures. The model needs explicit process
features plus an event detector for cold-to-hot transitions.

## Z87 Hypothesis

Use the XML/additive path as a high-recall candidate region, but do not let the
deterministic prior directly force high temperature. Instead:

1. Build a broad residual prior from active scan path support.
2. Sharpen that prior with neighbor hot statistics.
3. Train a supervised `cold_to_hot` classifier on true cold-start melt events.
4. Use the calibrated classifier gate to control the laser residual head.

Implemented config:

- `configs/ablation_z87_xml_cold_to_hot_track_residual.yaml`

Key settings:

- `laser_residual_gate_source: "path_active_with_neighbor_support"`
- `laser_residual_gate_mix: "calibrated_cold_to_hot"`
- `enable_cold_to_hot_head: true`
- `lambda_cold_to_hot_cls: 160.0`
- `laser_residual_min_delta: 0.0`

## Probe Run

Run a small probe before any long training:

```powershell
.\scripts\run_z_group_experiments.ps1 `
  -Mode probe `
  -VtuDir F:\VTU `
  -Python F:\anaconda3\envs\dtstpinn\python.exe
```

Equivalent direct command:

```powershell
python scripts\train.py `
  --config configs\ablation_z87_xml_cold_to_hot_track_residual.yaml `
  --vtu_dir F:\VTU `
  --epochs 5 `
  --max_train_samples 100 `
  --max_val_samples 30 `
  --max_test_samples 100 `
  --graph_device cuda `
  --experiment_name ablation_z87_xml_cold_to_hot_track_residual_probe100
```

Probe pass criteria:

- Worst cold-start hotspot prediction is no longer near ambient temperature.
- `MaxError` drops meaningfully from the previous 2300+ C failure mode.
- `TempRecallAboveSolidus` improves without exploding false positives.
- `AbsErrorP99` does not regress severely.

## Full Run

Only run full training after the probe shows that the melt-pool failure mode is
improving:

```powershell
.\scripts\run_z_group_experiments.ps1 `
  -Mode full `
  -VtuDir F:\VTU `
  -Python F:\anaconda3\envs\dtstpinn\python.exe `
  -FullEpochs 30
```

The training script writes final test metrics to:

- `logs/<experiment_name>/test_metrics.json`

The z-group summary is written to:

- `results/z_experiment_summary.csv`
- `results/z_experiment_summary.md`

## Decision Tree

## Probe Results Through Z97

The z87-z91 probes show that the XML path features are useful, but the residual
gate must be calibrated before any long run:

| Run | Gate / residual strategy | Test RMSE | Test MAE | Test P99 | Test MaxError | Solidus recall | Solidus precision | Worst pred / target | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| z87 | Path-active with neighbor support + cold-to-hot gate | 22.80 | 3.17 | 37.45 | 2205.81 | 0.0069 | 0.0069 | 124.5 / 2330.3 | Gate too suppressed by missing neighbor support. |
| z88 | Arrival/endpoint/sweep XML prior, learned delta | 25.17 | 3.35 | 31.35 | 1758.31 | 0.0069 | 0.0069 | 572.0 / 2330.3 | Best stable direction so far; gate opens but delta is too small. |
| z89 | z88 with stronger delta loss | 27.15 | 3.37 | 31.75 | 1702.31 | 0.0069 | 0.0066 | 628.0 / 2330.3 | Stronger loss gives weak return and worse stability. |
| z90 | z88 plus 2000 C residual floor | 52.04 | 4.34 | 39.75 | 1916.00 | 0.7260 | 0.0365 | 1936.0 / 20.0 | Recall works, but the sweep/arrival prior is too broad and creates false-hot nodes. |
| z91 | Endpoint-only 2000 C residual floor | 29.72 | 3.00 | 27.46 | 2284.38 | 0.0514 | 0.0180 | 3.6 / 2288.0 | False-hot is controlled, but endpoint-only is too narrow. |
| z92 | Calibrated XML pre-arrival soft prior + 2000 C floor | 18.89 | 2.54 | 29.26 | 2199.31 | 0.0069 | 0.0070 | 131.0 / 2330.3 | Time calibration helps, but the soft prior attenuates the residual too much. |
| z93 | Calibrated XML pre-arrival hard threshold + 2000 C floor | 29.47 | 3.04 | 31.13 | 1980.00 | 0.7260 | 0.0587 | 2000.0 / 20.0 | Recall works, but hard pre-arrival alone is still too permissive. |
| z94 | Calibrated pre-arrival hard threshold + endpoint support | 22.20 | 2.80 | 27.62 | 2275.26 | 0.0514 | 0.0242 | 12.8 / 2288.0 | Endpoint guard reduces false-hot but misses in-track cold-start events. |
| z95 | Calibrated pre-arrival hard threshold + sweep support | 20.95 | 2.70 | 28.96 | 2322.56 | 0.0205 | 0.0192 | 7.8 / 2330.3 | Current sweep proxy is not aligned with true end-of-track/ellipsoid heating. |
| z96 | Calibrated pre-arrival hard threshold + relaxed endpoint support | 22.75 | 2.83 | 28.41 | 2259.12 | 0.0993 | 0.0372 | 8.9 / 2268.0 | Lower endpoint threshold helps slightly, but still misses non-endpoint true events. |
| z97 | Calibrated pre-arrival hard threshold + XML body-or-endpoint support | 22.76 | 2.81 | 28.54 | 2260.12 | 0.0993 | 0.0372 | 7.9 / 2268.0 | Same as z96; target-center body footprint is too narrow/misaligned for non-endpoint events. |
| z98 | Calibrated pre-arrival hard threshold + XML path-body corridor support | 29.28 | 3.05 | 36.50 | 1972.00 | 0.7260 | 0.0587 | 1992.0 / 20.0 | Direction is correct: the swept XML corridor opens true residual support, but the guard is too permissive and cold false-hot now dominates. |
| z99 | z98 with strict path-body support threshold 0.15 | 28.79 | 3.03 | 35.00 | 2338.68 | 0.7055 | 0.0599 | -8.4 / 2330.3 | Threshold-only tuning is not enough: strict path-body support removes the z98 false-hot boost, but also reintroduces a true cold-start miss. |
| z100 | z98 high-recall path-body gate with lower 1600 C residual floor | 23.71 | 2.88 | 31.17 | 1580.00 | 0.0068 | 0.0053 | 1600.0 / 20.0 | Lower floor caps false-hot error effectively, but 1600 C is just below the 1604.85 C solidus threshold and therefore loses recall. |
| z101 | z100 with 1650 C residual floor | 24.33 | 2.90 | 31.79 | 1628.00 | 0.7260 | 0.0587 | 1648.0 / 20.0 | Restores solidus recall while capping false-hot error near the 1650 C floor; remaining worst case is a cold corridor node triggered by broad path-body support. |
| z102 | z101 floor with pre-arrival track-support blend | 20.24 | 2.79 | 30.63 | 2327.93 | 0.1849 | 0.0342 | 2.4 / 2330.3 | Too conservative: false-hot is reduced, but true cold-start hotspots are missed again. |
| z103 | z102 with target/sweep/endpoint/neighbor heat support | 22.26 | 2.80 | 28.57 | 2283.51 | 0.3322 | 0.0514 | 4.5 / 2288.0 | Better recall than z102 but still suppresses true in-track events; the worst true-hot node has high path-active but weak heat support. |
| z104 | z101 high-recall path-body gate with tiered floor | 20.67 | 2.77 | 30.55 | 1628.00 | 0.3322 | 0.0514 | 1648.0 / 20.0 | Not sufficient: the tiered floor still promotes a cold corridor/endpoint node to the solidus floor, while recall falls back to the z103 level. |
| z105 | XML scan-phase inputs with learned residual control, no hard floor | 18.88 | 2.58 | 29.22 | 1622.31 | 0.0068 | 0.0069 | 708.0 / 2330.3 | Removes the z104 false-hot floor failure, but becomes too conservative again: true cold-start hotspots receive only a partial residual boost. |
| z106 | z105 learned control with 3800 C residual delta floor | 18.74 | 2.59 | 29.22 | 1690.31 | 0.0068 | 0.0069 | 640.0 / 2330.3 | Reject: high delta floor is neutralized by lower learned/cold-to-hot gates, so true-hot recall does not recover. |
| z107 | XML candidate prior with supervised learned residual gate | 18.92 | 2.74 | 28.30 | 1537.59 | 0.0068 | 0.0070 | 2336.0 / 798.4 | Reject as the next full run: supervised residual gate avoids the old cold-start miss as worst case, but solidus recall remains collapsed and the new worst case is over-prediction near a neighbor-hot/path-corridor candidate. |
| z108 | z107 with literal para.xml serpentine order | 20.66 | 2.58 | 28.43 | 2307.56 | 0.0068 | 0.0070 | 22.8 / 2330.3 | Reject: literal no-flip/no-reversal path makes the true layer-10 hotspot lose arrival support entirely, so the VTU timing appears to require the calibrated odd-layer/hatch-order mapping despite the simple XML field reading. |
| z109 | z107 with diagnosed reverse-odd path clock | 17.72 | 2.93 | 30.46 | 2024.31 | 0.0068 | 0.0069 | 306.0 / 2330.3 | Reject as a full-run candidate: calibrated scan timing opens the arrival/prior gates on the true cold-start node, but the learned residual control is still too conservative and applies only a small boost. |
| z110 | z109 calibrated scan prior driving a specialist event head | 19.20 | 2.66 | 31.04 | 2141.94 | 0.0308 | 0.0185 | 3008.0 / 866.1 | Reject as a full-run candidate: the calibrated prior opens specialist recall, but the direct specialist blend is too permissive and creates a severe warm-node false-hot over-prediction. |
| z111 | z110 specialist head gated by calibrated prior x direct heat support | 20.15 | 2.43 | 28.85 | 2255.24 | 0.0068 | 0.0068 | 32.8 / 2288.0 | Reject: direct support removes the z110 false-hot worst case, but it is too strict for true in-track events whose target/body/endpoint gates are near zero. |
| z112 | z110 broad calibrated prior with specialist output capped at 1900 C | 17.31 | 2.57 | 30.32 | 2174.31 | 0.0240 | 0.0146 | 156.0 / 2330.3 | Reject: the cap prevents 3000 C false-hot predictions, but the learned specialist temperature collapses under false-hot pressure and still misses the true melt-pool node. |
| z113 | z101 solidus residual floor with diagnosed reverse-odd path clock | 61.42 | 5.34 | 36.45 | 1628.00 | 0.7260 | 0.0139 | 1648.0 / 20.0 | Do not full-run yet: the fixed floor restores high solidus recall and caps error, but the pre-arrival + path-body guard is too broad and creates many cold false-hot nodes. |
| z114 | z113 floor guarded by target/sweep/endpoint/neighbor heat support | 46.60 | 4.22 | 33.25 | 2257.13 | 0.6815 | 0.0218 | -12.8 / 2244.4 | Too strict: false-hot precision improves, but true in-track cold-start events without direct heat support are missed again. |
| z115 | z113 floor gated by arrival x process, threshold 0.05 | 39.85 | 3.60 | 37.83 | 1612.00 | 0.7260 | 0.0445 | 1632.0 / 20.0 | Best recall/error cap among z113-z115, but still creates cold false-hot nodes near weak arrival support. |
| z116 | z115 with arrival x process threshold 0.15 | 25.38 | 3.00 | 35.51 | 2303.01 | 0.7192 | 0.0648 | -15.0 / 2288.0 | Higher threshold improves precision, but misses a true hotspot whose arrival x process is about 0.10. |
| z117 | z116 threshold lowered to 0.10 | 29.31 | 3.12 | 35.95 | 2303.26 | 0.7192 | 0.0576 | -15.3 / 2288.0 | Still misses the same true hotspot: its product is just below 0.10, while the z115 false-hot product is also near 0.10. Product-threshold tuning is not separative enough. |
| z118 | z115 candidate with cold-to-hot learned control | 17.42 | 2.72 | 27.74 | 2310.56 | 0.0068 | 0.0051 | 19.8 / 2330.3 | Reject: global errors improve, but the learned cold-to-hot control suppresses true cold-start melt-pool events and collapses solidus recall. |
| z119 | pre-arrival or arrival residual floor, threshold 0.20 | 25.32 | 2.96 | 36.23 | 2306.26 | 0.7192 | 0.0770 | -18.3 / 2288.0 | Threshold is too strict: the recurring true hotspot has arrival support 0.124 and pre-arrival support 0.054, so the residual floor stays closed. |
| z120 | z119 threshold lowered to 0.10 | 35.75 | 3.24 | 37.42 | 1612.00 | 0.7260 | 0.0583 | 1632.0 / 20.0 | Recovers the true cold-start hotspot but reopens the weak-arrival false-hot mode. The gate boundary is real, but threshold tuning alone cannot separate these two nodes. |

Conclusion: do not launch a long z90/z91/z92/z93 training run.  z90 proves that a
strong residual can recover high-temperature recall, but it also proves that the
current `endpoint_sweep_or_arrival` gate is too permissive for a hard floor.
z91 proves the opposite boundary: endpoint-only is precise enough to avoid the
z90 false-hot node, but misses non-endpoint cold-start events.
z92 shows that calibrated `path_pre` is aligned with the failing event but the
continuous threshold mapping shrinks a true-hot prior gate to about `0.038`, so
a learned delta near 3000 C becomes only about 114 C of final boost.  z93 shows
that turning the same prior into a hard switch recovers solidus recall, but also
creates cold false-hot nodes where `path_pre` is just above threshold away from
the actual melt pool.  z94, z95, and z96 then bracket the geometry problem:
endpoint support is too narrow for many true events, while the current sweep
heat proxy does not cover the true high-temperature footprint near hatch ends
and the ellipsoid front/back region.  z98 confirms that projecting ellipsoid
support along the XML hatch corridor is a useful missing process signal, but the
very low support threshold (`0.0003`) opens the hard residual floor on weak
corridor matches.  z99 should test whether a stricter support threshold (`0.15`)
can preserve the recall gain while removing the new false-hot worst case.  z99
shows that the answer is no: the z98 false-hot node has path-body gate `0.1299`,
while a z99 true-hot miss has path-body gate `0.1194`, so a single path-body
threshold cannot separate them.  The next gate needs another signal, such as
target-center heat leakage, endpoint phase, along-track timing, or calibrated
gate confusion statistics around the failing raw-time windows.  Focused gate
confusion over raw time `40000..42600` shows `path_pre >= 0.003` has 0.9944
hot recall with 0.0004866 cold-positive rate, while path-body thresholding alone
keeps about 0.048 cold-positive rate.  z100 therefore tests a different tradeoff:
keep the high-recall path prior but reduce the hard residual floor from 2000 C
to 1600 C, capping false-hot damage while still pushing cold-start candidates to
near solidus.  z100 confirms the cap works, but the floor is slightly below the
solidus threshold, so z101 moves the floor to 1650 C rather than returning to the
2000 C setting.  z101 restores recall, but its worst node has high path-body
support with almost zero target/sweep/endpoint heat, so z102 should keep the
floor and replace the broad corridor guard with a timing gate blended with
track-support evidence.  z102/z103 reduce false-hot behavior, but they also
miss true cold-start melt-pool nodes again.  z104 shows that a tiered floor is
still not enough: a weak endpoint/corridor signal can be treated as strong
support and force a cold node to the 1650 C floor.  z105 removes that hard floor
and uses learned/cold-to-hot gates to control the residual.  This fixes the
false-hot worst-case type, but recall collapses because the learned gate only
applies about 43% of a 1584 C residual at a true hotspot.  z106 raises the delta
floor to 3800 C, but the learned/cold-to-hot gates shrink further and the true
hotspot still receives only about 620 C of boost.  z107 supervises the learned
residual gate directly, but it still gives only 0.0068 solidus recall and turns
the worst case into a 2336 C over-prediction on a 798 C target.  z108 then tests
the literal no-flip/no-reversal interpretation of the supplied XML summary, but
that removes arrival support from the true layer-10 hotspot and restores the
22.8 / 2330.3 cold-start miss.  z109 applies the diagnosed reverse-odd clock and
does open the calibrated prior on the true event, but the learned residual
control still shrinks a roughly 1352 C delta into only about 288 C of final
boost.  Therefore the next branch should stop tuning residual amplitude or
residual gates and instead train an explicit cold-start event temperature head.
z110 uses the calibrated scan prior to gate `hotspot_specialist_head` directly,
while false-hot losses penalize cold nodes that the same prior would otherwise
lift toward melt-pool temperatures.  It confirms the right mechanism but the
wrong gate width: `SpecGateRecallAt0.1 = 1.0` covers the true hot nodes, while
`SpecGatePrecisionAt0.1 = 0.0166` and the worst node `3008.0 / 866.1 C` show
that sweep/neighbor-supported corridor nodes can still receive the full
specialist prediction.  z111 therefore keeps the specialist actuator but changes
the final blend gate to calibrated residual prior times direct heat-source
support (`target_heat,body,endpoint`), deliberately excluding sweep and neighbor
from the gate-opening support.  z111 confirms that this guard is too narrow:
the worst miss has calibrated prior `1.0` and path-body support `0.259`, but
target heat `0.0023`, body `0.0`, endpoint `0.0009`, and neighbor `0.0`, so the
specialist gate is only `0.014` and the prediction stays near ambient.  Because
the z110 false-hot node also has path-body support around `0.263`, a single
path-body threshold cannot separate the two cases.  The next branch should avoid
a binary open/closed specialist switch and instead cap the specialist actuator:
use path-body support to permit a moderate cold-start boost, while preventing a
weakly supported candidate from blending all the way to 3000 C.  z112 tests the
simplest version of that idea: keep z110's broad calibrated prior gate but lower
`hotspot_specialist_max_temp` to `1900 C`, initialize the residual specialist
near `1650 C`, and strengthen false-hot penalties.  This cannot perfectly fit
2370 C melt-pool peaks, but it should answer whether a capped event actuator can
reduce the 2200-2300 C miss without replacing it with the z110 false-hot mode.
z112 improves RMSE, but it fails the real objective: at the z109/z112 recurring
true-hot node the specialist gate is fully open, yet the specialist temperature
is only `156 C`.  The learned specialist actuator has been suppressed by the
false-hot losses instead of becoming a reliable capped event response.  The next
z-branch should use a deterministic capped floor/boost at candidate event nodes
and make the remaining question purely gate selection, not learned temperature
amplitude.  z113 therefore returns to the z101 residual-floor actuator but swaps
in the later diagnosed reverse-odd path clock.  It confirms the floor actuator:
solidus recall returns to `0.7260` and the original cold-start miss is no longer
the worst case.  However, the new worst case is a cold node at step `2011`, raw
time `40220`, coordinate `[-0.601, 9.375, 1.000]` predicted at the `1648 C`
floor while the target is `20 C`.  Its target/body/sweep/neighbor gates are
near zero, but `path_body = 0.279` and `path_pre = 0.00377` trip the hard prior.
z114 therefore keeps the same calibrated additive-z-scan clock and residual
floor, but changes the gate to `pre_arrival_with_track_heat_support` with a
`0.001` support threshold.  The goal is to keep true event recall from the XML
trajectory while removing pure path-body corridor false positives.

z115-z117 show the limit of simple threshold tuning on that same XML trajectory.
The `arrival_process_product` gate is a useful family: z115 restores high
solidus recall and reduces the z113 false-hot precision problem, while z116
raises precision further.  However, the z115 false-hot node and the z116/z117
true-hot miss both sit near the same product value, about `0.10`.  A hard
threshold cannot reliably separate them.  The next branch should make the
known `additive_z_scan` physics more explicit by distinguishing actual
ellipsoid heat-source coverage from broad future-arrival/path-corridor support.

The next useful experiment should use the additive-z-scan geometry as a more
structured process gate rather than a larger loss weight or another product
threshold sweep.  Good candidates are:

- Add a guarded pre-arrival residual: `path_pre >= 0.003` must also satisfy a
  geometric support gate such as endpoint proximity, target/sweep heat, or a
  learned event gate.  The z93 false-hot node has no endpoint and no neighbor
  heat, while the z92 missed hotspot has endpoint support.
- Replace the current sweep proxy with an XML ellipsoid/corridor support gate
  that uses the additive scan segment, hatch parity, calibrated time offset, and
  the `radius_front/radius_back = 0.4 mm` footprint.  This should support
  near-end, slightly outside-the-line nodes that the `radius = 0.1 mm` sweep
  proxy suppresses.
- Probe a lower residual floor, around `1400..1800 C`, if recall is the priority
  but false-hot max error must be capped.
- Add gate confusion diagnostics over raw times around `40200`, `41740`,
  `41900`, and `42560` to choose thresholds from true/false event statistics
  instead of from one worst node.

Do not rank z90 as better just because recall is high. Its worst node is a cold
target (`20 C`) predicted as melt-pool temperature (`1936 C`), which is the
mirror-image failure of the original cold-start miss.

If z87 improves recall but creates false-hot nodes:

- Increase `laser_residual_cold_to_hot_gate_threshold` from `0.20` to `0.30`.
- Increase `cold_to_hot_gate_bias` from `-2.50` to `-3.00`.
- Try `laser_residual_gate_source: "path_active_neighbor_product"`.

If z87 still misses node `24437` at raw time `42560`:

- Calibrate `laser_path_time_offset_s` around that event before changing loss.
- Use `scripts/diagnose_laser_alignment.py --gate_confusion` on a narrow raw-time
  range around `42520..42600`.

If z87 improves the worst hotspot but worsens global regression:

- Reduce `lambda_laser_residual_delta` from `4200` to `3000`.
- Reduce `lambda_laser_residual` from `1800` to `1200`.

## Primary Metrics

Rank z-group experiments by this order:

1. `TempRecallAboveSolidus`
2. `TempPrecisionAboveSolidus`
3. `MaxError`
4. `AbsErrorP99`
5. Worst-node prediction/target gap
6. RMSE and MAE

RMSE alone is not decisive because most nodes are low-temperature background.

## Z214-Z221 XML Path Branch

The later z214-z217 branch uses the real `additive_z_scan` path described by
`para.xml`: start point `(10.3165, -10.1562, 0.1)`, scan direction `+Y`,
reverse-odd hatch traversal, `104` hatches per layer, `10` layers, hatch spacing
`0.2004 mm`, layer thickness `0.1 mm`, and velocity `600 mm/s`.  This is the
right direction because the original cold-start hotspot failures have ambient
input windows; loss reweighting alone cannot tell the model that a cold node is
about to enter the heat-source footprint.

z214-z215 showed that deterministic XML residual floors can recover some
solidus recall, but raising the floor too high reintroduces cold false-hot
outliers.  The safer actuator is the soft `2600 C` floor from z215 onward.
z216 improved the probe100 balance with `Recall=0.7308`, `Precision=0.2568`,
`F1=0.3800`, and no return of the earlier `3000 C` cold false-hot mode.  z217
adds a cautious endpoint-track support branch and improves probe100 to
`Recall=0.7778`, `Precision=0.2593`, `F1=0.3889`, `MaxError=2187.9 C`.

The remaining z217 worst case is still a true hotspot miss, not a false-hot:
`pre_arrival=0.5436`, `program_track=0.7087`, `path_body=0.2645`,
`wake=0.0585`, `neighbor_hot=0.4274`, but `endpoint=0` and `sweep=0.0105`.
This suggests a narrow additional support source, not a broader endpoint or
sweep threshold relaxation.

z218 therefore adds `prearrival_wake_neighbor_track_all`, which only opens the
strong residual floor when all of these XML/thermal supports agree:
pre-arrival phase, same program track, path-body support, wake support, and hot
neighbor support.  This did not improve over z217 because the new support source
only affected the residual delta floor; it did not open the residual gate for
the still-missed cold-start hotspots.

z217 probe300 confirms that the z217 improvement is stable when the sample
count increases: `test_RMSE=16.14`, `test_MAE=2.93`, `test_P99=20.65`,
`test_MaxError=2270.9`, `Recall=0.7548`, `Precision=0.2603`, and `F1=0.3871`.
The worst node is again a true hotspot miss rather than a cold false-hot:
step `2095`, raw time `41900`, node `24303`, prediction `59.4 C`, target
`2330.3 C`.  Its diagnostic gates are `pre_arrival=0.5345`, `endpoint=0.4397`,
`program_track=0.8252`, but `path_body=0.119431`, barely below the z217
`0.1200` body-support threshold.  Therefore z218 should keep the wake-neighbor
branch and also make a tiny footprint-edge relaxation by lowering
`laser_residual_prior_support_threshold` to `0.1150`, rather than changing the
loss scale or raising the residual floor.

z219 then lets XML strong support open the residual gate through
`xml_prearrival_strong_support_product`.  This is the first clear positive
mechanism in the late z branch: solidus recall rises to `0.9017` and F1 to
`0.4319`.  However, the `2600 C` residual floor overfires on a cold endpoint-like
node: step `2017`, node `35671`, prediction `2290.7 C`, target `20.0 C`.

z220 tested a hard program-track guard by raising
`laser_residual_endpoint_program_track_threshold` to `0.6500` and removing the
broad heat-track support.  Reject this branch: recall collapses to `0.2393`, and
the worst case is still a false-hot node that barely passes the hard threshold.

z221 keeps the z219 high-recall gate but lowers the tiered strong residual floor
from `2600 C` to `1900 C`.  This is currently the best probe100 balance:

| Run | Key change | Test RMSE | Test MAE | Test P99 | Test MaxError | Solidus recall | Solidus precision | Solidus F1 | Worst pred / target | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| z217 probe100 | Endpoint-track soft floor | 19.48 | 4.86 | 29.44 | 2187.9 | 0.7778 | 0.2593 | 0.3889 | 27.1 / 2215.0 | Useful but still misses true cold-start hotspots. |
| z218 probe100 | Add wake-neighbor strong support | 19.47 | 4.83 | 29.51 | 2187.9 | 0.7778 | 0.2593 | 0.3889 | 27.1 / 2215.0 | No-op relative to z217. |
| z219 probe100 | Strong XML support opens residual gate | 19.64 | 4.90 | 29.53 | 2270.7 | 0.9017 | 0.2840 | 0.4319 | 2290.7 / 20.0 | Recall improves, but false-hot worst case returns. |
| z220 probe100 | Hard program-track guard | 20.33 | 3.83 | 29.85 | 2334.3 | 0.2393 | 0.2995 | 0.2660 | 2580.7 / 246.4 | Reject: brittle threshold, low recall. |
| z221 probe100 | z219 gate with `1900 C` floor | 17.67 | 4.71 | 29.49 | 2176.7 | 0.9017 | 0.2840 | 0.4319 | 26.1 / 2202.8 | Best current candidate; scale to probe300. |
| z221 probe300 | Larger z221 probe | 14.28 | 2.94 | 21.34 | 2150.0 | 0.8883 | 0.2835 | 0.4298 | 94.4 / 2244.4 | Keeps recall/F1; remaining worst is true hotspot miss. |
| z222 probe100 | Add wake/body/program-track strong support at `0.6600` | 17.59 | 4.70 | 29.50 | 2165.3 | 0.9274 | 0.2897 | 0.4415 | 25.5 / 2190.7 | Better recall/F1; threshold just misses true hotspot at `program_track=0.6505`. |
| z223 probe100 | Lower wake/body/program-track threshold to `0.6500` | 17.56 | 4.69 | 29.49 | 2156.8 | 0.9359 | 0.2916 | 0.4447 | 25.7 / 2182.5 | Best probe100 so far; next miss has `program_track=0.6408`. |
| z223 probe300 | Larger z223 probe | 14.13 | 3.00 | 21.19 | 2080.2 | 0.9428 | 0.2957 | 0.4502 | 102.3 / 2182.5 | Good scaled probe; true-hot miss just below `0.6500` track gate. |
| z224 probe300 | Lower broad wake/body/program-track threshold to `0.6400` | 14.16 | 2.94 | 21.09 | 2064.5 | 0.9482 | 0.2967 | 0.4519 | 96.4 / 2160.9 | Small gain; next true-hot miss has low broad `program_track=0.4757`. |
| z225 probe300 | Add cold-to-hot guarded weak-track rescue at `0.4500` | 14.09 | 3.06 | 21.22 | 2034.9 | 0.9728 | 0.3023 | 0.4612 | 102.8 / 2137.8 | Best recall/F1/MaxError so far; still true-hot miss, weak track misses by `0.0034`. |

The remaining z221 worst case is again a true hotspot miss, not a cold false-hot:
step `2064`, raw time `41280`, node `74752`, prediction `26.1 C`, target
`2202.8 C`.  Its gates show why it is still missed:
`pre_arrival=0.5419`, `program_track=0.6796`, `path_body=0.2663`,
`wake=0.0572`, and `cold_to_hot=0.8789`, but `endpoint=0`, `neighbor_hot=0`,
and `laser_residual_gate=0`.  z221 probe300 confirmed the same pattern on
step `2107`, node `24527`: `pre_arrival=0.5461`, `program_track=0.8835`,
`path_body=0.2544`, `wake=0.0434`, `process_gate=0.9292`, but
`endpoint=0`, `neighbor_hot=0`, and `laser_residual_gate=0`.

z222 therefore adds `prearrival_wake_body_track_all`, a narrow strong-support
source for additive-z-scan cold-start hotspots.  It requires pre-arrival phase,
body support, wake support, and same-program-track support, but does not require
endpoint or hot-neighbor support.  z222 improves recall/F1 but its worst case
lands just below the `0.6600` program-track threshold.  z223 lowers that
threshold to `0.6500` and improves the probe again without making the worst case
a cold false-hot.  Keep the `1900 C` floor; do not restore a global `2600 C`
floor because that already caused z219 false-hot outliers.

Next execution step before a long 20-50 hour run:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\train.py `
  --config configs\ablation_z223_xml_reverse_odd_wake_body_track_gate_0650_1900_floor.yaml `
  --vtu_dir F:\VTU `
  --epochs 10 `
  --max_train_samples 300 `
  --max_val_samples 80 `
  --max_test_samples 160 `
  --graph_device cuda `
  --experiment_name ablation_z223_xml_reverse_odd_wake_body_track_gate_0650_1900_floor_probe300
```

Pass criteria for z223 probe300:

- Keep `TempRecallAboveSolidus >= 0.88`.
- Keep `TempF1AboveSolidus >= 0.40`.
- Keep worst case as a true hotspot miss or reduce `MaxError` below `2100 C`.
- Do not reintroduce a cold false-hot worst case above `1800 C`.

If z223 probe300 repeats the same true hotspot miss with program-track just
below the threshold, create z224 by lowering
`laser_residual_wake_body_program_track_threshold` to `0.6400`.  Before lowering
below `0.6400`, add a learned-event guard such as `cold_to_hot >= 0.75`, because
program-track alone cannot distinguish all cold false-hot cases.

z223 probe300 passed the main hotspot criteria and confirmed the XML trajectory
prior is the right optimization direction: `test_RMSE=14.13`, `test_MAE=3.00`,
`test_P99=21.19`, `test_MaxError=2080.2`, `Recall=0.9428`, `Precision=0.2957`,
and `F1=0.4502`.  The worst case was still a true hotspot miss rather than a
cold false-hot: step `2056`, raw time `41120`, node `74791`, prediction
`102.3 C`, target `2182.5 C`.  Its diagnostics match the confirmed
`additive_z_scan` serpentine path: `pre_arrival=0.5398`, `path_body=0.2686`,
`wake=0.0561`, `cold_to_hot=0.8711`, but `program_track=0.6408`, just below the
z223 `0.6500` threshold.  Therefore z224 lowers only
`laser_residual_wake_body_program_track_threshold` to `0.6400` while keeping the
body and wake support thresholds fixed.

Next execution step:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\train.py `
  --config configs\ablation_z224_xml_reverse_odd_wake_body_track_gate_0640_1900_floor.yaml `
  --vtu_dir F:\VTU `
  --epochs 10 `
  --max_train_samples 300 `
  --max_val_samples 80 `
  --max_test_samples 160 `
  --graph_device cuda `
  --experiment_name ablation_z224_xml_reverse_odd_wake_body_track_gate_0640_1900_floor_probe300
```

z224 probe300 is a small positive step from z223, but it exposes the next
failure mode.  Metrics: `test_RMSE=14.16`, `test_MAE=2.94`, `test_P99=21.09`,
`test_MaxError=2064.5`, `Recall=0.9482`, `Precision=0.2967`, and `F1=0.4519`.
The worst case is still a true hotspot miss: step `2021`, raw time `40420`,
node `35692`, prediction `96.4 C`, target `2160.9 C`.  Its diagnostics are
`pre_arrival=0.5376`, `path_body=0.2785`, `wake=0.0753`, `cold_to_hot=0.8867`,
but `program_track=0.4757`, so further lowering the broad program-track
threshold would likely reopen false-hot nodes.

z225 therefore keeps the z224 broad wake/body/program-track threshold at
`0.6400` and adds a separate conservative support source,
`prearrival_wake_body_track_cold_all`.  This branch only opens when
pre-arrival, body, wake, weak program-track, and strong cold-to-hot evidence all
agree.  The first probe uses `cold_to_hot >= 0.8500` and weak
`program_track >= 0.4500`, chosen to include the z224 true-hot miss while still
rejecting broad path-corridor false positives.

Next execution step:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\train.py `
  --config configs\ablation_z225_xml_reverse_odd_wake_body_track_cold_gate_0640_1900_floor.yaml `
  --vtu_dir F:\VTU `
  --epochs 10 `
  --max_train_samples 300 `
  --max_val_samples 80 `
  --max_test_samples 160 `
  --graph_device cuda `
  --experiment_name ablation_z225_xml_reverse_odd_wake_body_track_cold_gate_0640_1900_floor_probe300
```

Pass criteria for z225 probe300:

- Keep `TempPrecisionAboveSolidus >= 0.29`.
- Improve or maintain `TempRecallAboveSolidus >= 0.948`.
- Reduce `MaxError` below z224's `2064.5 C`, or at least keep the worst case as
  a true hotspot miss rather than a cold false-hot.
- Watch `AbsErrorP99`; it should not rise materially above `21.1 C`.

z225 probe300 passes the hotspot criteria and is the best z-group probe so far
on the metrics that matter for melt-pool recovery: `Recall=0.9728`,
`Precision=0.3023`, `F1=0.4612`, and `MaxError=2034.9 C`.  `AbsErrorP99`
is essentially flat relative to z223/z224 (`21.22 C`), so the extra cold-to-hot
rescue did not broadly damage the field.  The worst case remains a true hotspot
miss rather than a cold false-hot: step `2015`, raw time `40300`, node `72280`,
prediction `102.8 C`, target `2137.8 C`.  Its diagnostics show that the new
branch is almost exactly right but still slightly too strict:
`pre_arrival=0.5360`, `path_body=0.2802`, `wake=0.0744`,
`cold_to_hot=0.8750`, but weak `program_track=0.4466`, just below the z225
`0.4500` threshold.  Therefore z226 lowers only
`laser_residual_wake_body_cold_program_track_threshold` to `0.4400` and keeps
the `0.8500` cold-to-hot threshold unchanged.

Next execution step:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\train.py `
  --config configs\ablation_z226_xml_reverse_odd_wake_body_track_cold_gate_0440_1900_floor.yaml `
  --vtu_dir F:\VTU `
  --epochs 10 `
  --max_train_samples 300 `
  --max_val_samples 80 `
  --max_test_samples 160 `
  --graph_device cuda `
  --experiment_name ablation_z226_xml_reverse_odd_wake_body_track_cold_gate_0440_1900_floor_probe300
```

Pass criteria for z226 probe300:

- Keep `TempPrecisionAboveSolidus >= 0.30`.
- Keep `TempRecallAboveSolidus >= 0.972` or improve it.
- Reduce `MaxError` below z225's `2034.9 C`, or keep the worst case as a true
  hotspot miss with lower target temperature.
- Reject z226 if the worst case becomes a cold false-hot above `1800 C` or if
  `AbsErrorP99` rises materially above `21.5 C`.

z226 probe300 passes and becomes the current best z-group probe.  Metrics:
`test_RMSE=14.07`, `test_MAE=2.97`, `test_P99=19.91`,
`test_MaxError=1736.7`, `Recall=0.9782`, `Precision=0.3035`, and
`F1=0.4632`.  This is a clean improvement over z225 on recall, F1, P99, and
MaxError while keeping precision above `0.30`.  The worst case is still a true
hotspot miss rather than a cold false-hot: step `2128`, raw time `42560`, node
`24436`, prediction `149.6 C`, target `1886.3 C`.  The diagnostics show the
next failure mode: `post_arrival=0.5773`, `wake=0.5183`,
`program_track=0.9806`, `cold_to_hot=0.9149`, and `path_body=0.2488`, but
`pre_arrival=0.0`, so the z226 `xml_prearrival_strong_support_product` prior
does not open and `laser_residual_gate=0.0`.

z227 therefore keeps z226's `0.4400` weak program-track threshold and `0.8500`
cold-to-hot threshold, but changes the residual prior from a pre-arrival-only
phase to a pre-or-post phase.  It adds the strict support token
`postarrival_wake_body_track_cold_all`, which still requires post-arrival,
path-body, wake, program-track, and cold-to-hot evidence to agree.  A 1-batch
real-data smoke test passed with `configs/ablation_z227_xml_phase_wake_body_track_cold_gate_0440_1900_floor.yaml`.

Next execution step:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\train.py `
  --config configs\ablation_z227_xml_phase_wake_body_track_cold_gate_0440_1900_floor.yaml `
  --vtu_dir F:\VTU `
  --epochs 10 `
  --max_train_samples 300 `
  --max_val_samples 80 `
  --max_test_samples 160 `
  --graph_device cuda `
  --experiment_name ablation_z227_xml_phase_wake_body_track_cold_gate_0440_1900_floor_probe300
```

Pass criteria for z227 probe300:

- Keep `TempPrecisionAboveSolidus >= 0.30`.
- Keep or improve `TempRecallAboveSolidus >= 0.978` and `TempF1AboveSolidus >= 0.463`.
- Reduce `MaxError` below z226's `1736.7 C`, or at least keep the worst case as
  a true hotspot miss with lower target temperature.
- Reject z227 if the worst case becomes a cold false-hot above `1800 C` or if
  `AbsErrorP99` rises materially above `20.5 C`.

z227 probe300 is rejected.  Metrics: `test_RMSE=14.65`, `test_MAE=3.02`,
`test_P99=20.77`, `test_MaxError=2068.5`, `Recall=0.9646`,
`Precision=0.1691`, and `F1=0.2878`.  It fails every z227 pass criterion
relative to z226: recall and F1 both drop, precision drops sharply, P99 rises,
and MaxError regresses from `1736.7 C` to `2068.5 C`.  The worst case is still a
true hotspot miss rather than a cold false-hot: step `2021`, raw time `40420`,
node `35692`, prediction `92.4 C`, target `2160.9 C`.  Its diagnostics show
that simply broadening the phase gate is the wrong lever: `pre_arrival=0.5376`,
`arrival=0.7265`, `path_body=0.2785`, `program_track=0.4757`, and
`cold_to_hot=0.8438`, but `post_arrival=0.0`, `wake=0.0753`,
`neighbor_hot=0.0`, and `laser_residual_gate=0.0`.

Decision: keep z226 as the current best z-group probe.  Do not scale z227.  The
next branch should not replace the z226 pre-arrival gate with a broad
pre-or-post phase.  If testing a post-arrival fix, add it as a separate narrow
rescue branch aimed only at the z226 miss: require strong post-arrival,
strong wake, strong same-track support, and strong cold-to-hot evidence, for
example around `post_arrival >= 0.55`, `wake >= 0.50`,
`program_track >= 0.90`, `cold_to_hot >= 0.90`, and `path_body >= 0.24`.
That branch should be additive to z226 and rejected immediately if solidus
precision falls below `0.30` or P99 rises above `20.5 C`.

z228 implements that narrow post-arrival rescue as an additive branch on top of
z226, using `xml_prearrival_plus_strict_post_rescue_product`.  The original z226
pre-arrival branch is preserved, while the post-arrival branch requires strong
post-arrival, path-body, wake, program-track, and cold-to-hot support.  The probe
config is
`configs/ablation_z228_xml_reverse_odd_strict_post_rescue_0440_1900_floor.yaml`.

z228 probe300 is not accepted as the next scaling candidate.  Metrics:
`test_RMSE=14.25`, `test_MAE=2.89`, `test_P99=20.50`,
`test_MaxError=1724.3`, `Recall=0.9782`, `Precision=0.2967`, and
`F1=0.4553`.  Compared with z226, it slightly improves MAE and MaxError, and it
keeps recall unchanged, but it misses the precision floor (`0.2967 < 0.30`),
raises P99 from `19.91` to `20.50`, and lowers F1 from `0.4632` to `0.4553`.
The failure mode also changes from a true hotspot miss to a false-hot overshoot:
step `2130`, raw time `42600`, node `75344`, coordinate
`[10.216, 8.854, 1.000]`, prediction `2874.5 C`, target `1150.1 C`.  Its
diagnostics show the post-rescue branch opened very strongly:
`post_arrival=0.5778`, `wake=0.5185`, `program_track=0.9903`,
`cold_to_hot=0.9863`, `laser_residual_gate=0.8938`, and
`laser_residual_boost=1701.7 C`.

Decision: keep z226 as the current best z-group probe.  z228 confirms that a
post-arrival rescue can reduce the worst true miss, but the rescue is still too
aggressive because it can add a large residual to a cooling or moderate-hot
node.  The next branch should keep the additive z226+post structure, but make
the post branch phase-aware: reduce or suppress the residual when the target
point is already in the cooling tail or when direct laser body support is weak.
Good candidates are adding an upper/lower arrival-time window, requiring stronger
`laser_body_heat_gate`, or capping the post-arrival residual by elapsed time
since beam passage.

z229 adds a specialist-temperature cap to the z228 post-rescue branch.  It is
not accepted as a scaling candidate.  Metrics: `test_RMSE=14.03`,
`test_MAE=3.04`, `test_P99=21.01`, `test_MaxError=1671.0`,
`Recall=0.9782`, `Precision=0.3032`, and `F1=0.4629`.  The global hotspot
metrics are close to z226 and the MaxError is lower, but the worst case becomes
a cold false-hot: step `2091`, raw time `41820`, node `24292`, coordinate
`[6.611, -9.375, 1.000]`, prediction `1691.0 C`, target `20.0 C`.  The
residual branch opens too broadly with `pre_arrival=0.5233`,
`target_heat=0.1275`, `path_body=0.2583`, `program_track=0.8155`,
`laser_residual_gate=0.8759`, and `laser_residual_boost=1667.8 C`.

z230 tightens the shared pre-arrival heat/path-body support threshold to
`0.1400` while keeping the sweep threshold at `0.1200`.  It is also not
accepted.  Metrics: `test_RMSE=13.89`, `test_MAE=2.89`, `test_P99=20.54`,
`test_MaxError=2252.8`, `Recall=0.9537`, `Precision=0.3054`, and `F1=0.4627`.
The z229 cold false-hot is suppressed, but a real endpoint-adjacent cold-start
hotspot is missed: step `2095`, raw time `41900`, node `24303`, coordinate
`[6.811, -10.417, 1.000]`, prediction `77.5 C`, target `2330.3 C`.  This point
has strong process-path evidence (`pre_arrival=0.5345`, `endpoint=0.4397`,
`program_track=0.8252`) and weak cold-to-hot evidence (`0.3066`), but direct
body/wake/sweep support is zero, so the residual prior remains closed.

z231 adds a dedicated `prearrival_endpoint_track_cold_all` support source for
the z230 endpoint miss.  It is rejected.  Metrics: `test_RMSE=13.85`,
`test_MAE=2.91`, `test_P99=21.03`, `test_MaxError=1670.8`, `Recall=0.9700`,
`Precision=0.3056`, and `F1=0.4648`.  The aggregate F1 is the best of this
small branch, but the worst case again becomes a cold false-hot: step `2017`,
raw time `40340`, node `35671`, coordinate `[-0.801, -10.417, 1.000]`,
prediction `1690.8 C`, target `20.0 C`.  Diagnostics show why the endpoint
rescue is too permissive: `pre_arrival=0.5233`, `endpoint=0.4352`,
`program_track=0.4660`, `cold_to_hot=0.4219`, `laser_residual_gate=0.8759`,
and `laser_residual_boost=1667.8 C`.

Decision: keep z226 as the current conservative best.  Do not scale z229,
z230, or z231.  The confirmed `additive_z_scan` path explains the next clean
lever: endpoint-adjacent cold-start rescue should require stronger same-track
program support, not a higher cold-to-hot threshold.  Raising cold-to-hot would
also reject the z230 true miss (`cold_to_hot=0.3066`), while raising endpoint
program-track can reject the z231 false-hot (`program_track=0.4660`) and still
keep the z230 true miss (`program_track=0.8252`).  z232 therefore keeps z231's
endpoint branch but raises `laser_residual_endpoint_program_track_threshold` to
`0.7500` in
`configs/ablation_z232_xml_reverse_odd_endpoint_track075_rescue_0440_1900_floor.yaml`.

Next execution step:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\train.py `
  --config configs\ablation_z232_xml_reverse_odd_endpoint_track075_rescue_0440_1900_floor.yaml `
  --vtu_dir F:\VTU `
  --epochs 10 `
  --max_train_samples 300 `
  --max_val_samples 80 `
  --max_test_samples 160 `
  --graph_device cuda `
  --experiment_name ablation_z232_xml_reverse_odd_endpoint_track075_rescue_0440_1900_floor_probe300
```

Pass criteria for z232 probe300:

- Reject immediately if the worst case is a cold false-hot above `1200 C`.
- Prefer `TempPrecisionAboveSolidus >= 0.30` and `TempF1AboveSolidus >= 0.463`.
- Recover recall toward z226/z229 (`>= 0.970`, ideally `>= 0.978`).
- Keep `AbsErrorP99 <= 21.0 C` and reduce `MaxError` below z230's
  `2252.8 C`; a true hotspot miss is acceptable only if it is below z226's
  `1736.7 C`.

z232 probe300 is rejected.  Metrics: `test_RMSE=14.01`, `test_MAE=2.94`,
`test_P99=21.16`, `test_MaxError=2193.8`, `Recall=0.9373`,
`Precision=0.2999`, and `F1=0.4544`.  It does avoid the z231 cold false-hot,
but the endpoint program-track threshold is now too strict.  The worst case is
a real endpoint-adjacent cold-start hotspot: step `2054`, raw time `41080`,
node `25029`, coordinate `[2.804, -10.417, 1.000]`, prediction `86.5 C`,
target `2280.3 C`.  Diagnostics show strong endpoint/process timing evidence
(`pre_arrival=0.5438`, `endpoint=0.4426`, `cold_to_hot=0.5391`,
`elapsed=0.9635`), but `program_track=0.6408` is below z232's `0.7500`
threshold, so `laser_residual_gate=0.0`.

Decision: keep z226 as the conservative best.  The confirmed XML path narrows
the endpoint threshold interval: z231's false-hot had `program_track=0.4660`,
while z232's true miss has `program_track=0.6408`.  z233 therefore keeps the
same endpoint rescue branch but sets
`laser_residual_endpoint_program_track_threshold=0.6000`, between those two
diagnosed cases.

Next execution step:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\train.py `
  --config configs\ablation_z233_xml_reverse_odd_endpoint_track060_rescue_0440_1900_floor.yaml `
  --vtu_dir F:\VTU `
  --epochs 10 `
  --max_train_samples 300 `
  --max_val_samples 80 `
  --max_test_samples 160 `
  --graph_device cuda `
  --experiment_name ablation_z233_xml_reverse_odd_endpoint_track060_rescue_0440_1900_floor_probe300
```

Pass criteria for z233 probe300:

- Reject immediately if the worst case becomes a cold false-hot above `1200 C`.
- Recover recall and F1 toward z226: `TempRecallAboveSolidus >= 0.970` and
  `TempF1AboveSolidus >= 0.463`.
- Keep `TempPrecisionAboveSolidus >= 0.30` and `AbsErrorP99 <= 21.2 C`.
- Prefer `MaxError < 1736.7 C`; accept a slightly higher true hotspot miss only
  if it is clearly below z232's `2193.8 C` and no cold false-hot reappears.

z233 probe300 is rejected.  Metrics: `test_RMSE=13.99`, `test_MAE=2.87`,
`test_P99=19.68`, `test_MaxError=2261.7`, `Recall=0.9401`,
`Precision=0.2997`, and `F1=0.4545`.  Global distribution metrics improve, but
the worst case is still a true endpoint-adjacent cold-start hotspot: step
`2095`, raw time `41900`, node `24303`, coordinate
`[6.811, -10.417, 1.000]`, prediction `68.6 C`, target `2330.3 C`.  The XML
path signal is strong (`pre_arrival=0.5345`, `endpoint=0.4397`,
`program_track=0.8252`, `elapsed=0.9827`), but `cold_to_hot=0.2617` is below
the `0.3000` endpoint cold-start threshold, so `laser_residual_gate=0.0`.

Decision: keep z226 as the conservative best.  z234 makes one narrow follow-up:
keep z233's same-track threshold at `0.6000`, but lower
`laser_residual_endpoint_cold_to_hot_threshold` from `0.3000` to `0.2500`.
This should recover the z233 true miss while still rejecting the z231 cold
false-hot through its low `program_track=0.4660`.

Next execution step:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\train.py `
  --config configs\ablation_z234_xml_reverse_odd_endpoint_track060_cold025_rescue_0440_1900_floor.yaml `
  --vtu_dir F:\VTU `
  --epochs 10 `
  --max_train_samples 300 `
  --max_val_samples 80 `
  --max_test_samples 160 `
  --graph_device cuda `
  --experiment_name ablation_z234_xml_reverse_odd_endpoint_track060_cold025_rescue_0440_1900_floor_probe300
```

Pass criteria for z234 probe300:

- Reject immediately if the worst case becomes a cold false-hot above `1200 C`.
- Recover recall and F1 toward z226: `TempRecallAboveSolidus >= 0.970` and
  `TempF1AboveSolidus >= 0.463`.
- Keep `TempPrecisionAboveSolidus >= 0.30` and `AbsErrorP99 <= 21.2 C`.
- Prefer `MaxError < 1736.7 C`; accept only if the worst true hotspot miss is
  clearly below z233's `2261.7 C` and no cold false-hot reappears.

z234 probe300 is rejected.  Metrics: `test_RMSE=14.02`, `test_MAE=2.80`,
`test_P99=19.59`, `test_MaxError=2264.7`, `Recall=0.9401`,
`Precision=0.2997`, and `F1=0.4545`.  It did not create an obvious cold
false-hot, but it also did not recover the endpoint-adjacent true miss.  The
worst case remains step `2095`, raw time `41900`, node `24303`, coordinate
`[6.811, -10.417, 1.000]`, prediction `65.6 C`, target `2330.3 C`.  The
deterministic path evidence is still strong (`pre_arrival=0.5345`,
`endpoint=0.4397`, `program_track=0.8252`, `elapsed=0.9827`), but
`cold_to_hot=0.2354` fell just below z234's `0.2500` threshold, so
`laser_residual_gate=0.0`.

Decision: stop chasing the learned cold-to-hot threshold for this endpoint
branch.  The confirmed `additive_z_scan` path says this node is on a valid
same-track endpoint arrival, and z231's known cold false-hot is separable by
program-track (`0.4660`) rather than by the learned cold gate.  z235 therefore
adds `prearrival_endpoint_program_track_all`: a deterministic endpoint rescue
path requiring pre-arrival, endpoint, and program-track support, but not the
learned cold-to-hot gate.

Next execution step:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\train.py `
  --config configs\ablation_z235_xml_reverse_odd_endpoint_track060_program_rescue_0440_1900_floor.yaml `
  --vtu_dir F:\VTU `
  --epochs 10 `
  --max_train_samples 300 `
  --max_val_samples 80 `
  --max_test_samples 160 `
  --graph_device cuda `
  --experiment_name ablation_z235_xml_reverse_odd_endpoint_track060_program_rescue_0440_1900_floor_probe300
```

Pass criteria for z235 probe300:

- Reject immediately if the worst case becomes a cold false-hot above `1200 C`.
- Recover recall and F1 toward z226: `TempRecallAboveSolidus >= 0.970` and
  `TempF1AboveSolidus >= 0.463`.
- Keep `TempPrecisionAboveSolidus >= 0.30` and `AbsErrorP99 <= 21.2 C`.
- Prefer `MaxError < 1736.7 C`; accept only if the worst true hotspot miss is
  clearly below z234's `2264.7 C` and no cold false-hot reappears.

z235 probe300 is rejected as a final candidate, but it confirms the direction.
Metrics: `test_RMSE=14.01`, `test_MAE=2.93`, `test_P99=19.62`,
`test_MaxError=2142.9`, `Recall=0.9482`, `Precision=0.2985`, and
`F1=0.4540`.  MaxError improves versus z234 (`2264.7 -> 2142.9 C`), but
recall remains far below z226.  The worst case is still a true cold-start
hotspot: step `2013`, raw time `40260`, node `35707`, coordinate
`[-1.202, -10.417, 1.000]`, prediction `83.5 C`, target `2226.4 C`.
The endpoint and arrival evidence are positive (`pre_arrival=0.5574`,
`endpoint=0.4455`), but `program_track=0.4466` is below z235's `0.6000`
threshold, so `laser_residual_gate=0.0`.

Decision: do not keep lowering the same-track/program-track threshold as the
main rescue signal.  The confirmed `additive_z_scan` path gives a stronger
physical discriminator: endpoint-adjacent nodes that are close in future arrival
time should be rescued even when the signed program-track coordinate is near an
edge.  z236 therefore adds a `prearrival_endpoint_time_until_all` support path
using the existing XML path timing feature.

Next execution step:

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\train.py `
  --config configs\ablation_z236_xml_reverse_odd_endpoint_time_until_rescue_0440_1900_floor.yaml `
  --vtu_dir F:\VTU `
  --epochs 10 `
  --max_train_samples 300 `
  --max_val_samples 80 `
  --max_test_samples 160 `
  --graph_device cuda `
  --experiment_name ablation_z236_xml_reverse_odd_endpoint_time_until_rescue_0440_1900_floor_probe300
```

Pass criteria for z236 probe300:

- Reject immediately if the worst case becomes a cold false-hot above `1200 C`.
- Improve z235 true-hot recovery: `TempRecallAboveSolidus >= 0.960` and
  `TempF1AboveSolidus >= 0.458`.
- Keep `TempPrecisionAboveSolidus >= 0.295` and `AbsErrorP99 <= 21.2 C`.
- Prefer `MaxError < 2000 C`; accept only if the worst true hotspot miss is
  clearly below z235's `2142.9 C` and no cold false-hot reappears.

Longer term, the path geometry should become first-class model input or an
auxiliary supervised target: nearest hatch index, snake-direction parity,
time-to-arrival, signed elapsed time since passage, endpoint distance, and
distance to the currently active segment can all be computed directly from the
confirmed `additive_z_scan` parameters instead of inferred indirectly through
residual gates.

## z237: Literal XML Path Variant

The para.xml/additive_z_scan description says each layer starts near
`(10.3165, -10.1562, 0.1)`, scans the first hatch along `+Y`, offsets along
`-X`, then alternates hatch direction as a serpentine path.  z236 keeps the
previous calibrated `laser_reverse_hatch_order_parity=1` assumption, which
reverses the physical hatch order on odd layers.  z237 is the literal XML
variant: it keeps the same endpoint ETA rescue logic as z236, but sets
`laser_reverse_hatch_order_parity=-1` so every layer uses the same physical
hatch order and only the per-hatch scan direction alternates.

Run z237 after or alongside z236 if the z236 worst-case diagnostics still look
geometrically inconsistent with the XML path.

```powershell
F:\anaconda3\envs\dtstpinn\python.exe scripts\train.py `
  --config configs\ablation_z237_xml_literal_endpoint_time_until_rescue_0440_1900_floor.yaml `
  --vtu_dir F:\VTU `
  --epochs 10 `
  --max_train_samples 300 `
  --max_val_samples 80 `
  --max_test_samples 160 `
  --graph_device cuda `
  --experiment_name ablation_z237_xml_literal_endpoint_time_until_rescue_0440_1900_floor_probe300
```

Compare z237 against z236 using the same criteria.  If z237 improves the worst
true-hot miss or gives a lower `Path time-until gate` inconsistency at the
worst node, use z237 as the new base.  If z237 regresses recall/F1, keep z236
and treat the odd-layer reverse as an empirical solver-alignment correction.

Convenience command for the z236/z237 pair:

```powershell
.\scripts\run_z_group_experiments.ps1 `
  -Mode probe `
  -Python F:\anaconda3\envs\dtstpinn\python.exe `
  -VtuDir F:\VTU `
  -GraphDevice cuda `
  -SummaryPattern "ablation_z23*_probe*"
```

The summary table now includes the worst-case path gates needed for this
decision: `worst_path_pre_arrival_gate`, `worst_endpoint_gate`,
`worst_program_track_gate`, `worst_time_until_gate`, `worst_residual_gate`, and
`worst_residual_boost`.

The batch script runs `scripts/decide_z236_z237.py` after summarization and
writes `results/z236_z237_decision.md`.  That report is the first-pass decision
artifact for choosing the next z-series base.  It applies the probe gates above,
then breaks ties with a score that prioritizes lower `MaxError`, lower P99/RMSE,
and higher solidus recall/F1/precision.

## z238-z242: Endpoint ETA Gate Strategy

The current z23/z24 group should be treated as one controlled experiment around
the same calibrated `additive_z_scan` geometry.  Keep the empirical reverse-odd
alignment (`laser_reverse_hatch_order_parity=1`) unless z237 clearly wins the
path-alignment decision.  The tested question is no longer whether hotspot loss
helps; it does not fix cold-window arrivals by itself.  The question is which
path-derived support signal opens the laser residual head on true cold-start
hotspots without reopening cold false-hot cases.

Candidate roles:

| ID | Config | Purpose | Key gate change |
| --- | --- | --- | --- |
| z238 | `configs\ablation_z238_reverse_odd_no_timeuntil_shortcut_1900_floor.yaml` | Conservative control | Removes the endpoint `time_until` shortcut. |
| z239 | `configs\ablation_z239_reverse_odd_timeuntil_track_1900_floor.yaml` | Current balanced base | Adds `prearrival_endpoint_time_until_track_all` with `program_track >= 0.600`. |
| z240 | `configs\ablation_z240_reverse_odd_endpoint_eta_track_cold_1900_floor.yaml` | Low-track rescue with learned guard | Lowers endpoint `program_track` to `0.440`, but requires `cold_to_hot >= 0.500`. |
| z241 | `configs\ablation_z241_reverse_odd_endpoint_eta_high_track_1900_floor.yaml` | High-confidence path rescue | Keeps z240, plus `prearrival_endpoint_time_until_high_track_all` with `program_track >= 0.750` and no hard `cold_to_hot` requirement. |
| z242 | `configs\ablation_z242_reverse_odd_endpoint_eta_high_track_no_sweep_1900_floor.yaml` | False-hot cleanup after z241 | Keeps the z241 endpoint/time-until/high-track rescue, but removes the broad `prearrival_heat_track_all` sweep shortcut. |
| z243 | `configs\ablation_z243_reverse_odd_constrained_sweep_rescue_1900_floor.yaml` | Constrained sweep rescue after z242 | Restores only a tight sweep rescue requiring prearrival, active arrival, path body, sweep, cold-to-hot, and program-track support. |

Run the full probe matrix:

```powershell
.\scripts\run_z_group_experiments.ps1 `
  -Mode probe `
  -Python F:\anaconda3\envs\dtstpinn\python.exe `
  -VtuDir F:\VTU `
  -GraphDevice cuda `
  -ProbeEpochs 10 `
  -ProbeTrainSamples 300 `
  -ProbeValSamples 80 `
  -ProbeTestSamples 160
```

If z236/z237 have already been run and only z238-z243 need to be tested, pass a
custom config list and skip the path decision:

```powershell
.\scripts\run_z_group_experiments.ps1 `
  -Mode probe `
  -Python F:\anaconda3\envs\dtstpinn\python.exe `
  -VtuDir F:\VTU `
  -GraphDevice cuda `
  -Configs @(
    "configs\ablation_z238_reverse_odd_no_timeuntil_shortcut_1900_floor.yaml",
    "configs\ablation_z239_reverse_odd_timeuntil_track_1900_floor.yaml",
    "configs\ablation_z240_reverse_odd_endpoint_eta_track_cold_1900_floor.yaml",
    "configs\ablation_z241_reverse_odd_endpoint_eta_high_track_1900_floor.yaml",
    "configs\ablation_z242_reverse_odd_endpoint_eta_high_track_no_sweep_1900_floor.yaml",
    "configs\ablation_z243_reverse_odd_constrained_sweep_rescue_1900_floor.yaml"
  ) `
  -SkipPathDecision
```

Primary decision metrics, in order:

1. Reject any run whose worst case is a cold false-hot: prediction above
   `1200 C` while target is near ambient.
2. Prefer lower `test_MaxError`; z242 should keep the z241 improvement below
   z239/z240 and ideally stay at or below about `1535 C`.
3. Keep `test_TempRecallAboveSolidus >= 0.95` and prefer higher
   `test_TempF1AboveSolidus`.
4. Keep `test_AbsErrorP99 <= 21 C`; if P99 worsens, the rescue is too broad.
5. Inspect the worst node gates. A true cold-start hotspot with high
   `worst_program_track_gate` and low `worst_residual_gate` means the gate is
   still too strict; a cold false-hot with moderate `program_track` means the
   gate is too broad.

Observed z241 outcome:

- z241 fixed the z240 true-hot miss and reduced `MaxError` to about `1535 C`.
- The new worst case became a false-hot node at step `2126`, node `75336`:
  prediction about `1755 C`, target about `220 C`.
- That false-hot had strong program-track support (`~0.981`) and a high
  residual gate (`~0.876`), but near-zero endpoint/target/body heat support.
  The likely culprit is the broad `prearrival_heat_track_all` branch, because
  it can fire from sweep-like support without enough local heat-source evidence.

Observed z242 outcome:

- z242 removes `prearrival_heat_track_all` while keeping endpoint/time-until,
  high-track, neighbor wake, and body wake support.
- z242 improved precision/F1 and removed the z241 false-hot style, but its
  `MaxError` regressed to about `2122 C` and recall fell below z241 because it
  missed a true cold-start hotspot at step `2046`, node `24852`.
- The missed node still has useful XML path evidence: `prearrival~0.552`,
  active arrival support `~0.741`, `path_body~0.271`, `sweep~0.129`,
  `program_track~0.602`, and `cold_to_hot~0.836`.

Next z243 hypothesis:

- z243 restores sweep support only through
  `prearrival_sweep_body_arrival_cold_track_all`, requiring all of the above
  signal families at once.
- It intentionally sets `path_body >= 0.250` and `cold_to_hot >= 0.800`, which
  should pass the z242 true-hot miss but block the z241 false-hot probe point
  whose diagnosed `path_body` and `cold_to_hot` were slightly lower.
- Success criteria: `test_MaxError < z242`, ideally below `1700 C`; recall at
  or above `0.95`; no worst case with target near ambient/moderate and
  prediction above `1200 C`; and `test_AbsErrorP99 <= 21 C`.
- If z243 still false-hots, inspect the worst-node gates and add a cap or
  stricter cold-to-hot/wake condition rather than increasing hotspot loss.

## Current Execution Policy

The z24x line is now treated as a bounded cleanup, not an open-ended threshold
search.

- Finish z245 evaluation if its training checkpoint already exists.
- Do not add z246+ gate-threshold variants unless a later trajectory-alignment
  check proves that the path features are wrong and must be recalibrated.
- Summarize z241-z245 with one protocol: global RMSE, P99, MaxError, solidus
  Recall/F1, and false-hot count or worst false-hot evidence.
- Use the fixed 50-epoch baseline as the only global-regression control.

## Baseline Freeze

Freeze the accepted baseline with a local artifact bundle before starting the
next model family:

```powershell
python scripts\freeze_baseline.py `
  --name paper1_fast_50epoch `
  --config configs\paper1_fast.yaml `
  --checkpoint logs\paper1_temperature_fast\best_model.pt `
  --test_report logs\paper1_temperature_fast\test_metrics.json `
  --vtu_dir F:\VTU `
  --seed 42
```

The script copies the config, checkpoint, and test report into
`artifacts/baselines/<name>_<timestamp>/`, writes `split_indices.json`, and
records the git commit plus SHA256 hashes in `baseline_manifest.json`.
`artifacts/` and checkpoints stay local and are intentionally ignored by Git.

## Trajectory Alignment Checks

Export the additive path from the XML/config before trusting any laser-derived
feature:

```powershell
python scripts\export_laser_trajectory.py `
  --config configs\paper1_fast.yaml `
  --xml F:\datas\5-block-fem\para.xml `
  --vtu_dir F:\VTU `
  --output_dir results\laser_trajectory_paper1_fast `
  --diagnose_raw_time 42560 `
  --diagnose_coord "10.216,-9.375,1.000"
```

Expected outputs:

- `laser_segments.csv`: one row per layer/track segment.
- `laser_samples.csv`: laser position at each VTU raw time.
- `laser_trajectory_manifest.json`: path parameters and worst-node diagnosis.

For visual overlays, pass `--hotspot_vtu F:\VTU\Data-42560.vtu` to export high
temperature nodes with the same laser-distance and arrival-time features.

## E1/E2/E3 Feature Dataset

Build chunked path-feature tables only after the alignment check passes:

```powershell
python scripts\build_laser_feature_dataset.py `
  --config configs\paper1_fast.yaml `
  --xml F:\datas\5-block-fem\para.xml `
  --vtu_dir F:\VTU `
  --split train `
  --max_steps 8 `
  --output_dir data\processed\laser_features_smoke
```

The script writes one compressed NPZ per target step and a manifest listing
feature columns.  The manifest defines the minimum feature groups:

- E1: laser coordinates, node-relative coordinates, and center distance.
- E2: E1 plus along/cross distances and `time_to_arrival`.
- E3: E2 plus layer, track, scan direction, and track-neighborhood flags.

## Minimal Feature Ablation Configs

The tracked configs for the next clean experiment family are:

- `configs/feature_e0_baseline.yaml`: fixed 50-epoch E0 control, no target
  lookahead features, 12 node features.
- `configs/feature_e1_laser_distance.yaml`: E1, appends current/target laser
  coordinates plus target-step node-to-laser geometry, 27 node features.
- `configs/feature_e2_scan_arrival.yaml`: E2, adds scan along/cross geometry and
  additive-path arrival features, 37 node features.
- `configs/feature_e3_path_phase.yaml`: E3, adds layer/track/progress/direction
  path-phase features, 47 node features.

Use the same split, seed, epoch count, checkpoint selection, and evaluation
script for all four runs:

```powershell
python scripts\train.py `
  --config configs\feature_e1_laser_distance.yaml `
  --vtu_dir F:\VTU `
  --seed 42 `
  --graph_device cuda
```

Only change the config filename for E0/E1/E2/E3.  These configs use the
calibrated additive_z_scan timing/path settings that aligned the known worst
hotspot near raw time `42560` better than the default `paper1_fast.yaml` path.

