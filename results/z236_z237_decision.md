# z236/z237 Decision

Summary CSV: `results\z_experiment_summary.csv`

Decision: **No run fully passes the probe gates; keep z236 only as the next diagnostic base.**

## Selected Runs

| run | test_RMSE | test_MAE | test_AbsErrorP99 | test_MaxError | test_TempRecallAboveSolidus | test_TempPrecisionAboveSolidus | test_TempF1AboveSolidus | test_worst_step | test_worst_node | test_worst_pred | test_worst_target | test_worst_abs_error | test_worst_path_pre_arrival_gate | test_worst_endpoint_gate | test_worst_program_track_gate | test_worst_time_until_gate | test_worst_residual_gate | test_worst_residual_boost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ablation_z236_xml_reverse_odd_endpoint_time_until_rescue_0440_1900_floor_probe300 | 14.0493 | 2.97406 | 21.2483 | 1.670e+03 | 0.956403 | 0.298978 | 0.455548 | 2017 | 35671 | 1.690e+03 | 20 | 1.670e+03 | 0.523283 | 0.435232 | 0.466019 | 0.00222143 | 0.875925 | 1.668e+03 |
| ablation_z237_xml_literal_endpoint_time_until_rescue_0440_1900_floor_probe300 | 19.9057 | 3.18105 | 28.6296 | 2.751e+03 | 0.027248 | 0.00687285 | 0.0109769 | 2131 | 24219 | 197.431 | 2.948e+03 | 2.751e+03 | 0 | 0 | 0 | 0 | 0 | 0 |

## Probe Gate Failures

- z236: worst case is a cold false-hot above 1200 C; solidus recall 0.9564 < 0.960; solidus F1 0.4555 < 0.458; AbsErrorP99 21.25 > 21.2 C
- z237: solidus recall 0.0272 < 0.960; solidus F1 0.0110 < 0.458; solidus precision 0.0069 < 0.295; AbsErrorP99 28.63 > 21.2 C; MaxError 2750.9 C is not below 2000 C

## Diagnostic Notes

- z236 worst time-until/residual gate: 0.00222143 / 0.875925
- z237 worst time-until/residual gate: 0 / 0
- z236 winning means the odd-layer hatch-order reversal remains the better empirical solver alignment.
- z237: ETA says near-future arrival but residual gate stayed closed; relax/trace support logic.
