# z241-z245 closing summary

This note closes the z24x threshold/gating sweep. Do not extend to z246+ unless the
laser-feature ablation fails and points back to a specific trajectory-alignment bug.

All rows below are from the local test_metrics.json reports under logs/, using
160 test samples. z241-z244 did not store explicit solidus TP/FP/FN, so their
false-hot/FN counts are estimated from the shared z245 solidus support of 367 and
their recorded recall/precision. z245 uses the exact reported counts.

| run | RMSE | P99 | MaxError | solidus Recall | solidus F1 | false-hot | FN | worst step | worst node | worst pred / target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| z241 | 13.953 | 20.808 | 1535.0 | 0.9700 | 0.4638 | ~812 | ~11 | 2126 | 75336 | 1755.0 / 220.4 |
| z242 | 13.777 | 22.913 | 2122.0 | 0.8937 | 0.5408 | ~518 | ~39 | 2046 | 24852 | 127.3 / 2249.0 |
| z243 | 13.651 | 22.845 | 2095.3 | 0.9183 | 0.5270 | ~575 | ~30 | 2131 | 24218 | 129.1 / 2224.4 |
| z244 | 13.617 | 23.123 | 2056.4 | 0.9346 | 0.5249 | ~597 | ~24 | 2043 | 75126 | 141.0 / 2197.4 |
| z245 | 13.388 | 22.407 | 2045.7 | 0.9455 | 0.5254 | 607 | 20 | 2041 | 24549 | 151.0 / 2196.7 |

## Decision

z245 is the best z24x candidate by RMSE and missed-hotspot count, but the family
has reached diminishing returns. The remaining failures are not solved by more
threshold or hand-gate tuning: MaxError is still above 2000 C for z242-z245, and
solidus false-hot count remains around 500-600. This supports stopping z24x and
moving to the fixed E0-E3 laser-feature ablation protocol.

## Next gate

Evaluate E0/E1/E2/E3 with scripts/run_feature_ablation.py and compare the
canonical evaluation_report.json metrics. If E1/E2 do not improve solidus or
liquidus recall, re-check laser time/coordinate alignment before adding more
model complexity.
