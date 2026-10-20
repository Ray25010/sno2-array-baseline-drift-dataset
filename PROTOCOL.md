# Experimental and evaluation protocol

## Historical task

Seven recordings from a single 25-channel array form the historical cohort.
Paper labels S1-S7 map to unchanged original identifiers in
`data/historical/session_map.csv`. The recordings began on August 2, 8, 9, 10,
11 and 27, and September 3, 2026. A microcontroller replacement occurred
between S5 and S6. Sessions are the validation and statistical sampling units.

For each target channel:

1. Leave one complete session out and train on the other six.
2. Fit constant-feature removal, centering and population-standard-deviation
   scaling inside each training fold.
3. Select Ridge alpha from `{0.01, 1, 100, 10000}` by uncalibrated RMSE
   macro-averaged over six inner held-out sessions. A tie within `1e-12 V`
   favors larger alpha.
4. Fit the selected predictor on all retained outer-training rows. It receives
   the other 24 simultaneous voltages and `T_C`, `RH_pct`, `AH_gm3`; the
   target voltage is excluded from the input. Legacy constant metadata drop out.
5. Freeze this predictor and compare uncalibrated, offset and bounded-affine
   output updates using the held-out session's four calibration windows.
6. Score only that session's predefined evaluation rows.

Environment-only Ridge and environment-plus-peer-mean Ridge use the same
nesting and alpha grid and select their own alphas. The first omits all peers;
the second replaces individual peer coordinates with their arithmetic mean.
Additional controls use four-window target-mean persistence and direct peer
mean with/without output calibration.

## Four-window update

The first continuous 300 seconds of each A1-A4 window contribute calibration
labels: 286 samples per window in S1-S6 and 285 in S7. Every window is reduced
to one equally weighted mean prediction/observation pair `(p_j, y_j)`.
Window timing and measured conditions are in `anchor_windows.csv`.

Offset is `mean(y_j - p_j)`. For the affine update, minimize

```text
mean_j (y_j - a*p_j - b)^2 + 0.01*(a - 1)^2,
subject to 0.8 <= a <= 1.2.
```

Shrinkage is in squared volts. This changes only the slope `a` and intercept
`b`; it does not refit the historical predictor. No evaluation target values
enter scaling, alpha selection or calibration. A target is excluded from its
own synchronous inputs in every target rotation.

## Metrics

Fold RMSE is computed on each target-session pair. The primary macro score
equally averages the 175 fold RMSEs (25 targets by seven sessions); with this
balanced design it also equals the mean of seven session means. It is not a
pooled pointwise RMSE. Target means and session means answer different questions.

The primary no-update/offset/affine RMSEs are 0.0527642972, 0.0354931848 and
0.0322430328 V. The raw-to-affine gain is 0.0205212644 V (38.9%).
Target means improve in 23/25 targets; session means in 4/7 sessions;
75/175 folds worsen. Session bootstrap and exact paired tests in
`metadata/original_experiments/historical_manifest.json` use seven sessions,
not 175 independent experiments. The paired bootstrap interval is
[-0.00040598, 0.04440313] V and the exact Wilcoxon p-value is 0.375.

## Chronological results

`results/forward/fold_metrics.csv` contains historical expanding-window
validation: train on S1...S(k-1), predict Sk. S2 uses alpha 100 because one
preceding session cannot support session-level selection. Later alphas are
chosen by expanding-window validation on preceding sessions. Uncalibrated and
affine macro RMSEs are 0.08466 and 0.03704 V.

The long recording began September 15, 2026 and spans approximately 90.85 hours
and 91 hourly segments. Segments 1-36 supply calibration windows for segments
37-72; segments 37-72 supply refresh windows for segments 73-91. The selected
segments are 2/17/22/36 and 38/53/58/72, respectively, using their final 300
valid samples. Historical data alone select the predictor penalty.

The long-recording `affine` result is an **environmental affine residual
correction**, not the historical scalar map `a*p+b`. It fits the residual
against environmental coordinates. Its macro RMSEs are 0.025020762 and
0.020075424 V in the two test scopes. The supplied per-target and macro tables
contain only the historical-selected configurations; fixed-alpha sensitivity
runs are not a second primary result.
