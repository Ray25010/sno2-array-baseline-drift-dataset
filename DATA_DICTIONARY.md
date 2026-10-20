# Data dictionary

## Historical measurements

`data/historical/recordings.csv.gz` is a lossless copy of the original
`input_25ch.csv`. Each row is one synchronized measurement. Join the role table
on `row_id`, not on file position or timestamps alone.

| Field | Meaning | Unit / format |
| --- | --- | --- |
| `row_id` | Unique historical observation key | Integer |
| `session_id` | Original session identifier; S1-S7 mapping is in `session_map.csv` | String |
| `timestamp_dt` | Host timestamp with UTC offset | ISO datetime, UTC |
| `source_block` | Acquisition stage, including A1-A4 calibration and subsequent test stages | String |
| `source_session` | Original acquisition grouping | String |
| `T_C` | External temperature measurement | Degrees Celsius |
| `RH_pct` | External relative humidity measurement | Percent RH |
| `AH_gm3` | Derived absolute humidity | Grams per cubic metre |
| `sensor_001_V` ... `sensor_025_V` | Measured divider-output voltages | Volts |
| `heater_V` | Recorded nominal heater setting metadata | Volts |
| `Pd_amount`, `Pt_amount` | Legacy formulation metadata | Nominal preparation proportions |
| `load_resistance_kohm` | Legacy load-resistor metadata | Kilohms |
| `supply_voltage_V` | Legacy supply metadata | Volts |

The legacy hardware/formulation columns are constant in this historical table
and are removed by training-fold constant-feature filtering. In particular,
`Pd_amount` and `Pt_amount` do **not** describe the composition of each of the
25 individual channels. These columns should not be used to infer channel
composition or actual operating-voltage measurements. `supply_voltage_V` is
not an input to the released historical predictor.

Environmental inputs were acquired by the external temperature/humidity
sensor, rather than the array board's onboard environmental sensor.

## Row roles

`data/historical/row_roles.csv.gz` has the same 74,183 unique `row_id` values:

| `role` | Count | Use when the session is held out |
| --- | ---: | --- |
| `calibration` | 8,004 | Four window means supply target labels for output updating. |
| `evaluation` | 43,791 | Target labels are withheld from prediction and used for scoring. |
| `excluded` | 22,388 | Unscored anchor-stage rows outside the selected calibration windows. |

The role table also repeats `session_id`, `source_block` and `timestamp_dt` for
join verification. `anchor_name` is A1, A2, A3 or A4 on calibration rows and
empty otherwise. A held-out row's role does not restrict its use if its session
instead belongs to a training fold: all retained training-session rows are used.

## Long recording

`data/long_recording/recordings.csv.gz` retains the original segmented table:

| Field | Meaning |
| --- | --- |
| `sensor_001_V` ... `sensor_025_V` | Measured voltages in volts. |
| `host_time_utc` | Host UTC timestamp; use this explicit UTC field for interpretation. |
| `timestamp` | Companion timestamp field stored without a timezone suffix. |
| `T_C`, `RH_pct` | External environmental measurements, degrees Celsius and percent RH. |
| `session_id` | Recording identifier. |
| `sample_index` | Original sample sequence identifier. |
| `frame_status` | Recorded frame status. |
| `hour_segment`, `segment` | One-hour plan-aligned segment indices. |
| `dT_5min`, `dRH_5min` | Rolling temperature and humidity change diagnostics. |
| `stable_like` | Archived stability indicator; a diagnostic rather than a sampling gate. |

The source table contains 304,103 rows. Fourteen rows in segment 35 lack sensor
values, leaving 304,089 complete target-peer observations. Empty fields denote
missing values; voltage values are not filled or interpolated. Rolling
diagnostics are also missing during their initial warm-up interval. The table
includes environmental transitions, rather than only stable points.

`segment_summary.csv` provides each segment's first/last timestamp, sample count,
temperature and relative-humidity means/standard deviations, and stable fraction.

## Predictions and scores

`results/historical/predictions/sensor_NNN_V.csv.gz` contains that target's
43,791 evaluation rows. `y_true` is its measured voltage. The other numeric
columns are predictions in volts. Keys are `row_id`, `session_id`,
`source_block` and `timestamp_dt`. Family prefixes are `environment`,
`mean_peer` and `full_peer`; correction suffixes are `none`, `offset` and
`affine`. Extra controls are `anchor_constant`, `direct_mean__none` and
`direct_mean__affine`.

`fold_metrics.csv` has one row per target, held-out session and method.
`n` is the number of scored samples; `rmse_V`, `mae_V` and `bias_V` are in
volts, `mse_V2` and `centered_variance_V2` in squared volts.
`normalized_rmse_pct` is 100 times RMSE divided by the absolute mean target
voltage on that fold's evaluation rows.

`metadata/dataset_inventory.json` records the byte-level source hash for each
measurement file. The original CSV schema and values are unchanged.
