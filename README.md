# SnO2 Array Baseline Drift Dataset

Multi-session baseline-voltage recordings from a 25-channel SnO2 gas sensor
array under varying temperature and humidity conditions, with row-level
calibration/evaluation assignments and prediction results.

This repository accompanies the manuscript **Peer-Assisted Residual Calibration
for Cross-Session Baseline Voltage Reconstruction in a SnO2 Gas Sensor Array**
by Bairui Zhang, Puhao He, Xin Lan, Shuojin Zhao and Yingbang Yao,
School of Integrated Circuits, Guangdong University of Technology.
Corresponding author: Yingbang Yao, ybyao@gdut.edu.cn.

## Contents

| Path | Contents |
| --- | --- |
| `data/historical/recordings.csv.gz` | 74,183 synchronized observations: 25 voltage channels, environmental measurements and acquisition metadata. |
| `data/historical/row_roles.csv.gz` | One-to-one `row_id` assignments: 8,004 calibration, 43,791 evaluation and 22,388 unscored rows. |
| `data/historical/anchor_windows.csv` | Timing, sample count and measured conditions for each of the four calibration windows per session. |
| `data/historical/session_map.csv` | Paper labels S1-S7, retained internal identifiers, acquisition times and row counts. |
| `data/long_recording/recordings.csv.gz` | The 90.85-hour September 15-19 recording, including 91 hourly segments. |
| `data/long_recording/segment_summary.csv` | Measured temperature/humidity and stability summaries by segment. |
| `results/historical/` | Shared-predictor nested-LOSO results for the primary method and information controls. |
| `results/historical/predictions/` | All 43,791 evaluation predictions for each of 25 targets and all 12 method branches. |
| `results/forward/` | Chronological results using preceding sessions for training. |
| `results/long_recording/` | Historical-only penalty-selected long-recording scores and figure source data. |
| `code/train_historical.py` | Portable, from-data reproduction of the historical predictor and information controls. |
| `code/verify_release.py` | Checksums, dataset joins, prediction-to-measurement alignment and independent score reconstruction. |
| `metadata/` | Dataset inventory, input hashes and original experiment provenance. |

The published tables are processed, synchronized measurement records. They
preserve the source CSV bytes through lossless gzip compression; they are not
raw serial packets. Acquisition firmware, circuit design files, manuscript
drafts and third-party article PDFs are outside this dataset.

## Quick start

Python 3.10 or newer is required. Install NumPy and pandas, then validate the
downloaded repository:

```bash
python -m pip install -r requirements.txt
python code/verify_release.py
```

Read the measured voltage table directly without extracting it:

```python
import pandas as pd

measurements = pd.read_csv("data/historical/recordings.csv.gz")
roles = pd.read_csv("data/historical/row_roles.csv.gz")
data = measurements.merge(
    roles[["row_id", "role", "anchor_name"]], on="row_id", validate="one_to_one"
)
print(data.groupby(["session_id", "role"]).size())
```

Reproduce all historical fits, including fresh inner-session parameter selection:

```bash
python code/train_historical.py --output rerun_historical
python code/verify_release.py --prediction-directory rerun_historical/predictions
```

For a shorter reproduction run:

```bash
python code/train_historical.py --targets 10 --output rerun_sensor10
```

The training script writes to a new directory and checks its predictions against
the frozen, per-row reference. It does not read archived predictions or selected
parameters while fitting. Runtime depends on the machine and BLAS library.

## Task and evaluation

For each target, a historical Ridge predictor uses the **other 24 synchronous
voltages** and environmental measurements. Four brief target-observation windows
update the predictor's output level and scale. Later target readings are used
only for offline evaluation. This is synchronous reconstruction, rather than
forecasting from past observations alone.

The historical experiment leaves out one entire session at a time and rotates
the target over all 25 channels (175 target-session folds). Inner selection uses
only the six outer-training sessions. The uncalibrated predictor selects the
Ridge penalty; the same predictor is then used for no correction, offset and
bounded-affine correction. Environment-only and peer-mean controls each select
their own penalty by the same procedure.

The shared-predictor macro RMSE is **0.0527643 V before calibration** and
**0.0322430 V after bounded-affine calibration** (38.9% reduction). Target means
improve for 23/25 channels, while session means improve in 4/7 sessions. Scores
and their statistical units are documented in [PROTOCOL.md](PROTOCOL.md).

The measurements come from one array, without controlled gas dosing. Voltages
are measured divider outputs, not gas concentrations. Long-recording scores
use a separate chronological protocol; its environmental residual correction
is distinct from the historical scalar output-affine map.

## Documentation and provenance

- [DATA_DICTIONARY.md](DATA_DICTIONARY.md): fields, units and missing values.
- [PROTOCOL.md](PROTOCOL.md): splits, calibration budget and score definitions.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): reproduction coverage and commands.
- `metadata/dataset_inventory.json`: original CSV hashes, compression hashes and sizes.
- `checksums.sha256`: SHA-256 checksums of release files.
- [CITATION.cff](CITATION.cff): dataset authors and repository citation metadata.

This deposit publishes the study's analysis-ready measurements and supporting
scores. Use the Git commit identifier to identify the version used in an
analysis. No dataset DOI has been assigned.

## Licensing

| Material | Licence |
| --- | --- |
| Measurements, results, metadata and accompanying documentation | [CC BY-NC 4.0](LICENSE-DATA.txt) |
| Software in `code/` | [PolyForm Noncommercial 1.0.0](LICENSE-CODE.md) |

Both licences permit reuse for their specified noncommercial purposes. Data
reuse requires attribution; software redistribution must preserve the licence
and required notice. The software is source available under a noncommercial
licence. See [LICENSE.md](LICENSE.md) for scope and attribution details.

For commercial permissions, contact Yingbang Yao (ybyao@gdut.edu.cn).
