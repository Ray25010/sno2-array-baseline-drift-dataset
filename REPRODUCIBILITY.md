# Reproduction coverage

## Validate the download

```bash
python -m pip install -r requirements.txt
python code/verify_release.py
```

This verifies every recorded SHA-256 checksum, decompresses measurement files
and checks their original byte hashes, checks unique row joins and role counts,
and matches each target's saved evaluation labels to the measured values. It
independently recomputes RMSE, MAE, bias, MSE, residual variance and normalized
RMSE for every historical fold and all macro results.

## Refit the historical models

```bash
python code/train_historical.py --output rerun_historical
python code/verify_release.py --prediction-directory rerun_historical/predictions
```

Only NumPy and pandas are required. This portable implementation is derived
from the archived information-control experiment. It reselects parameters in
the training sessions for all three input families and refits each outer model.
It does not depend on the original workstation's directories or archived
selected alphas. After fitting, it checks new predictions against the frozen
per-row reference (absolute tolerance `1e-8 V`). Standardization, the Ridge
sum-of-squares objective, alpha grid and tie rule match the archived experiment.

`--targets 10` runs one target; multiple numbers such as `--targets 1 10 22 23`
are accepted. `--families full_peer` restricts fitting to the primary input
family. Such runs are subsets for convenience, rather than replacements for
the 25-target evaluation. Frozen data and outputs are never overwritten.

## Other evidence

- `results/forward/`: saved fold scores from historical expanding-window fits.
- `results/long_recording/`: saved per-target chronological scores and figure
  source data from historical-selected configurations.
- `results/historical/`: includes saved output-component ablations, inner
  selections and feature audits alongside the per-row predictions.

The portable refit entry point covers the primary historical experiment and
its information controls. Full refits for every supplementary comparator and
the long-recording residual models are not included in this initial deposit.
Their supplied score tables can be reaggregated; they are not presented as
independent from-data training reproductions. There are no learned-model
binary files or raw serial packets in this release.

## Release validation

Before the initial upload, all 25 targets and all three historical input
families were refitted from the released measurements, including fresh inner
parameter selection. The largest difference from the archived predictions was
`8.88e-16 V`. Both the archived and refitted predictions passed the independent
score checks for all 2,100 target-session-method combinations. Details are in
`metadata/release_validation.json` and environment versions are in
`metadata/validation_environment.json`.

## Integrity and versions

Compression is lossless and uses reproducible gzip headers. Original and
compressed hashes are in `metadata/dataset_inventory.json`. Git line-ending
conversion is disabled so file checksums remain stable across operating
systems. Original
experiment manifests retain input hashes, statistical tests and historical
protocol records. Local workstation paths in these manifests are replaced by
repository paths where an equivalent file is included, or by a source basename
otherwise; this transformation does not alter the archived measurement files
or result tables.

Use a Git commit identifier when citing a particular snapshot. No DOI has been
assigned by this deposit. No software environment beyond the versions recorded
in `metadata/validation_environment.json` is claimed to have been tested.
