"""Verify released file integrity and reconstruct historical scores from predictions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def digest(path, decompress=False):
    hasher = hashlib.sha256()
    opener = gzip.open if decompress else open
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-directory", type=Path,
                        default=ROOT / "results/historical/predictions")
    args = parser.parse_args()
    lines = (ROOT / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        assert digest(ROOT / relative) == expected, relative
    inventory = json.loads((ROOT / "metadata/dataset_inventory.json").read_text(encoding="utf-8"))
    for record in inventory:
        path = ROOT / record["repository_path"]
        assert digest(path) == record["sha256"], record["repository_path"]
        assert digest(path, path.suffix == ".gz") == record["source_csv_sha256"], record["repository_path"]
    data = pd.read_csv(ROOT / "data/historical/recordings.csv.gz", low_memory=False)
    roles = pd.read_csv(ROOT / "data/historical/row_roles.csv.gz")
    assert len(data) == len(roles) == 74183
    assert data.row_id.is_unique and roles.row_id.is_unique
    joined = data.merge(roles, on="row_id", suffixes=("", "_role"), validate="one_to_one")
    for column in ["session_id", "source_block", "timestamp_dt"]:
        assert joined[column].eq(joined[column + "_role"]).all(), column
    assert roles.role.value_counts().to_dict() == {
        "evaluation": 43791, "excluded": 22388, "calibration": 8004
    }
    data = data.set_index("row_id")
    evaluation_ids = set(roles.loc[roles.role.eq("evaluation"), "row_id"])
    long = pd.read_csv(ROOT / "data/long_recording/recordings.csv.gz", low_memory=False)
    sensors = [f"sensor_{i:03d}_V" for i in range(1, 26)]
    assert len(long) == 304103
    complete = long[sensors].notna().all(axis=1)
    assert int(complete.sum()) == 304089
    assert set(long.loc[~complete, "segment"]) == {35}

    expected = pd.read_csv(ROOT / "results/historical/fold_metrics.csv").set_index(["target", "session", "method"])
    measured = []
    maxima = {name: 0.0 for name in ["rmse_V", "mae_V", "bias_V", "mse_V2",
                                    "centered_variance_V2", "normalized_rmse_pct"]}
    prediction_paths = sorted(args.prediction_directory.glob("sensor_*_V.csv.gz"))
    assert prediction_paths, "No per-target prediction files found"
    for path in prediction_paths:
        target = path.name.removesuffix(".csv.gz")
        prediction = pd.read_csv(path)
        assert prediction.row_id.is_unique and set(prediction.row_id) == evaluation_ids, target
        actual = data.loc[prediction.row_id]
        np.testing.assert_allclose(prediction.y_true, actual[target], rtol=0, atol=1e-12)
        for column in ["session_id", "source_block", "timestamp_dt"]:
            assert np.array_equal(prediction[column].to_numpy(), actual[column].to_numpy()), (target, column)
        methods = [column for column in prediction if column not in (
            "row_id", "session_id", "source_block", "timestamp_dt", "y_true")]
        for session, group in prediction.groupby("session_id"):
            truth = group.y_true.to_numpy()
            for method in methods:
                error = group[method].to_numpy() - truth
                assert np.isfinite(error).all(), (target, session, method)
                values = {"rmse_V": np.sqrt(np.mean(error ** 2)), "mae_V": np.mean(abs(error)),
                          "bias_V": error.mean(), "mse_V2": np.mean(error ** 2),
                          "centered_variance_V2": error.var(),
                          "normalized_rmse_pct": 100 * np.sqrt(np.mean(error ** 2)) / abs(truth.mean())}
                reference = expected.loc[(target, session, method)]
                assert len(group) == int(reference.n)
                for metric, value in values.items():
                    difference = abs(value - reference[metric])
                    maxima[metric] = max(maxima[metric], float(difference))
                    assert difference < 1e-7, (target, session, method, metric, difference)
                measured.append({"target": target, "session": session, "method": method, **values})

    frame = pd.DataFrame(measured)
    result = {"checksums_checked": len(lines), "historical_rows": len(data),
              "roles": roles.role.value_counts().to_dict(), "long_recording_rows": len(long),
              "complete_long_rows": int(complete.sum()), "prediction_targets_checked": len(prediction_paths),
              "fold_metrics_checked": len(frame), "max_score_differences": maxima}
    if len(prediction_paths) == 25 and frame.method.nunique() == 12:
        aggregate = frame.groupby("method").rmse_V.mean()
        summary = pd.read_csv(ROOT / "results/historical/summary.csv").set_index("method")
        np.testing.assert_allclose(aggregate, summary.loc[aggregate.index, "rmse_V"], rtol=0, atol=1e-8)
        scores = frame.loc[frame.method.isin(["full_peer__none", "full_peer__affine"])]
        session_means = scores.pivot_table(index="session", columns="method", values="rmse_V")
        target_means = scores.pivot_table(index="target", columns="method", values="rmse_V")
        folds = scores.pivot_table(index=["target", "session"], columns="method", values="rmse_V")
        improved_sessions = int((session_means.full_peer__affine < session_means.full_peer__none).sum())
        improved_targets = int((target_means.full_peer__affine < target_means.full_peer__none).sum())
        worse_folds = int((folds.full_peer__affine > folds.full_peer__none).sum())
        assert (improved_sessions, improved_targets, worse_folds) == (4, 23, 75)
        result.update({"raw_macro_rmse_V": float(aggregate["full_peer__none"]),
                       "affine_macro_rmse_V": float(aggregate["full_peer__affine"]),
                       "improved_sessions": improved_sessions, "improved_targets": improved_targets,
                       "worse_target_session_folds": worse_folds})
    result["status"] = "PASSED"
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
