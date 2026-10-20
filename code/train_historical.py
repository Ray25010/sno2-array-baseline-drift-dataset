"""Refit the shared-predictor historical experiment from released measurements."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ["D1", "D2", "D3", "D4", "D5", "D7_2_formal", "D8_2_20260903_formal"]
SENSORS = [f"sensor_{i:03d}_V" for i in range(1, 26)]
ENV = ["T_C", "RH_pct", "AH_gm3"]
ALPHAS = [0.01, 1.0, 100.0, 10000.0]


def ridge_path(x, y):
    active = np.ptp(x, axis=0) > 0
    mean, scale = x[:, active].mean(0), x[:, active].std(0)
    z = (x[:, active] - mean) / scale
    ym = y.mean()
    gram, rhs = z.T @ z, z.T @ (y - ym)
    coef = np.column_stack([
        np.linalg.solve(gram + alpha * np.eye(len(mean)), rhs) for alpha in ALPHAS
    ])
    return active, mean, scale, ym, coef


def predict(model, x):
    active, mean, scale, ym, coef = model
    return ((x[:, active] - mean) / scale) @ coef + ym


def calibrate(prediction, target, masks):
    pm = np.array([prediction[mask].mean() for mask in masks])
    ym = np.array([target[mask].mean() for mask in masks])
    var = np.mean((pm - pm.mean()) ** 2)
    slope = (np.mean((pm - pm.mean()) * (ym - ym.mean())) + 0.01) / (var + 0.01)
    slope = float(np.clip(slope, 0.8, 1.2))
    intercept = float(ym.mean() - slope * pm.mean())
    return {
        "none": prediction,
        "offset": prediction + (ym - pm).mean(),
        "affine": slope * prediction + intercept,
    }, {"slope": slope, "intercept_V": intercept}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=list(range(1, 26)))
    parser.add_argument("--families", nargs="+", choices=["environment", "mean_peer", "full_peer"],
                        default=["environment", "mean_peer", "full_peer"])
    args = parser.parse_args()
    if len(set(args.targets)) != len(args.targets) or not all(1 <= i <= 25 for i in args.targets):
        parser.error("targets must be distinct integers from 1 to 25")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()

    data = pd.read_csv(ROOT / "data/historical/recordings.csv.gz", low_memory=False)
    roles = pd.read_csv(ROOT / "data/historical/row_roles.csv.gz")
    assert data.row_id.is_unique and roles.row_id.is_unique
    assert set(data.row_id) == set(roles.row_id)
    data = data.merge(roles[["row_id", "role", "anchor_name"]], on="row_id", validate="one_to_one")
    data = data.sort_values(["session_id", "timestamp_dt", "row_id"]).reset_index(drop=True)
    assert data.role.value_counts().to_dict() == {
        "evaluation": 43791, "excluded": 22388, "calibration": 8004
    }
    env = data[ENV].to_numpy(float)
    voltage = data[SENSORS].to_numpy(float)
    assert np.isfinite(env).all() and np.isfinite(voltage).all()
    session_ids = data.session_id.to_numpy()
    evaluation = data.role.eq("evaluation").to_numpy()
    folds, selections, inner_rows, coefficients, parity = [], [], [], [], []

    for target_number in args.targets:
        target = SENSORS[target_number - 1]
        print(f"Fitting {target}", flush=True)
        y = voltage[:, target_number - 1]
        peers = np.delete(voltage, target_number - 1, axis=1)
        peer_mean = peers.mean(axis=1)
        designs = {
            "environment": env,
            "mean_peer": np.column_stack([env, peer_mean]),
            "full_peer": np.column_stack([env, peers]),
        }
        prediction_rows = data.loc[evaluation, ["row_id", "session_id", "source_block", "timestamp_dt"]].copy()
        prediction_rows["y_true"] = y[evaluation]
        prediction_rows = prediction_rows.set_index("row_id")
        for family in args.families:
            x = designs[family]
            # A five-session fit serves both possible outer/inner omission orders.
            pair_models = {}
            for omitted in itertools.combinations(SESSIONS, 2):
                train = ~np.isin(session_ids, omitted)
                pair_models[frozenset(omitted)] = ridge_path(x[train], y[train])
            for outer in SESSIONS:
                scores = []
                for inner in SESSIONS:
                    if inner == outer:
                        continue
                    valid = (session_ids == inner) & evaluation
                    prediction = predict(pair_models[frozenset([outer, inner])], x[valid])
                    rmses = np.sqrt(np.mean((prediction - y[valid, None]) ** 2, axis=0))
                    scores.append(rmses)
                    for alpha, score in zip(ALPHAS, rmses):
                        inner_rows.append({"target": target, "family": family, "outer": outer,
                                           "inner": inner, "alpha": alpha, "rmse_V": score})
                average = np.mean(scores, axis=0)
                alpha = max(a for a, v in zip(ALPHAS, average) if v <= min(average) + 1e-12)
                selections.append({"target": target, "session": outer, "family": family, "alpha": alpha})
                train, test = session_ids != outer, session_ids == outer
                model = ridge_path(x[train], y[train])
                prediction = predict(model, x[test])[:, ALPHAS.index(alpha)]
                local = data.loc[test]
                masks = [(local.role.eq("calibration") & local.anchor_name.eq(anchor)).to_numpy()
                         for anchor in ["A1", "A2", "A3", "A4"]]
                assert [int(m.sum()) for m in masks] == [285 if outer == SESSIONS[-1] else 286] * 4
                corrected, fitted = calibrate(prediction, y[test], masks)
                coefficients.append({"target": target, "session": outer, "family": family, **fitted})
                local_eval = local.role.eq("evaluation").to_numpy()
                row_ids = local.loc[local_eval, "row_id"].to_numpy()
                truth = y[test][local_eval]
                for strategy, values in corrected.items():
                    method = family + "__" + strategy
                    values = values[local_eval]
                    error = values - truth
                    prediction_rows.loc[row_ids, method] = values
                    folds.append({"target": target, "session": outer, "method": method, "n": len(values),
                                  "rmse_V": float(np.sqrt(np.mean(error ** 2))),
                                  "mae_V": float(np.mean(np.abs(error)))})

        # Target-only persistence and direct-mean controls use the same four windows.
        for outer in SESSIONS:
            test = session_ids == outer
            local = data.loc[test]
            masks = [(local.role.eq("calibration") & local.anchor_name.eq(anchor)).to_numpy()
                     for anchor in ["A1", "A2", "A3", "A4"]]
            corrected, _ = calibrate(peer_mean[test], y[test], masks)
            constant = np.mean([y[test][mask].mean() for mask in masks])
            local_eval = local.role.eq("evaluation").to_numpy()
            row_ids = local.loc[local_eval, "row_id"].to_numpy()
            for method, values in {"anchor_constant": np.full(test.sum(), constant),
                                   "direct_mean__none": corrected["none"],
                                   "direct_mean__affine": corrected["affine"]}.items():
                prediction_rows.loc[row_ids, method] = values[local_eval]

        assert not prediction_rows.isna().any().any()
        prediction_rows.reset_index().to_csv(output / "predictions" / f"{target}.csv.gz", index=False)
        # Frozen references are read only after fitting and prediction are complete.
        frozen = pd.read_csv(ROOT / "results/historical/predictions" / f"{target}.csv.gz").set_index("row_id")
        frozen = frozen.loc[prediction_rows.index]
        numeric = [c for c in prediction_rows.columns if c not in ("session_id", "source_block", "timestamp_dt")]
        difference = float(np.max(np.abs(prediction_rows[numeric].to_numpy() - frozen[numeric].to_numpy())))
        assert difference < 1e-8, (target, difference)
        parity.append({"target": target, "max_abs_difference_V": difference})
        print(f"  archived-prediction agreement: {difference:.3g} V", flush=True)

    for name, rows in [("fold_metrics", folds), ("selected_parameters", selections),
                       ("inner_scores", inner_rows), ("coefficients", coefficients), ("parity", parity)]:
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    summary = pd.DataFrame(folds).groupby("method", as_index=False).rmse_V.mean()
    summary.to_csv(output / "summary.csv", index=False)
    (output / "run.json").write_text(json.dumps({"targets": args.targets, "families": args.families,
                                                "alphas": ALPHAS, "numpy": np.__version__,
                                                "pandas": pd.__version__}, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
