"""Predictions of the ensemble, the Cox benchmark and SENECA, computed once.

For the internal test set and the temporal cohort: ensemble risk score
(ml_risk_score) and survival probabilities S(t) and P(event <= t) at 6, 12,
18 and 24 months, the same for the Cox benchmark (cox_ prefix), and for the
temporal cohort the SENECA score, group and complete-case flag. The training
risk scores of both models are saved too: the risk-group cutoffs are their
percentiles. compute_metrics.py reads these files and never re-predicts.

The ensemble is loaded from the MLflow registry (models:/<name>/<version>);
outside Azure ML, set MLFLOW_TRACKING_URI to the store where
4_ensemble/evaluate_models.py registered it.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from lib.cox_benchmark import load_cox_benchmark
from lib.seneca_model import SENECAModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SURVIVAL_TIMEPOINTS = [6, 12, 18, 24]


def _load_data(path: str) -> pd.DataFrame:
    """CSV or Parquet file, or the first such file in a folder."""
    if os.path.isdir(path):
        path = os.path.join(path, [f for f in os.listdir(path) if f.endswith((".csv", ".parquet"))][0])
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def _risk_scores(predictions) -> np.ndarray:
    """model.predict() output as a 1-D array."""
    if isinstance(predictions, pd.DataFrame):
        return predictions.iloc[:, 0].values
    arr = np.asarray(predictions)
    return arr.flatten() if arr.ndim == 2 and arr.shape[1] == 1 else arr


def _unwrap_model(model):
    """Object exposing predict_survival_function (inside the MLflow pyfunc wrapper)."""
    if hasattr(model, "predict_survival_function"):
        return model
    if hasattr(model, "_model_impl") and hasattr(model._model_impl, "python_model"):
        return model._model_impl.python_model
    return model


def _survival_at_timepoints(model, X: pd.DataFrame) -> dict:
    """S(t) and P(event <= t) at each time point, clipped to [0, 1]."""
    surv_funcs = _unwrap_model(model).predict_survival_function(X)
    result = {}
    for t in SURVIVAL_TIMEPOINTS:
        s = np.array([float(fn(t)) for fn in surv_funcs])
        result[f"S_{t}m"] = np.clip(s, 0, 1)
        result[f"P_event_{t}m"] = np.clip(1.0 - s, 0, 1)
    return result


def _score_summary(scores: np.ndarray) -> dict:
    return {"scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std()),
            "min": float(scores.min()), "max": float(scores.max()), "n": len(scores)}


def _predictions_table(df: pd.DataFrame, X: pd.DataFrame, model) -> pd.DataFrame:
    table = pd.DataFrame({"tte": df["tte"].values, "event": df["event"].values,
                          "ml_risk_score": _risk_scores(model.predict(X))})
    for col, values in _survival_at_timepoints(model, X).items():
        table[col] = values
    return table


def _add_cox_columns(table: pd.DataFrame, X: pd.DataFrame, cox_model) -> None:
    table["cox_risk_score"] = cox_model.predict(X)
    for col, values in _survival_at_timepoints(cox_model, X).items():
        table[f"cox_{col}"] = values


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training_data", required=True, help="Training set (data_prep.py)")
    parser.add_argument("--test_data", required=True, help="Internal test set (data_prep.py)")
    parser.add_argument("--external_data", required=True, help="Temporal cohort (prepare_external.py)")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--cox_model_dir", required=True, help="Output of fit_cox_benchmark.py")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mlflow.start_run()
    mlflow.log_params({"model_name": args.model_name, "model_version": args.model_version})

    data = {"train": _load_data(args.training_data), "test": _load_data(args.test_data),
            "external": _load_data(args.external_data)}
    for name, df in data.items():
        if (df["tte"] <= 0).any():
            raise ValueError(f"{name}: non-positive follow-up")
        logger.info(f"{name}: n={len(df)}, events={int(df['event'].sum())}")
    X = {name: df.drop(columns=[c for c in ["tte", "event", "id", "center"] if c in df.columns])
         for name, df in data.items()}

    model = mlflow.pyfunc.load_model(f"models:/{args.model_name}/{args.model_version}")
    cox_model = load_cox_benchmark(args.cox_model_dir)
    ensemble_features = sorted({f for feats in _unwrap_model(model).selected_features.values() for f in feats})
    if cox_model.features != ensemble_features:
        raise ValueError(f"Cox benchmark predictors {cox_model.features} differ from the ensemble's "
                         f"{ensemble_features}")

    # Training risk scores: used only for the risk-group cutoffs
    with open(out / "training_risk_scores.json", "w") as f:
        json.dump(_score_summary(_risk_scores(model.predict(X["train"]))), f, indent=2)
    with open(out / "training_risk_scores_cox.json", "w") as f:
        json.dump(_score_summary(cox_model.predict(X["train"])), f, indent=2)

    internal = _predictions_table(data["test"], X["test"], model)
    _add_cox_columns(internal, X["test"], cox_model)
    internal.to_csv(out / "internal_predictions.csv", index=False, float_format="%.6f")

    temporal = _predictions_table(data["external"], X["external"], model)
    seneca = SENECAModel().predict(data["external"])
    temporal["seneca_risk_score"] = seneca["seneca_risk_score"].values
    temporal["seneca_group"] = seneca["seneca_group"].values
    temporal["seneca_complete_case"] = seneca["complete_case"].values
    _add_cox_columns(temporal, X["external"], cox_model)
    temporal.to_csv(out / "temporal_predictions.csv", index=False, float_format="%.6f")

    meta = {
        "model_name": args.model_name,
        "model_version": args.model_version,
        "internal": {"n": len(data["test"]), "events": int(data["test"]["event"].sum())},
        "temporal": {"n": len(data["external"]), "events": int(data["external"]["event"].sum()),
                     "seneca_complete_cases": int(seneca["complete_case"].sum())},
        "training": {"n": len(data["train"]), "events": int(data["train"]["event"].sum())},
        "survival_timepoints": SURVIVAL_TIMEPOINTS,
        "cox_benchmark": {"predictors": cox_model.features, "columns_prefix": "cox_"},
    }
    with open(out / "prediction_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    mlflow.log_artifacts(str(out), artifact_path="predictions")
    mlflow.end_run()


if __name__ == "__main__":
    main()
