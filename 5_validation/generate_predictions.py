"""
Generate Predictions — Unified Prediction Step

Generates ALL predictions upfront for both cohorts (internal test + temporal
validation) and both models (ML ensemble + SENECA), decoupling prediction
from evaluation so that metrics can be re-computed without re-predicting.

Outputs
-------
predictions/ directory with:
  - internal_predictions.csv
  - temporal_predictions.csv
  - training_risk_scores.json   (for threshold computation — no leakage)
  - prediction_metadata.json

Note on model loading
---------------------
The script loads the ensemble model from an MLflow Model Registry using the
``models:/<name>/<version>`` URI convention.  For local reproduction, pass
a local run URI or filesystem path instead:

    --model_name runs:/<run_id>/model        (MLflow run artifact)
    --model_name file:///path/to/mlmodel     (local directory)
"""

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from lib.seneca_model import SENECAModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DCA_TIMEPOINTS = [6, 12, 18, 24]
SURVIVAL_TIMEPOINTS = [6, 12, 18, 24]


# ------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------

def _load_data(path: str) -> pd.DataFrame:
    """Load CSV or Parquet from a file or AML output directory."""
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith((".csv", ".parquet"))]
        if not files:
            raise FileNotFoundError(f"No data files in {path}")
        path = os.path.join(path, files[0])
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _extract_risk_scores(predictions) -> np.ndarray:
    """Normalise model.predict() output to a 1-D risk-score array."""
    if isinstance(predictions, pd.DataFrame):
        return predictions.iloc[:, 0].values
    arr = np.asarray(predictions)
    return arr.flatten() if arr.ndim == 2 and arr.shape[1] == 1 else arr


def _unwrap_model(model):
    """Extract underlying Python model from MLflow PyFunc wrapper."""
    if hasattr(model, "predict_survival_function"):
        return model
    if hasattr(model, "_model_impl") and hasattr(model._model_impl, "python_model"):
        return model._model_impl.python_model
    return model


def _survival_at_timepoints(model, X: pd.DataFrame, timepoints: list) -> dict:
    """Evaluate S(t) and P(event ≤ t) at each timepoint.  Returns dict of arrays."""
    actual = _unwrap_model(model)
    if not hasattr(actual, "predict_survival_function"):
        logger.warning("predict_survival_function unavailable")
        return {}
    surv_funcs = actual.predict_survival_function(X)
    result = {}
    for t in timepoints:
        s = np.array([float(fn(t)) for fn in surv_funcs])
        result[f"S_{int(t)}m"] = np.clip(s, 0, 1)
        result[f"P_event_{int(t)}m"] = np.clip(1.0 - s, 0, 1)
    return result


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--external_data", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mlflow.start_run()
    mlflow.log_param("model_name", args.model_name)
    mlflow.log_param("model_version", args.model_version)

    # ── Load datasets ────────────────────────────────────────────────
    df_train = _load_data(args.training_data)
    df_test = _load_data(args.test_data)
    df_ext = _load_data(args.external_data)

    # Normalise survival columns
    for df in [df_train, df_test, df_ext]:
        if "os" in df.columns and "death" in df.columns:
            df.rename(columns={"os": "tte", "death": "event"}, inplace=True)

    # Validate
    for name, df in [("train", df_train), ("test", df_test), ("external", df_ext)]:
        assert "tte" in df.columns and "event" in df.columns, f"{name}: missing tte/event"
        invalid = (df["tte"] <= 0).sum()
        if invalid:
            raise ValueError(f"{name}: {invalid} samples with tte ≤ 0")
        logger.info(f"{name}: n={len(df)}, events={int(df['event'].sum())} "
                    f"({100*df['event'].mean():.1f}%)")

    # ── Feature matrices ─────────────────────────────────────────────
    drop_cols = ["tte", "event", "id"]
    X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])
    X_ext = df_ext.drop(columns=[c for c in drop_cols if c in df_ext.columns])

    # ── Load ML model ────────────────────────────────────────────────
    model_uri = f"models:/{args.model_name}/{args.model_version}"
    logger.info(f"Loading model: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    # ── ML predictions on TRAINING set (for threshold computation) ──
    logger.info("Predicting on training set (thresholds only)...")
    train_scores = _extract_risk_scores(model.predict(X_train))
    train_meta = {
        "scores": train_scores.tolist(),
        "mean": float(train_scores.mean()),
        "std": float(train_scores.std()),
        "min": float(train_scores.min()),
        "max": float(train_scores.max()),
        "n": len(train_scores),
    }
    with open(out / "training_risk_scores.json", "w") as f:
        json.dump(train_meta, f, indent=2)

    # ── ML predictions on INTERNAL test set ──────────────────────────
    logger.info("Predicting on internal test set...")
    test_scores = _extract_risk_scores(model.predict(X_test))
    surv_test = _survival_at_timepoints(model, X_test, SURVIVAL_TIMEPOINTS)

    internal_df = pd.DataFrame({"tte": df_test["tte"].values,
                                "event": df_test["event"].values,
                                "ml_risk_score": test_scores})
    for col, arr in surv_test.items():
        internal_df[col] = arr
    internal_df.to_csv(out / "internal_predictions.csv", index=False, float_format="%.6f")

    # ── ML predictions on TEMPORAL validation set ────────────────────
    logger.info("Predicting on temporal validation set...")
    ext_scores = _extract_risk_scores(model.predict(X_ext))
    surv_ext = _survival_at_timepoints(model, X_ext, SURVIVAL_TIMEPOINTS)

    temporal_df = pd.DataFrame({"tte": df_ext["tte"].values,
                                "event": df_ext["event"].values,
                                "ml_risk_score": ext_scores})
    for col, arr in surv_ext.items():
        temporal_df[col] = arr

    # ── SENECA predictions on temporal set ───────────────────────────
    logger.info("Computing SENECA predictions...")
    seneca = SENECAModel()
    seneca_preds = seneca.predict(df_ext)
    temporal_df["seneca_risk_score"] = seneca_preds["seneca_risk_score"].values
    temporal_df["seneca_group"] = seneca_preds["seneca_group"].values
    temporal_df["seneca_complete_case"] = seneca_preds["complete_case"].values

    temporal_df.to_csv(out / "temporal_predictions.csv", index=False, float_format="%.6f")

    # ── Metadata ─────────────────────────────────────────────────────
    meta = {
        "model_name": args.model_name,
        "model_version": args.model_version,
        "internal": {"n": len(df_test), "events": int(df_test["event"].sum())},
        "temporal": {
            "n": len(df_ext),
            "events": int(df_ext["event"].sum()),
            "seneca_complete_cases": int(seneca_preds["complete_case"].sum()),
        },
        "training": {"n": len(df_train), "events": int(df_train["event"].sum())},
        "survival_timepoints": SURVIVAL_TIMEPOINTS,
    }
    with open(out / "prediction_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── MLflow metrics ───────────────────────────────────────────────
    mlflow.log_metric("n_internal", len(df_test))
    mlflow.log_metric("n_temporal", len(df_ext))
    mlflow.log_metric("n_seneca_complete", int(seneca_preds["complete_case"].sum()))
    mlflow.log_metric("ml_risk_internal_mean", float(test_scores.mean()))
    mlflow.log_metric("ml_risk_temporal_mean", float(ext_scores.mean()))
    mlflow.log_artifacts(str(out), artifact_path="predictions")

    mlflow.end_run()
    logger.info("=== Generate Predictions Complete ===")


if __name__ == "__main__":
    main()
