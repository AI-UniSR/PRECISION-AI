"""Unpenalised Cox benchmark on the ten predictors of the ensemble.

The benchmark differs from the ensemble only in the modelling method:

- patients: the training set of data_prep.py, checked against fingerprints
  stored in the registered ensemble (the column means learnt by each base
  learner's imputer and the follow-up times of the Cox baseline hazards);
- predictors: the union of the base learners' features (selected_features.json);
- preprocessing: the ensemble's 'cox' learner (IterativeImputer ->
  RobustScaler -> CoxPHSurvivalAnalysis, alpha = 0) is cloned, unfitted, and
  refitted here on all ten predictors.

The ensemble-Cox difference therefore measures the effect of the modelling
method; the ensemble-SENECA difference also includes the choice of
predictors. Validation data are not used here.

Outputs: the fitted pipeline (joblib), the ordered predictor list, hazard
ratios with 95% Wald CIs, and a JSON report of the checks.
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import sksurv
from sklearn.base import clone

from lib.cox_benchmark import COEFFICIENTS_FILE, FEATURES_FILE, FIT_REPORT_FILE, PIPELINE_FILE, cox_coefficient_table
from lib.survival_metrics import compute_c_index, make_structured_array

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_STEPS = ["IterativeImputer", "RobustScaler", "CoxPHSurvivalAnalysis"]


def _load_data(path: str) -> pd.DataFrame:
    """CSV or Parquet file, or the first such file in a folder."""
    if os.path.isdir(path):
        path = os.path.join(path, [f for f in os.listdir(path) if f.endswith((".csv", ".parquet"))][0])
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def _unwrap_model(model):
    """Python model inside an MLflow pyfunc wrapper."""
    if hasattr(model, "_model_impl") and hasattr(model._model_impl, "python_model"):
        return model._model_impl.python_model
    return model


def _times_check(label, expected_times, observed_times):
    expected_times = np.asarray(expected_times, dtype=float)
    same_shape = expected_times.shape == observed_times.shape
    return {"fingerprint": label,
            "match": bool(same_shape and np.allclose(expected_times, observed_times, rtol=0, atol=1e-9)),
            "n_expected": int(expected_times.size), "n_observed": int(observed_times.size)}


def _verify_derivation_set(ensemble, df: pd.DataFrame) -> list:
    """Compare df with the training data the registered ensemble was fitted on.

    Fingerprints: the column means of each base learner's IterativeImputer
    (initial_imputer_.statistics_) and the unique follow-up times of the Breslow
    baseline hazard of the 'cox' learner and of the meta-learner.
    """
    observed_times = np.unique(df["tte"].astype(float).to_numpy())
    checks = []
    for name in sorted(ensemble.models):
        steps = getattr(ensemble.models[name], "steps", None) or []
        first, final = (steps[0][1], steps[-1][1]) if steps else (None, None)
        features = list(ensemble.selected_features.get(name, []))
        imputer = getattr(first, "initial_imputer_", None)
        if imputer is not None and features:
            expected = np.asarray(imputer.statistics_, dtype=float)
            observed = df[features].astype(float).mean(axis=0, skipna=True).to_numpy()
            same_shape = expected.shape == observed.shape
            checks.append({
                "fingerprint": f"{name}: imputer column means",
                "match": bool(same_shape and np.allclose(expected, observed, rtol=1e-9, atol=1e-9)),
                "max_abs_diff": float(np.max(np.abs(expected - observed))) if same_shape else None,
            })
        names_in = getattr(first, "feature_names_in_", None)
        if names_in is not None and features:
            checks.append({"fingerprint": f"{name}: imputer feature names", "match": list(names_in) == features})
        hazard = getattr(final, "cum_baseline_hazard_", None)
        if hazard is not None:
            checks.append(_times_check(f"{name}: baseline-hazard follow-up times", hazard.x, observed_times))
    meta_hazard = getattr(getattr(ensemble, "meta_learner", None), "cum_baseline_hazard_", None)
    if meta_hazard is not None:
        checks.append(_times_check("meta_learner: baseline-hazard follow-up times", meta_hazard.x, observed_times))
    return checks


def _benchmark_pipeline(ensemble):
    """Unfitted clone of the ensemble's linear learner (same steps and settings)."""
    base = ensemble.models["cox"]
    steps = [type(step).__name__ for _, step in getattr(base, "steps", [])]
    if steps != EXPECTED_STEPS:
        raise RuntimeError(f"The 'cox' learner is {steps}, expected {EXPECTED_STEPS}")
    estimator = base.steps[-1][1]
    if not np.all(np.asarray(estimator.alpha, dtype=float) == 0):
        raise RuntimeError(f"The 'cox' learner is penalised (alpha={estimator.alpha})")
    if getattr(estimator, "ties", "breslow") != "breslow":
        raise RuntimeError("The coefficient CIs assume Breslow ties")
    return clone(base)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training_data", required=True, help="Training set (data_prep.py)")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mlflow.start_run()

    df = _load_data(args.training_data)
    if df[["tte", "event"]].isna().any().any() or (df["tte"] <= 0).any():
        raise ValueError("Training set has missing or non-positive follow-up.")

    model_uri = f"models:/{args.model_name}/{args.model_version}"
    ensemble = _unwrap_model(mlflow.pyfunc.load_model(model_uri))
    features = sorted({f for feats in ensemble.selected_features.values() for f in feats})
    logger.info(f"Ensemble predictors ({len(features)}): {features}")

    fingerprints = _verify_derivation_set(ensemble, df)
    for check in fingerprints:
        logger.info(f"  {'OK  ' if check['match'] else 'FAIL'} {check}")
    if not fingerprints or not all(check["match"] for check in fingerprints):
        raise RuntimeError("The training set does not match the data the registered ensemble was fitted on.")

    pipeline = _benchmark_pipeline(ensemble)
    X = df[features]
    y = make_structured_array(df["event"].values, df["tte"].values)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(X, y)
    fit_warnings = sorted({f"{w.category.__name__}: {w.message}" for w in caught})

    coefficients, diagnostics = cox_coefficient_table(pipeline, X, df["tte"].to_numpy(), df["event"].to_numpy())
    if not diagnostics["linear_predictor_matches_pipeline"]:
        raise RuntimeError("The coefficient table does not reproduce the pipeline's predictions.")
    logger.info("Hazard ratios per IQR:\n" + coefficients[
        ["feature", "hr_per_iqr", "hr_per_iqr_ci_lower", "hr_per_iqr_ci_upper", "p_value"]
    ].round(4).to_string(index=False))
    apparent_c = compute_c_index(y, pipeline.predict(X))

    joblib.dump(pipeline, out / PIPELINE_FILE)
    with open(out / FEATURES_FILE, "w") as f:
        json.dump(features, f, indent=2)
    coefficients.to_csv(out / COEFFICIENTS_FILE, index=False, float_format="%.6g")
    report = {
        "model": "Unpenalised Cox PH benchmark (same predictors as the ensemble)",
        "ensemble_model_uri": model_uri,
        "derivation": {"n": len(df), "events": int(df["event"].sum())},
        "predictors": features,
        "ensemble_base_learner_predictors": {k: list(v) for k, v in sorted(ensemble.selected_features.items())},
        "pipeline_source": "sklearn.base.clone(ensemble.models['cox']), refitted on the training set",
        "pipeline_steps": [{"step": name, "class": type(step).__name__,
                            "params": {k: repr(v) for k, v in step.get_params(deep=False).items()}}
                           for name, step in pipeline.steps],
        "derivation_fingerprints": fingerprints,
        "fit_warnings": fit_warnings,
        "coefficient_diagnostics": diagnostics,
        "apparent_c_index_derivation": float(apparent_c),
        "versions": {"scikit-learn": sklearn.__version__, "scikit-survival": sksurv.__version__},
    }
    with open(out / FIT_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    mlflow.log_params({"cox_benchmark_n_predictors": len(features)})
    mlflow.log_metric("cox_benchmark_apparent_c_index", float(apparent_c))
    mlflow.log_artifacts(str(out), artifact_path="cox_benchmark")
    mlflow.end_run()


if __name__ == "__main__":
    main()
