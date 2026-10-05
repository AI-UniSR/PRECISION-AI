"""Unpenalised Cox benchmark: loading the fitted model and its coefficient table.

The benchmark is a Cox proportional hazards model on the ten predictors of
the ensemble, fitted once on the training set by fit_cox_benchmark.py;
generate_predictions.py and compute_metrics.py only load it. CoxBenchmarkModel
gives it the interface of the registered ensemble (predict and
predict_survival_function on the full feature matrix), so that the same
evaluation code serves both models.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sksurv  # noqa: F401  (adds predict_survival_function to sklearn Pipeline)
from scipy import stats

logger = logging.getLogger(__name__)

PIPELINE_FILE = "cox_benchmark_pipeline.pkl"
FEATURES_FILE = "cox_benchmark_features.json"
COEFFICIENTS_FILE = "cox_benchmark_coefficients.csv"
FIT_REPORT_FILE = "cox_benchmark_fit_report.json"


class CoxBenchmarkModel:
    """Fitted Cox pipeline applied to its own predictors."""

    def __init__(self, pipeline, features: List[str]):
        self.pipeline = pipeline
        self.features = list(features)

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.features if f not in X.columns]
        if missing:
            raise ValueError(f"Cox benchmark predictors missing from input: {missing}")
        return X[self.features]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Linear predictor (log relative hazard); higher means higher risk."""
        return np.asarray(self.pipeline.predict(self._select(X)), dtype=float)

    def predict_survival_function(self, X: pd.DataFrame):
        """Breslow survival curves, one per patient."""
        return self.pipeline.predict_survival_function(self._select(X))


def load_cox_benchmark(model_dir: str) -> CoxBenchmarkModel:
    """Model written by fit_cox_benchmark.py."""
    model_dir = Path(model_dir)
    pipeline = joblib.load(model_dir / PIPELINE_FILE)
    with open(model_dir / FEATURES_FILE) as f:
        features = json.load(f)
    logger.info(f"Cox benchmark, {len(features)} predictors: {features}")
    return CoxBenchmarkModel(pipeline, features)


def breslow_information(Z: np.ndarray, time: np.ndarray, event: np.ndarray, coef: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Observed information and score of the Breslow partial likelihood at coef.

    Same likelihood as CoxPHSurvivalAnalysis(ties="breslow"): the risk set of a
    death at t_i is {j : t_j >= t_i}, tied deaths included. At the maximum
    likelihood estimate the score is close to 0.
    """
    Z = np.asarray(Z, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    eta = Z @ np.asarray(coef, dtype=float)
    w = np.exp(eta - eta.max())  # the constant factor cancels in every ratio

    order = np.argsort(time, kind="mergesort")
    t, e, Zs, ws = time[order], event[order], Z[order], w[order]
    # Reverse cumulative sums are sums over positions >= i; at the first
    # position of a block of tied times they cover the whole risk set.
    s0 = np.cumsum(ws[::-1])[::-1]
    s1 = np.cumsum((ws[:, None] * Zs)[::-1], axis=0)[::-1]
    s2 = np.cumsum((ws[:, None, None] * Zs[:, :, None] * Zs[:, None, :])[::-1], axis=0)[::-1]
    first = np.searchsorted(t, t, side="left")

    deaths = np.where(e)[0]
    risk0 = s0[first[deaths]]
    zbar = s1[first[deaths]] / risk0[:, None]
    information = ((s2[first[deaths]] / risk0[:, None, None]).sum(axis=0)
                   - np.einsum("ij,ik->jk", zbar, zbar))
    score = (Zs[deaths] - zbar).sum(axis=0)
    return information, score


def cox_coefficient_table(pipeline, X: pd.DataFrame, time: np.ndarray, event: np.ndarray
                          ) -> Tuple[pd.DataFrame, Dict]:
    """Hazard ratios with 95% Wald CIs of a fitted benchmark pipeline.

    The design matrix is the pipeline's own preprocessing of the training data
    (imputation, then RobustScaler), so the coefficients are per scaled unit,
    i.e. per interquartile range (IQR); since the scaler is linear, the HR per
    original unit is exp(beta / IQR). Standard errors treat the single
    imputation as fixed.
    """
    Z = np.asarray(pipeline[:-1].transform(X), dtype=float)
    beta = np.asarray(pipeline[-1].coef_, dtype=float).ravel()
    scaler = pipeline[-2]
    information, score = breslow_information(Z, time, event, beta)

    diagnostics = {
        "max_abs_score_at_estimate": float(np.max(np.abs(score))),
        "information_condition_number": float(np.linalg.cond(information)),
        "linear_predictor_matches_pipeline": bool(np.allclose(pipeline.predict(X), Z @ beta,
                                                              rtol=1e-8, atol=1e-8)),
    }
    try:
        covariance = np.linalg.inv(information)
        diagnostics["covariance"] = "inverse"
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(information)
        diagnostics["covariance"] = "pseudo-inverse (singular information)"
        logger.warning("Singular information matrix: standard errors from the pseudo-inverse.")

    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    z_crit = stats.norm.ppf(0.975)
    center = getattr(scaler, "center_", None)
    scale = getattr(scaler, "scale_", None)
    center = np.zeros_like(beta) if center is None else np.asarray(center, dtype=float)
    scale = np.ones_like(beta) if scale is None else np.asarray(scale, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_value = beta / se
    coef_unit, se_unit = beta / scale, se / scale

    table = pd.DataFrame({
        "feature": list(X.columns),
        "n_missing_derivation": X.isna().sum().to_numpy(),
        "pct_missing_derivation": 100 * X.isna().mean().to_numpy(),
        "scaler_center_median": center,
        "scaler_scale_iqr": scale,
        "coef_per_iqr": beta,
        "se_per_iqr": se,
        "hr_per_iqr": np.exp(beta),
        "hr_per_iqr_ci_lower": np.exp(beta - z_crit * se),
        "hr_per_iqr_ci_upper": np.exp(beta + z_crit * se),
        "coef_per_unit": coef_unit,
        "hr_per_unit": np.exp(coef_unit),
        "hr_per_unit_ci_lower": np.exp(coef_unit - z_crit * se_unit),
        "hr_per_unit_ci_upper": np.exp(coef_unit + z_crit * se_unit),
        "z": z_value,
        "p_value": 2 * stats.norm.sf(np.abs(z_value)),
    })
    return table, diagnostics
