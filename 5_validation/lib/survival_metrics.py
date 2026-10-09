"""Discrimination and prediction-error metrics of the validation pipeline.

Conventions (times in months):

- The censoring distribution G(t) of every IPCW estimator (Uno's C, time-dependent
  AUC, Brier score, IBS) is estimated by Kaplan-Meier in the cohort being evaluated
  (internal test set, temporal cohort, subgroup or bootstrap resample), not on the
  training set (changed 9 Oct 2026). The temporal cohort comes from a later registry
  data freeze with shorter follow-up (median 10.1 vs 18.0 months), so training-set
  weights under-weighted the patients followed beyond the horizon and biased the
  Brier score and IBS downwards and the AUC and Uno's C upwards. G(t) describes the
  evaluation data, not the model, so estimating it there involves no leakage. The
  training set still sets the time grid and the evaluation horizon.
- Follow-up is truncated at the evaluation horizon tau = min(24 months,
  maximum training follow-up, maximum follow-up of the evaluated cohort):
  later times are censored at tau. Uno's C uses tau = 24 months.
- The time-dependent AUC (cumulative/dynamic) and the Brier score are
  evaluated at 6, 12, 18 and 24 months. A time point equal to tau is moved
  just below it, because the IPCW estimators need t < tau.
- "Mean AUC" is the summary returned by scikit-survival's
  cumulative_dynamic_auc: the AUCs weighted by the drop of the Kaplan-Meier
  estimate of the evaluated cohort between time points, not their plain
  average.
- The IBS integrates the Brier score over a grid of 40 percentiles (5th-95th)
  of the training follow-up times plus the four time points, and divides by
  the span of the grid.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from sksurv.metrics import (
    brier_score as sksurv_brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)

logger = logging.getLogger(__name__)

CLINICAL_HORIZON_MONTHS: float = 24.0
CLINICAL_TIMEPOINTS_MONTHS: List[float] = [6, 12, 18, 24]
CLINICAL_IBS_INTERVALS: List[Tuple[float, float]] = [(0, 6), (6, 12), (12, 18), (18, 24)]
PPV_TIMEPOINT_MONTHS: float = 6.0
NPV_TIMEPOINT_MONTHS: float = 18.0


def make_structured_array(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    """scikit-survival structured array (fields 'event', 'time')."""
    return np.array(list(zip(np.asarray(event, dtype=bool), np.asarray(time, dtype=float))),
                    dtype=[("event", bool), ("time", "<f8")])


def truncate_survival_times(y: np.ndarray, horizon: float) -> np.ndarray:
    """Censor at *horizon* every patient followed beyond it."""
    y_truncated = y.copy()
    beyond = y_truncated["time"] > horizon
    y_truncated["time"][beyond] = horizon
    y_truncated["event"][beyond] = False
    logger.debug(f"Truncated {int(beyond.sum())}/{len(y)} patients at {horizon:.2f} months")
    return y_truncated


def create_temporal_grid(y_train: np.ndarray, y_test: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Evaluation grid, the time points within follow-up, and the horizon tau.

    tau = min(24, max training time, max test time). Every grid point is
    strictly below tau; a time point equal to tau is moved just below it.
    """
    max_train = float(y_train["time"].max())
    min_time = float(y_train["time"].min())
    if y_test is not None:
        horizon = min(max_train, float(y_test["time"].max()), CLINICAL_HORIZON_MONTHS)
    else:
        horizon = min(max_train, CLINICAL_HORIZON_MONTHS)
    grid_max = np.nextafter(horizon, 0)

    clinical = [t if t < grid_max else grid_max for t in CLINICAL_TIMEPOINTS_MONTHS if t <= horizon]
    clinical = np.array(clinical, dtype=float) if clinical else np.array([horizon * 0.5])

    valid_times = y_train["time"][(y_train["time"] >= min_time) & (y_train["time"] < grid_max)]
    if len(valid_times) > 10:
        percentiles = np.percentile(valid_times, np.linspace(5, 95, 40))
    else:
        percentiles = np.percentile(valid_times, [10, 25, 50, 75, 90])
    grid = np.unique(np.concatenate([percentiles, clinical]))
    grid = np.sort(grid[(grid >= min_time) & (grid <= grid_max)])
    logger.debug(f"Grid: {len(grid)} points in [{grid.min():.2f}, {grid.max():.2f}], tau={horizon:.2f}")
    return grid, clinical, horizon


def compute_c_index(y: np.ndarray, risk_scores: np.ndarray) -> float:
    """Harrell's C-index."""
    return concordance_index_censored(y["event"], y["time"], risk_scores)[0]


def compute_c_index_ipcw(y_train: np.ndarray, y_test: np.ndarray, risk_scores: np.ndarray,
                         tau: float = CLINICAL_HORIZON_MONTHS) -> float:
    """Uno's C-index truncated at tau (NaN if not computable). The censoring distribution is
    estimated in the evaluated sample y_test (module docstring); y_train is kept for the callers."""
    y_test_t = truncate_survival_times(y_test, tau)
    try:
        return concordance_index_ipcw(y_test_t, y_test_t, risk_scores, tau=tau)[0]
    except Exception as e:
        logger.error(f"Uno's C-index not computed: {e}")
        return float("nan")


def compute_time_dependent_auc(y_train: np.ndarray, y_test: np.ndarray, risk_scores: np.ndarray,
                               time_points: Optional[List[float]] = None) -> Dict[str, float]:
    """Cumulative/dynamic AUC at each time point and the mean AUC (keys auc_{t}m, mean_auc).
    Censoring distribution from y_test; y_train sets the evaluation horizon only."""
    if time_points is None:
        time_points = CLINICAL_TIMEPOINTS_MONTHS
    _, _, horizon = create_temporal_grid(y_train, y_test)
    grid_max = np.nextafter(horizon, 0)
    pairs = [(t, t if t < horizon else grid_max) for t in time_points if t <= horizon]
    if not pairs:
        return {}
    try:
        y_test_t = truncate_survival_times(y_test, horizon)
        auc, mean_auc = cumulative_dynamic_auc(y_test_t, y_test_t, risk_scores,
                                               np.array([p[1] for p in pairs]))
    except Exception as e:
        logger.error(f"Time-dependent AUC not computed: {e}")
        return {}
    results = {"mean_auc": float(mean_auc)}
    for (label, _), value in zip(pairs, auc):
        results[f"auc_{int(round(label))}m"] = float(value)
    return results


def _unwrap_model(model):
    """Python model inside an MLflow pyfunc wrapper (the object with predict_survival_function)."""
    if hasattr(model, "predict_survival_function"):
        return model
    if hasattr(model, "_model_impl") and hasattr(model._model_impl, "python_model"):
        return model._model_impl.python_model
    return model


def predict_survival_matrix(model, X: pd.DataFrame, grid: np.ndarray) -> Optional[np.ndarray]:
    """S(t) for every patient (rows) at every grid point (columns)."""
    model = _unwrap_model(model)
    if not hasattr(model, "predict_survival_function"):
        logger.warning("Model has no predict_survival_function: Brier score not available.")
        return None
    try:
        return np.asarray([[fn(t) for t in grid] for fn in model.predict_survival_function(X)])
    except Exception as e:
        logger.error(f"predict_survival_function failed: {e}")
        return None


def compute_brier_scores(model, X_test: pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray
                         ) -> Dict[str, float]:
    """Brier score at the time points and IBS (keys brier_{t}m, ibs_{a}_{b}m, ibs_overall).
    Censoring distribution from y_test; y_train sets the time grid and the horizon."""
    grid, clinical, horizon = create_temporal_grid(y_train, y_test)
    if len(grid) == 0:
        return {}
    preds = predict_survival_matrix(model, X_test, grid)
    if preds is None:
        return {}
    y_test_t = truncate_survival_times(y_test, horizon)
    brier = np.array([sksurv_brier_score(y_test_t, y_test_t, preds[:, i], t)[1][0]
                      for i, t in enumerate(grid)])

    results: Dict[str, float] = {}
    for t in clinical:
        idx = int(np.argmin(np.abs(grid - t)))
        if np.abs(grid[idx] - t) < 0.5:
            results[f"brier_{int(round(t))}m"] = float(brier[idx])
    for start, end in CLINICAL_IBS_INTERVALS:
        if start >= grid.max():
            continue
        end = min(end, grid.max())
        mask = (grid >= start) & (grid <= end)
        if mask.sum() >= 3:
            results[f"ibs_{int(round(start))}_{int(round(end))}m"] = float(
                trapezoid(brier[mask], grid[mask]) / (end - start))
    if len(grid) >= 3:
        results["ibs_overall"] = float(trapezoid(brier, grid) / (grid.max() - grid.min()))
    return results
