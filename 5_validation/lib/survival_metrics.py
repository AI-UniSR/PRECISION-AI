"""
Survival Analysis Metrics for BTC Validation Pipeline

Consolidates the three copies of survival_analysis_utils.py into a single
authoritative module.  Uses the extended version with ``truncate_survival_times``
and the 3-value ``create_temporal_grid`` return signature.

All IPCW computations use **y_train** for the censoring distribution.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from scipy.integrate import trapezoid

from sksurv.metrics import (
    brier_score as sksurv_brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score as sksurv_integrated_brier_score,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CLINICAL CONSTANTS
# ============================================================================

CLINICAL_HORIZON_MONTHS: float = 24.0
CLINICAL_TIMEPOINTS_MONTHS: List[float] = [6, 12, 18, 24]
CLINICAL_IBS_INTERVALS: List[Tuple[float, float]] = [
    (0, 6),
    (6, 12),
    (12, 18),
    (18, 24),
]
PPV_TIMEPOINT_MONTHS: float = 6.0
NPV_TIMEPOINT_MONTHS: float = 18.0


# ============================================================================
# HELPERS
# ============================================================================

def make_structured_array(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Create a scikit-survival structured array from event/time arrays."""
    return np.array(
        list(zip(np.asarray(event, dtype=bool), np.asarray(time, dtype=float))),
        dtype=[("event", bool), ("time", "<f8")],
    )


def truncate_survival_times(y: np.ndarray, horizon: float) -> np.ndarray:
    """
    Truncate survival times to *horizon* for IPCW computation.

    Samples with ``time > horizon`` are censored at ``time = horizon``.
    This prevents IPCW weight errors when test follow-up exceeds training.
    """
    y_truncated = y.copy()
    beyond = y_truncated["time"] > horizon
    n_truncated = int(beyond.sum())
    if n_truncated > 0:
        y_truncated["time"][beyond] = horizon
        y_truncated["event"][beyond] = False
        logger.info(
            f"Truncated {n_truncated}/{len(y)} samples beyond {horizon:.2f} months"
        )
    return y_truncated


# ============================================================================
# TEMPORAL GRID
# ============================================================================

def create_temporal_grid(
    y_train: np.ndarray,
    y_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Create a unified temporal grid combining training-set percentiles and
    clinical timepoints, constrained to valid follow-up.

    Returns
    -------
    grid_months : np.ndarray
        Sorted, deduplicated grid (all points strictly < evaluation_horizon).
    clinical_months : np.ndarray
        Subset of ``CLINICAL_TIMEPOINTS_MONTHS`` within range.
    evaluation_horizon : float
        ``min(max_train, max_test, CLINICAL_HORIZON_MONTHS)`` — the upper
        bound used for IPCW truncation.
    """
    max_train = float(y_train["time"].max())
    min_time = float(y_train["time"].min())

    if y_test is not None:
        max_test = float(y_test["time"].max())
        evaluation_horizon = min(max_train, max_test, CLINICAL_HORIZON_MONTHS)
    else:
        evaluation_horizon = min(max_train, CLINICAL_HORIZON_MONTHS)

    # Strict upper bound (IPCW requires t < tau)
    grid_max = np.nextafter(evaluation_horizon, 0)

    # Clinical timepoints within range.
    # Timepoints that fall exactly on evaluation_horizon are snapped to
    # grid_max (epsilon below) so they satisfy the strict-inequality
    # constraint while still being evaluated at effectively the same time.
    snapped: List[float] = []
    for t in CLINICAL_TIMEPOINTS_MONTHS:
        if t < grid_max:
            snapped.append(t)
        elif t <= evaluation_horizon:          # boundary case (e.g. 24 == 24)
            snapped.append(grid_max)
            logger.info(
                f"Clinical timepoint {t:.0f}m snapped to grid_max "
                f"({grid_max:.16f}) for IPCW safety"
            )
    clinical_months = np.array(snapped, dtype=float) if snapped else None
    if clinical_months is None or len(clinical_months) == 0:
        clinical_months = np.array([evaluation_horizon * 0.5])
        logger.warning("No clinical timepoints within follow-up — using midpoint.")

    # Percentile grid from training data
    valid_times = y_train["time"][
        (y_train["time"] >= min_time) & (y_train["time"] < grid_max)
    ]
    if len(valid_times) > 10:
        percentile_months = np.percentile(valid_times, np.linspace(5, 95, 40))
    else:
        percentile_months = np.percentile(valid_times, [10, 25, 50, 75, 90])

    grid_months = np.unique(np.concatenate([percentile_months, clinical_months]))
    grid_months = grid_months[(grid_months >= min_time) & (grid_months <= grid_max)]
    grid_months = np.sort(grid_months)

    logger.info(
        f"Temporal grid: {len(grid_months)} points, "
        f"range=[{grid_months.min():.2f}, {grid_months.max():.2f}], "
        f"horizon={evaluation_horizon:.2f}, "
        f"clinical=[{', '.join(f'{int(round(c))}' for c in clinical_months)}]"
    )
    return grid_months, clinical_months, evaluation_horizon


# ============================================================================
# DISCRIMINATION
# ============================================================================

def compute_c_index(
    y: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    """Standard Harrell C-index (no IPCW)."""
    return concordance_index_censored(
        y["event"], y["time"], risk_scores
    )[0]


def compute_c_index_ipcw(
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk_scores: np.ndarray,
    tau: float = CLINICAL_HORIZON_MONTHS,
) -> float:
    """
    IPCW-corrected C-index.

    Censoring distribution estimated from *y_train* only (no leakage).
    Data is truncated to *tau* before computation.
    """
    y_train_t = truncate_survival_times(y_train, tau)
    y_test_t = truncate_survival_times(y_test, tau)
    try:
        return concordance_index_ipcw(y_train_t, y_test_t, risk_scores, tau=tau)[0]
    except Exception as e:
        logger.error(f"C-index IPCW failed: {e}")
        return float("nan")


def compute_time_dependent_auc(
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk_scores: np.ndarray,
    time_points: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Cumulative/dynamic time-dependent AUC (IPCW-corrected).

    Parameters
    ----------
    y_train : structured array
        Training data — used for IPCW censoring weights.
    y_test : structured array
        Test data to evaluate.
    risk_scores : array-like
        Predicted risk scores for test data.
    time_points : list of float, optional
        Timepoints in months.  Defaults to ``CLINICAL_TIMEPOINTS_MONTHS``.

    Returns
    -------
    dict with keys ``auc_{t}m`` and ``mean_auc``.
    """
    if time_points is None:
        time_points = CLINICAL_TIMEPOINTS_MONTHS

    _, _, horizon = create_temporal_grid(y_train, y_test)
    y_train_t = truncate_survival_times(y_train, horizon)
    y_test_t = truncate_survival_times(y_test, horizon)

    grid_max = np.nextafter(horizon, 0)
    # Snap boundary timepoints (t == horizon) to grid_max so they satisfy
    # the strict-inequality IPCW constraint.  Keep (original, snapped) pairs
    # so dict keys use the intended integer label.
    pairs = []  # (original_label, evaluation_value)
    for t in time_points:
        if t < horizon:
            pairs.append((t, t))
        elif t <= horizon:                    # boundary case
            pairs.append((t, grid_max))
    if not pairs:
        logger.warning("No valid timepoints for AUC computation.")
        return {}
    eval_times = np.array([p[1] for p in pairs])

    try:
        auc_values, mean_auc = cumulative_dynamic_auc(
            y_train_t, y_test_t, risk_scores, eval_times
        )
        results: Dict[str, float] = {"mean_auc": float(mean_auc)}
        for (t_label, _), a in zip(pairs, auc_values):
            results[f"auc_{int(round(t_label))}m"] = float(a)
        return results
    except Exception as e:
        logger.error(f"Time-dependent AUC failed: {e}")
        return {}


# ============================================================================
# CALIBRATION (Brier / IBS)
# ============================================================================

def _unwrap_model(model):
    """Extract the underlying Python model from an MLflow PyFunc wrapper."""
    if hasattr(model, "predict_survival_function"):
        return model
    if hasattr(model, "_model_impl") and hasattr(model._model_impl, "python_model"):
        return model._model_impl.python_model
    return model


def _predict_survival_matrix(model, X: pd.DataFrame, grid: np.ndarray) -> Optional[np.ndarray]:
    """
    Evaluate S(t) for every patient at every grid point.

    Returns
    -------
    preds : np.ndarray of shape (n_patients, len(grid)), or None on failure.
    """
    actual = _unwrap_model(model)
    if not hasattr(actual, "predict_survival_function"):
        logger.warning("Model lacks predict_survival_function — Brier scores unavailable.")
        return None
    try:
        surv_funcs = actual.predict_survival_function(X)
        return np.asarray([[fn(t) for t in grid] for fn in surv_funcs])
    except Exception as e:
        logger.error(f"predict_survival_function failed: {e}")
        return None


def compute_brier_scores(
    model,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Brier scores at clinical timepoints + IBS over clinical intervals.

    Uses ``sksurv.metrics.brier_score`` (IPCW-corrected, censoring from *y_train*).
    Both *y_train* and *y_test* are truncated to the evaluation horizon.

    Returns dict with keys ``brier_{t}m``, ``ibs_{a}_{b}m``, ``ibs_overall``.
    """
    grid, clinical, horizon = create_temporal_grid(y_train, y_test)
    if len(grid) == 0:
        return {}

    y_train_t = truncate_survival_times(y_train, horizon)
    y_test_t = truncate_survival_times(y_test, horizon)

    preds = _predict_survival_matrix(model, X_test, grid)
    if preds is None:
        return {}

    # Point-wise Brier scores on the full grid
    brier = np.zeros(len(grid))
    for i, t in enumerate(grid):
        brier[i] = sksurv_brier_score(y_train_t, y_test_t, preds[:, i], t)[1]

    results: Dict[str, float] = {}

    # Extract scores at clinical timepoints
    for t in clinical:
        idx = int(np.argmin(np.abs(grid - t)))
        if np.abs(grid[idx] - t) < 0.5:
            results[f"brier_{int(round(t))}m"] = float(brier[idx])

    # IBS over clinical intervals
    for start, end in CLINICAL_IBS_INTERVALS:
        if start >= grid.max():
            continue
        eff_end = min(end, grid.max())
        mask = (grid >= start) & (grid <= eff_end)
        if mask.sum() >= 3:
            results[f"ibs_{int(round(start))}_{int(round(eff_end))}m"] = float(
                trapezoid(brier[mask], grid[mask]) / (eff_end - start)
            )

    # Overall IBS
    if len(grid) >= 3:
        results["ibs_overall"] = float(
            trapezoid(brier, grid) / (grid.max() - grid.min())
        )

    return results


def compute_event_probabilities(
    model,
    X: pd.DataFrame,
    timepoints: Optional[List[float]] = None,
) -> Optional[Dict[int, np.ndarray]]:
    """
    Compute P(event ≤ t) = 1 − S(t) at each clinical timepoint.

    Returns dict  {timepoint_months: array_of_probabilities}  or None.
    """
    if timepoints is None:
        timepoints = [6, 18]

    actual = _unwrap_model(model)
    if not hasattr(actual, "predict_survival_function"):
        logger.warning("predict_survival_function unavailable — cannot compute event probs.")
        return None

    surv_funcs = actual.predict_survival_function(X)
    probs: Dict[int, np.ndarray] = {}
    for t in timepoints:
        p = np.clip(np.array([1.0 - float(fn(t)) for fn in surv_funcs]), 0.0, 1.0)
        probs[int(t)] = p
        logger.info(f"P(event ≤ {t}m): mean={p.mean():.4f}, range=[{p.min():.4f}, {p.max():.4f}]")
    return probs
