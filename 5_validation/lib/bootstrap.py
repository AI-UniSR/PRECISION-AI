"""
Bootstrap Utilities for Survival Analysis Validation

Stratified resampling, per-iteration metric computation, CI aggregation,
and paired statistical comparison (ML vs SENECA).

Consolidated from the two identical copies of bootstrap_utils.py.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

from scipy import stats
from sklearn.utils import resample

from sksurv.metrics import (
    brier_score as sksurv_brier_score,
    integrated_brier_score as sksurv_integrated_brier_score,
)

from lib.survival_metrics import (
    CLINICAL_HORIZON_MONTHS,
    CLINICAL_TIMEPOINTS_MONTHS,
    compute_c_index,
    compute_c_index_ipcw,
    compute_time_dependent_auc,
    make_structured_array,
    truncate_survival_times,
)

logger = logging.getLogger(__name__)


def stratified_bootstrap_sample(
    df: pd.DataFrame,
    stratify_col: str = "event",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a stratified bootstrap sample preserving event rate.

    Uses ``sklearn.utils.resample`` with the ``stratify`` argument.
    """
    if stratify_col not in df.columns:
        logger.warning(f"Column '{stratify_col}' not found; falling back to simple bootstrap.")
        return resample(df, replace=True, n_samples=len(df), random_state=random_state)

    return resample(
        df,
        replace=True,
        n_samples=len(df),
        stratify=df[stratify_col].values,
        random_state=random_state,
    )


def compute_bootstrap_metrics(
    y_train: np.ndarray,
    tte: np.ndarray,
    event: np.ndarray,
    risk_ml: np.ndarray,
    risk_seneca: Optional[np.ndarray] = None,
    surv_preds_ml: Optional[np.ndarray] = None,
    grid: Optional[np.ndarray] = None,
    horizon: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all survival metrics for one bootstrap iteration.

    Parameters
    ----------
    y_train : structured array
        Training censoring distribution (for IPCW).
    tte, event : arrays
        Bootstrap-sample survival data.
    risk_ml : array
        ML risk scores for the bootstrap sample.
    risk_seneca : array, optional
        SENECA risk scores.  ``None`` skips SENECA metrics.
    surv_preds_ml : np.ndarray, optional
        Pre-computed S(t) matrix for ML (n_boot_samples × n_grid).
        Required for Brier / IBS.
    grid : np.ndarray, optional
        Temporal grid corresponding to columns of *surv_preds_ml*.
    horizon : float, optional
        Evaluation horizon for IPCW truncation.

    Returns
    -------
    dict with C-index (Harrell + Uno), time-dependent AUC, Brier at
    clinical timepoints, IBS, and all pairwise differences.
    """
    y_boot = make_structured_array(event, tte)
    metrics: Dict[str, float] = {}
    eff_horizon = horizon if horizon is not None else CLINICAL_HORIZON_MONTHS

    # ── Harrell's C-index ────────────────────────────────────────────
    try:
        metrics["c_index_ml"] = compute_c_index(y_boot, risk_ml)
    except Exception:
        metrics["c_index_ml"] = np.nan
    if risk_seneca is not None:
        try:
            metrics["c_index_seneca"] = compute_c_index(y_boot, risk_seneca)
            metrics["c_index_diff"] = metrics["c_index_ml"] - metrics["c_index_seneca"]
        except Exception:
            metrics["c_index_seneca"] = np.nan
            metrics["c_index_diff"] = np.nan

    # ── Uno's C-index (IPCW) ────────────────────────────────────────
    try:
        metrics["c_index_ipcw_ml"] = compute_c_index_ipcw(
            y_train, y_boot, risk_ml, tau=eff_horizon,
        )
    except Exception:
        metrics["c_index_ipcw_ml"] = np.nan
    if risk_seneca is not None:
        try:
            metrics["c_index_ipcw_seneca"] = compute_c_index_ipcw(
                y_train, y_boot, risk_seneca, tau=eff_horizon,
            )
            metrics["c_index_ipcw_diff"] = (
                metrics["c_index_ipcw_ml"] - metrics["c_index_ipcw_seneca"]
            )
        except Exception:
            metrics["c_index_ipcw_seneca"] = np.nan
            metrics["c_index_ipcw_diff"] = np.nan

    # ── Time-dependent AUC (IPCW) ───────────────────────────────────
    try:
        auc_ml = compute_time_dependent_auc(y_train, y_boot, risk_ml)
        for key, val in auc_ml.items():
            metrics[f"{key}_ml"] = val
    except Exception:
        pass
    if risk_seneca is not None:
        try:
            auc_seneca = compute_time_dependent_auc(y_train, y_boot, risk_seneca)
            for key, val in auc_seneca.items():
                metrics[f"{key}_seneca"] = val
            if "mean_auc_ml" in metrics and "mean_auc_seneca" in metrics:
                metrics["mean_auc_diff"] = (
                    metrics["mean_auc_ml"] - metrics["mean_auc_seneca"]
                )
        except Exception:
            pass

    # ── Brier scores + IBS (IPCW, ML only — SENECA has no S(t)) ────
    if surv_preds_ml is not None and grid is not None:
        try:
            y_train_t = truncate_survival_times(y_train, eff_horizon)
            y_boot_t = truncate_survival_times(y_boot, eff_horizon)
            boot_max = float(y_boot_t["time"].max())
            valid = (grid > 0) & (grid < boot_max)
            g = grid[valid]
            sp = surv_preds_ml[:, valid]

            if len(g) >= 3:
                for t in CLINICAL_TIMEPOINTS_MONTHS:
                    idx = int(np.argmin(np.abs(g - t)))
                    if np.abs(g[idx] - t) < 0.5:
                        try:
                            _, bs = sksurv_brier_score(
                                y_train_t, y_boot_t, sp[:, idx], g[idx],
                            )
                            metrics[f"brier_{int(round(t))}m_ml"] = float(bs[0])
                        except Exception:
                            pass

                try:
                    ibs = sksurv_integrated_brier_score(
                        y_train_t, y_boot_t, sp, g,
                    )
                    metrics["ibs_overall_ml"] = float(ibs)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Brier/IBS failed in bootstrap: {e}")

    return metrics


def aggregate_bootstrap_results(
    distributions: List[Dict[str, float]],
    ci_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate bootstrap distributions → median, mean, 95 % CI.

    Returns dict[metric_name → {mean, median, std, ci_lower, ci_upper, n_valid}].
    """
    df = pd.DataFrame(distributions)
    alpha = (1 - ci_level) / 2
    aggregated: Dict[str, Dict[str, float]] = {}

    for col in df.columns:
        vals = df[col].dropna()
        if len(vals) == 0:
            aggregated[col] = {k: np.nan for k in ["mean", "median", "std", "ci_lower", "ci_upper"]}
            aggregated[col]["n_valid"] = 0
            continue
        aggregated[col] = {
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "std": float(vals.std()),
            "ci_lower": float(np.percentile(vals, 100 * alpha)),
            "ci_upper": float(np.percentile(vals, 100 * (1 - alpha))),
            "n_valid": int(len(vals)),
        }
    return aggregated


def compare_bootstrap_distributions(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    test: str = "paired_t",
) -> Dict[str, float]:
    """
    Test whether two paired bootstrap distributions differ significantly.

    Parameters
    ----------
    dist_a, dist_b : 1-D arrays of bootstrap metric values.
    test : 'paired_t' or 'wilcoxon'.
    """
    valid = ~(np.isnan(dist_a) | np.isnan(dist_b))
    a, b = dist_a[valid], dist_b[valid]
    if len(a) < 3:
        return {"test": test, "statistic": np.nan, "p_value": np.nan, "n_pairs": 0}

    if test == "paired_t":
        stat, p = stats.ttest_rel(a, b)
    elif test == "wilcoxon":
        stat, p = stats.wilcoxon(a, b)
    else:
        raise ValueError(f"Unknown test: {test}")

    return {"test": test, "statistic": float(stat), "p_value": float(p), "n_pairs": int(len(a))}
