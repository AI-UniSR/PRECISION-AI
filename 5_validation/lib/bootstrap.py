"""Event-stratified bootstrap of the validation metrics.

Each resample has the size of the cohort and keeps its number of events.
Resample i is drawn with random_state = i, so every model compared on a
cohort (ensemble, SENECA, Cox benchmark) is evaluated on the same resamples
and differences between models are paired. Predictions are never refitted:
the cached risk scores and survival curves are resampled with the patients.
95% CIs are percentile intervals of the resampled values.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
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

# Metrics for which the paired difference ensemble - Cox is kept for every resample
PAIRED_DIFF_BASE_METRICS: List[str] = (
    ["c_index", "c_index_ipcw", "mean_auc"]
    + [f"auc_{int(t)}m" for t in CLINICAL_TIMEPOINTS_MONTHS]
    + [f"brier_{int(t)}m" for t in CLINICAL_TIMEPOINTS_MONTHS]
    + ["ibs_overall"]
)


def stratified_bootstrap_sample(df: pd.DataFrame, stratify_col: str = "event",
                                random_state: Optional[int] = None) -> pd.DataFrame:
    """Resample of the rows of df with replacement, stratified on stratify_col."""
    return resample(df, replace=True, n_samples=len(df), stratify=df[stratify_col].values,
                    random_state=random_state)


def compute_bootstrap_metrics(
    y_train: np.ndarray,
    tte: np.ndarray,
    event: np.ndarray,
    risk_ml: np.ndarray,
    risk_seneca: Optional[np.ndarray] = None,
    surv_preds_ml: Optional[np.ndarray] = None,
    grid: Optional[np.ndarray] = None,
    horizon: Optional[float] = None,
    risk_cox: Optional[np.ndarray] = None,
    surv_preds_cox: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """All metrics of one resample.

    Keys end in _ml (ensemble), _seneca or _cox; _diff is ensemble - SENECA and
    _diff_ml_cox is ensemble - Cox. SENECA has no survival function, so it has
    no Brier score or IBS. surv_preds_* are S(t) matrices (patients x grid).
    """
    y_boot = make_structured_array(event, tte)
    tau = horizon if horizon is not None else CLINICAL_HORIZON_MONTHS
    metrics: Dict[str, float] = {}

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

    metrics["c_index_ipcw_ml"] = compute_c_index_ipcw(y_train, y_boot, risk_ml, tau=tau)
    if risk_seneca is not None:
        metrics["c_index_ipcw_seneca"] = compute_c_index_ipcw(y_train, y_boot, risk_seneca, tau=tau)
        metrics["c_index_ipcw_diff"] = metrics["c_index_ipcw_ml"] - metrics["c_index_ipcw_seneca"]

    metrics.update({f"{k}_ml": v for k, v in compute_time_dependent_auc(y_train, y_boot, risk_ml).items()})
    if risk_seneca is not None:
        metrics.update({f"{k}_seneca": v
                        for k, v in compute_time_dependent_auc(y_train, y_boot, risk_seneca).items()})
        if "mean_auc_ml" in metrics and "mean_auc_seneca" in metrics:
            metrics["mean_auc_diff"] = metrics["mean_auc_ml"] - metrics["mean_auc_seneca"]

    if surv_preds_ml is not None and grid is not None:
        metrics.update(_brier_ibs_metrics(y_train, y_boot, surv_preds_ml, grid, tau, "ml"))

    if risk_cox is not None:
        try:
            metrics["c_index_cox"] = compute_c_index(y_boot, risk_cox)
        except Exception:
            metrics["c_index_cox"] = np.nan
        metrics["c_index_ipcw_cox"] = compute_c_index_ipcw(y_train, y_boot, risk_cox, tau=tau)
        metrics.update({f"{k}_cox": v for k, v in compute_time_dependent_auc(y_train, y_boot, risk_cox).items()})
        if surv_preds_cox is not None and grid is not None:
            metrics.update(_brier_ibs_metrics(y_train, y_boot, surv_preds_cox, grid, tau, "cox"))
        for base in PAIRED_DIFF_BASE_METRICS:
            if f"{base}_ml" in metrics and f"{base}_cox" in metrics:
                metrics[f"{base}_diff_ml_cox"] = metrics[f"{base}_ml"] - metrics[f"{base}_cox"]
    return metrics


def _brier_ibs_metrics(y_train, y_boot, surv_preds, grid, tau, suffix) -> Dict[str, float]:
    """Brier score at the time points and IBS on one resample (keys brier_{t}m_{suffix},
    ibs_overall_{suffix}); grid points beyond the resample's follow-up are dropped."""
    out: Dict[str, float] = {}
    try:
        y_train_t = truncate_survival_times(y_train, tau)
        y_boot_t = truncate_survival_times(y_boot, tau)
        valid = (grid > 0) & (grid < float(y_boot_t["time"].max()))
        g, sp = grid[valid], surv_preds[:, valid]
        if len(g) >= 3:
            for t in CLINICAL_TIMEPOINTS_MONTHS:
                idx = int(np.argmin(np.abs(g - t)))
                if np.abs(g[idx] - t) < 0.5:
                    try:
                        _, bs = sksurv_brier_score(y_train_t, y_boot_t, sp[:, idx], g[idx])
                        out[f"brier_{int(round(t))}m_{suffix}"] = float(bs[0])
                    except Exception:
                        pass
            try:
                out[f"ibs_overall_{suffix}"] = float(sksurv_integrated_brier_score(y_train_t, y_boot_t, sp, g))
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Brier/IBS not computed on this resample: {e}")
    return out


def aggregate_bootstrap_results(distributions: List[Dict[str, float]], ci_level: float = 0.95
                                ) -> Dict[str, Dict[str, float]]:
    """Mean, median, SD, percentile CI and number of valid resamples of every metric."""
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
