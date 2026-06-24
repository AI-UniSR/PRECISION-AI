"""
Risk Stratification Utilities

Threshold definition (from training set), group assignment, Kaplan-Meier,
log-rank tests, Cox PH models, and KM statistics.

Extracted from validate_internal / validate_external duplicated code.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lifelines import CoxPHFitter, KaplanMeierFitter
from sksurv.compare import compare_survival
from sksurv.metrics import concordance_index_censored

logger = logging.getLogger(__name__)


# ============================================================================
# THRESHOLD DEFINITION & GROUP ASSIGNMENT
# ============================================================================

def define_risk_groups(
    risk_scores: np.ndarray,
    low_percentile: float = 33,
    high_percentile: float = 66,
) -> Dict[str, float]:
    """Compute risk-group thresholds from training-set risk scores."""
    low_threshold = float(np.percentile(risk_scores, low_percentile))
    high_threshold = float(np.percentile(risk_scores, high_percentile))
    logger.info(
        f"Risk thresholds: low ({low_percentile}th)={low_threshold:.4f}, "
        f"high ({high_percentile}th)={high_threshold:.4f}"
    )
    return {"low_threshold": low_threshold, "high_threshold": high_threshold}


def apply_risk_thresholds(
    risk_scores: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    """Assign risk groups: 0=low, 1=medium, 2=high."""
    groups = np.ones(len(risk_scores), dtype=int)  # default medium
    groups[risk_scores <= low_threshold] = 0
    groups[risk_scores > high_threshold] = 2
    for g, label in enumerate(["Low", "Medium", "High"]):
        n = int((groups == g).sum())
        logger.info(f"  {label}: {n} ({100 * n / len(groups):.1f}%)")
    return groups


# ============================================================================
# KAPLAN-MEIER
# ============================================================================

def fit_kaplan_meier(
    y: np.ndarray,
    risk_groups: np.ndarray,
) -> Dict[int, KaplanMeierFitter]:
    """
    Fit a ``KaplanMeierFitter`` for each risk group.

    Returns dict {group_int: fitted KaplanMeierFitter}.
    """
    kmfs: Dict[int, KaplanMeierFitter] = {}
    group_labels = {0: "Low", 1: "Medium", 2: "High"}
    for g in sorted(np.unique(risk_groups)):
        mask = risk_groups == g
        n = int(mask.sum())
        if n == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(
            y["time"][mask],
            y["event"][mask],
            label=f"{group_labels.get(g, f'Group {g}')} (n={n})",
        )
        kmfs[g] = kmf
        logger.info(
            f"  KM group {group_labels.get(g, g)}: n={n}, "
            f"events={int(y['event'][mask].sum())}, "
            f"median={kmf.median_survival_time_:.1f}"
        )
    return kmfs


def compute_km_statistics(
    y: np.ndarray,
    risk_groups: np.ndarray,
    timepoints: List[float] = [6, 12, 18, 24],
) -> pd.DataFrame:
    """
    Kaplan-Meier summary table per risk group.

    Columns: group, n, events, event_rate, median_survival, median_ci_lower,
    median_ci_upper, surv_{t}m (survival probability at each timepoint).
    """
    group_labels = {0: "Low", 1: "Medium", 2: "High"}
    rows = []
    for g in sorted(np.unique(risk_groups)):
        mask = risk_groups == g
        n = int(mask.sum())
        events = int(y["event"][mask].sum())
        kmf = KaplanMeierFitter()
        kmf.fit(y["time"][mask], y["event"][mask])
        median = kmf.median_survival_time_

        # Extract median CI from the KM confidence interval bands.
        # lifelines >= 0.27 exposes confidence_interval_median_survival_time_,
        # but it may not exist in older versions. The robust fallback is to
        # interpolate where the lower/upper CI bands of the survival function
        # cross 0.5: the lower band crossing 0.5 gives ci_upper of the median,
        # and the upper band crossing 0.5 gives ci_lower of the median.
        ci_lower_val = "NR"
        ci_upper_val = "NR"
        try:
            attr = getattr(kmf, "confidence_interval_median_survival_time_", None)
            if attr is not None and hasattr(attr, "iloc") and attr.shape[0] > 0:
                lo = float(attr.iloc[0, 0])
                hi = float(attr.iloc[0, 1])
                ci_lower_val = round(lo, 2) if np.isfinite(lo) else "NR"
                ci_upper_val = round(hi, 2) if np.isfinite(hi) else "NR"
            else:
                # Fallback: interpolate from the KM confidence interval bands
                ci_df = kmf.confidence_interval_
                times = ci_df.index.values.astype(float)
                cols = ci_df.columns.tolist()  # [label_lower_0.95, label_upper_0.95]
                if len(cols) >= 2:
                    lower_band = ci_df.iloc[:, 0].values.astype(float)
                    upper_band = ci_df.iloc[:, 1].values.astype(float)
                    # ci_lower of median = time where lower_band (pessimistic) drops to 0.5
                    # lower S(t) → earlier crossing → smaller median CI bound
                    above_low = lower_band >= 0.5
                    if above_low.any():
                        idx = int(np.where(above_low)[0][-1])
                        ci_lower_val = round(float(times[idx]), 2)
                    # ci_upper of median = time where upper_band (optimistic) drops to 0.5
                    # higher S(t) → later crossing → larger median CI bound
                    above = upper_band >= 0.5
                    if above.any():
                        idx2 = int(np.where(above)[0][-1])
                        ci_upper_val = round(float(times[idx2]), 2)
        except Exception as _e:
            logger.debug(f"Median CI extraction failed: {_e}")

        row = {
            "group": group_labels.get(g, f"Group {g}"),
            "group_idx": g,
            "n": n,
            "events": events,
            "event_rate": round(events / n, 4) if n else float("nan"),
            "median_survival": round(float(median), 2) if np.isfinite(median) else "NR",
            "median_ci_lower": ci_lower_val,
            "median_ci_upper": ci_upper_val,
        }
        for t in timepoints:
            s = float(kmf.predict(t))
            row[f"surv_{int(t)}m"] = round(s, 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# LOG-RANK & COX
# ============================================================================

def log_rank_test(
    y: np.ndarray,
    risk_groups: np.ndarray,
) -> Dict[str, float]:
    """Omnibus log-rank test across risk groups."""
    if len(np.unique(risk_groups)) < 2:
        return {"test_statistic": float("nan"), "p_value": float("nan")}
    result = compare_survival(y, risk_groups)
    # Older scikit-survival returns (chi2, pvalue); newer returns an object
    if isinstance(result, tuple):
        return {"test_statistic": float(result[0]), "p_value": float(result[1])}
    return {
        "test_statistic": float(result.test_statistic),
        "p_value": float(result.pvalue),
    }


def fit_cox_model(
    y: np.ndarray,
    risk_groups: np.ndarray,
) -> Dict[str, float]:
    """
    Cox PH with risk groups as dummies (reference = low = 0).

    Uses lifelines.CoxPHFitter for proper SE / CI / p-values.
    """
    df = pd.DataFrame(
        {
            "time": y["time"],
            "event": y["event"].astype(int),
            "medium": (risk_groups == 1).astype(int),
            "high": (risk_groups == 2).astype(int),
        }
    )
    cph = CoxPHFitter()
    try:
        cph.fit(df, duration_col="time", event_col="event")
        summary = cph.summary
        result = {}
        for covar in ["medium", "high"]:
            if covar in summary.index:
                result[f"hr_{covar}_vs_low"] = float(summary.loc[covar, "exp(coef)"])
                result[f"hr_{covar}_vs_low_ci_lower"] = float(
                    summary.loc[covar, "exp(coef) lower 95%"]
                )
                result[f"hr_{covar}_vs_low_ci_upper"] = float(
                    summary.loc[covar, "exp(coef) upper 95%"]
                )
                result[f"hr_{covar}_vs_low_p"] = float(summary.loc[covar, "p"])
        return result
    except Exception as e:
        logger.error(f"Cox model fitting failed: {e}")
        return {}


def compute_concordance_by_group(
    y: np.ndarray,
    risk_scores: np.ndarray,
    risk_groups: np.ndarray,
    min_n: int = 10,
) -> Dict[str, float]:
    """C-index within each risk group (standard, no IPCW)."""
    results = {}
    for g in sorted(np.unique(risk_groups)):
        mask = risk_groups == g
        if mask.sum() < min_n:
            results[f"c_index_group_{g}"] = float("nan")
            continue
        results[f"c_index_group_{g}"] = float(
            concordance_index_censored(
                y["event"][mask], y["time"][mask], risk_scores[mask]
            )[0]
        )
    return results


# ============================================================================
# PPV / NPV
# ============================================================================

def compute_ppv_npv(
    y: np.ndarray,
    risk_groups: np.ndarray,
    ppv_timepoint: float = 6.0,
    npv_timepoint: float = 18.0,
) -> Dict[str, Optional[float]]:
    """
    PPV for high-risk group at *ppv_timepoint* and NPV for low-risk at
    *npv_timepoint*, computed with Kaplan-Meier (censoring-aware).
    """
    results: Dict[str, Optional[float]] = {}
    kmf = KaplanMeierFitter()

    # PPV = 1 − S_KM(ppv_timepoint) in high-risk group
    high_mask = risk_groups == 2
    if high_mask.sum() >= 5:
        kmf.fit(y["time"][high_mask], y["event"][high_mask])
        results["ppv"] = round(1.0 - float(kmf.predict(ppv_timepoint)), 4)
    else:
        results["ppv"] = None

    # NPV = S_KM(npv_timepoint) in low-risk group
    low_mask = risk_groups == 0
    if low_mask.sum() >= 5:
        kmf.fit(y["time"][low_mask], y["event"][low_mask])
        results["npv"] = round(float(kmf.predict(npv_timepoint)), 4)
    else:
        results["npv"] = None

    return results


# ============================================================================
# HAZARD RATIOS (pairwise)
# ============================================================================

def compute_pairwise_hazard_ratios(
    y: np.ndarray,
    risk_groups: np.ndarray,
    pairs: List[Tuple[int, int]] = [(2, 0), (2, 1), (1, 0)],
) -> pd.DataFrame:
    """Pairwise HRs using lifelines CoxPHFitter (proper SE/CI/p)."""
    labels = {0: "Low", 1: "Medium", 2: "High"}
    rows = []
    for g1, g2 in pairs:
        mask = (risk_groups == g1) | (risk_groups == g2)
        df = pd.DataFrame(
            {
                "time": y["time"][mask],
                "event": y["event"][mask].astype(int),
                "is_g1": (risk_groups[mask] == g1).astype(int),
            }
        )
        cph = CoxPHFitter()
        try:
            cph.fit(df, duration_col="time", event_col="event")
            s = cph.summary.loc["is_g1"]
            rows.append(
                {
                    "comparison": f"{labels[g1]} vs {labels[g2]}",
                    "HR": round(float(s["exp(coef)"]), 3),
                    "CI_lower": round(float(s["exp(coef) lower 95%"]), 3),
                    "CI_upper": round(float(s["exp(coef) upper 95%"]), 3),
                    "p_value": float(s["p"]),
                }
            )
        except Exception as e:
            logger.warning(f"HR {labels[g1]} vs {labels[g2]} failed: {e}")
    return pd.DataFrame(rows)
