"""Risk groups: training-set cutoffs, Kaplan-Meier summaries, log-rank test,
pairwise hazard ratios, and censoring-aware PPV/NPV.

Groups (0 low, 1 intermediate, 2 high): score <= lower cutoff is low risk,
score > upper cutoff is high risk. The cutoffs are percentiles of the
training-set risk scores and are applied unchanged to the other cohorts.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import median_survival_times
from sksurv.compare import compare_survival

logger = logging.getLogger(__name__)

GROUP_LABELS = {0: "Low", 1: "Medium", 2: "High"}


def define_risk_groups(risk_scores: np.ndarray, low_percentile: float = 33,
                       high_percentile: float = 66) -> Dict[str, float]:
    """Cutoffs at two percentiles of the training risk scores (linear interpolation)."""
    low = float(np.percentile(risk_scores, low_percentile))
    high = float(np.percentile(risk_scores, high_percentile))
    logger.info(f"Cutoffs: P{low_percentile} = {low:.4f}, P{high_percentile} = {high:.4f}")
    return {"low_threshold": low, "high_threshold": high}


def apply_risk_thresholds(risk_scores: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    """0 = low (<= low_threshold), 2 = high (> high_threshold), 1 = intermediate."""
    groups = np.ones(len(risk_scores), dtype=int)
    groups[risk_scores <= low_threshold] = 0
    groups[risk_scores > high_threshold] = 2
    logger.info("Group sizes: " + ", ".join(f"{GROUP_LABELS[g]} {int((groups == g).sum())}" for g in (0, 1, 2)))
    return groups


def _round_or_nr(x: float):
    return round(float(x), 2) if np.isfinite(x) else "NR"


def compute_km_statistics(y: np.ndarray, risk_groups: np.ndarray,
                          timepoints: Tuple[float, ...] = (6, 12, 18, 24)) -> pd.DataFrame:
    """Kaplan-Meier summary of each risk group.

    Median survival with 95% CI: the limits are the first times at which the
    lower and the upper pointwise 95% confidence band of the curve (lifelines,
    log(-log) transformation) fall to 0.5 or below; "NR" (not reached) when a
    band, or the curve itself, stays above 0.5. surv_{t}m is the Kaplan-Meier
    estimate at t months.
    """
    rows = []
    for g in sorted(np.unique(risk_groups)):
        mask = risk_groups == g
        n, events = int(mask.sum()), int(y["event"][mask].sum())
        kmf = KaplanMeierFitter().fit(y["time"][mask], y["event"][mask])
        ci = median_survival_times(kmf.confidence_interval_)  # one row: lower, upper band
        row = {
            "group": GROUP_LABELS.get(g, f"Group {g}"),
            "group_idx": g,
            "n": n,
            "events": events,
            "event_rate": round(events / n, 4) if n else float("nan"),
            "median_survival": _round_or_nr(kmf.median_survival_time_),
            "median_ci_lower": _round_or_nr(ci.iloc[0, 0]),
            "median_ci_upper": _round_or_nr(ci.iloc[0, 1]),
        }
        for t in timepoints:
            row[f"surv_{int(t)}m"] = round(float(kmf.predict(t)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def log_rank_test(y: np.ndarray, risk_groups: np.ndarray) -> Dict[str, float]:
    """Omnibus log-rank test across the risk groups."""
    if len(np.unique(risk_groups)) < 2:
        return {"test_statistic": float("nan"), "p_value": float("nan")}
    chisq, p_value = compare_survival(y, risk_groups)[:2]
    return {"test_statistic": float(chisq), "p_value": float(p_value)}


def km_survival_at(time: np.ndarray, event: np.ndarray, timepoint: float) -> float:
    """Kaplan-Meier S(timepoint); NaN when timepoint is beyond the last follow-up time."""
    time = np.asarray(time, dtype=float)
    if time.size == 0 or timepoint > time.max():
        return float("nan")
    return float(KaplanMeierFitter().fit(time, np.asarray(event)).predict(timepoint))


def compute_ppv_npv(y: np.ndarray, risk_groups: np.ndarray, ppv_timepoint: float = 6.0,
                    npv_timepoint: float = 18.0) -> Dict[str, Optional[float]]:
    """PPV = 1 - S(ppv_timepoint) in the high-risk group; NPV = S(npv_timepoint) in the
    low-risk group (Kaplan-Meier). None for a group with fewer than 5 patients."""
    results: Dict[str, Optional[float]] = {"ppv": None, "npv": None}
    high, low = risk_groups == 2, risk_groups == 0
    if high.sum() >= 5:
        results["ppv"] = round(1.0 - km_survival_at(y["time"][high], y["event"][high], ppv_timepoint), 4)
    if low.sum() >= 5:
        results["npv"] = round(km_survival_at(y["time"][low], y["event"][low], npv_timepoint), 4)
    return results


def compute_pairwise_hazard_ratios(y: np.ndarray, risk_groups: np.ndarray,
                                   pairs: List[Tuple[int, int]] = [(2, 0), (2, 1), (1, 0)]) -> pd.DataFrame:
    """Hazard ratio of group g1 vs g2 from a Cox model fitted on the patients of the two
    groups only, with Wald 95% CI and p value (lifelines)."""
    rows = []
    for g1, g2 in pairs:
        mask = (risk_groups == g1) | (risk_groups == g2)
        df = pd.DataFrame({"time": y["time"][mask], "event": y["event"][mask].astype(int),
                           "is_g1": (risk_groups[mask] == g1).astype(int)})
        try:
            s = CoxPHFitter().fit(df, duration_col="time", event_col="event").summary.loc["is_g1"]
        except Exception as e:
            logger.warning(f"HR {GROUP_LABELS[g1]} vs {GROUP_LABELS[g2]} not estimated: {e}")
            continue
        rows.append({
            "comparison": f"{GROUP_LABELS[g1]} vs {GROUP_LABELS[g2]}",
            "HR": round(float(s["exp(coef)"]), 3),
            "CI_lower": round(float(s["exp(coef) lower 95%"]), 3),
            "CI_upper": round(float(s["exp(coef) upper 95%"]), 3),
            "p_value": float(s["p"]),
        })
    return pd.DataFrame(rows)
