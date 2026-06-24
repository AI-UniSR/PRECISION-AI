"""
ROC and Precision-Recall Curve Analysis (Standard + IPCW)

Consolidates roc_analysis.py and roc_analysis_ipcw.py into a single module.

Two approaches:
  1. **Standard** — excludes patients censored before the evaluation horizon
     (simple but biased if censoring is informative).
  2. **IPCW** — Inverse Probability of Censoring Weighted curves following
     Heagerty & Zheng (2004).  Censoring distribution G(t) estimated via KM
     from the evaluation-set censoring pattern.

Both return numerical curve data (FPR/TPR/thresholds) suitable for downstream
R plotting.  No matplotlib plots are produced here.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple

from lifelines import KaplanMeierFitter
from sklearn.metrics import auc as sklearn_auc, roc_curve, precision_recall_curve

from lib.seneca_model import SENECA_RISK_THRESHOLDS

logger = logging.getLogger(__name__)

# Percentile sets used for operating-point annotation
ML_PERCENTILES_HIGH_RISK = [90, 85, 80, 75, 67]
ML_PERCENTILES_LOW_RISK = [10, 15, 20, 25, 33]


# ============================================================================
# STANDARD (censoring-naive) ROC / PR
# ============================================================================

def compute_roc_at_timepoint(
    tte: np.ndarray,
    event: np.ndarray,
    risk_scores: np.ndarray,
    timepoint: float,
    invert_for_low_risk: bool = False,
) -> Dict:
    """
    Standard time-dependent ROC at a fixed horizon.

    Patients censored before *timepoint* are **excluded** (outcome unknown).

    Returns dict with keys: fpr, tpr, thresholds, auc, n_valid, n_positive.
    """
    valid = (tte > timepoint) | (event == 1)
    if invert_for_low_risk:
        y_bin = (tte > timepoint).astype(int)
        scores = -risk_scores
    else:
        y_bin = ((event == 1) & (tte <= timepoint)).astype(int)
        scores = risk_scores

    y_v, s_v = y_bin[valid], scores[valid]
    if len(np.unique(y_v)) < 2:
        return {"fpr": [0, 1], "tpr": [0, 1], "thresholds": [], "auc": 0.5,
                "n_valid": int(valid.sum()), "n_positive": int(y_v.sum())}

    fpr, tpr, thr = roc_curve(y_v, s_v)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thr.tolist(),
        "auc": float(sklearn_auc(fpr, tpr)),
        "n_valid": int(valid.sum()),
        "n_positive": int(y_v.sum()),
    }


def compute_pr_at_timepoint(
    tte: np.ndarray,
    event: np.ndarray,
    risk_scores: np.ndarray,
    timepoint: float,
    invert_for_low_risk: bool = False,
) -> Dict:
    """Standard time-dependent Precision-Recall at a fixed horizon."""
    valid = (tte > timepoint) | (event == 1)
    if invert_for_low_risk:
        y_bin = (tte > timepoint).astype(int)
        scores = -risk_scores
    else:
        y_bin = ((event == 1) & (tte <= timepoint)).astype(int)
        scores = risk_scores

    y_v, s_v = y_bin[valid], scores[valid]
    if len(np.unique(y_v)) < 2:
        return {"recall": [0, 1], "precision": [1, 0], "thresholds": [],
                "average_precision": 0.0, "n_valid": int(valid.sum())}

    prec, rec, thr = precision_recall_curve(y_v, s_v)
    from sklearn.metrics import average_precision_score as ap
    return {
        "recall": rec.tolist(),
        "precision": prec.tolist(),
        "thresholds": thr.tolist(),
        "average_precision": float(ap(y_v, s_v)),
        "n_valid": int(valid.sum()),
    }


# ============================================================================
# IPCW-weighted ROC / PR
# ============================================================================

def _censoring_km(times: np.ndarray, events: np.ndarray) -> KaplanMeierFitter:
    """KM estimator for the *censoring* distribution G(t) = P(C ≥ t)."""
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=(1 - events))
    return kmf


def _ipcw_weights(
    times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    timepoint: float,
    kmf_cens: KaplanMeierFitter,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    IPCW weights for a single timepoint.

    Returns (filtered_scores, labels, weights, n_effective).
    """
    t_eval = min(timepoint, times.max())
    G_t = float(kmf_cens.survival_function_at_times(t_eval).iloc[0])

    scores_out, labels_out, w_out = [], [], []
    for i in range(len(times)):
        if events[i] == 1 and times[i] <= timepoint:
            G_Xi = float(kmf_cens.survival_function_at_times(times[i]).iloc[0])
            w = 1.0 / G_Xi if G_Xi > 0 else 0.0
            labels_out.append(1)
            w_out.append(w)
            scores_out.append(risk_scores[i])
        elif times[i] > timepoint:
            w = 1.0 / G_t if G_t > 0 else 0.0
            labels_out.append(0)
            w_out.append(w)
            scores_out.append(risk_scores[i])
        # censored before timepoint → excluded

    scores_out = np.asarray(scores_out)
    labels_out = np.asarray(labels_out)
    w_out = np.asarray(w_out, dtype=float)
    n_eff = len(labels_out)
    if n_eff > 0 and w_out.sum() > 0:
        w_out *= n_eff / w_out.sum()
    return scores_out, labels_out, w_out, float(n_eff)


def _weighted_roc(scores, labels, weights):
    """O(n) cumulative-sum weighted ROC curve."""
    idx = np.argsort(-scores)
    s, l, w = scores[idx], labels[idx], weights[idx]
    cp = np.cumsum(w * l)
    cn = np.cumsum(w * (1 - l))
    tp, tn = (cp[-1] if len(cp) else 0), (cn[-1] if len(cn) else 0)
    tpr = cp / tp if tp > 0 else np.zeros_like(cp)
    fpr = cn / tn if tn > 0 else np.zeros_like(cn)
    ui = np.unique(np.concatenate([[0], np.where(np.diff(s) != 0)[0] + 1, [len(s) - 1]]))
    tpr_out = np.concatenate([[0], tpr[ui], [1]])
    fpr_out = np.concatenate([[0], fpr[ui], [1]])
    thr_out = np.concatenate([[np.inf], s[ui], [s[-1] - 1]])
    return fpr_out, tpr_out, thr_out


def _weighted_pr(scores, labels, weights):
    """O(n) cumulative-sum weighted PR curve."""
    idx = np.argsort(-scores)
    s, l, w = scores[idx], labels[idx], weights[idx]
    cp = np.cumsum(w * l)
    ct = np.cumsum(w)
    tp = cp[-1] if len(cp) else 0
    rec = cp / tp if tp > 0 else np.zeros_like(cp)
    prec = np.divide(cp, ct, out=np.zeros_like(cp, dtype=float), where=ct != 0)
    ui = np.unique(np.concatenate([[0], np.where(np.diff(s) != 0)[0] + 1, [len(s) - 1]]))
    return rec[ui], prec[ui], s[ui]


def compute_ipcw_roc_at_timepoint(
    times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    timepoint: float,
) -> Dict:
    """IPCW-weighted ROC at a fixed timepoint. Returns dict with fpr, tpr, thresholds, auc, n_eff."""
    kmf = _censoring_km(times, events)
    fs, fl, fw, n_eff = _ipcw_weights(times, events, risk_scores, timepoint, kmf)
    if len(fs) == 0:
        return {"fpr": [], "tpr": [], "thresholds": [], "auc": float("nan"), "n_eff": 0}
    fpr, tpr, thr = _weighted_roc(fs, fl, fw)
    a = float(np.trapz(tpr[np.argsort(fpr)], np.sort(fpr)))
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist(),
            "auc": a, "n_eff": n_eff}


def compute_ipcw_pr_at_timepoint(
    times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    timepoint: float,
    invert: bool = False,
) -> Dict:
    """IPCW-weighted PR at a fixed timepoint.

    Parameters
    ----------
    invert : bool
        If True, the positive class becomes *survival past timepoint*
        (i.e.  P(T > t)) and scores are negated so that lower risk ⇒
        higher "positive" likelihood.  Useful for long horizons where
        survival is the clinically interesting outcome.
    """
    kmf = _censoring_km(times, events)
    fs, fl, fw, n_eff = _ipcw_weights(times, events, risk_scores, timepoint, kmf)
    if len(fs) == 0:
        return {"recall": [], "precision": [], "thresholds": [], "average_precision": float("nan"), "n_eff": 0}
    if invert:
        fl = 1 - fl          # swap positive class
        fs = -fs             # lower risk → higher score for survival
    rec, prec, thr = _weighted_pr(fs, fl, fw)
    ap = float(np.trapz(prec[np.argsort(rec)], np.sort(rec))) if len(rec) > 1 else 0.0
    return {"recall": rec.tolist(), "precision": prec.tolist(),
            "thresholds": thr.tolist(), "average_precision": ap, "n_eff": n_eff}


# ============================================================================
# OPERATING-POINT HELPERS
# ============================================================================

def compute_ml_threshold_values(
    risk_scores: np.ndarray,
    percentiles: List[int],
) -> Dict[int, float]:
    """Map percentiles to actual risk-score thresholds."""
    return {p: float(np.percentile(risk_scores, p)) for p in percentiles}


def find_operating_point(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    target_threshold: float,
) -> Dict[str, float]:
    """Locate the operating point closest to *target_threshold* on a ROC curve."""
    fpr, tpr, thresholds = np.asarray(fpr), np.asarray(tpr), np.asarray(thresholds)
    if len(thresholds) == 0:
        return {"fpr": 0.5, "tpr": 0.5, "threshold": target_threshold}
    idx = int(np.argmin(np.abs(thresholds - target_threshold)))
    return {
        "fpr": float(fpr[idx]),
        "tpr": float(tpr[idx]),
        "specificity": float(1 - fpr[idx]),
        "sensitivity": float(tpr[idx]),
        "threshold": float(thresholds[idx]),
    }


def create_roc_summary_table(
    tte: np.ndarray,
    event: np.ndarray,
    ml_scores: np.ndarray,
    seneca_scores: np.ndarray,
    ml_train_scores: Optional[np.ndarray] = None,
    use_ipcw: bool = True,
) -> pd.DataFrame:
    """
    Summary table of ROC operating points at 6 m (high-risk) and 18 m (low-risk).

    Includes both ML percentile thresholds and SENECA clinical thresholds.
    """
    ref = ml_train_scores if ml_train_scores is not None else ml_scores
    roc_func = compute_ipcw_roc_at_timepoint if use_ipcw else (
        lambda t, e, r, tp: compute_roc_at_timepoint(t, e, r, tp, invert_for_low_risk=False)
    )

    rows: List[Dict] = []

    # --- 6 m high-risk ---
    for model, scores, label in [("ML", ml_scores, "ML"), ("SENECA", seneca_scores, "SENECA")]:
        roc = roc_func(tte, event, scores, 6.0)
        fpr_arr, tpr_arr, thr_arr = np.asarray(roc["fpr"]), np.asarray(roc["tpr"]), np.asarray(roc["thresholds"])

        if label == "ML":
            for p in ML_PERCENTILES_HIGH_RISK:
                tv = float(np.percentile(ref, p))
                pt = find_operating_point(fpr_arr, tpr_arr, thr_arr, tv)
                rows.append({"timepoint": "6m", "model": label,
                             "threshold_type": f"P{100 - p}", "threshold_value": tv, **pt,
                             "auc": roc["auc"]})
        else:
            pt = find_operating_point(fpr_arr, tpr_arr, thr_arr, SENECA_RISK_THRESHOLDS["high"])
            rows.append({"timepoint": "6m", "model": label,
                         "threshold_type": "Clinical (High)", "threshold_value": SENECA_RISK_THRESHOLDS["high"],
                         **pt, "auc": roc["auc"]})

    # --- 18 m low-risk ---
    for model, scores, label in [("ML", ml_scores, "ML"), ("SENECA", seneca_scores, "SENECA")]:
        if use_ipcw:
            roc = compute_ipcw_roc_at_timepoint(tte, event, scores, 18.0)
        else:
            roc = compute_roc_at_timepoint(tte, event, scores, 18.0, invert_for_low_risk=True)
        fpr_arr, tpr_arr, thr_arr = np.asarray(roc["fpr"]), np.asarray(roc["tpr"]), np.asarray(roc["thresholds"])

        if label == "ML":
            for p in ML_PERCENTILES_LOW_RISK:
                tv = float(np.percentile(ref, p))
                target = -tv if not use_ipcw else tv
                pt = find_operating_point(fpr_arr, tpr_arr, thr_arr, target)
                rows.append({"timepoint": "18m", "model": label,
                             "threshold_type": f"P{p}", "threshold_value": tv, **pt,
                             "auc": roc["auc"]})
        else:
            target = -SENECA_RISK_THRESHOLDS["low"] if not use_ipcw else SENECA_RISK_THRESHOLDS["low"]
            pt = find_operating_point(fpr_arr, tpr_arr, thr_arr, target)
            rows.append({"timepoint": "18m", "model": label,
                         "threshold_type": "Clinical (Low)", "threshold_value": SENECA_RISK_THRESHOLDS["low"],
                         **pt, "auc": roc["auc"]})

    return pd.DataFrame(rows)
