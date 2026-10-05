"""IPCW-weighted ROC and precision-recall curves at a fixed horizon t.

Cases are patients who died by t, weighted by 1/G(T_i); controls are patients
alive beyond t, weighted by 1/G(t); patients censored before t are excluded.
G is the Kaplan-Meier estimate of the censoring distribution in the evaluated
cohort itself. The weights are rescaled to sum to the number of patients kept.
The same weights give the event-status densities of Fig. S3A-B
(compute_metrics.py); the curves themselves are drawn by
publication_figures.R and do not enter any reported statistic.
"""

from typing import Dict, Tuple

import numpy as np
from lifelines import KaplanMeierFitter
from scipy.integrate import trapezoid


def censoring_km(times: np.ndarray, events: np.ndarray) -> KaplanMeierFitter:
    """Kaplan-Meier estimate of the censoring survival function G(t)."""
    return KaplanMeierFitter().fit(times, event_observed=(1 - events))


def ipcw_weights(times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray, timepoint: float,
                 kmf_cens: KaplanMeierFitter) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Scores, labels (1 = death by timepoint) and weights of the patients kept, and their number."""
    G_t = float(kmf_cens.survival_function_at_times(min(timepoint, times.max())).iloc[0])
    scores, labels, weights = [], [], []
    for i in range(len(times)):
        if events[i] == 1 and times[i] <= timepoint:
            G_i = float(kmf_cens.survival_function_at_times(times[i]).iloc[0])
            labels.append(1)
            weights.append(1.0 / G_i if G_i > 0 else 0.0)
            scores.append(risk_scores[i])
        elif times[i] > timepoint:
            labels.append(0)
            weights.append(1.0 / G_t if G_t > 0 else 0.0)
            scores.append(risk_scores[i])
    scores, labels, weights = np.asarray(scores), np.asarray(labels), np.asarray(weights, dtype=float)
    n_kept = len(labels)
    if n_kept > 0 and weights.sum() > 0:
        weights *= n_kept / weights.sum()
    return scores, labels, weights, float(n_kept)


def _weighted_roc(scores, labels, weights):
    order = np.argsort(-scores)
    s, l, w = scores[order], labels[order], weights[order]
    cum_pos, cum_neg = np.cumsum(w * l), np.cumsum(w * (1 - l))
    tp = cum_pos[-1] if len(cum_pos) else 0
    tn = cum_neg[-1] if len(cum_neg) else 0
    tpr = cum_pos / tp if tp > 0 else np.zeros_like(cum_pos)
    fpr = cum_neg / tn if tn > 0 else np.zeros_like(cum_neg)
    cut = np.unique(np.concatenate([[0], np.where(np.diff(s) != 0)[0] + 1, [len(s) - 1]]))
    return (np.concatenate([[0], fpr[cut], [1]]), np.concatenate([[0], tpr[cut], [1]]),
            np.concatenate([[np.inf], s[cut], [s[-1] - 1]]))


def _weighted_pr(scores, labels, weights):
    order = np.argsort(-scores)
    s, l, w = scores[order], labels[order], weights[order]
    cum_pos, cum_all = np.cumsum(w * l), np.cumsum(w)
    tp = cum_pos[-1] if len(cum_pos) else 0
    recall = cum_pos / tp if tp > 0 else np.zeros_like(cum_pos)
    precision = np.divide(cum_pos, cum_all, out=np.zeros_like(cum_pos, dtype=float), where=cum_all != 0)
    cut = np.unique(np.concatenate([[0], np.where(np.diff(s) != 0)[0] + 1, [len(s) - 1]]))
    return recall[cut], precision[cut], s[cut]


def compute_ipcw_roc_at_timepoint(times, events, risk_scores, timepoint) -> Dict:
    """ROC curve (fpr, tpr, thresholds) and its trapezoidal AUC."""
    s, l, w, n_kept = ipcw_weights(times, events, risk_scores, timepoint, censoring_km(times, events))
    if len(s) == 0:
        return {"fpr": [], "tpr": [], "thresholds": [], "auc": float("nan"), "n_eff": 0}
    fpr, tpr, thr = _weighted_roc(s, l, w)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist(),
            "auc": float(trapezoid(tpr[np.argsort(fpr)], np.sort(fpr))), "n_eff": n_kept}


def compute_ipcw_pr_at_timepoint(times, events, risk_scores, timepoint, invert=False) -> Dict:
    """Precision-recall curve and its trapezoidal area.

    With invert=True the positive class is survival beyond timepoint and the
    scores are negated (used at 18 and 24 months).
    """
    s, l, w, n_kept = ipcw_weights(times, events, risk_scores, timepoint, censoring_km(times, events))
    if len(s) == 0:
        return {"recall": [], "precision": [], "thresholds": [], "average_precision": float("nan"), "n_eff": 0}
    if invert:
        l, s = 1 - l, -s
    recall, precision, thr = _weighted_pr(s, l, w)
    ap = float(trapezoid(precision[np.argsort(recall)], np.sort(recall))) if len(recall) > 1 else 0.0
    return {"recall": recall.tolist(), "precision": precision.tolist(), "thresholds": thr.tolist(),
            "average_precision": ap, "n_eff": n_kept}
