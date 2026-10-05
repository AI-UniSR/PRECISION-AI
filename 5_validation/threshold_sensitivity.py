"""Sensitivity of the risk groups to the choice of percentile cutoffs (reviewer analysis).

Symmetric cutoffs k/(100 - k), k in 5, 10, 15, 20, 25, 30, 33, are percentiles
of the training scores and are applied unchanged to the temporal cohort, with
the group rule of the main analysis (score <= lower cutoff: low; score > upper
cutoff: high).

- SENECA complete cases (n = 579): the ensemble (cutoffs from the scores of all
  training patients) and SENECA (cutoffs from the training patients with all
  five SENECA inputs), plus SENECA with its published cutoffs (<= 2.14 /
  > 2.89). Supplementary table, figure and caption.
- Full temporal cohort (n = 698): the ensemble alone at k = 10, 15, 20, 33
  (threshold_sensitivity_temporal_full.{json,csv}, formatted as a Word table by
  publication_tables.R).

Per model and k: censoring-aware PPV (1 - Kaplan-Meier survival at 6 months,
high-risk group), NPV (Kaplan-Meier survival at 18 months, low-risk group),
Harrell's C of the three groups used as an ordinal predictor, group sizes,
median OS and the high vs low hazard ratio. CIs use the event-stratified
resamples of compute_metrics.py (random_state = 0 .. n-1) with the cutoffs, and
so each patient's group, held fixed; ensemble - SENECA differences are paired.

Nothing is written unless the rows at k = 15 and k = 33 reproduce the
corresponding files of compute_metrics.py (33-66 there uses the 33rd and 67th
percentiles) and the resamples are those of compute_metrics.py (checked on the
ensemble's C-index distribution). Run both scripts with the same --n_bootstrap.
"""

import argparse
import io
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.bootstrap import aggregate_bootstrap_results, stratified_bootstrap_sample
from lib.risk_stratification import (
    apply_risk_thresholds,
    compute_km_statistics,
    compute_pairwise_hazard_ratios,
    compute_ppv_npv,
    define_risk_groups,
)
from lib.seneca_model import SENECA_RISK_THRESHOLDS, SENECAModel
from lib.survival_metrics import (
    NPV_TIMEPOINT_MONTHS,
    PPV_TIMEPOINT_MONTHS,
    compute_c_index,
    make_structured_array,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

K_GRID: List[int] = [5, 10, 15, 20, 25, 30, 33]
TABLE_K: List[int] = [10, 15, 20, 33]
REFERENCE_K: Dict[int, str] = {15: "15-85", 33: "33-66"}   # main-analysis scheme names
MIN_AT_RISK: int = 10
PROPORTION_DEVIATION_WARN: float = 0.05

SCORE_COLUMNS = {
    "ensemble": "ml_risk_score",
    "seneca": "seneca_risk_score",
    "seneca_published": "seneca_risk_score",
}
MODEL_LABELS = {
    "ensemble": "ML ensemble",
    "seneca": "SENECA",
    "seneca_published": "SENECA (published cutoffs)",
}
METRICS = ["ppv_6m", "npv_18m", "c_index_3group"]

# Colours of the manuscript figures (publication_figures.R: col_ml, col_seneca)
COLOR_ML = "#0072B2"
COLOR_SENECA = "#D55E00"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"

OUTPUT_JSON = "threshold_sensitivity_temporal_cc.json"
OUTPUT_GRID_CSV = "threshold_sensitivity_temporal_cc.csv"
OUTPUT_TABLE_CSV = "supplementary_table_threshold_sensitivity.csv"
OUTPUT_TABLE_NOTES = "supplementary_table_threshold_sensitivity_footnotes.txt"
OUTPUT_FIGURE = "supplementary_figure_threshold_sensitivity"

# Full temporal cohort, ensemble only
FULL_TABLE_K: List[int] = [10, 15, 20, 33]
OUTPUT_FULL_JSON = "threshold_sensitivity_temporal_full.json"
OUTPUT_FULL_CSV = "threshold_sensitivity_temporal_full.csv"


# ----------------------------------------------------------------------------
# Cutoffs and point estimates
# ----------------------------------------------------------------------------

def build_configs(
    train_scores: Dict[str, np.ndarray],
    k_grid: List[int] = K_GRID,
    models: Tuple[str, ...] = ("ensemble", "seneca"),
    include_published: bool = True,
) -> List[Dict]:
    """Cutoffs per (model, k) from training percentiles [+ the SENECA published reference]."""
    configs = []
    for k in k_grid:
        for model in models:
            thr = define_risk_groups(train_scores[model], low_percentile=k,
                                     high_percentile=100 - k)
            configs.append({"config_id": f"{model}_k{k}", "model": model, "k": k,
                            "scheme": f"{k}-{100 - k}", **thr})
    if not include_published:
        return configs
    configs.append({
        "config_id": "seneca_published", "model": "seneca_published", "k": None,
        "scheme": "published",
        "low_threshold": float(SENECA_RISK_THRESHOLDS["low"]),
        "high_threshold": float(SENECA_RISK_THRESHOLDS["high"]),
    })
    return configs


def _nan_if_none(x) -> float:
    return float("nan") if x is None else float(x)


def _c_index_3group(y: np.ndarray, groups: np.ndarray) -> float:
    try:
        return float(compute_c_index(y, groups.astype(float)))
    except Exception:
        return float("nan")


def _stratification_metrics(y: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    ppv_npv = compute_ppv_npv(y, groups, PPV_TIMEPOINT_MONTHS, NPV_TIMEPOINT_MONTHS)
    return {
        "ppv_6m": _nan_if_none(ppv_npv["ppv"]),
        "npv_18m": _nan_if_none(ppv_npv["npv"]),
        "c_index_3group": _c_index_3group(y, groups),
    }


def point_estimates(df_cc: pd.DataFrame, y_cc: np.ndarray, configs: List[Dict]) -> Dict[str, Dict]:
    """Groups, sizes, at-risk counts, KM / HR tables and metrics on the original sample."""
    n = len(df_cc)
    results: Dict[str, Dict] = {}
    for cfg in configs:
        logger.info(f"  {cfg['config_id']}: cutoffs {cfg['low_threshold']:.4f} / "
                    f"{cfg['high_threshold']:.4f}")
        groups = apply_risk_thresholds(
            df_cc[SCORE_COLUMNS[cfg["model"]]].to_numpy(),
            cfg["low_threshold"], cfg["high_threshold"],
        )
        counts = {g: int((groups == g).sum()) for g in (0, 1, 2)}
        km = compute_km_statistics(y_cc, groups)
        hr = compute_pairwise_hazard_ratios(y_cc, groups, pairs=[(2, 0)])
        results[cfg["config_id"]] = {
            "config": cfg,
            "groups": groups,
            "km_statistics": km,
            "hr_high_vs_low": hr,
            "n": n,
            "counts": counts,
            "at_risk_high_6m": int((y_cc["time"][groups == 2] >= PPV_TIMEPOINT_MONTHS).sum()),
            "at_risk_low_18m": int((y_cc["time"][groups == 0] >= NPV_TIMEPOINT_MONTHS).sum()),
            "metrics": _stratification_metrics(y_cc, groups),
        }
    return results


# ----------------------------------------------------------------------------
# Reproduction check: k = 15 and 33 against the files of compute_metrics.py
# ----------------------------------------------------------------------------

def _roundtrip_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Same writer as the main-analysis tables (index=False, default float format)."""
    return pd.read_csv(io.StringIO(df.to_csv(index=False)))


def _same(a, b, tol: float) -> bool:
    a, b = _nan_if_none(a), _nan_if_none(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(abs(a - b) <= tol)


def check_reproduction(
    point: Dict[str, Dict],
    tables_dir: Path,
    tol: float = 1e-9,
    cohort_tag: str = "temporal_cc",
) -> Dict:
    """Ensemble k = 15 / 33 vs ``*_{cohort_tag}_{15-85,33-66}`` and thresholds files."""
    mismatches: List[Dict] = []
    n_compared = 0
    for k, scheme in REFERENCE_K.items():
        res = point[f"ensemble_k{k}"]
        cfg = res["config"]

        thr_ref = json.loads((tables_dir / f"risk_thresholds_{scheme}.json").read_text())
        for key in ["low_threshold", "high_threshold"]:
            n_compared += 1
            if not _same(cfg[key], thr_ref[key], tol):
                mismatches.append({"k": k, "file": f"risk_thresholds_{scheme}.json",
                                   "field": key, "expected": thr_ref[key], "observed": cfg[key]})

        km_file = f"km_statistics_{cohort_tag}_{scheme}.csv"
        km_ref = pd.read_csv(tables_dir / km_file)
        n_compared += int(km_ref.size)
        try:
            pd.testing.assert_frame_equal(
                _roundtrip_csv(res["km_statistics"]), km_ref,
                check_dtype=False, check_exact=False, atol=tol, rtol=0,
            )
        except AssertionError as e:
            mismatches.append({"k": k, "file": km_file, "field": "table",
                               "detail": str(e).splitlines()[:6]})

        hr_file = f"hazard_ratios_{cohort_tag}_{scheme}.csv"
        hr_ref = pd.read_csv(tables_dir / hr_file).set_index("comparison").loc["High vs Low"]
        hr_new = res["hr_high_vs_low"].set_index("comparison").loc["High vs Low"]
        for key in ["HR", "CI_lower", "CI_upper", "p_value"]:
            n_compared += 1
            if not _same(hr_new[key], hr_ref[key], tol):
                mismatches.append({"k": k, "file": hr_file, "field": key,
                                   "expected": float(hr_ref[key]), "observed": float(hr_new[key])})

        pn_file = f"ppv_npv_{cohort_tag}_{scheme}.csv"
        pn_ref = pd.read_csv(tables_dir / pn_file).iloc[0]
        for key, metric in [("ppv", "ppv_6m"), ("npv", "npv_18m")]:
            n_compared += 1
            if not _same(res["metrics"][metric], pn_ref[key], tol):
                mismatches.append({"k": k, "file": pn_file, "field": key,
                                   "expected": _nan_if_none(pn_ref[key]),
                                   "observed": res["metrics"][metric]})

    report = {
        "status": "passed" if not mismatches else "failed",
        "reference": "main-analysis files of the same run: risk_thresholds_{15-85,33-66}.json, "
                     f"{{km_statistics,hazard_ratios,ppv_npv}}_{cohort_tag}_{{15-85,33-66}}.csv",
        "n_compared": n_compared,
        "mismatches": mismatches,
    }
    if mismatches:
        for m in mismatches:
            logger.error(f"  REPRODUCTION MISMATCH: {m}")
    else:
        logger.info(f"  Reproduction check passed: k=15 and k=33 match the main analysis "
                    f"({cohort_tag}, {n_compared} values).")
    return report


# ----------------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------------

def bootstrap_distributions(
    df_cc: pd.DataFrame,
    y_cc: np.ndarray,
    groups_by_config: Dict[str, np.ndarray],
    n_iter: int,
) -> Tuple[Dict[str, List[Dict[str, float]]], List[Dict[str, float]]]:
    """Per-resample metrics with fixed group assignments.

    Resamples are drawn exactly as in ``compute_metrics._run_bootstrap`` for the
    same patient set (temporal CC or full cohort: same rows, same ``event``
    stratification, ``random_state = i``).  The ML C-index of each resample is returned as well,
    so that ``check_bootstrap_resamples`` can confirm they are the same resamples.
    """
    frame = df_cc[["event"]].reset_index(drop=True)
    frame["_pos_idx"] = np.arange(len(frame))
    ml_scores = df_cc["ml_risk_score"].to_numpy()
    dists: Dict[str, List[Dict[str, float]]] = {cid: [] for cid in groups_by_config}
    c_index_ml: List[Dict[str, float]] = []
    for i in tqdm(range(n_iter), desc="Bootstrap (threshold sensitivity)"):
        idx = stratified_bootstrap_sample(frame, stratify_col="event",
                                          random_state=i)["_pos_idx"].to_numpy()
        y_b = y_cc[idx]
        c_index_ml.append({"c_index_ml": compute_c_index(y_b, ml_scores[idx])})
        for cid, groups in groups_by_config.items():
            dists[cid].append(_stratification_metrics(y_b, groups[idx]))
    return dists, c_index_ml


def check_bootstrap_resamples(
    c_index_ml: List[Dict[str, float]],
    reference: Dict,
    n_iter: int,
    tol: float = 1e-9,
    reference_name: str = "bootstrap_temporal_cc.json",
) -> Dict:
    """ML C-index over our resamples vs *reference_name* (same resamples)."""
    mismatches: List[Dict] = []
    if reference.get("n_iterations") != n_iter:
        mismatches.append({"field": "n_iterations", "expected": reference.get("n_iterations"),
                           "observed": n_iter})
    ref = reference["aggregated"]["c_index_ml"]
    new = aggregate_bootstrap_results(c_index_ml)["c_index_ml"]
    for key in ["median", "ci_lower", "ci_upper", "n_valid"]:
        if not _same(new[key], ref[key], tol):
            mismatches.append({"field": f"c_index_ml.{key}", "expected": ref[key],
                               "observed": new[key]})
    for m in mismatches:
        logger.error(f"  RESAMPLE MISMATCH vs {reference_name}: {m}")
    if not mismatches:
        logger.info(f"  Resample check passed: same resamples as {reference_name} "
                    f"(ML C-index median {new['median']:.6f}, 95% CI "
                    f"{new['ci_lower']:.6f}–{new['ci_upper']:.6f}).")
    return {
        "status": "passed" if not mismatches else "failed",
        "reference": f"metrics/{reference_name}, aggregated c_index_ml",
        "mismatches": mismatches,
    }


def paired_differences(point: Dict[str, Dict], dists: Dict[str, List[Dict]], k_grid: List[int]) -> List[Dict]:
    """Ensemble − SENECA (training-percentile cutoffs) per k, same resamples."""
    rows = []
    for k in k_grid:
        ens, sen = point[f"ensemble_k{k}"], point[f"seneca_k{k}"]
        diff_dist = [
            {m: d_e[m] - d_s[m] for m in METRICS}
            for d_e, d_s in zip(dists[f"ensemble_k{k}"], dists[f"seneca_k{k}"])
        ]
        agg = aggregate_bootstrap_results(diff_dist)
        for m in METRICS:
            rows.append({
                "k": k, "metric": m,
                "difference": ens["metrics"][m] - sen["metrics"][m],
                "ci_lower": agg[m]["ci_lower"], "ci_upper": agg[m]["ci_upper"],
                "n_valid": agg[m]["n_valid"],
                "ensemble_tail_proportion": _tail_proportion(ens, m),
                "seneca_tail_proportion": _tail_proportion(sen, m),
            })
    return rows


def _tail_proportion(res: Dict, metric: str) -> Optional[float]:
    if metric == "ppv_6m":
        return res["counts"][2] / res["n"]
    if metric == "npv_18m":
        return res["counts"][0] / res["n"]
    return None


# ----------------------------------------------------------------------------
# Result rows and summary
# ----------------------------------------------------------------------------

def _median_fields(km: pd.DataFrame) -> Dict:
    out = {}
    for g, tag in [(0, "low"), (1, "intermediate"), (2, "high")]:
        row = km[km["group_idx"] == g]
        for col in ["events", "median_survival", "median_ci_lower", "median_ci_upper"]:
            out[f"{col}_{tag}"] = row[col].iloc[0] if len(row) else None
    return out


def grid_rows(point: Dict[str, Dict], dists: Dict[str, List[Dict]], n_iter: int) -> List[Dict]:
    rows = []
    for cid, res in point.items():
        cfg = res["config"]
        agg = aggregate_bootstrap_results(dists[cid])
        hr = res["hr_high_vs_low"]
        hr_row = hr.iloc[0] if len(hr) else {}
        row = {
            "config_id": cid, "model": cfg["model"], "k": cfg["k"], "scheme": cfg["scheme"],
            "cutoff_low": cfg["low_threshold"], "cutoff_high": cfg["high_threshold"],
            "n": res["n"],
            "n_low": res["counts"][0], "n_intermediate": res["counts"][1], "n_high": res["counts"][2],
            "prop_low": res["counts"][0] / res["n"],
            "prop_intermediate": res["counts"][1] / res["n"],
            "prop_high": res["counts"][2] / res["n"],
            "at_risk_high_6m": res["at_risk_high_6m"],
            "at_risk_low_18m": res["at_risk_low_18m"],
            **_median_fields(res["km_statistics"]),
            "hr_high_vs_low": hr_row.get("HR"),
            "hr_high_vs_low_ci_lower": hr_row.get("CI_lower"),
            "hr_high_vs_low_ci_upper": hr_row.get("CI_upper"),
            "hr_high_vs_low_p": hr_row.get("p_value"),
        }
        for m in METRICS:
            row[m] = res["metrics"][m]
            row[f"{m}_ci_lower"] = agg[m]["ci_lower"]
            row[f"{m}_ci_upper"] = agg[m]["ci_upper"]
            row[f"{m}_n_valid"] = agg[m]["n_valid"]
        row["ppv_6m_plotted"] = bool(res["at_risk_high_6m"] >= MIN_AT_RISK
                                     and np.isfinite(row["ppv_6m"]))
        row["npv_18m_plotted"] = bool(res["at_risk_low_18m"] >= MIN_AT_RISK
                                      and np.isfinite(row["npv_18m"]))
        row["n_bootstrap"] = n_iter
        rows.append(row)
    return rows


def summarize(rows: List[Dict], diffs: List[Dict], n_iter: int) -> Dict:
    by_id = {r["config_id"]: r for r in rows}
    summary: Dict = {"c_index_3group_argmax": {}, "ensemble_vs_seneca": {}}
    warnings: List[str] = []

    for model in ["ensemble", "seneca"]:
        cands = [r for r in rows if r["model"] == model and np.isfinite(r["c_index_3group"])]
        best = max(cands, key=lambda r: r["c_index_3group"])
        summary["c_index_3group_argmax"][model] = {
            "k": best["k"], "c_index_3group": best["c_index_3group"],
            "ci_lower": best["c_index_3group_ci_lower"], "ci_upper": best["c_index_3group_ci_upper"],
            "c_index_3group_at_k15": by_id[f"{model}_k15"]["c_index_3group"],
        }

    for m in METRICS:
        eligible, excluded = [], []
        for d in [d for d in diffs if d["metric"] == m]:
            k = d["k"]
            ok = np.isfinite(d["difference"])
            if m != "c_index_3group":
                flag = f"{m}_plotted"
                ok = ok and by_id[f"ensemble_k{k}"][flag] and by_id[f"seneca_k{k}"][flag]
            (eligible if ok else excluded).append(d)
        summary["ensemble_vs_seneca"][m] = {
            "n_k_compared": len(eligible),
            "k_compared": [d["k"] for d in eligible],
            "k_excluded": [d["k"] for d in excluded],
            "n_k_ensemble_higher": sum(d["difference"] > 0 for d in eligible),
            "k_ensemble_higher": [d["k"] for d in eligible if d["difference"] > 0],
            "n_k_ci_above_zero": sum(d["ci_lower"] > 0 for d in eligible),
            "k_ci_above_zero": [d["k"] for d in eligible if d["ci_lower"] > 0],
            "n_k_ci_below_zero": sum(d["ci_upper"] < 0 for d in eligible),
            "k_ci_below_zero": [d["k"] for d in eligible if d["ci_upper"] < 0],
        }

    for r in rows:
        label = f"{MODEL_LABELS[r['model']]} {r['scheme']}"
        if r["at_risk_high_6m"] < MIN_AT_RISK:
            warnings.append(f"{label}: {r['at_risk_high_6m']} high-risk patients at risk at "
                            f"6 m (<{MIN_AT_RISK}): PPV not plotted, flagged in the table")
        if r["at_risk_low_18m"] < MIN_AT_RISK:
            warnings.append(f"{label}: {r['at_risk_low_18m']} low-risk patients at risk at "
                            f"18 m (<{MIN_AT_RISK}): NPV not plotted, flagged in the table")
        for m in METRICS:
            if not np.isfinite(r[m]):
                warnings.append(f"{label}: {m} not estimable on the original sample")
            elif r[f"{m}_n_valid"] < n_iter:
                warnings.append(f"{label}: {m} estimable in {r[f'{m}_n_valid']}/{n_iter} "
                                "resamples; CI from the estimable ones only")
        if r["k"] is not None:
            for tail, prop in [("low", r["prop_low"]), ("high", r["prop_high"])]:
                if abs(prop - r["k"] / 100) >= PROPORTION_DEVIATION_WARN:
                    warnings.append(f"{label}: {tail}-risk group holds {100 * prop:.1f}% of "
                                    f"the cohort vs nominal {r['k']}%")
    warnings.append(
        "At the same nominal k the two models have different group sizes; PPV/NPV "
        "paired differences are therefore not at equal group size (the figure plots "
        "against the observed tail proportion instead)."
    )
    summary["warnings"] = warnings
    return summary


def _log_summary(summary: Dict) -> None:
    for model, best in summary["c_index_3group_argmax"].items():
        logger.info(f"  3-group C-index maximum, {MODEL_LABELS[model]}: k={best['k']} "
                    f"(C={best['c_index_3group']:.4f}; at k=15 C={best['c_index_3group_at_k15']:.4f})")
    for m, s in summary["ensemble_vs_seneca"].items():
        logger.info(f"  Ensemble vs SENECA [{m}]: higher at {s['n_k_ensemble_higher']}/"
                    f"{s['n_k_compared']} k; CI > 0 at {s['n_k_ci_above_zero']}, "
                    f"CI < 0 at {s['n_k_ci_below_zero']} (excluded k: {s['k_excluded']})")
    for w in summary["warnings"]:
        logger.warning(f"  {w}")


# ----------------------------------------------------------------------------
# Supplementary table
# ----------------------------------------------------------------------------

def _fmt_num(x, digits: int) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NE"
    if isinstance(x, str):
        return x
    return f"{x:.{digits}f}"


def _fmt_est_ci(est, lo, hi, digits: int) -> str:
    if est is None or (isinstance(est, float) and not np.isfinite(est)):
        return "NE"
    return f"{_fmt_num(est, digits)} ({_fmt_num(lo, digits)}–{_fmt_num(hi, digits)})"


def supplementary_table(rows: List[Dict]) -> pd.DataFrame:
    by_id = {r["config_id"]: r for r in rows}
    ordered = [by_id[f"{m}_k{k}"] for k in TABLE_K for m in ["ensemble", "seneca"]]
    ordered.append(by_id["seneca_published"])
    out = []
    for r in ordered:
        k = r["k"]
        ppv = _fmt_est_ci(r["ppv_6m"], r["ppv_6m_ci_lower"], r["ppv_6m_ci_upper"], 3)
        npv = _fmt_est_ci(r["npv_18m"], r["npv_18m_ci_lower"], r["npv_18m_ci_upper"], 3)
        if r["at_risk_high_6m"] < MIN_AT_RISK:
            ppv += " †"
        if r["at_risk_low_18m"] < MIN_AT_RISK:
            npv += " †"
        row = {
            "Cutoffs": f"{k}/{100 - k}" if k is not None else "Published (≤2.14 / >2.89)",
            "Model": MODEL_LABELS[r["model"]],
            "Cutoff values (low / high)": f"{r['cutoff_low']:.3f} / {r['cutoff_high']:.3f}",
        }
        for tag, label in [("low", "Low"), ("intermediate", "Intermediate"), ("high", "High")]:
            row[f"{label} risk, n (%)"] = f"{r[f'n_{tag}']} ({100 * r[f'prop_{tag}']:.1f}%)"
        for tag, label in [("low", "Low"), ("intermediate", "Intermediate"), ("high", "High")]:
            row[f"Median OS {label.lower()}, months (95% CI)"] = _fmt_est_ci(
                r[f"median_survival_{tag}"], r[f"median_ci_lower_{tag}"],
                r[f"median_ci_upper_{tag}"], 1,
            )
        row["HR high vs low (95% CI)"] = _fmt_est_ci(
            r["hr_high_vs_low"], r["hr_high_vs_low_ci_lower"], r["hr_high_vs_low_ci_upper"], 2,
        )
        row["PPV, death ≤6 m (95% CI)"] = ppv
        row["High risk at risk at 6 m, n"] = r["at_risk_high_6m"]
        row["NPV, alive at 18 m (95% CI)"] = npv
        row["Low risk at risk at 18 m, n"] = r["at_risk_low_18m"]
        row["3-group C-index (95% CI)"] = _fmt_est_ci(
            r["c_index_3group"], r["c_index_3group_ci_lower"], r["c_index_3group_ci_upper"], 3,
        )
        out.append(row)
    return pd.DataFrame(out)


def table_footnotes(n_cohort: int, n_events: int, n_train: Dict[str, int], n_iter: int) -> str:
    return "\n".join([
        f"Temporal validation cohort, patients with complete SENECA variables "
        f"(n = {n_cohort}, {n_events} deaths); the same patients for every row.",
        "Cutoffs k/(100 − k) are the k-th and (100 − k)-th percentiles of the training-set "
        "scores (linear interpolation), fixed before evaluation and not re-estimated in the "
        "temporal cohort or in bootstrap resamples. Score ≤ lower cutoff: low risk; "
        "score > upper cutoff: high risk; otherwise intermediate.",
        f"Ensemble cutoffs: all {n_train['ensemble']} training patients (as in the main "
        f"analysis). SENECA cutoffs: the {n_train['seneca']} training patients with complete "
        "SENECA variables. SENECA published cutoffs: score ≤ 2.14 low, > 2.89 high.",
        "PPV: 1 − Kaplan–Meier survival at 6 months in the high-risk group. NPV: "
        "Kaplan–Meier survival at 18 months in the low-risk group. Not estimated (NE) when "
        "the horizon is beyond the group's last observed follow-up.",
        f"† Fewer than {MIN_AT_RISK} patients at risk at the horizon: estimate unstable, not "
        "shown in the supplementary figure.",
        "3-group C-index: Harrell's C with the risk group (low / intermediate / high) as an "
        "ordinal predictor; pairs in the same group count as ties (0.5).",
        f"95% CIs for PPV, NPV and C-index: percentile intervals from {n_iter} event-stratified "
        "bootstrap resamples (the same resamples as the main temporal analysis), with fixed "
        "cutoffs. HR: univariable Cox model, high vs low risk group, Wald 95% CI. Median OS: "
        "Kaplan–Meier, 95% CI; NR = not reached.",
    ]) + "\n"


# ----------------------------------------------------------------------------
# Supplementary figure
# ----------------------------------------------------------------------------

def plot_figure(rows: List[Dict], out_dir: Path) -> List[Dict]:
    """Two panels (A PPV 6 m, B NPV 18 m) vs observed tail proportion; returns hidden points."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    hidden: List[Dict] = []
    panels = [
        ("ppv_6m", "prop_high", "at_risk_high_6m",
         "A  PPV at 6 months (high-risk group)",
         "Patients classified high risk (%)", "Death within 6 months (1 − KM)"),
        ("npv_18m", "prop_low", "at_risk_low_18m",
         "B  NPV at 18 months (low-risk group)",
         "Patients classified low risk (%)", "Alive at 18 months (KM)"),
    ]
    styles = {
        "ensemble": dict(color=COLOR_ML, marker="o", linestyle="-", label_dy=7, va="bottom"),
        "seneca": dict(color=COLOR_SENECA, marker="s", linestyle=(0, (4, 2)), label_dy=-8, va="top"),
    }
    for ax, (metric, xcol, atrisk_col, title, xlabel, ylabel) in zip(axes, panels):
        for model, st in styles.items():
            pts = sorted([r for r in rows if r["model"] == model], key=lambda r: r["k"])
            x = np.array([100 * r[xcol] for r in pts])
            ok = np.array([r[f"{metric}_plotted"] for r in pts])
            y = np.where(ok, [r[metric] for r in pts], np.nan)
            lo = np.where(ok, [r[f"{metric}_ci_lower"] for r in pts], np.nan)
            hi = np.where(ok, [r[f"{metric}_ci_upper"] for r in pts], np.nan)
            ax.fill_between(x, lo, hi, color=st["color"], alpha=0.14, linewidth=0)
            ax.plot(x, y, color=st["color"], linestyle=st["linestyle"], linewidth=1.6,
                    marker=st["marker"], markersize=5.5, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=3)
            for r, xi, yi, shown in zip(pts, x, y, ok):
                if not shown:
                    hidden.append({"model": model, "k": r["k"], "metric": metric,
                                   "n_at_risk": r[atrisk_col], "estimate": r[metric]})
                    continue
                ax.annotate(f"{r['k']}", (xi, yi), xytext=(0, st["label_dy"]),
                            textcoords="offset points", ha="center", va=st["va"],
                            fontsize=7, color=TEXT_SECONDARY,
                            fontweight="bold" if r["k"] == 15 else "normal")
        ref = next(r for r in rows if r["model"] == "seneca_published")
        if ref[f"{metric}_plotted"]:
            yv = ref[metric]
            ax.errorbar(100 * ref[xcol], yv,
                        yerr=[[yv - ref[f"{metric}_ci_lower"]], [ref[f"{metric}_ci_upper"] - yv]],
                        fmt="D", markersize=7, markerfacecolor="white",
                        markeredgecolor=COLOR_SENECA, markeredgewidth=1.4,
                        ecolor=COLOR_SENECA, elinewidth=1.2, capsize=3, zorder=4)
        else:
            hidden.append({"model": "seneca_published", "k": None, "metric": metric,
                           "n_at_risk": ref[atrisk_col], "estimate": ref[metric]})
        ax.axvline(15, color=TEXT_SECONDARY, linestyle=(0, (2, 2)), linewidth=0.9, zorder=1)
        ax.text(15.6, 0.985, "15%", fontsize=7, color=TEXT_SECONDARY, va="top", ha="left")
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
        ax.grid(True, color=GRID_COLOR, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color("#b5b4ae")

    handles = [
        Line2D([], [], color=COLOR_ML, marker="o", linestyle="-", linewidth=1.6,
               markersize=5.5, label="ML ensemble (training-percentile cutoffs)"),
        Line2D([], [], color=COLOR_SENECA, marker="s", linestyle=(0, (4, 2)), linewidth=1.6,
               markersize=5.5, label="SENECA (training-percentile cutoffs)"),
        Line2D([], [], color=COLOR_SENECA, marker="D", linestyle="none", markersize=7,
               markerfacecolor="white", markeredgewidth=1.4,
               label="SENECA (published cutoffs)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_dir / f"{OUTPUT_FIGURE}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{OUTPUT_FIGURE}.pdf", bbox_inches="tight")
    plt.close(fig)
    return hidden


def figure_caption(n_cohort: int, n_train: Dict[str, int], n_iter: int, hidden: List[Dict]) -> str:
    if hidden:
        names = {"ppv_6m": "PPV", "npv_18m": "NPV"}
        items = []
        for h in hidden:
            k_txt = "" if h["k"] is None else f" k = {h['k']}"
            items.append(f"{MODEL_LABELS[h['model']]}{k_txt} ({names[h['metric']]}, "
                         f"{h['n_at_risk']} at risk)")
        items = "; ".join(items)
        hidden_txt = (f"Points with fewer than {MIN_AT_RISK} patients at risk at the horizon are "
                      f"not shown (marked † in Supplementary Table): {items}.")
    else:
        hidden_txt = (f"All points had at least {MIN_AT_RISK} patients at risk at the horizon.")
    return (
        "Sensitivity of risk-group performance to the choice of percentile cutoffs. "
        f"Temporal validation cohort, patients with complete SENECA variables (n = {n_cohort}). "
        "Symmetric cutoffs k/(100 − k), k = " + ", ".join(str(k) for k in K_GRID) + ", were "
        "derived from training-set score percentiles and fixed before evaluation "
        f"(ML ensemble: all {n_train['ensemble']} training patients; SENECA: "
        f"{n_train['seneca']} training patients with complete SENECA variables). "
        "(A) Censoring-aware positive predictive value for death within 6 months "
        "(1 − Kaplan–Meier survival) in the high-risk group, against the observed proportion "
        "of patients classified as high risk. (B) Censoring-aware negative predictive value "
        "for survival beyond 18 months (Kaplan–Meier survival) in the low-risk group, against "
        "the observed proportion classified as low risk. Point labels give k; shaded bands "
        f"are 95% percentile intervals from {n_iter} event-stratified bootstrap resamples with "
        "fixed cutoffs. The open diamond shows SENECA with its published thresholds "
        "(≤ 2.14 / > 2.89), with 95% CI. The dotted vertical line marks 15%, the nominal "
        f"proportion of the prespecified 15/85 cutoffs. {hidden_txt}\n"
    )


# ----------------------------------------------------------------------------
# Complete-case analysis
# ----------------------------------------------------------------------------

def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def run_threshold_sensitivity(
    df_cc: pd.DataFrame,
    y_cc: np.ndarray,
    train_scores: Dict[str, np.ndarray],
    n_iter: int,
    tables_dir: Path,
    bootstrap_reference: Dict,
    output_dir: Path,
) -> Dict:
    """Complete-case analysis. Raises, before writing any output, if k = 15 / 33 do
    not reproduce the main analysis or the resamples differ from the main bootstrap."""
    n_train = {m: int(len(s)) for m, s in train_scores.items()}
    configs = build_configs(train_scores)
    point = point_estimates(df_cc, y_cc, configs)

    reproduction = check_reproduction(point, tables_dir)
    if reproduction["status"] != "passed":
        raise RuntimeError(
            "STOPPED: threshold-sensitivity rows k=15 / k=33 do not reproduce the main "
            f"analysis ({len(reproduction['mismatches'])} mismatches: "
            f"{reproduction['mismatches']}). Nothing was written."
        )

    dists, c_index_ml = bootstrap_distributions(
        df_cc, y_cc, {cid: res["groups"] for cid, res in point.items()}, n_iter,
    )
    resamples = check_bootstrap_resamples(c_index_ml, bootstrap_reference, n_iter)
    if resamples["status"] != "passed":
        raise RuntimeError(
            "STOPPED: bootstrap resamples differ from bootstrap_temporal_cc.json "
            f"({resamples['mismatches']}). Nothing was written."
        )

    rows = grid_rows(point, dists, n_iter)
    diffs = paired_differences(point, dists, K_GRID)
    summary = summarize(rows, diffs, n_iter)

    n_events = int(np.sum(y_cc["event"]))
    hidden = plot_figure(rows, output_dir)
    (output_dir / f"{OUTPUT_FIGURE}_caption.txt").write_text(
        figure_caption(len(df_cc), n_train, n_iter, hidden), encoding="utf-8")
    # utf-8-sig so that Excel shows ≤ – † correctly
    supplementary_table(rows).to_csv(output_dir / OUTPUT_TABLE_CSV, index=False,
                                     encoding="utf-8-sig")
    (output_dir / OUTPUT_TABLE_NOTES).write_text(
        table_footnotes(len(df_cc), n_events, n_train, n_iter), encoding="utf-8")

    grid_df = pd.concat([
        pd.DataFrame(rows),
        pd.DataFrame(diffs).assign(model="ensemble_minus_seneca", config_id=lambda d: (
            "diff_k" + d["k"].astype(str) + "_" + d["metric"])),
    ], ignore_index=True)
    grid_df.to_csv(output_dir / OUTPUT_GRID_CSV, index=False, float_format="%.6f")

    result = {
        "description": "Threshold sensitivity of the 3-group risk stratification, temporal "
                       "validation SENECA complete cases.",
        "cohort": {"name": "temporal_validation_seneca_complete_cases",
                   "n": len(df_cc), "events": n_events},
        "settings": {
            "k_grid": K_GRID, "table_k": TABLE_K,
            "ppv_timepoint_months": PPV_TIMEPOINT_MONTHS,
            "npv_timepoint_months": NPV_TIMEPOINT_MONTHS,
            "n_bootstrap": n_iter, "bootstrap": "event-stratified, random_state = 0..n-1, "
                                                "same resamples as bootstrap_temporal_cc.json",
            "min_at_risk_for_plot": MIN_AT_RISK,
            "group_rule": "score <= low cutoff -> low; score > high cutoff -> high",
            "percentile_method": "numpy.percentile (linear interpolation)",
            "km_lookup": "NaN beyond the group's last observed time (no carry-forward)",
            "training_cutoff_sources": {
                "ensemble": {"n": n_train["ensemble"], "patients": "all training patients"},
                "seneca": {"n": n_train["seneca"],
                           "patients": "training patients with complete SENECA variables"},
                "seneca_published": {"low": SENECA_RISK_THRESHOLDS["low"],
                                     "high": SENECA_RISK_THRESHOLDS["high"]},
            },
        },
        "grid": rows,
        "paired_differences_ensemble_minus_seneca": diffs,
        "reproduction_check": reproduction,
        "bootstrap_resample_check": resamples,
        "figure_hidden_points": hidden,
        "summary": summary,
    }
    with open(output_dir / OUTPUT_JSON, "w") as f:
        json.dump(_json_ready(result), f, indent=2)
    _log_summary(summary)
    return result


# ----------------------------------------------------------------------------
# Full temporal cohort, ensemble only
# ----------------------------------------------------------------------------

def run_full_cohort_sensitivity(
    df_full: pd.DataFrame,
    y_full: np.ndarray,
    ensemble_train_scores: np.ndarray,
    n_iter: int,
    tables_dir: Path,
    bootstrap_reference: Dict,
    output_dir: Path,
) -> Dict:
    """Ensemble risk groups at k ∈ ``FULL_TABLE_K`` on the full temporal cohort.

    Same cutoffs (training percentiles), group rule, KM / HR / PPV / NPV code
    and bootstrap as the CC analysis, on the full-cohort resamples of
    ``bootstrap_temporal_full.json``. Raises, before writing its outputs, if
    k = 15 / 33 do not reproduce ``*_temporal_{15-85,33-66}`` or if the resamples
    differ.
    """
    logger.info(f"Full temporal cohort, ensemble, k = {FULL_TABLE_K} (n = {len(df_full)})")
    configs = build_configs({"ensemble": ensemble_train_scores}, FULL_TABLE_K,
                            models=("ensemble",), include_published=False)
    point = point_estimates(df_full, y_full, configs)

    reproduction = check_reproduction(point, tables_dir, cohort_tag="temporal")
    if reproduction["status"] != "passed":
        raise RuntimeError(
            "STOPPED: full-cohort rows k=15 / k=33 do not reproduce the main analysis "
            f"({len(reproduction['mismatches'])} mismatches: {reproduction['mismatches']})."
        )

    dists, c_index_ml = bootstrap_distributions(
        df_full, y_full, {cid: res["groups"] for cid, res in point.items()}, n_iter,
    )
    resamples = check_bootstrap_resamples(c_index_ml, bootstrap_reference, n_iter,
                                          reference_name="bootstrap_temporal_full.json")
    if resamples["status"] != "passed":
        raise RuntimeError(
            "STOPPED: full-cohort resamples differ from bootstrap_temporal_full.json "
            f"({resamples['mismatches']})."
        )

    rows = grid_rows(point, dists, n_iter)
    for r in rows:
        groups = point[r["config_id"]]["groups"]
        r["min_at_risk_reported"] = MIN_AT_RISK
        for metric, group, t, at_risk in [
            ("ppv_6m", 2, PPV_TIMEPOINT_MONTHS, "at_risk_high_6m"),
            ("npv_18m", 0, NPV_TIMEPOINT_MONTHS, "at_risk_low_18m"),
        ]:
            # Word table: "NE" unless >= MIN_AT_RISK patients at risk and estimable
            r[f"{metric}_reported"] = bool(r[at_risk] >= MIN_AT_RISK and np.isfinite(r[metric]))
            in_group = y_full["time"][groups == group]
            r[f"{metric}_horizon_beyond_last_follow_up"] = bool(
                len(in_group) == 0 or t > float(in_group.max()))
        for flag in ["ppv_6m_plotted", "npv_18m_plotted"]:
            r.pop(flag)

    pd.DataFrame(rows).to_csv(output_dir / OUTPUT_FULL_CSV, index=False, float_format="%.6f")
    n_events = int(np.sum(y_full["event"]))
    result = {
        "description": "Threshold sensitivity of the ensemble 3-group risk stratification, "
                       "temporal validation full cohort.",
        "cohort": {"name": "temporal_validation_full", "n": len(df_full), "events": n_events},
        "settings": {
            "table_k": FULL_TABLE_K,
            "ppv_timepoint_months": PPV_TIMEPOINT_MONTHS,
            "npv_timepoint_months": NPV_TIMEPOINT_MONTHS,
            "n_bootstrap": n_iter,
            "bootstrap": "event-stratified, random_state = 0..n-1, same resamples as "
                         "bootstrap_temporal_full.json",
            "min_at_risk_reported": MIN_AT_RISK,
            "group_rule": "score <= low cutoff -> low; score > high cutoff -> high",
            "percentile_method": "numpy.percentile (linear interpolation)",
            "km_lookup": "NaN beyond the group's last observed time (no carry-forward)",
            "training_cutoff_source": {"n": int(len(ensemble_train_scores)),
                                       "patients": "all training patients"},
        },
        "grid": rows,
        "reproduction_check": reproduction,
        "bootstrap_resample_check": resamples,
    }
    with open(output_dir / OUTPUT_FULL_JSON, "w") as f:
        json.dump(_json_ready(result), f, indent=2)
    for r in rows:
        logger.info(f"  {r['scheme']}: n = {r['n_low']}/{r['n_intermediate']}/{r['n_high']}, "
                    f"HR {r['hr_high_vs_low']}, PPV {r['ppv_6m']:.4f} "
                    f"({r['at_risk_high_6m']} at risk), NPV {r['npv_18m']:.4f} "
                    f"({r['at_risk_low_18m']} at risk)")
    return result


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def _load_data(path: str) -> pd.DataFrame:
    """Load CSV or Parquet from a file or AML output directory."""
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith((".csv", ".parquet"))]
        if not files:
            raise FileNotFoundError(f"No data files in {path}")
        path = os.path.join(path, files[0])
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", required=True,
                        help="generate_predictions output (temporal predictions, training scores)")
    parser.add_argument("--training_data", required=True,
                        help="data_prep training data (SENECA score of training patients)")
    parser.add_argument("--metrics_dir", required=True,
                        help="compute_metrics output_dir (bootstrap_temporal_{cc,full}.json)")
    parser.add_argument("--tables_dir", required=True,
                        help="compute_metrics tables_dir (main 15-85 / 33-66 files, CC and full)")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.start_run()
    mlflow.log_param("n_bootstrap", args.n_bootstrap)
    mlflow.log_param("k_grid", str(K_GRID))

    # Temporal SENECA complete cases, in the row order of compute_metrics.py
    pred_dir = Path(args.predictions_dir)
    df_temporal = pd.read_csv(pred_dir / "temporal_predictions.csv")
    df_cc = df_temporal[df_temporal["seneca_complete_case"] == 1].reset_index(drop=True)
    y_cc = make_structured_array(df_cc["event"].values, df_cc["tte"].values)
    logger.info(f"Temporal SENECA complete cases: {len(df_cc)}/{len(df_temporal)}")

    # Training scores: ensemble (all training patients, as in compute_metrics)
    # and SENECA (training complete cases)
    with open(pred_dir / "training_risk_scores.json") as f:
        ensemble_train = np.array(json.load(f)["scores"])
    seneca_train = SENECAModel().predict(_load_data(args.training_data))
    seneca_train_scores = seneca_train.loc[
        seneca_train["complete_case"] == 1, "seneca_risk_score"
    ].to_numpy()
    if len(seneca_train_scores) == 0:
        raise ValueError("No training patient has complete SENECA variables.")

    with open(Path(args.metrics_dir) / "bootstrap_temporal_cc.json") as f:
        bootstrap_reference = json.load(f)

    result = run_threshold_sensitivity(
        df_cc=df_cc,
        y_cc=y_cc,
        train_scores={"ensemble": ensemble_train, "seneca": seneca_train_scores},
        n_iter=args.n_bootstrap,
        tables_dir=Path(args.tables_dir),
        bootstrap_reference=bootstrap_reference,
        output_dir=output_dir,
    )

    # Full temporal cohort, ensemble only, in the row order of compute_metrics.py
    df_full = df_temporal.reset_index(drop=True)
    y_full = make_structured_array(df_full["event"].values, df_full["tte"].values)
    with open(Path(args.metrics_dir) / "bootstrap_temporal_full.json") as f:
        bootstrap_reference_full = json.load(f)
    full = run_full_cohort_sensitivity(
        df_full=df_full,
        y_full=y_full,
        ensemble_train_scores=ensemble_train,
        n_iter=args.n_bootstrap,
        tables_dir=Path(args.tables_dir),
        bootstrap_reference=bootstrap_reference_full,
        output_dir=output_dir,
    )

    mlflow.set_tag("reproduction_check", result["reproduction_check"]["status"])
    mlflow.set_tag("bootstrap_resample_check", result["bootstrap_resample_check"]["status"])
    mlflow.set_tag("full_cohort_reproduction_check", full["reproduction_check"]["status"])
    mlflow.set_tag("full_cohort_bootstrap_resample_check",
                   full["bootstrap_resample_check"]["status"])
    for model, best in result["summary"]["c_index_3group_argmax"].items():
        mlflow.log_metric(f"c_index_3group_argmax_k_{model}", best["k"])
    mlflow.log_artifacts(str(output_dir), artifact_path="threshold_sensitivity")
    mlflow.end_run()
    logger.info("=== Threshold Sensitivity Complete ===")


if __name__ == "__main__":
    main()
