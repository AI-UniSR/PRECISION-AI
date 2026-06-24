"""
Compute Metrics — Consolidated Evaluation Step

Consumes pre-computed predictions from ``generate_predictions`` and produces
all numerical metrics, bootstrap CIs, risk-stratification tables, and CSV
data files that the R publication-figures step will consume.

Outputs
-------
output_dir/
├── metrics/
│   ├── performance_internal.json
│   ├── performance_temporal.json
│   ├── bootstrap_results.json
│   ├── comparison_internal_vs_temporal.json
│   └── cohort_characteristics.json
├── tables/
│   ├── performance_summary.csv         (all metrics, both cohorts, both models)
│   ├── km_statistics_{scheme}.csv      (one per stratification scheme)
│   ├── roc_operating_points.csv
│   ├── hazard_ratios_{scheme}.csv
│   └── ppv_npv_{scheme}.csv
├── r_input/
│   ├── km_data_internal_{scheme}.csv   (individual-level KM input for R survminer)
│   ├── km_data_temporal_{scheme}.csv
│   ├── calibration_input_internal.csv  (predicted + observed for R calibration)
│   ├── calibration_input_temporal.csv
│   ├── roc_curves_{timepoint}.csv      (FPR / TPR for R ggplot)
│   ├── pr_curves_{timepoint}.csv
│   ├── temporal_auc.csv                (timepoint / model / auc / ci_lower / ci_upper)
│   ├── temporal_brier.csv
│   ├── risk_density_internal.csv       (risk score + event for density plots)
│   └── risk_density_temporal.csv
└── python_figures/
    └── km/
        ├── km_{scheme}_{cohort}.png    (300 DPI Kaplan-Meier curves)
        └── km_{scheme}_{cohort}.pdf    (vector Kaplan-Meier curves)
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from lib.bootstrap import (
    aggregate_bootstrap_results,
    compare_bootstrap_distributions,
    compute_bootstrap_metrics,
    stratified_bootstrap_sample,
)
from lib.risk_stratification import (
    apply_risk_thresholds,
    compute_concordance_by_group,
    compute_km_statistics,
    compute_pairwise_hazard_ratios,
    compute_ppv_npv,
    define_risk_groups,
    fit_cox_model,
    fit_kaplan_meier,
    log_rank_test,
)
from lib.roc_pr import (
    compute_ipcw_pr_at_timepoint,
    compute_ipcw_roc_at_timepoint,
    compute_pr_at_timepoint,
    compute_roc_at_timepoint,
    create_roc_summary_table,
)
from lib.seneca_model import SENECA_RISK_THRESHOLDS, SENECAModel
from lib.survival_metrics import (
    CLINICAL_HORIZON_MONTHS,
    CLINICAL_TIMEPOINTS_MONTHS,
    compute_brier_scores,
    compute_c_index,
    compute_c_index_ipcw,
    compute_time_dependent_auc,
    create_temporal_grid,
    make_structured_array,
    truncate_survival_times,
)
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Risk-stratification schemes requested by the user
STRATIFICATION_SCHEMES = [
    {"name": "33-66", "low_percentile": 33, "high_percentile": 67},
    {"name": "15-85", "low_percentile": 15, "high_percentile": 85},
]


# ======================================================================
# I/O helpers
# ======================================================================

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


def _to_native(obj):
    """Recursively convert numpy types > native Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _precompute_survival_matrix(model, X, grid):
    """Evaluate S(t) for every patient at every grid point.

    Re-uses the model-unwrapping logic from ``lib.survival_metrics`` but
    avoids importing a private helper.  Returns ``None`` when the model
    does not expose ``predict_survival_function``.
    """
    actual = model
    if hasattr(model, "_model_impl") and hasattr(model._model_impl, "python_model"):
        actual = model._model_impl.python_model
    if not hasattr(actual, "predict_survival_function"):
        logger.warning("Model lacks predict_survival_function — Brier bootstrap unavailable.")
        return None
    try:
        surv_funcs = actual.predict_survival_function(X)
        return np.asarray([[fn(t) for t in grid] for fn in surv_funcs])
    except Exception as e:
        logger.error(f"predict_survival_function failed: {e}")
        return None


# ======================================================================
# COHORT CHARACTERISATION
# ======================================================================

def _cohort_characteristics(
    df_train: pd.DataFrame,
    df_internal: pd.DataFrame,
    df_temporal: pd.DataFrame,
) -> Dict:
    """Basic descriptive statistics for each cohort."""
    def _desc(df, name):
        return {
            "cohort": name,
            "n": len(df),
            "events": int(df["event"].sum()),
            "event_rate": round(float(df["event"].mean()), 4),
            "tte_median": round(float(df["tte"].median()), 2),
            "tte_mean": round(float(df["tte"].mean()), 2),
            "tte_min": round(float(df["tte"].min()), 2),
            "tte_max": round(float(df["tte"].max()), 2),
        }

    return {
        "training": _desc(df_train, "training"),
        "internal_test": _desc(df_internal, "internal_test"),
        "temporal_validation": _desc(df_temporal, "temporal_validation"),
    }


# ======================================================================
# PERFORMANCE METRICS — single cohort
# ======================================================================

def _compute_cohort_performance(
    model,
    X: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk_scores: np.ndarray,
    label: str,
) -> Dict:
    """All discrimination + calibration metrics for one cohort / one model."""
    logger.info(f"Computing performance metrics: {label}")
    metrics: Dict = {}

    # C-index (standard)
    metrics["c_index"] = compute_c_index(y_test, risk_scores)

    # C-index IPCW
    metrics["c_index_ipcw"] = compute_c_index_ipcw(y_train, y_test, risk_scores)

    # Time-dependent AUC
    auc_dict = compute_time_dependent_auc(y_train, y_test, risk_scores)
    metrics.update(auc_dict)

    # Brier scores + IBS (requires model for S(t))
    if model is not None:
        brier_dict = compute_brier_scores(model, X, y_train, y_test)
        metrics.update(brier_dict)

    logger.info(f"  C-index={metrics['c_index']:.4f}, "
                f"IPCW={metrics.get('c_index_ipcw', 'N/A')}")
    return metrics


# ======================================================================
# RISK STRATIFICATION — single cohort
# ======================================================================

def _evaluate_stratification(
    y: np.ndarray,
    risk_scores: np.ndarray,
    thresholds: Dict[str, float],
    scheme_name: str,
) -> Dict:
    """KM, log-rank, Cox, PPV/NPV for one stratification scheme."""
    groups = apply_risk_thresholds(
        risk_scores, thresholds["low_threshold"], thresholds["high_threshold"]
    )
    km_stats = compute_km_statistics(y, groups)
    lr = log_rank_test(y, groups)
    cox = fit_cox_model(y, groups)
    c_by_grp = compute_concordance_by_group(y, risk_scores, groups)
    ppv_npv = compute_ppv_npv(y, groups)

    # Pairwise HRs
    try:
        pairwise_hr = compute_pairwise_hazard_ratios(y, groups)
    except Exception:
        pairwise_hr = []

    return {
        "scheme_name": scheme_name,
        "thresholds": thresholds,
        "groups": groups,  # excluded from JSON — used at runtime only
        "km_statistics": km_stats,
        "logrank": lr,
        "cox": cox,
        "concordance_by_group": c_by_grp,
        "ppv_npv": ppv_npv,
        "pairwise_hr": pairwise_hr,
    }


# ======================================================================
# ROC / PR CURVES — prepare data for R
# ======================================================================

ROC_PR_TIMEPOINTS = [6, 12, 18, 24]


def _roc_pr_data(
    tte: np.ndarray,
    event: np.ndarray,
    ml_scores: np.ndarray,
    seneca_scores: np.ndarray,
) -> Dict[str, pd.DataFrame]:
    """Compute curve data for R plotting — ROC + PR at 6 m and 18 m."""
    out: Dict[str, pd.DataFrame] = {}

    for tp in ROC_PR_TIMEPOINTS:
        # --- ROC ---
        roc_rows = []
        for label, scores in [("ML", ml_scores), ("SENECA", seneca_scores)]:
            roc = compute_ipcw_roc_at_timepoint(tte, event, scores, float(tp))
            for fpr_v, tpr_v, thr_v in zip(roc["fpr"], roc["tpr"], roc["thresholds"]):
                roc_rows.append({
                    "model": label, "timepoint": tp,
                    "fpr": fpr_v, "tpr": tpr_v, "threshold": thr_v,
                    "auc": roc["auc"],
                })
        out[f"roc_curves_{tp}m"] = pd.DataFrame(roc_rows)

        # --- PR ---
        pr_rows = []
        invert_pr = tp >= 18  # long horizons: positive = survival
        for label, scores in [("ML", ml_scores), ("SENECA", seneca_scores)]:
            pr = compute_ipcw_pr_at_timepoint(tte, event, scores, float(tp),
                                              invert=invert_pr)
            for rec_v, prec_v, thr_v in zip(pr["recall"], pr["precision"], pr["thresholds"]):
                pr_rows.append({
                    "model": label, "timepoint": tp,
                    "recall": rec_v, "precision": prec_v, "threshold": thr_v,
                    "average_precision": pr["average_precision"],
                    "inverted": invert_pr,
                })
        out[f"pr_curves_{tp}m"] = pd.DataFrame(pr_rows)

    return out


# ======================================================================
# KAPLAN-MEIER PYTHON FIGURES
# ======================================================================

# Colors matching R palette
_KM_COLORS = {0: "#4DAF4A", 1: "#FF7F00", 2: "#E41A1C"}
_KM_LABELS = {0: "Low", 1: "Intermediate", 2: "High"}


def _generate_km_figures(
    df: pd.DataFrame,
    risk_groups: np.ndarray,
    y: np.ndarray,
    scheme_name: str,
    cohort: str,
    output_dir: Path,
) -> None:
    """Generate Kaplan-Meier figures in PNG + PDF using lifelines."""
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts

    fig, ax_main = plt.subplots(figsize=(8, 6))

    kmfs = []
    unique_groups = sorted(np.unique(risk_groups))
    for g in unique_groups:
        mask = risk_groups == g
        n = int(mask.sum())
        if n == 0:
            continue
        kmf = KaplanMeierFitter()
        label = f"{_KM_LABELS.get(g, f'Group {g}')} (n={n})"
        kmf.fit(
            y["time"][mask], y["event"][mask], label=label,
        )
        kmf.plot_survival_function(
            ax=ax_main,
            color=_KM_COLORS.get(g, "grey"),
            ci_alpha=0.15,
            linewidth=1.2,
        )
        kmfs.append(kmf)

    # Truncate x-axis where any stratum < 10 at risk
    max_time = 24.0
    for kmf in kmfs:
        timeline = kmf.survival_function_.index.values
        at_risk = kmf.event_table["at_risk"].values
        below_10 = timeline[at_risk < 10]
        if len(below_10) > 0:
            max_time = min(max_time, float(below_10[0]))
    max_time = max(max_time, 6.0)  # never below 6m

    ax_main.set_xlim(0, max_time)
    ax_main.xaxis.set_major_locator(MultipleLocator(6))
    ax_main.set_xlabel("Time (months)", fontweight="bold")
    ax_main.set_ylabel("Survival probability", fontweight="bold")
    ax_main.set_title(f"{scheme_name} \u2014 {cohort}", fontweight="bold", fontsize=13)
    ax_main.set_xticks([6, 12, 18, 24])
    ax_main.legend(loc="lower left", frameon=False)

    # Numbers-at-risk table (At risk + Censored)
    if kmfs:
        add_at_risk_counts(*kmfs, ax=ax_main, rows_to_show=["At risk", "Censored"])

    plt.tight_layout()

    base = f"km_{scheme_name}_{cohort}"
    fig.savefig(output_dir / f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  KM figure saved: {base} (.png + .pdf)")


# ======================================================================
# R-INPUT DATA — KM, calibration, density
# ======================================================================

def _km_r_data(
    df: pd.DataFrame,
    risk_groups: np.ndarray,
    scheme_name: str,
    cohort: str,
) -> pd.DataFrame:
    """Individual-level data for R survminer ggsurvplot."""
    return pd.DataFrame({
        "tte": df["tte"].values,
        "event": df["event"].values.astype(int),
        "risk_group": risk_groups,
        "scheme": scheme_name,
        "cohort": cohort,
    })


def _calibration_r_data(
    df: pd.DataFrame,
    cohort: str,
) -> pd.DataFrame:
    """Predicted event probabilities + observed outcomes for R calibration."""
    cols = {"tte": df["tte"].values, "event": df["event"].values.astype(int), "cohort": cohort}
    for tp in [6, 12, 18, 24]:
        p_col = f"P_event_{tp}m"
        if p_col in df.columns:
            cols[f"p_event_{tp}m"] = df[p_col].values
    return pd.DataFrame(cols)


def _risk_density_r_data(
    df: pd.DataFrame,
    cohort: str,
) -> pd.DataFrame:
    """Risk scores + event status for R density plots."""
    cols = {
        "ml_risk_score": df["ml_risk_score"].values,
        "event": df["event"].values.astype(int),
        "cohort": cohort,
    }
    if "seneca_risk_score" in df.columns:
        cols["seneca_risk_score"] = df["seneca_risk_score"].values
    return pd.DataFrame(cols)


def _predicted_risk_distributions(
    df_internal: pd.DataFrame,
    df_temporal: pd.DataFrame,
) -> pd.DataFrame:
    """Predicted P(T <= t) at 6 m and 18 m for both cohorts (R density plots)."""
    rows = []
    for cohort, df in [("internal", df_internal), ("temporal", df_temporal)]:
        rec = {"cohort": cohort}
        for tp in [6, 18]:
            col = f"P_event_{tp}m"
            if col in df.columns:
                rec[f"p_event_{tp}m"] = df[col].values
            else:
                rec[f"p_event_{tp}m"] = np.full(len(df), np.nan)
        rows.append(pd.DataFrame(rec))
    return pd.concat(rows, ignore_index=True)


def _ipcw_stratified_distributions(
    tte: np.ndarray,
    event: np.ndarray,
    risk_scores: np.ndarray,
    p_event_cols: Dict[int, np.ndarray],
    cohort: str = "temporal",
) -> pd.DataFrame:
    """
    IPCW-weighted event-status-stratified P(T <= t) distributions for R.

    For each horizon (6 m, 18 m), assigns each patient an event_status
    ('event' or 'no_event') and an IPCW weight.  Censored-before-horizon
    patients are excluded.
    """
    from lib.roc_pr import _censoring_km, _ipcw_weights

    rows = []
    kmf = _censoring_km(tte, event)
    for tp in [6, 18]:
        p_col = p_event_cols.get(tp)
        if p_col is None:
            continue
        fs, fl, fw, _ = _ipcw_weights(tte, event, risk_scores, float(tp), kmf)
        # Map back to the p_event values for the same patients
        # _ipcw_weights filters to patients with event<=tp or tte>tp
        idx = []
        for i in range(len(tte)):
            if (event[i] == 1 and tte[i] <= tp) or (tte[i] > tp):
                idx.append(i)
        idx = np.array(idx)
        if len(idx) != len(fs):
            continue  # safety check

        for j in range(len(idx)):
            rows.append({
                "cohort": cohort,
                "timepoint": tp,
                f"p_event_{tp}m": float(p_col[idx[j]]),
                "event_status": "event" if fl[j] == 1 else "no_event",
                "ipcw_weight": float(fw[j]),
            })
    return pd.DataFrame(rows)


# ======================================================================
# BOOTSTRAP ANALYSIS
# ======================================================================

def _run_bootstrap(
    df: pd.DataFrame,
    y_train: np.ndarray,
    n_iter: int,
    surv_matrix: np.ndarray = None,
    grid: np.ndarray = None,
    horizon: float = None,
    include_seneca: bool = True,
    label: str = "Bootstrap",
) -> Dict:
    """Bootstrap comparison ML [vs SENECA] on a given cohort.

    When *include_seneca* is True the DataFrame is filtered to SENECA
    complete cases.  *surv_matrix* rows must already match *df* rows
    (caller is responsible for any pre-filtering).
    """
    logger.info(f"Running {label} with {n_iter} iterations...")

    if include_seneca:
        complete = df["seneca_complete_case"] == 1
        df_boot = df[complete].copy()
        if surv_matrix is not None:
            surv_matrix = surv_matrix[complete.values]
        logger.info(f"  Complete cases: {len(df_boot)} / {len(df)}")
    else:
        df_boot = df.copy()

    df_boot = df_boot.reset_index(drop=True)
    df_boot["_pos_idx"] = np.arange(len(df_boot))

    distributions: List[Dict[str, float]] = []
    for i in tqdm(range(n_iter), desc=label):
        boot = stratified_bootstrap_sample(df_boot, stratify_col="event", random_state=i)

        boot_surv = None
        if surv_matrix is not None:
            boot_surv = surv_matrix[boot["_pos_idx"].values]

        metrics = compute_bootstrap_metrics(
            y_train=y_train,
            tte=boot["tte"].values,
            event=boot["event"].values,
            risk_ml=boot["ml_risk_score"].values,
            risk_seneca=(
                boot["seneca_risk_score"].values if include_seneca else None
            ),
            surv_preds_ml=boot_surv,
            grid=grid,
            horizon=horizon,
        )
        distributions.append(metrics)

    aggregated = aggregate_bootstrap_results(distributions)

    result: Dict = {
        "n_samples": len(df_boot),
        "n_iterations": n_iter,
        "aggregated": aggregated,
    }

    if include_seneca:
        for a_key, b_key, tag in [
            ("c_index_ml", "c_index_seneca", "comparison_c_index"),
            ("c_index_ipcw_ml", "c_index_ipcw_seneca", "comparison_c_index_ipcw"),
            ("mean_auc_ml", "mean_auc_seneca", "comparison_mean_auc"),
        ]:
            a_vals = np.array([d.get(a_key, np.nan) for d in distributions])
            b_vals = np.array([d.get(b_key, np.nan) for d in distributions])
            result[tag] = compare_bootstrap_distributions(a_vals, b_vals)

    return result


# ======================================================================
# TEMPORAL AUC/BRIER CURVES FOR R PLOTTING
# ======================================================================

def _temporal_curves_for_r(
    perf_internal: Dict,
    perf_temporal_ml: Dict,
    perf_temporal_seneca: Dict,
    boot_internal_agg: Dict,
    boot_temporal_agg: Dict,
) -> Dict[str, pd.DataFrame]:
    """Assemble temporal AUC and Brier data for R line plots."""
    timepoints = [6, 12, 18, 24]

    # --- Temporal AUC ---
    auc_rows = []
    for tp in timepoints:
        key = f"auc_{tp}m"
        # Internal ML
        val = perf_internal.get(key)
        ci_int = boot_internal_agg.get(f"{key}_ml", {})
        if val is not None:
            auc_rows.append({"timepoint": tp, "model": "ML", "cohort": "internal",
                             "auc": val,
                             "ci_lower": ci_int.get("ci_lower"),
                             "ci_upper": ci_int.get("ci_upper")})
        # Temporal ML
        val = perf_temporal_ml.get(key)
        ci = boot_temporal_agg.get(f"{key}_ml", {})
        if val is not None:
            auc_rows.append({"timepoint": tp, "model": "ML", "cohort": "temporal",
                             "auc": val, "ci_lower": ci.get("ci_lower"),
                             "ci_upper": ci.get("ci_upper")})
        # Temporal SENECA
        val_s = perf_temporal_seneca.get(key)
        ci_s = boot_temporal_agg.get(f"{key}_seneca", {})
        if val_s is not None:
            auc_rows.append({"timepoint": tp, "model": "SENECA", "cohort": "temporal",
                             "auc": val_s, "ci_lower": ci_s.get("ci_lower"),
                             "ci_upper": ci_s.get("ci_upper")})

    # --- Temporal Brier (per-cohort with CIs) ---
    brier_rows = []
    for tp in timepoints:
        key = f"brier_{tp}m"
        # Internal ML
        val_i = perf_internal.get(key)
        ci_i = boot_internal_agg.get(f"{key}_ml", {})
        if val_i is not None:
            brier_rows.append({
                "timepoint": tp, "model": "ML", "cohort": "internal",
                "brier": val_i,
                "ci_lower": ci_i.get("ci_lower"),
                "ci_upper": ci_i.get("ci_upper"),
            })
        # Temporal ML
        val_t = perf_temporal_ml.get(key)
        ci_t = boot_temporal_agg.get(f"{key}_ml", {})
        if val_t is not None:
            brier_rows.append({
                "timepoint": tp, "model": "ML", "cohort": "temporal",
                "brier": val_t,
                "ci_lower": ci_t.get("ci_lower"),
                "ci_upper": ci_t.get("ci_upper"),
            })

    return {
        "temporal_auc": pd.DataFrame(auc_rows),
        "temporal_brier": pd.DataFrame(brier_rows),
    }


# ======================================================================
# MLFLOW LOGGING
# ======================================================================

def _log_metrics_mlflow(
    perf_internal: Dict,
    perf_internal_seneca: Dict,
    perf_temporal_ml: Dict,
    perf_temporal_seneca: Dict,
    boot_internal: Dict,
    boot_temporal: Dict,
    strat_internal: Dict[str, Dict],
    strat_temporal: Dict[str, Dict],
):
    """Log key scalar metrics to MLflow for experiment comparison."""
    # -- Internal point estimates --
    for k, v in perf_internal.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v)

    # -- Internal SENECA point estimates --
    for k, v in perf_internal_seneca.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"{k}_internal_seneca", v)

    # -- Temporal ML --
    for k, v in perf_temporal_ml.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"{k}_temporal", v)

    # -- Temporal SENECA --
    for k, v in perf_temporal_seneca.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"{k}_seneca", v)

    # -- Bootstrap CIs for both cohorts --
    for scope, boot in [("internal", boot_internal), ("temporal", boot_temporal)]:
        agg = boot.get("aggregated", {})
        # Key metrics with CIs
        for key in [
            "c_index_ml", "c_index_seneca", "c_index_diff",
            "c_index_ipcw_ml", "c_index_ipcw_seneca", "c_index_ipcw_diff",
            "mean_auc_ml", "mean_auc_seneca", "mean_auc_diff",
            "ibs_overall_ml",
        ]:
            if key in agg:
                for stat in ["median", "ci_lower", "ci_upper"]:
                    val = agg[key].get(stat)
                    if val is not None:
                        mlflow.log_metric(f"boot_{scope}_{key}_{stat}", val)

        # Per-timepoint AUC CIs
        for tp in [6, 12, 18, 24]:
            for suffix in ["ml", "seneca"]:
                auc_key = f"auc_{tp}m_{suffix}"
                if auc_key in agg:
                    for stat in ["median", "ci_lower", "ci_upper"]:
                        val = agg[auc_key].get(stat)
                        if val is not None:
                            mlflow.log_metric(
                                f"boot_{scope}_{auc_key}_{stat}", val)

        # Per-timepoint Brier CIs (ML only)
        for tp in [6, 12, 18, 24]:
            brier_key = f"brier_{tp}m_ml"
            if brier_key in agg:
                for stat in ["median", "ci_lower", "ci_upper"]:
                    val = agg[brier_key].get(stat)
                    if val is not None:
                        mlflow.log_metric(
                            f"boot_{scope}_{brier_key}_{stat}", val)

        # Comparison p-values
        for comp_key in [
            "comparison_c_index", "comparison_c_index_ipcw",
            "comparison_mean_auc",
        ]:
            comp = boot.get(comp_key, {})
            if "p_value" in comp:
                mlflow.log_metric(f"boot_{scope}_{comp_key}_pvalue",
                                  comp["p_value"])

    # -- Stratification --
    for scope, strat_dict in [("", strat_internal), ("_temporal", strat_temporal)]:
        for scheme_name, sdata in strat_dict.items():
            lr = sdata.get("logrank", {})
            if "p_value" in lr:
                mlflow.log_metric(f"logrank_p{scope}_{scheme_name}", lr["p_value"])
            ppv_npv = sdata.get("ppv_npv", {})
            for met in ["ppv", "npv"]:
                val = ppv_npv.get(met)
                if val is not None:
                    mlflow.log_metric(f"{met}{scope}_{scheme_name}", val)
            cox = sdata.get("cox", {})
            for ckey, cval in cox.items():
                if isinstance(cval, (int, float)):
                    mlflow.log_metric(f"cox_{ckey}{scope}_{scheme_name}", cval)


# ======================================================================
# SUMMARY TABLES (Tables 1–4)
# ======================================================================

def _build_overall_summary_table(
    perf_ml: Dict,
    perf_seneca: Dict,
    boot_agg: Dict,
    cohort: str,
) -> pd.DataFrame:
    """Tables 1 / 2 — overall discrimination + calibration metrics.

    One row per metric (Harrell C, Uno C, Mean AUC, IBS).
    Columns: metric, ml, ml_ci_lower, ml_ci_upper,
             seneca, seneca_ci_lower, seneca_ci_upper,
             delta, delta_ci_lower, delta_ci_upper.
    """
    rows = []
    spec = [
        ("harrell_c_index", "c_index",      "c_index_ml",      "c_index_seneca",      "c_index_diff"),
        ("uno_c_index",     "c_index_ipcw", "c_index_ipcw_ml", "c_index_ipcw_seneca", "c_index_ipcw_diff"),
        ("mean_auc",        "mean_auc",     "mean_auc_ml",     "mean_auc_seneca",     "mean_auc_diff"),
        ("ibs",             "ibs_overall",  "ibs_overall_ml",  None,                  None),
    ]
    for label, perf_key, boot_ml_key, boot_sen_key, boot_diff_key in spec:
        ml_val = perf_ml.get(perf_key)
        sen_val = perf_seneca.get(perf_key)
        ml_ci = boot_agg.get(boot_ml_key, {})
        sen_ci = boot_agg.get(boot_sen_key, {}) if boot_sen_key else {}
        diff_ci = boot_agg.get(boot_diff_key, {}) if boot_diff_key else {}
        delta = None
        if ml_val is not None and sen_val is not None:
            delta = ml_val - sen_val
        rows.append({
            "cohort": cohort,
            "metric": label,
            "ml": ml_val,
            "ml_ci_lower": ml_ci.get("ci_lower"),
            "ml_ci_upper": ml_ci.get("ci_upper"),
            "seneca": sen_val,
            "seneca_ci_lower": sen_ci.get("ci_lower"),
            "seneca_ci_upper": sen_ci.get("ci_upper"),
            "delta": delta,
            "delta_ci_lower": diff_ci.get("ci_lower"),
            "delta_ci_upper": diff_ci.get("ci_upper"),
        })
    return pd.DataFrame(rows)


def _build_timedep_summary_table(
    perf_ml: Dict,
    perf_seneca: Dict,
    boot_agg: Dict,
    cohort: str,
) -> pd.DataFrame:
    """Tables 3 / 4 — AUC + Brier at 6, 12, 18, 24 months.

    Columns: cohort, timepoint, metric,
             ml, ml_ci_lower, ml_ci_upper,
             seneca, seneca_ci_lower, seneca_ci_upper.
    """
    rows = []
    for tp in [6, 12, 18, 24]:
        # AUC
        auc_ml = perf_ml.get(f"auc_{tp}m")
        auc_sen = perf_seneca.get(f"auc_{tp}m")
        auc_ml_ci = boot_agg.get(f"auc_{tp}m_ml", {})
        auc_sen_ci = boot_agg.get(f"auc_{tp}m_seneca", {})
        rows.append({
            "cohort": cohort, "timepoint": tp, "metric": "auc",
            "ml": auc_ml,
            "ml_ci_lower": auc_ml_ci.get("ci_lower"),
            "ml_ci_upper": auc_ml_ci.get("ci_upper"),
            "seneca": auc_sen,
            "seneca_ci_lower": auc_sen_ci.get("ci_lower"),
            "seneca_ci_upper": auc_sen_ci.get("ci_upper"),
        })
        # Brier (ML only — SENECA has no S(t))
        brier_ml = perf_ml.get(f"brier_{tp}m")
        brier_ml_ci = boot_agg.get(f"brier_{tp}m_ml", {})
        rows.append({
            "cohort": cohort, "timepoint": tp, "metric": "brier",
            "ml": brier_ml,
            "ml_ci_lower": brier_ml_ci.get("ci_lower"),
            "ml_ci_upper": brier_ml_ci.get("ci_upper"),
            "seneca": None,
            "seneca_ci_lower": None,
            "seneca_ci_upper": None,
        })
    return pd.DataFrame(rows)


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", required=True)
    parser.add_argument("--training_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--external_data", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--n_bootstrap", type=int, default=200)
    parser.add_argument("--output_dir", required=True,
                        help="Metrics JSON output directory")
    parser.add_argument("--tables_dir", required=True,
                        help="Tables CSV output directory")
    parser.add_argument("--r_input_dir", required=True,
                        help="R input CSV output directory")
    parser.add_argument("--python_figures_dir", required=True,
                        help="Python-generated figures output directory")
    args = parser.parse_args()

    # Output directories
    metrics_dir = Path(args.output_dir)
    tables_dir = Path(args.tables_dir)
    r_input_dir = Path(args.r_input_dir)
    python_figures_dir = Path(args.python_figures_dir)
    km_figures_dir = python_figures_dir / "km"
    for d in [metrics_dir, tables_dir, r_input_dir, km_figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Subdirectories for full internal test set (all patients, ML only)
    internal_full_metrics_dir = metrics_dir / "internal_full"
    internal_full_tables_dir = tables_dir / "internal_full"
    for d in [internal_full_metrics_dir, internal_full_tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    mlflow.start_run()
    mlflow.log_param("n_bootstrap", args.n_bootstrap)
    mlflow.log_param("stratification_schemes",
                     str([s["name"] for s in STRATIFICATION_SCHEMES]))

    # ── Load predictions from generate_predictions step ──────────────
    pred_dir = Path(args.predictions_dir)
    df_internal = pd.read_csv(pred_dir / "internal_predictions.csv")
    df_temporal = pd.read_csv(pred_dir / "temporal_predictions.csv")
    with open(pred_dir / "training_risk_scores.json") as f:
        train_scores_data = json.load(f)
    train_risk_scores = np.array(train_scores_data["scores"])

    # ── Load training data (for IPCW y_train) ───────────────────────
    df_train = _load_data(args.training_data)
    if "os" in df_train.columns and "death" in df_train.columns:
        df_train.rename(columns={"os": "tte", "death": "event"}, inplace=True)

    # ── Load raw test/external data (for model Brier computation) ────
    df_test_raw = _load_data(args.test_data)
    df_ext_raw = _load_data(args.external_data)
    for df in [df_test_raw, df_ext_raw]:
        if "os" in df.columns and "death" in df.columns:
            df.rename(columns={"os": "tte", "death": "event"}, inplace=True)

    # ── Structured arrays ────────────────────────────────────────────
    y_train = make_structured_array(df_train["event"].values, df_train["tte"].values)
    y_internal = make_structured_array(
        df_internal["event"].values, df_internal["tte"].values
    )
    y_temporal = make_structured_array(
        df_temporal["event"].values, df_temporal["tte"].values
    )

    # ── Load ML model (for Brier / S(t) computation) ─────────────────
    model_uri = f"models:/{args.model_name}/{args.model_version}"
    logger.info(f"Loading model: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    # Feature matrices from raw data (Brier score needs model predictions at grid)
    drop_cols = ["tte", "event", "id"]
    X_internal = df_test_raw.drop(
        columns=[c for c in drop_cols if c in df_test_raw.columns]
    )
    X_temporal = df_ext_raw.drop(
        columns=[c for c in drop_cols if c in df_ext_raw.columns]
    )

    # ── Compute SENECA scores on internal set (if not present) ───────
    if "seneca_risk_score" not in df_internal.columns:
        logger.info("Computing SENECA scores for internal test set...")
        seneca_model_obj = SENECAModel()
        seneca_preds = seneca_model_obj.predict(df_test_raw)
        df_internal["seneca_risk_score"] = seneca_preds["seneca_risk_score"].values
        df_internal["seneca_complete_case"] = seneca_preds["complete_case"].values
        n_int_cc = int(df_internal["seneca_complete_case"].sum())
        logger.info(f"  SENECA complete cases (internal): {n_int_cc}/{len(df_internal)}")

    # ── Pre-compute temporal grids and survival matrices ─────────────
    grid_int, _, horizon_int = create_temporal_grid(y_train, y_internal)
    grid_temp, _, horizon_temp = create_temporal_grid(y_train, y_temporal)
    surv_matrix_internal = _precompute_survival_matrix(model, X_internal, grid_int)
    surv_matrix_temporal = _precompute_survival_matrix(model, X_temporal, grid_temp)

    # ── Censoring-awareness audit ────────────────────────────────────
    logger.info("=" * 70)
    logger.info("CENSORING-AWARENESS AUDIT")
    logger.info("=" * 70)
    logger.info("  Harrell C-index:       concordance_index_censored  (censoring-aware, no IPCW)")
    logger.info("  Uno C-index:           concordance_index_ipcw      (IPCW from y_train)")
    logger.info("  Time-dep AUC:          cumulative_dynamic_auc      (IPCW, Uno 2007)")
    logger.info("  Brier score:           sksurv.brier_score          (IPCW, Graf 1999)")
    logger.info("  IBS:                   integrated_brier_score      (IPCW integral)")
    logger.info("  ROC/PR curves:         IPCW-weighted               (Heagerty & Zheng 2004)")
    logger.info("  Bootstrap CIs:         stratified resampling, percentile method")
    logger.info("  No naive binary approximations are used.")
    logger.info("=" * 70)

    # ── Diagnostic: follow-up summary per cohort ────────────────────
    logger.info("=" * 70)
    logger.info("FOLLOW-UP DIAGNOSTICS (24-month grid verification)")
    logger.info("=" * 70)
    for label, y_arr, df in [
        ("training", y_train, df_train),
        ("internal", y_internal, df_internal),
        ("temporal", y_temporal, df_temporal),
    ]:
        tte = df["tte"].values
        ev = df["event"].values.astype(bool)
        logger.info(f"  [{label}] n={len(tte)}")
        logger.info(f"    max follow-up (all):        {tte.max():.2f} months")
        logger.info(f"    max follow-up (events):     {tte[ev].max():.2f} months" if ev.any() else f"    max follow-up (events):     N/A (no events)")
        logger.info(f"    max follow-up (censored):   {tte[~ev].max():.2f} months" if (~ev).any() else f"    max follow-up (censored):   N/A (none censored)")
        logger.info(f"    n with follow-up >= 24m:    {int((tte >= 24).sum())}")
        logger.info(f"    n events by 24m:            {int(((ev) & (tte <= 24)).sum())}")
        logger.info(f"    n events beyond 24m:        {int(((ev) & (tte > 24)).sum())}")
    eval_horizon = min(
        float(y_train["time"].max()),
        float(y_temporal["time"].max()),
        24.0,
    )
    logger.info(f"  Computed evaluation_horizon = min(train_max, temporal_max, 24.0) = {eval_horizon:.4f}")
    logger.info(f"  24m included in grid? {'YES' if eval_horizon >= 24.0 else 'NO — max follow-up < 24m'}")
    logger.info("=" * 70)

    # ── Cohort characteristics ───────────────────────────────────────
    logger.info("=" * 70)
    logger.info("COHORT CHARACTERISTICS")
    logger.info("=" * 70)
    cohort_chars = _cohort_characteristics(df_train, df_internal, df_temporal)
    with open(metrics_dir / "cohort_characteristics.json", "w") as f:
        json.dump(_to_native(cohort_chars), f, indent=2)

    # ================================================================
    # PERFORMANCE — Internal test set (ML + SENECA)
    # ================================================================
    logger.info("=" * 70)
    logger.info("INTERNAL TEST SET PERFORMANCE")
    logger.info("=" * 70)
    perf_internal = _compute_cohort_performance(
        model, X_internal, y_train, y_internal,
        df_internal["ml_risk_score"].values, "internal_ml",
    )

    # SENECA on internal (complete cases)
    cc_mask_int = df_internal["seneca_complete_case"] == 1
    y_internal_cc = make_structured_array(
        df_internal.loc[cc_mask_int, "event"].values,
        df_internal.loc[cc_mask_int, "tte"].values,
    )
    perf_internal_seneca: Dict = {}
    if cc_mask_int.any():
        perf_internal_seneca["c_index"] = compute_c_index(
            y_internal_cc,
            df_internal.loc[cc_mask_int, "seneca_risk_score"].values,
        )
        perf_internal_seneca["c_index_ipcw"] = compute_c_index_ipcw(
            y_train, y_internal_cc,
            df_internal.loc[cc_mask_int, "seneca_risk_score"].values,
        )
        seneca_auc_int = compute_time_dependent_auc(
            y_train, y_internal_cc,
            df_internal.loc[cc_mask_int, "seneca_risk_score"].values,
        )
        perf_internal_seneca.update(seneca_auc_int)
        perf_internal_seneca["n_complete_cases"] = int(cc_mask_int.sum())
        logger.info(f"  SENECA internal: C={perf_internal_seneca.get('c_index', 'N/A'):.4f}, "
                     f"Uno={perf_internal_seneca.get('c_index_ipcw', 'N/A'):.4f}")

    internal_perf = {"ml": perf_internal, "seneca": perf_internal_seneca}
    with open(metrics_dir / "performance_internal.json", "w") as f:
        json.dump(_to_native(internal_perf), f, indent=2)

    # ================================================================
    # PERFORMANCE — Temporal validation (ML + SENECA)
    # ================================================================
    logger.info("=" * 70)
    logger.info("TEMPORAL VALIDATION PERFORMANCE")
    logger.info("=" * 70)

    # ML metrics on FULL temporal set
    perf_temporal_ml = _compute_cohort_performance(
        model, X_temporal, y_train, y_temporal,
        df_temporal["ml_risk_score"].values, "temporal_ml_full",
    )

    # Complete-case mask
    cc_mask = df_temporal["seneca_complete_case"] == 1
    y_temporal_cc = make_structured_array(
        df_temporal.loc[cc_mask, "event"].values,
        df_temporal.loc[cc_mask, "tte"].values,
    )
    logger.info(f"  Complete cases (temporal): {int(cc_mask.sum())}/{len(df_temporal)}")

    # ML metrics on COMPLETE-CASE subset (head-to-head with SENECA)
    X_temporal_cc = X_temporal.loc[cc_mask]
    perf_temporal_ml_cc = _compute_cohort_performance(
        model, X_temporal_cc, y_train, y_temporal_cc,
        df_temporal.loc[cc_mask, "ml_risk_score"].values, "temporal_ml_cc",
    )
    perf_temporal_ml_cc["n_complete_cases"] = int(cc_mask.sum())

    # SENECA metrics (complete cases only)
    perf_temporal_seneca: Dict = {}
    perf_temporal_seneca["c_index"] = compute_c_index(
        y_temporal_cc, df_temporal.loc[cc_mask, "seneca_risk_score"].values
    )
    perf_temporal_seneca["c_index_ipcw"] = compute_c_index_ipcw(
        y_train, y_temporal_cc,
        df_temporal.loc[cc_mask, "seneca_risk_score"].values,
    )
    seneca_auc = compute_time_dependent_auc(
        y_train, y_temporal_cc,
        df_temporal.loc[cc_mask, "seneca_risk_score"].values,
    )
    perf_temporal_seneca.update(seneca_auc)
    perf_temporal_seneca["n_complete_cases"] = int(cc_mask.sum())

    temporal_perf = {
        "ml_full": perf_temporal_ml,
        "ml_cc": perf_temporal_ml_cc,
        "seneca": perf_temporal_seneca,
    }
    with open(metrics_dir / "performance_temporal.json", "w") as f:
        json.dump(_to_native(temporal_perf), f, indent=2)

    # ================================================================
    # BOOTSTRAP — Internal test set (ML + SENECA)
    # ================================================================
    logger.info("=" * 70)
    logger.info("BOOTSTRAP — INTERNAL")
    logger.info("=" * 70)
    has_seneca_int = cc_mask_int.any() and "seneca_risk_score" in df_internal.columns
    boot_internal = _run_bootstrap(
        df_internal, y_train, args.n_bootstrap,
        surv_matrix=surv_matrix_internal,
        grid=grid_int, horizon=horizon_int,
        include_seneca=has_seneca_int,
        label="Bootstrap (internal)",
    )
    with open(metrics_dir / "bootstrap_internal.json", "w") as f:
        json.dump(_to_native(boot_internal), f, indent=2)

    # ================================================================
    # BOOTSTRAP — Temporal validation: FULL set (ML only, no CC filter)
    # ================================================================
    logger.info("=" * 70)
    logger.info("BOOTSTRAP — TEMPORAL FULL (ML only)")
    logger.info("=" * 70)
    boot_temporal_full = _run_bootstrap(
        df_temporal, y_train, args.n_bootstrap,
        surv_matrix=surv_matrix_temporal,
        grid=grid_temp, horizon=horizon_temp,
        include_seneca=False,
        label="Bootstrap (temporal full — ML)",
    )
    with open(metrics_dir / "bootstrap_temporal_full.json", "w") as f:
        json.dump(_to_native(boot_temporal_full), f, indent=2)
    # Backward-compat alias used by R scripts (full-cohort CIs for ML)
    with open(metrics_dir / "bootstrap_temporal.json", "w") as f:
        json.dump(_to_native(boot_temporal_full), f, indent=2)

    # ================================================================
    # BOOTSTRAP — Temporal validation: CC set (ML vs SENECA)
    # ================================================================
    logger.info("=" * 70)
    logger.info("BOOTSTRAP — TEMPORAL CC (ML vs SENECA)")
    logger.info("=" * 70)
    # Pre-compute survival matrix for CC subset
    surv_matrix_temporal_cc = None
    if surv_matrix_temporal is not None:
        surv_matrix_temporal_cc = surv_matrix_temporal[cc_mask.values]
    boot_temporal_cc = _run_bootstrap(
        df_temporal[cc_mask].reset_index(drop=True), y_train, args.n_bootstrap,
        surv_matrix=surv_matrix_temporal_cc,
        grid=grid_temp, horizon=horizon_temp,
        include_seneca=True,
        label="Bootstrap (temporal CC — ML vs SENECA)",
    )
    with open(metrics_dir / "bootstrap_temporal_cc.json", "w") as f:
        json.dump(_to_native(boot_temporal_cc), f, indent=2)

    # Backward compatibility alias
    boot_temporal = boot_temporal_cc

    # ================================================================
    # BOOTSTRAP — Internal test set FULL (ML only, no CC filter)
    # ================================================================
    logger.info("=" * 70)
    logger.info("BOOTSTRAP — INTERNAL FULL (ML only)")
    logger.info("=" * 70)
    boot_internal_full = _run_bootstrap(
        df_internal, y_train, args.n_bootstrap,
        surv_matrix=surv_matrix_internal,
        grid=grid_int, horizon=horizon_int,
        include_seneca=False,
        label="Bootstrap (internal full — ML only)",
    )
    with open(internal_full_metrics_dir / "bootstrap.json", "w") as f:
        json.dump(_to_native(boot_internal_full), f, indent=2)

    with open(internal_full_metrics_dir / "performance.json", "w") as f:
        json.dump(_to_native({"ml": perf_internal}), f, indent=2)

    # ================================================================
    # RISK STRATIFICATION
    # ================================================================
    logger.info("=" * 70)
    logger.info("RISK STRATIFICATION")
    logger.info("=" * 70)

    strat_internal: Dict[str, Dict] = {}
    strat_temporal: Dict[str, Dict] = {}
    strat_temporal_cc: Dict[str, Dict] = {}

    for scheme in STRATIFICATION_SCHEMES:
        name = scheme["name"]
        logger.info(f"--- Scheme: {name} ---")

        # Compute thresholds on TRAINING risk scores (no leakage)
        thresholds = define_risk_groups(
            train_risk_scores,
            low_percentile=scheme["low_percentile"],
            high_percentile=scheme["high_percentile"],
        )

        # Internal
        si = _evaluate_stratification(
            y_internal, df_internal["ml_risk_score"].values, thresholds, name
        )
        strat_internal[name] = si

        # Save KM statistics table
        si["km_statistics"].to_csv(
            tables_dir / f"km_statistics_internal_{name}.csv", index=False
        )

        # Save HR table
        hr_data = si.get("pairwise_hr", [])
        if isinstance(hr_data, pd.DataFrame) and not hr_data.empty:
            hr_data.to_csv(
                tables_dir / f"hazard_ratios_internal_{name}.csv", index=False
            )
        elif isinstance(hr_data, list) and len(hr_data) > 0:
            pd.DataFrame(hr_data).to_csv(
                tables_dir / f"hazard_ratios_internal_{name}.csv", index=False
            )

        # PPV/NPV table
        pd.DataFrame([si["ppv_npv"]]).to_csv(
            tables_dir / f"ppv_npv_internal_{name}.csv", index=False
        )

        # R input — individual-level KM data
        _km_r_data(df_internal, si["groups"], name, "internal").to_csv(
            r_input_dir / f"km_data_internal_{name}.csv", index=False
        )
        _generate_km_figures(
            df_internal, si["groups"], y_internal, name, "internal", km_figures_dir
        )

        # Temporal — ML on FULL set
        st_ml = _evaluate_stratification(
            y_temporal, df_temporal["ml_risk_score"].values, thresholds, name
        )
        strat_temporal[name] = st_ml
        st_ml["km_statistics"].to_csv(
            tables_dir / f"km_statistics_temporal_{name}.csv", index=False
        )
        hr_data_t = st_ml.get("pairwise_hr", [])
        if isinstance(hr_data_t, pd.DataFrame) and not hr_data_t.empty:
            hr_data_t.to_csv(
                tables_dir / f"hazard_ratios_temporal_{name}.csv", index=False
            )
        elif isinstance(hr_data_t, list) and len(hr_data_t) > 0:
            pd.DataFrame(hr_data_t).to_csv(
                tables_dir / f"hazard_ratios_temporal_{name}.csv", index=False
            )
        pd.DataFrame([st_ml["ppv_npv"]]).to_csv(
            tables_dir / f"ppv_npv_temporal_{name}.csv", index=False
        )
        _km_r_data(df_temporal, st_ml["groups"], name, "temporal").to_csv(
            r_input_dir / f"km_data_temporal_{name}.csv", index=False
        )
        _generate_km_figures(
            df_temporal, st_ml["groups"], y_temporal, name, "temporal", km_figures_dir
        )

        # Temporal — ML on COMPLETE-CASE subset (head-to-head with SENECA)
        st_ml_cc = _evaluate_stratification(
            y_temporal_cc,
            df_temporal.loc[cc_mask, "ml_risk_score"].values,
            thresholds, name,
        )
        strat_temporal_cc[name] = st_ml_cc
        st_ml_cc["km_statistics"].to_csv(
            tables_dir / f"km_statistics_temporal_cc_{name}.csv", index=False
        )
        hr_data_cc = st_ml_cc.get("pairwise_hr", [])
        if isinstance(hr_data_cc, pd.DataFrame) and not hr_data_cc.empty:
            hr_data_cc.to_csv(
                tables_dir / f"hazard_ratios_temporal_cc_{name}.csv", index=False
            )
        elif isinstance(hr_data_cc, list) and len(hr_data_cc) > 0:
            pd.DataFrame(hr_data_cc).to_csv(
                tables_dir / f"hazard_ratios_temporal_cc_{name}.csv", index=False
            )
        pd.DataFrame([st_ml_cc["ppv_npv"]]).to_csv(
            tables_dir / f"ppv_npv_temporal_cc_{name}.csv", index=False
        )
        df_temporal_cc_for_km = df_temporal[cc_mask].copy()
        _km_r_data(df_temporal_cc_for_km, st_ml_cc["groups"], name, "temporal_cc").to_csv(
            r_input_dir / f"km_data_temporal_cc_{name}.csv", index=False
        )
        _generate_km_figures(
            df_temporal_cc_for_km, st_ml_cc["groups"], y_temporal_cc,
            name, "temporal_cc", km_figures_dir,
        )

        # Temporal — SENECA risk groups (published thresholds, CC only)
        seneca_groups_temporal = apply_risk_thresholds(
            df_temporal[cc_mask]["seneca_risk_score"].values,
            SENECA_RISK_THRESHOLDS["low"],
            SENECA_RISK_THRESHOLDS["high"],
        )
        km_seneca = compute_km_statistics(y_temporal_cc, seneca_groups_temporal)
        km_seneca.to_csv(
            tables_dir / f"km_statistics_temporal_seneca_{name}.csv", index=False
        )
        # Individual-level SENECA KM data for R
        df_temporal_cc_copy = df_temporal[cc_mask].copy()
        _km_r_data(df_temporal_cc_copy, seneca_groups_temporal, f"{name}_seneca", "temporal").to_csv(
            r_input_dir / f"km_data_temporal_seneca_{name}.csv", index=False
        )
        _generate_km_figures(
            df_temporal_cc_copy, seneca_groups_temporal, y_temporal_cc,
            f"{name}_seneca", "temporal", km_figures_dir,
        )

        # Save thresholds for reproducibility
        thresholds_out = {
            "scheme": name,
            "low_percentile": scheme["low_percentile"],
            "high_percentile": scheme["high_percentile"],
            **thresholds,
        }
        with open(tables_dir / f"risk_thresholds_{name}.json", "w") as f:
            json.dump(thresholds_out, f, indent=2)

    # ================================================================
    # ROC / PR CURVE DATA FOR R
    # ================================================================
    logger.info("=" * 70)
    logger.info("ROC / PR CURVES")
    logger.info("=" * 70)

    # Temporal FULL set — ML only (standalone ML presentation)
    logger.info("  ROC/PR: temporal full (ML only)")
    for tp in ROC_PR_TIMEPOINTS:
        roc_full = compute_ipcw_roc_at_timepoint(
            df_temporal["tte"].values, df_temporal["event"].values,
            df_temporal["ml_risk_score"].values, float(tp),
        )
        n_pts = len(roc_full["fpr"])
        pd.DataFrame({
            "model": ["ML"] * n_pts,
            "timepoint": [tp] * n_pts,
            "fpr": roc_full["fpr"], "tpr": roc_full["tpr"],
            "threshold": roc_full["thresholds"],
            "auc": [roc_full["auc"]] * n_pts,
        }).to_csv(r_input_dir / f"roc_curves_temporal_full_{tp}m.csv",
                  index=False, float_format="%.6f")

        pr_full = compute_ipcw_pr_at_timepoint(
            df_temporal["tte"].values, df_temporal["event"].values,
            df_temporal["ml_risk_score"].values, float(tp),
            invert=(tp >= 18),
        )
        n_pts_pr = len(pr_full["recall"])
        pd.DataFrame({
            "model": ["ML"] * n_pts_pr,
            "timepoint": [tp] * n_pts_pr,
            "recall": pr_full["recall"], "precision": pr_full["precision"],
            "threshold": pr_full["thresholds"],
            "average_precision": [pr_full["average_precision"]] * n_pts_pr,
            "inverted": [tp >= 18] * n_pts_pr,
        }).to_csv(r_input_dir / f"pr_curves_temporal_full_{tp}m.csv",
                  index=False, float_format="%.6f")

    # Temporal CC set — ML vs SENECA (head-to-head comparison)
    logger.info("  ROC/PR: temporal CC (ML vs SENECA)")
    roc_pr = _roc_pr_data(
        tte=df_temporal[cc_mask]["tte"].values,
        event=df_temporal[cc_mask]["event"].values,
        ml_scores=df_temporal[cc_mask]["ml_risk_score"].values,
        seneca_scores=df_temporal[cc_mask]["seneca_risk_score"].values,
    )
    for fname, rdf in roc_pr.items():
        # Save with _cc suffix for clarity
        rdf.to_csv(r_input_dir / f"{fname}_cc.csv", index=False, float_format="%.6f")
        # Backward compatibility: also save without suffix (R scripts expect these)
        rdf.to_csv(r_input_dir / f"{fname}.csv", index=False, float_format="%.6f")

    # Operating-point summary table (CC set, ML vs SENECA)
    roc_summary = create_roc_summary_table(
        tte=df_temporal[cc_mask]["tte"].values,
        event=df_temporal[cc_mask]["event"].values,
        ml_scores=df_temporal[cc_mask]["ml_risk_score"].values,
        seneca_scores=df_temporal[cc_mask]["seneca_risk_score"].values,
        ml_train_scores=train_risk_scores,
        use_ipcw=True,
    )
    roc_summary.to_csv(tables_dir / "roc_operating_points.csv",
                       index=False, float_format="%.4f")

    # Internal ROC / PR (ML only, no SENECA on internal set)
    for tp in ROC_PR_TIMEPOINTS:
        roc_int = compute_ipcw_roc_at_timepoint(
            df_internal["tte"].values, df_internal["event"].values,
            df_internal["ml_risk_score"].values, float(tp),
        )
        n_pts = len(roc_int["fpr"])
        pd.DataFrame({
            "model": ["ML"] * n_pts,
            "timepoint": [tp] * n_pts,
            "fpr": roc_int["fpr"], "tpr": roc_int["tpr"],
            "threshold": roc_int["thresholds"],
            "auc": [roc_int["auc"]] * n_pts,
        }).to_csv(r_input_dir / f"roc_curves_internal_{tp}m.csv",
                  index=False, float_format="%.6f")

    # ================================================================
    # CALIBRATION INPUT FOR R
    # ================================================================
    logger.info("Preparing calibration input for R...")
    _calibration_r_data(df_internal, "internal").to_csv(
        r_input_dir / "calibration_input_internal.csv", index=False, float_format="%.6f"
    )
    _calibration_r_data(df_temporal, "temporal").to_csv(
        r_input_dir / "calibration_input_temporal.csv", index=False, float_format="%.6f"
    )

    # ================================================================
    # RISK DENSITY DATA FOR R
    # ================================================================
    _risk_density_r_data(df_internal, "internal").to_csv(
        r_input_dir / "risk_density_internal.csv", index=False, float_format="%.6f"
    )
    _risk_density_r_data(df_temporal, "temporal").to_csv(
        r_input_dir / "risk_density_temporal.csv", index=False, float_format="%.6f"
    )

    # Predicted-risk distributions P(T<=t) at 6 m & 18 m (both cohorts)
    _predicted_risk_distributions(df_internal, df_temporal).to_csv(
        r_input_dir / "predicted_risk_distributions.csv", index=False, float_format="%.6f"
    )

    # IPCW-stratified distributions (temporal cohort)
    p_event_cols_temporal = {}
    for tp in [6, 18]:
        col = f"P_event_{tp}m"
        if col in df_temporal.columns:
            p_event_cols_temporal[tp] = df_temporal[col].values
    if p_event_cols_temporal:
        _ipcw_stratified_distributions(
            df_temporal["tte"].values,
            df_temporal["event"].values,
            df_temporal["ml_risk_score"].values,
            p_event_cols_temporal,
            cohort="temporal",
        ).to_csv(
            r_input_dir / "ipcw_stratified_distributions.csv",
            index=False, float_format="%.6f",
        )

    # ================================================================
    # TEMPORAL AUC / BRIER CURVES FOR R
    # ================================================================
    boot_int_agg = boot_internal.get("aggregated", {})
    boot_temp_full_agg = boot_temporal_full.get("aggregated", {})
    boot_temp_cc_agg = boot_temporal_cc.get("aggregated", {})

    # Temporal curves: include ML-full, ML-CC, and SENECA
    temporal_curves = _temporal_curves_for_r(
        perf_internal, perf_temporal_ml, perf_temporal_seneca,
        boot_int_agg, boot_temp_cc_agg,
    )
    for tname, tdf in temporal_curves.items():
        tdf.to_csv(r_input_dir / f"{tname}.csv", index=False, float_format="%.6f")

    # Additional: ML-full temporal AUC/Brier with proper CIs from boot_temporal_full
    auc_rows_full = []
    brier_rows_full = []
    for tp in [6, 12, 18, 24]:
        auc_val = perf_temporal_ml.get(f"auc_{tp}m")
        auc_ci = boot_temp_full_agg.get(f"auc_{tp}m_ml", {})
        if auc_val is not None:
            auc_rows_full.append({
                "timepoint": tp, "model": "ML", "cohort": "temporal_full",
                "auc": auc_val,
                "ci_lower": auc_ci.get("ci_lower"),
                "ci_upper": auc_ci.get("ci_upper"),
            })
        brier_val = perf_temporal_ml.get(f"brier_{tp}m")
        brier_ci = boot_temp_full_agg.get(f"brier_{tp}m_ml", {})
        if brier_val is not None:
            brier_rows_full.append({
                "timepoint": tp, "model": "ML", "cohort": "temporal_full",
                "brier": brier_val,
                "ci_lower": brier_ci.get("ci_lower"),
                "ci_upper": brier_ci.get("ci_upper"),
            })
    # ML-CC temporal AUC/Brier
    auc_rows_cc = []
    brier_rows_cc = []
    for tp in [6, 12, 18, 24]:
        auc_val = perf_temporal_ml_cc.get(f"auc_{tp}m")
        auc_ci = boot_temp_cc_agg.get(f"auc_{tp}m_ml", {})
        if auc_val is not None:
            auc_rows_cc.append({
                "timepoint": tp, "model": "ML", "cohort": "temporal_cc",
                "auc": auc_val,
                "ci_lower": auc_ci.get("ci_lower"),
                "ci_upper": auc_ci.get("ci_upper"),
            })
        brier_val = perf_temporal_ml_cc.get(f"brier_{tp}m")
        brier_ci = boot_temp_cc_agg.get(f"brier_{tp}m_ml", {})
        if brier_val is not None:
            brier_rows_cc.append({
                "timepoint": tp, "model": "ML", "cohort": "temporal_cc",
                "brier": brier_val,
                "ci_lower": brier_ci.get("ci_lower"),
                "ci_upper": brier_ci.get("ci_upper"),
            })
    if auc_rows_full or auc_rows_cc:
        pd.DataFrame(auc_rows_full + auc_rows_cc).to_csv(
            r_input_dir / "temporal_auc_extended.csv", index=False, float_format="%.6f"
        )
    if brier_rows_full or brier_rows_cc:
        pd.DataFrame(brier_rows_full + brier_rows_cc).to_csv(
            r_input_dir / "temporal_brier_extended.csv", index=False, float_format="%.6f"
        )

    # ================================================================
    # COMPARISON TABLE (Internal vs Temporal)
    # ================================================================
    logger.info("Creating comparison table...")
    comparison = {}
    for key in ["c_index", "c_index_ipcw", "mean_auc"]:
        vi = perf_internal.get(key)
        vt = perf_temporal_ml.get(key)
        if vi is not None and vt is not None:
            comparison[key] = {
                "internal": vi, "temporal_full": vt,
                "difference": round(vt - vi, 4),
            }
    for tp in [6, 12, 18, 24]:
        key = f"auc_{tp}m"
        vi = perf_internal.get(key)
        vt = perf_temporal_ml.get(key)
        if vi is not None and vt is not None:
            comparison[key] = {
                "internal": vi, "temporal_full": vt,
                "difference": round(vt - vi, 4),
            }
    with open(metrics_dir / "comparison_internal_vs_temporal.json", "w") as f:
        json.dump(_to_native(comparison), f, indent=2)

    # ================================================================
    # TABLES 1–4: Overall + Time-Dependent Summary
    # ================================================================
    logger.info("Building summary tables (Tables 1–4)...")

    # Table 1: Internal overall
    tbl1 = _build_overall_summary_table(
        perf_internal, perf_internal_seneca, boot_int_agg, "internal",
    )
    tbl1.to_csv(tables_dir / "table1_overall_internal.csv",
                index=False, float_format="%.4f")

    # Table 2a: Temporal overall — ML full (standalone)
    tbl2a = _build_overall_summary_table(
        perf_temporal_ml, {}, boot_temp_full_agg, "temporal_full",
    )
    tbl2a.to_csv(tables_dir / "table2a_overall_temporal_full.csv",
                 index=False, float_format="%.4f")

    # Table 2b: Temporal overall — ML CC vs SENECA (head-to-head)
    tbl2b = _build_overall_summary_table(
        perf_temporal_ml_cc, perf_temporal_seneca, boot_temp_cc_agg, "temporal_cc",
    )
    tbl2b.to_csv(tables_dir / "table2b_overall_temporal_cc.csv",
                 index=False, float_format="%.4f")

    # Backward compatibility: Table 2 (old name) = CC comparison
    tbl2b.to_csv(tables_dir / "table2_overall_temporal.csv",
                 index=False, float_format="%.4f")

    # Table 3: Internal time-dependent
    tbl3 = _build_timedep_summary_table(
        perf_internal, perf_internal_seneca, boot_int_agg, "internal",
    )
    tbl3.to_csv(tables_dir / "table3_timedep_internal.csv",
                index=False, float_format="%.4f")

    # Table 4a: Temporal time-dependent — ML full
    tbl4a = _build_timedep_summary_table(
        perf_temporal_ml, {}, boot_temp_full_agg, "temporal_full",
    )
    tbl4a.to_csv(tables_dir / "table4a_timedep_temporal_full.csv",
                 index=False, float_format="%.4f")

    # Table 4b: Temporal time-dependent — ML CC vs SENECA
    tbl4b = _build_timedep_summary_table(
        perf_temporal_ml_cc, perf_temporal_seneca, boot_temp_cc_agg, "temporal_cc",
    )
    tbl4b.to_csv(tables_dir / "table4b_timedep_temporal_cc.csv",
                 index=False, float_format="%.4f")

    # Backward compatibility
    tbl4b.to_csv(tables_dir / "table4_timedep_temporal.csv",
                 index=False, float_format="%.4f")

    # Internal full — overall (all patients, ML only, bootstrap CIs consistent with population)
    _build_overall_summary_table(
        perf_internal, {}, boot_internal_full.get("aggregated", {}), "internal_full",
    ).to_csv(
        internal_full_tables_dir / "table_overall.csv",
        index=False, float_format="%.4f",
    )

    # Internal full — time-dependent (all patients, ML only)
    _build_timedep_summary_table(
        perf_internal, {}, boot_internal_full.get("aggregated", {}), "internal_full",
    ).to_csv(
        internal_full_tables_dir / "table_timedep.csv",
        index=False, float_format="%.4f",
    )

    # ================================================================
    # CONSOLIDATED PERFORMANCE SUMMARY TABLE
    # ================================================================
    logger.info("Building consolidated performance summary...")
    n_internal = len(df_internal)
    n_internal_cc = int(cc_mask_int.sum())
    n_temporal_full = len(df_temporal)
    n_temporal_cc = int(cc_mask.sum())

    summary_rows = []
    for key, val in perf_internal.items():
        if isinstance(val, (int, float)) and not np.isnan(val):
            ci = boot_int_agg.get(f"{key}_ml", boot_int_agg.get(key, {}))
            summary_rows.append({
                "cohort": "internal", "model": "ML", "n": n_internal,
                "metric": key, "value": round(val, 4),
                "ci_lower": ci.get("ci_lower"),
                "ci_upper": ci.get("ci_upper"),
            })
    for key, val in perf_internal_seneca.items():
        if isinstance(val, (int, float)) and not np.isnan(val):
            ci = boot_int_agg.get(f"{key}_seneca", {})
            summary_rows.append({
                "cohort": "internal", "model": "SENECA", "n": n_internal_cc,
                "metric": key, "value": round(val, 4),
                "ci_lower": ci.get("ci_lower"),
                "ci_upper": ci.get("ci_upper"),
            })
    # Temporal FULL — ML
    for key, val in perf_temporal_ml.items():
        if isinstance(val, (int, float)) and not np.isnan(val):
            ci = boot_temp_full_agg.get(f"{key}_ml", {})
            summary_rows.append({
                "cohort": "temporal_full", "model": "ML", "n": n_temporal_full,
                "metric": key, "value": round(val, 4),
                "ci_lower": ci.get("ci_lower"),
                "ci_upper": ci.get("ci_upper"),
            })
    # Temporal CC — ML
    for key, val in perf_temporal_ml_cc.items():
        if isinstance(val, (int, float)) and not np.isnan(val):
            ci = boot_temp_cc_agg.get(f"{key}_ml", {})
            summary_rows.append({
                "cohort": "temporal_cc", "model": "ML", "n": n_temporal_cc,
                "metric": key, "value": round(val, 4),
                "ci_lower": ci.get("ci_lower"),
                "ci_upper": ci.get("ci_upper"),
            })
    # Temporal CC — SENECA
    for key, val in perf_temporal_seneca.items():
        if isinstance(val, (int, float)) and not np.isnan(val):
            ci = boot_temp_cc_agg.get(f"{key}_seneca", {})
            summary_rows.append({
                "cohort": "temporal_cc", "model": "SENECA", "n": n_temporal_cc,
                "metric": key, "value": round(val, 4),
                "ci_lower": ci.get("ci_lower"),
                "ci_upper": ci.get("ci_upper"),
            })
    # Internal full — ML (all patients, bootstrap CIs from boot_internal_full)
    boot_int_full_agg = boot_internal_full.get("aggregated", {})
    for key, val in perf_internal.items():
        if isinstance(val, (int, float)) and not np.isnan(val):
            ci = boot_int_full_agg.get(f"{key}_ml", boot_int_full_agg.get(key, {}))
            summary_rows.append({
                "cohort": "internal_full", "model": "ML", "n": n_internal,
                "metric": key, "value": round(val, 4),
                "ci_lower": ci.get("ci_lower"),
                "ci_upper": ci.get("ci_upper"),
            })
    pd.DataFrame(summary_rows).to_csv(
        tables_dir / "performance_summary.csv", index=False, float_format="%.4f"
    )

    # ================================================================
    # MLFLOW LOGGING
    # ================================================================
    _log_metrics_mlflow(
        perf_internal, perf_internal_seneca,
        perf_temporal_ml, perf_temporal_seneca,
        boot_internal, boot_temporal,
        strat_internal, strat_temporal,
    )
    # Additional: ML-CC point estimates + full-set bootstrap CIs
    for k, v in perf_temporal_ml_cc.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"{k}_temporal_cc", v)
    for key in ["c_index_ml", "c_index_ipcw_ml", "mean_auc_ml", "ibs_overall_ml"]:
        if key in boot_temp_full_agg:
            for stat in ["median", "ci_lower", "ci_upper"]:
                val = boot_temp_full_agg[key].get(stat)
                if val is not None:
                    mlflow.log_metric(f"boot_temporal_full_{key}_{stat}", val)
    mlflow.log_metric("n_temporal_full", n_temporal_full)
    mlflow.log_metric("n_temporal_cc", n_temporal_cc)
    mlflow.log_metric("n_internal", n_internal)
    mlflow.log_metric("n_internal_cc", n_internal_cc)
    # Internal full bootstrap CIs
    for key in ["c_index_ml", "c_index_ipcw_ml", "mean_auc_ml", "ibs_overall_ml"]:
        if key in boot_int_full_agg:
            for stat in ["median", "ci_lower", "ci_upper"]:
                val = boot_int_full_agg[key].get(stat)
                if val is not None:
                    mlflow.log_metric(f"boot_internal_full_{key}_{stat}", val)
    for tp in [6, 12, 18, 24]:
        auc_key = f"auc_{tp}m_ml"
        if auc_key in boot_int_full_agg:
            for stat in ["median", "ci_lower", "ci_upper"]:
                val = boot_int_full_agg[auc_key].get(stat)
                if val is not None:
                    mlflow.log_metric(f"boot_internal_full_{auc_key}_{stat}", val)

    mlflow.log_artifacts(str(metrics_dir), artifact_path="metrics")
    mlflow.log_artifacts(str(tables_dir), artifact_path="tables")
    mlflow.log_artifacts(str(r_input_dir), artifact_path="r_input")
    mlflow.log_artifacts(str(python_figures_dir), artifact_path="python_figures")

    mlflow.end_run()
    logger.info("=== Compute Metrics Complete ===")


if __name__ == "__main__":
    main()
