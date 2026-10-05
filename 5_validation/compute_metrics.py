"""Validation metrics of the ensemble, the Cox benchmark and SENECA.

Reads the cached predictions of generate_predictions.py and computes, for the
internal test set and the temporal validation cohort:

- Harrell's and Uno's C, time-dependent AUC and Brier score at 6, 12, 18 and
  24 months, IBS (definitions in lib/survival_metrics.py);
- 95% CIs from event-stratified bootstrap resamples (lib/bootstrap.py); all
  models compared on a cohort share the resamples, so their differences are
  paired;
- the comparison with SENECA on the temporal patients with all five SENECA
  inputs observed (579 of 698);
- risk groups at the 15th/85th and 33rd/67th percentiles of the training risk
  scores: Kaplan-Meier summaries, log-rank test, pairwise hazard ratios, PPV
  at 6 months and NPV at 18 months, agreement with the SENECA groups;
- completeness of the ensemble and SENECA inputs before imputation, and
  Harrell's/Uno's C in patients with and without imputed predictors;
- the inputs of the R figure and table scripts (7_publication_figures).

Paper items: Tables 2-4 (overall_*, timedep_* files), Figure 3 and Table S4
(km_statistics_temporal_15-85, hazard_ratios_temporal_15-85,
ppv_npv_temporal_15-85, Kaplan-Meier figure), Tables S7-S9 (completeness and
missingness files). The Cox-benchmark results are in the same files, under
keys and columns marked 'cox' and rows with model == "Cox".
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402
from lifelines.plotting import add_at_risk_counts  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402
from tqdm import tqdm  # noqa: E402

from lib.bootstrap import (  # noqa: E402
    PAIRED_DIFF_BASE_METRICS,
    aggregate_bootstrap_results,
    compute_bootstrap_metrics,
    stratified_bootstrap_sample,
)
from lib.cox_benchmark import COEFFICIENTS_FILE, FIT_REPORT_FILE, load_cox_benchmark  # noqa: E402
from lib.risk_stratification import (  # noqa: E402
    apply_risk_thresholds,
    compute_km_statistics,
    compute_pairwise_hazard_ratios,
    compute_ppv_npv,
    define_risk_groups,
    log_rank_test,
)
from lib.roc_pr import (  # noqa: E402
    censoring_km,
    compute_ipcw_pr_at_timepoint,
    compute_ipcw_roc_at_timepoint,
    ipcw_weights,
)
from lib.seneca_model import SENECA_RISK_THRESHOLDS  # noqa: E402
from lib.survival_metrics import (  # noqa: E402
    compute_brier_scores,
    compute_c_index,
    compute_c_index_ipcw,
    compute_time_dependent_auc,
    create_temporal_grid,
    make_structured_array,
    predict_survival_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Risk-group cutoffs: percentiles of the training risk scores. The paper uses 15-85;
# "33-66" is the tertile split at the 33rd and 67th percentiles.
STRATIFICATION_SCHEMES = [
    {"name": "33-66", "low_percentile": 33, "high_percentile": 67},
    {"name": "15-85", "low_percentile": 15, "high_percentile": 85},
]
TIMEPOINTS = [6, 12, 18, 24]


# ----------------------------------------------------------------------------
# Input / output helpers
# ----------------------------------------------------------------------------

def _load_data(path: str) -> pd.DataFrame:
    """CSV or Parquet file, or the first such file in a folder."""
    if os.path.isdir(path):
        path = os.path.join(path, [f for f in os.listdir(path) if f.endswith((".csv", ".parquet"))][0])
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def _load_selected_features(model_name: str, model_version: str) -> List[str]:
    """The ten predictors of the ensemble (union of the base learners' features)."""
    path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{model_name}/{model_version}/artifacts/selected_features.json")
    with open(path) as f:
        selected = json.load(f)
    if isinstance(selected, dict):
        selected = sorted({feature for features in selected.values() for feature in features})
    if not isinstance(selected, list) or len(selected) != 10:
        raise ValueError("Expected selected_features.json to resolve to exactly 10 model features.")
    return selected


def _to_native(obj):
    """numpy types -> Python types, recursively (for JSON)."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _nan_to_none(obj):
    """NaN -> None (JSON null), recursively."""
    obj = _to_native(obj)
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(x) for x in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _write_json(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(_to_native(obj), f, indent=2)


# ----------------------------------------------------------------------------
# Cohort description and completeness (Tables S7-S9)
# ----------------------------------------------------------------------------

def _cohort_characteristics(cohorts: Dict[str, pd.DataFrame]) -> Dict:
    return {name: {"cohort": name, "n": len(df), "events": int(df["event"].sum()),
                   "event_rate": round(float(df["event"].mean()), 4),
                   "tte_median": round(float(df["tte"].median()), 2),
                   "tte_mean": round(float(df["tte"].mean()), 2),
                   "tte_min": round(float(df["tte"].min()), 2),
                   "tte_max": round(float(df["tte"].max()), 2)}
            for name, df in cohorts.items()}


def _temporal_missingness_tables(df: pd.DataFrame, model_features: List[str]):
    """Completeness of the ensemble's and SENECA's inputs in the temporal cohort,
    before imputation: patients by number of missing inputs, missingness by
    feature and by centre, and the mask of patients with all ten predictors."""
    ecog = "ecog bin" if "ecog bin" in df.columns else "ps ecog"
    feature_sets = {"ml_model_10": model_features,
                    "seneca_5": ["neutrophils", ecog, "locally advanced/metastatic", "ca199", "cea"]}
    missing_columns = sorted({f for fs in feature_sets.values() for f in fs} - set(df.columns))
    if missing_columns:
        raise ValueError(f"Prepared temporal data lacks required columns: {missing_columns}")

    completeness, by_feature, by_centre = [], [], []
    for feature_set, features in feature_sets.items():
        n_missing = df[features].isna().sum(axis=1)
        completeness.append({
            "cohort": "temporal_validation", "feature_set": feature_set, "row_type": "complete_case_summary",
            "n_patients": len(df), "n_complete": int(n_missing.eq(0).sum()),
            "pct_complete": 100 * float(n_missing.eq(0).mean()), "missing_feature_count": np.nan,
            "n_patients_at_missing_count": np.nan, "pct_patients_at_missing_count": np.nan,
        })
        for count, n_patients in n_missing.value_counts().sort_index().items():
            completeness.append({
                "cohort": "temporal_validation", "feature_set": feature_set,
                "row_type": "missing_feature_count_distribution", "n_patients": len(df),
                "n_complete": np.nan, "pct_complete": np.nan, "missing_feature_count": int(count),
                "n_patients_at_missing_count": int(n_patients),
                "pct_patients_at_missing_count": 100 * n_patients / len(df),
            })
        for feature in features:
            n = int(df[feature].isna().sum())
            by_feature.append({"cohort": "temporal_validation", "feature_set": feature_set, "feature": feature,
                               "n_patients": len(df), "n_missing": n, "pct_missing": 100 * n / len(df)})
    for centre, centre_df in df.groupby("center", dropna=False):
        for feature_set, features in feature_sets.items():
            for feature in features:
                n = int(centre_df[feature].isna().sum())
                by_centre.append({"cohort": "temporal_validation",
                                  "center": "missing" if pd.isna(centre) else str(centre),
                                  "n_patients": len(centre_df), "feature_set": feature_set,
                                  "feature": feature, "n_missing": n,
                                  "pct_missing": 100 * n / len(centre_df)})
    return (pd.DataFrame(completeness), pd.DataFrame(by_feature), pd.DataFrame(by_centre),
            df[model_features].notna().all(axis=1).to_numpy())


def _subgroup_c_indices(y_train, df, mask, horizon, score_col="ml_risk_score") -> Dict[str, float]:
    """Harrell's and Uno's C in a subgroup (NaN when not computable)."""
    y_sub = make_structured_array(df.loc[mask, "event"].values, df.loc[mask, "tte"].values)
    scores = df.loc[mask, score_col].values
    out = {"c_index": np.nan, "c_index_ipcw": np.nan}
    try:
        out["c_index"] = compute_c_index(y_sub, scores)
    except Exception:
        pass
    out["c_index_ipcw"] = compute_c_index_ipcw(y_train, y_sub, scores, tau=horizon)
    return out


# ----------------------------------------------------------------------------
# Point estimates and risk groups
# ----------------------------------------------------------------------------

def _compute_cohort_performance(model, X, y_train, y_test, risk_scores, label) -> Dict:
    """C-indices, time-dependent AUC and (if the model gives survival curves) Brier/IBS."""
    metrics = {"c_index": compute_c_index(y_test, risk_scores),
               "c_index_ipcw": compute_c_index_ipcw(y_train, y_test, risk_scores)}
    metrics.update(compute_time_dependent_auc(y_train, y_test, risk_scores))
    if model is not None:
        metrics.update(compute_brier_scores(model, X, y_train, y_test))
    logger.info(f"{label}: Harrell's C {metrics['c_index']:.4f}, Uno's C {metrics['c_index_ipcw']:.4f}")
    return metrics


def _evaluate_stratification(y, risk_scores, thresholds, scheme_name) -> Dict:
    """Groups, Kaplan-Meier summary, log-rank test, PPV/NPV and pairwise HRs."""
    groups = apply_risk_thresholds(risk_scores, thresholds["low_threshold"], thresholds["high_threshold"])
    return {
        "scheme_name": scheme_name,
        "thresholds": thresholds,
        "groups": groups,
        "km_statistics": compute_km_statistics(y, groups),
        "logrank": log_rank_test(y, groups),
        "ppv_npv": compute_ppv_npv(y, groups),
        "pairwise_hr": compute_pairwise_hazard_ratios(y, groups),
    }


def _write_stratification(result, df, y, tag, scheme_label, tables_dir, r_input_dir, km_figures_dir,
                          km_title_scheme=None, km_title_cohort=None) -> None:
    """Kaplan-Meier summary, HRs, PPV/NPV, patient-level groups and the KM figure."""
    result["km_statistics"].to_csv(tables_dir / f"km_statistics_{tag}.csv", index=False)
    if not result["pairwise_hr"].empty:
        result["pairwise_hr"].to_csv(tables_dir / f"hazard_ratios_{tag}.csv", index=False)
    pd.DataFrame([result["ppv_npv"]]).to_csv(tables_dir / f"ppv_npv_{tag}.csv", index=False)
    cohort = km_title_cohort or tag
    pd.DataFrame({"tte": df["tte"].values, "event": df["event"].values.astype(int),
                  "risk_group": result["groups"], "scheme": scheme_label, "cohort": cohort}).to_csv(
        r_input_dir / f"km_data_{tag}.csv", index=False)
    _km_figure(result["groups"], y, km_title_scheme or scheme_label, cohort, km_figures_dir)


def _km_figure(groups, y, scheme_name, cohort, output_dir: Path) -> None:
    """Kaplan-Meier curves by risk group with numbers at risk and censored (Figure 3)."""
    colors = {0: "#4DAF4A", 1: "#FF7F00", 2: "#E41A1C"}
    labels = {0: "Low", 1: "Intermediate", 2: "High"}
    fig, ax = plt.subplots(figsize=(8, 6))
    fitters = []
    for g in sorted(np.unique(groups)):
        mask = groups == g
        kmf = KaplanMeierFitter().fit(y["time"][mask], y["event"][mask],
                                      label=f"{labels.get(g, f'Group {g}')} (n={int(mask.sum())})")
        kmf.plot_survival_function(ax=ax, color=colors.get(g, "grey"), ci_alpha=0.15, linewidth=1.2)
        fitters.append(kmf)

    # Show at most 24 months, and stop where a group has fewer than 10 patients at risk
    max_time = 24.0
    for kmf in fitters:
        below_10 = kmf.survival_function_.index.values[kmf.event_table["at_risk"].values < 10]
        if len(below_10) > 0:
            max_time = min(max_time, float(below_10[0]))
    ax.set_xlim(0, max(max_time, 6.0))
    ax.xaxis.set_major_locator(MultipleLocator(6))
    ax.set_xticks([6, 12, 18, 24])
    ax.set_xlabel("Time (months)", fontweight="bold")
    ax.set_ylabel("Survival probability", fontweight="bold")
    ax.set_title(f"{scheme_name} — {cohort}", fontweight="bold", fontsize=13)
    ax.legend(loc="lower left", frameon=False)
    if fitters:
        add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk", "Censored"])
    plt.tight_layout()
    fig.savefig(output_dir / f"km_{scheme_name}_{cohort}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"km_{scheme_name}_{cohort}.pdf", bbox_inches="tight")
    plt.close(fig)


def _risk_group_agreement(df, model_groups, scheme_name, score_col="ml_risk_score", group_col="ml_risk_group"):
    """Agreement between the model's and SENECA's groups (published cutoffs) in the
    SENECA complete cases: Cohen's kappa (linear weights and unweighted) and the 3x3 table."""
    valid = ((df["seneca_complete_case"].to_numpy() == 1) & np.isfinite(df["seneca_risk_score"].to_numpy())
             & np.isfinite(df["seneca_group"].to_numpy()) & np.isfinite(df[score_col].to_numpy()))
    paired = df.loc[valid, ["tte", "event", score_col, "seneca_risk_score", "seneca_group"]].copy()
    paired.insert(0, "prediction_row", df.index[valid])
    paired.insert(4, group_col, model_groups[valid])
    paired["seneca_group"] = paired["seneca_group"].astype(int)

    labels = [0, 1, 2]
    crosstab = pd.crosstab(paired[group_col], paired["seneca_group"], dropna=False).reindex(
        index=labels, columns=labels, fill_value=0)
    crosstab.index.name, crosstab.columns.name = group_col, "seneca_group"
    if len(paired):
        kappa_linear = cohen_kappa_score(paired[group_col], paired["seneca_group"], labels=labels, weights="linear")
        kappa = cohen_kappa_score(paired[group_col], paired["seneca_group"], labels=labels)
    else:
        kappa_linear = kappa = np.nan
    result = {"cohort": "temporal_validation", "stratification_scheme": scheme_name,
              "n_paired_complete_cases": len(paired), "weighting_primary": "linear",
              "cohen_kappa_linear": kappa_linear, "cohen_kappa_unweighted": kappa,
              "group_labels": {"0": "low", "1": "intermediate", "2": "high"}}
    melted = crosstab.reset_index().melt(id_vars=group_col, var_name="seneca_group",
                                         value_name="n_patients").assign(stratification_scheme=scheme_name)
    paired.insert(1, "stratification_scheme", scheme_name)
    return result, paired, melted


# ----------------------------------------------------------------------------
# Inputs of the R figure script
# ----------------------------------------------------------------------------

def _roc_frame(tte, event, models, tp) -> pd.DataFrame:
    """IPCW ROC curves at tp months, one block of rows per model."""
    frames = []
    for label, scores in models:
        roc = compute_ipcw_roc_at_timepoint(tte, event, scores, float(tp))
        n = len(roc["fpr"])
        frames.append(pd.DataFrame({"model": [label] * n, "timepoint": [tp] * n, "fpr": roc["fpr"],
                                    "tpr": roc["tpr"], "threshold": roc["thresholds"], "auc": [roc["auc"]] * n}))
    return pd.concat(frames, ignore_index=True)


def _pr_frame(tte, event, models, tp) -> pd.DataFrame:
    """IPCW precision-recall curves at tp months; from 18 months the positive class is survival."""
    frames = []
    for label, scores in models:
        pr = compute_ipcw_pr_at_timepoint(tte, event, scores, float(tp), invert=tp >= 18)
        n = len(pr["recall"])
        frames.append(pd.DataFrame({"model": [label] * n, "timepoint": [tp] * n, "recall": pr["recall"],
                                    "precision": pr["precision"], "threshold": pr["thresholds"],
                                    "average_precision": [pr["average_precision"]] * n,
                                    "inverted": [tp >= 18] * n}))
    return pd.concat(frames, ignore_index=True)


def _calibration_r_data(df, cohort) -> pd.DataFrame:
    """Follow-up and predicted P(event <= t) of the ensemble and the Cox benchmark."""
    cols = {"tte": df["tte"].values, "event": df["event"].values.astype(int), "cohort": cohort}
    for prefix in ["", "cox_"]:
        for tp in TIMEPOINTS:
            if f"{prefix}P_event_{tp}m" in df.columns:
                cols[f"{prefix}p_event_{tp}m"] = df[f"{prefix}P_event_{tp}m"].values
    return pd.DataFrame(cols)


def _risk_density_r_data(df, cohort) -> pd.DataFrame:
    cols = {"ml_risk_score": df["ml_risk_score"].values, "event": df["event"].values.astype(int),
            "cohort": cohort}
    for col in ["seneca_risk_score", "cox_risk_score"]:
        if col in df.columns:
            cols[col] = df[col].values
    return pd.DataFrame(cols)


def _predicted_risk_distributions(df_internal, df_temporal) -> pd.DataFrame:
    """Predicted P(event <= t) at 6 and 18 months in both cohorts (Fig. S3C-D)."""
    rows = []
    for cohort, df in [("internal", df_internal), ("temporal", df_temporal)]:
        rec = {"cohort": cohort}
        for tp in [6, 18]:
            rec[f"p_event_{tp}m"] = df[f"P_event_{tp}m"].values if f"P_event_{tp}m" in df.columns \
                else np.full(len(df), np.nan)
        for tp in [6, 18]:
            if f"cox_P_event_{tp}m" in df.columns:
                rec[f"cox_p_event_{tp}m"] = df[f"cox_P_event_{tp}m"].values
        rows.append(pd.DataFrame(rec))
    return pd.concat(rows, ignore_index=True)


def _ipcw_stratified_distributions(tte, event, risk_scores, p_event_cols, cohort="temporal") -> pd.DataFrame:
    """Predicted P(event <= t) at 6 and 18 months by event status at t, with the IPCW
    weights of lib/roc_pr.py; patients censored before t are excluded (Fig. S3A-B)."""
    rows = []
    kmf = censoring_km(tte, event)
    for tp in [6, 18]:
        p_col = p_event_cols.get(tp)
        if p_col is None:
            continue
        _, labels, weights, _ = ipcw_weights(tte, event, risk_scores, float(tp), kmf)
        kept = np.array([i for i in range(len(tte)) if (event[i] == 1 and tte[i] <= tp) or tte[i] > tp])
        if len(kept) != len(labels):
            continue
        for j, i in enumerate(kept):
            rows.append({"cohort": cohort, "timepoint": tp, f"p_event_{tp}m": float(p_col[i]),
                         "event_status": "event" if labels[j] == 1 else "no_event",
                         "ipcw_weight": float(weights[j])})
    return pd.DataFrame(rows)


def _temporal_curves_for_r(point: Dict[str, Dict], boot: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
    """AUC and Brier score over time with bootstrap CIs, for the R line plots.

    point and boot map (cohort, model) to the point estimates and to the
    bootstrap aggregate of the same patients.
    """
    rows = {"auc": [], "brier": []}
    for metric in rows:
        for (cohort, model), perf in point.items():
            if metric == "brier" and model == "SENECA":
                continue
            suffix = {"ML": "ml", "SENECA": "seneca", "Cox": "cox"}[model]
            for tp in TIMEPOINTS:
                value = perf.get(f"{metric}_{tp}m")
                ci = boot[(cohort, model)].get(f"{metric}_{tp}m_{suffix}") or {}
                if value is not None:
                    rows[metric].append({"timepoint": tp, "model": model, "cohort": cohort, metric: value,
                                         "ci_lower": ci.get("ci_lower"), "ci_upper": ci.get("ci_upper")})
    return {"temporal_auc": pd.DataFrame(rows["auc"]), "temporal_brier": pd.DataFrame(rows["brier"])}


# ----------------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------------

def _run_bootstrap(df, y_train, n_iter, surv_matrix=None, grid=None, horizon=None, include_seneca=True,
                   ml_complete_mask=None, label="bootstrap", include_cox=False, surv_matrix_cox=None) -> Dict:
    """Bootstrap of one cohort.

    With include_seneca the cohort is restricted to the SENECA complete cases.
    With ml_complete_mask, Harrell's and Uno's C are also computed, on every
    resample, in the patients with all ten predictors observed and in those
    with at least one imputed. surv_matrix rows must match the rows of df.
    """
    if include_seneca:
        complete = (df["seneca_complete_case"] == 1).values
        df = df[complete]
        surv_matrix = surv_matrix[complete] if surv_matrix is not None else None
        surv_matrix_cox = surv_matrix_cox[complete] if surv_matrix_cox is not None else None
    df_boot = df.reset_index(drop=True).copy()
    if ml_complete_mask is not None:
        if len(ml_complete_mask) != len(df_boot):
            raise ValueError("Completeness mask must align with the bootstrap cohort.")
        df_boot["_ml_complete"] = np.asarray(ml_complete_mask, dtype=bool)
    df_boot["_pos_idx"] = np.arange(len(df_boot))
    logger.info(f"Bootstrap {label}: {len(df_boot)} patients, {n_iter} resamples")

    distributions = []
    for i in tqdm(range(n_iter), desc=label):
        boot = stratified_bootstrap_sample(df_boot, stratify_col="event", random_state=i)
        idx = boot["_pos_idx"].values
        metrics = compute_bootstrap_metrics(
            y_train=y_train,
            tte=boot["tte"].values,
            event=boot["event"].values,
            risk_ml=boot["ml_risk_score"].values,
            risk_seneca=boot["seneca_risk_score"].values if include_seneca else None,
            surv_preds_ml=surv_matrix[idx] if surv_matrix is not None else None,
            grid=grid,
            horizon=horizon,
            risk_cox=boot["cox_risk_score"].values if include_cox else None,
            surv_preds_cox=surv_matrix_cox[idx] if include_cox and surv_matrix_cox is not None else None,
        )
        if ml_complete_mask is not None:
            score_cols = [("", "ml_risk_score")] + ([("cox_", "cox_risk_score")] if include_cox else [])
            for subgroup, mask in [("ml_complete", boot["_ml_complete"].values),
                                   ("ml_incomplete_imputed", ~boot["_ml_complete"].values)]:
                y_sub = make_structured_array(boot.loc[mask, "event"].values, boot.loc[mask, "tte"].values)
                for prefix, col in score_cols:
                    c_key, uno_key = f"c_index_{prefix}{subgroup}", f"c_index_ipcw_{prefix}{subgroup}"
                    metrics[c_key] = metrics[uno_key] = np.nan
                    if mask.any():
                        scores = boot.loc[mask, col].values
                        try:
                            metrics[c_key] = compute_c_index(y_sub, scores)
                        except Exception:
                            pass
                        metrics[uno_key] = compute_c_index_ipcw(y_train, y_sub, scores, tau=horizon)
                if include_cox:
                    for metric in ["c_index", "c_index_ipcw"]:
                        metrics[f"{metric}_diff_ml_cox_{subgroup}"] = (
                            metrics[f"{metric}_{subgroup}"] - metrics[f"{metric}_cox_{subgroup}"])
        distributions.append(metrics)

    aggregated = aggregate_bootstrap_results(distributions)
    if include_cox:
        for key in [k for k in aggregated if "cox" in k]:
            aggregated[key] = _nan_to_none(aggregated[key])
    return {"n_samples": len(df_boot), "n_iterations": n_iter, "aggregated": aggregated}


# ----------------------------------------------------------------------------
# Summary tables
# ----------------------------------------------------------------------------

def _cox_columns(ml_val, perf_cox: Dict, boot_agg: Dict, key: str) -> Dict:
    """Cox benchmark and paired ensemble - Cox difference for one row (same resamples as the ML CI)."""
    cox_val = perf_cox.get(key)
    cox_ci = boot_agg.get(f"{key}_cox") or {}
    diff_ci = boot_agg.get(f"{key}_diff_ml_cox") or {}
    return {"cox": cox_val, "cox_ci_lower": cox_ci.get("ci_lower"), "cox_ci_upper": cox_ci.get("ci_upper"),
            "delta_ml_cox": ml_val - cox_val if ml_val is not None and cox_val is not None else None,
            "delta_ml_cox_ci_lower": diff_ci.get("ci_lower"), "delta_ml_cox_ci_upper": diff_ci.get("ci_upper")}


def _build_overall_summary_table(perf_ml, perf_seneca, boot_agg, cohort, perf_cox=None) -> pd.DataFrame:
    """Harrell's C, Uno's C, mean AUC and IBS of the ensemble [, SENECA and their
    difference] [, Cox benchmark and ensemble - Cox], with bootstrap CIs."""
    rows = []
    for label, key, ml_key, sen_key, diff_key in [
        ("harrell_c_index", "c_index", "c_index_ml", "c_index_seneca", "c_index_diff"),
        ("uno_c_index", "c_index_ipcw", "c_index_ipcw_ml", "c_index_ipcw_seneca", "c_index_ipcw_diff"),
        ("mean_auc", "mean_auc", "mean_auc_ml", "mean_auc_seneca", "mean_auc_diff"),
        ("ibs", "ibs_overall", "ibs_overall_ml", None, None),
    ]:
        ml_val, sen_val = perf_ml.get(key), perf_seneca.get(key)
        ml_ci = boot_agg.get(ml_key, {})
        sen_ci = boot_agg.get(sen_key, {}) if sen_key else {}
        diff_ci = boot_agg.get(diff_key, {}) if diff_key else {}
        row = {"cohort": cohort, "metric": label,
               "ml": ml_val, "ml_ci_lower": ml_ci.get("ci_lower"), "ml_ci_upper": ml_ci.get("ci_upper"),
               "seneca": sen_val, "seneca_ci_lower": sen_ci.get("ci_lower"), "seneca_ci_upper": sen_ci.get("ci_upper"),
               "delta": ml_val - sen_val if ml_val is not None and sen_val is not None else None,
               "delta_ci_lower": diff_ci.get("ci_lower"), "delta_ci_upper": diff_ci.get("ci_upper")}
        if perf_cox is not None:
            row.update(_cox_columns(ml_val, perf_cox, boot_agg, key))
        rows.append(row)
    return pd.DataFrame(rows)


def _build_timedep_summary_table(perf_ml, perf_seneca, boot_agg, cohort, perf_cox=None) -> pd.DataFrame:
    """AUC and Brier score at 6, 12, 18 and 24 months, with bootstrap CIs."""
    rows = []
    for tp in TIMEPOINTS:
        for metric in ["auc", "brier"]:
            key = f"{metric}_{tp}m"
            ml_ci = boot_agg.get(f"{key}_ml", {})
            sen_ci = boot_agg.get(f"{key}_seneca", {}) if metric == "auc" else {}
            row = {"cohort": cohort, "timepoint": tp, "metric": metric, "ml": perf_ml.get(key),
                   "ml_ci_lower": ml_ci.get("ci_lower"), "ml_ci_upper": ml_ci.get("ci_upper"),
                   "seneca": perf_seneca.get(key) if metric == "auc" else None,
                   "seneca_ci_lower": sen_ci.get("ci_lower"), "seneca_ci_upper": sen_ci.get("ci_upper")}
            if perf_cox is not None:
                row.update(_cox_columns(perf_ml.get(key), perf_cox, boot_agg, key))
            rows.append(row)
    return pd.DataFrame(rows)


def _paired_difference_rows(cohort, n_patients, perf_ml, perf_cox, boot_result) -> List[Dict]:
    """Ensemble - Cox differences with paired-bootstrap 95% CIs."""
    agg = boot_result.get("aggregated", {})
    rows = []
    for metric in PAIRED_DIFF_BASE_METRICS:
        ml_val, cox_val = perf_ml.get(metric), perf_cox.get(metric)
        diff = agg.get(f"{metric}_diff_ml_cox") or {}
        rows.append({
            "cohort": cohort, "n_patients": n_patients, "n_bootstrap": boot_result.get("n_iterations"),
            "metric": metric, "higher_is_better": not metric.startswith(("brier", "ibs")),
            "ml": ml_val, "cox": cox_val,
            "difference_ml_minus_cox": ml_val - cox_val if ml_val is not None and cox_val is not None else None,
            "diff_ci_lower": diff.get("ci_lower"), "diff_ci_upper": diff.get("ci_upper"),
            "diff_bootstrap_median": diff.get("median"), "n_valid_bootstrap": diff.get("n_valid"),
        })
    return rows


def _performance_summary(entries) -> pd.DataFrame:
    """Long table of every point estimate with its bootstrap CI."""
    rows = []
    for cohort, model, n, perf, boot_agg, suffix in entries:
        for key, val in perf.items():
            if isinstance(val, (int, float)) and not np.isnan(val):
                ci = boot_agg.get(f"{key}_{suffix}") or {}
                rows.append({"cohort": cohort, "model": model, "n": n, "metric": key, "value": round(val, 4),
                             "ci_lower": ci.get("ci_lower"), "ci_upper": ci.get("ci_upper")})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions_dir", required=True, help="Output of generate_predictions.py")
    parser.add_argument("--training_data", required=True, help="Training set (data_prep.py)")
    parser.add_argument("--test_data", required=True, help="Internal test set (data_prep.py)")
    parser.add_argument("--external_data", required=True, help="Temporal cohort (prepare_external.py)")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--cox_model_dir", required=True, help="Output of fit_cox_benchmark.py")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--output_dir", required=True, help="Metrics (JSON)")
    parser.add_argument("--tables_dir", required=True, help="Tables (CSV)")
    parser.add_argument("--r_input_dir", required=True, help="Inputs of publication_figures.R")
    parser.add_argument("--python_figures_dir", required=True, help="Kaplan-Meier figures (km/)")
    args = parser.parse_args()

    metrics_dir, tables_dir = Path(args.output_dir), Path(args.tables_dir)
    r_input_dir, km_figures_dir = Path(args.r_input_dir), Path(args.python_figures_dir) / "km"
    for d in [metrics_dir, tables_dir, r_input_dir, km_figures_dir]:
        d.mkdir(parents=True, exist_ok=True)
    mlflow.start_run()
    mlflow.log_param("n_bootstrap", args.n_bootstrap)

    # ---- Data, cached predictions and models -------------------------------
    pred_dir = Path(args.predictions_dir)
    df_internal = pd.read_csv(pred_dir / "internal_predictions.csv")
    df_temporal = pd.read_csv(pred_dir / "temporal_predictions.csv")
    with open(pred_dir / "training_risk_scores.json") as f:
        train_risk_scores = np.array(json.load(f)["scores"])
    with open(pred_dir / "training_risk_scores_cox.json") as f:
        cox_train_risk_scores = np.array(json.load(f)["scores"])

    df_train = _load_data(args.training_data)
    df_test_raw, df_ext_raw = _load_data(args.test_data), _load_data(args.external_data)
    y_train = make_structured_array(df_train["event"].values, df_train["tte"].values)
    y_internal = make_structured_array(df_internal["event"].values, df_internal["tte"].values)
    y_temporal = make_structured_array(df_temporal["event"].values, df_temporal["tte"].values)

    model = mlflow.pyfunc.load_model(f"models:/{args.model_name}/{args.model_version}")
    model_features = _load_selected_features(args.model_name, args.model_version)
    cox_model = load_cox_benchmark(args.cox_model_dir)
    if cox_model.features != sorted(model_features):
        raise ValueError(f"Cox benchmark predictors {cox_model.features} differ from the ensemble's")
    shutil.copy(Path(args.cox_model_dir) / COEFFICIENTS_FILE, tables_dir / COEFFICIENTS_FILE)
    shutil.copy(Path(args.cox_model_dir) / FIT_REPORT_FILE, metrics_dir / FIT_REPORT_FILE)

    drop_cols = ["tte", "event", "id", "center"]
    X_internal = df_test_raw.drop(columns=[c for c in drop_cols if c in df_test_raw.columns])
    X_temporal = df_ext_raw.drop(columns=[c for c in drop_cols if c in df_ext_raw.columns])
    for name, X, df in [("internal", X_internal, df_internal), ("temporal", X_temporal, df_temporal)]:
        if not np.allclose(cox_model.predict(X), df["cox_risk_score"].values, rtol=0, atol=1e-5):
            raise ValueError(f"Cached predictions are not row-aligned with the {name} data.")

    # Survival curves on the evaluation grids, computed once and resampled with the patients
    grid_int, _, horizon_int = create_temporal_grid(y_train, y_internal)
    grid_temp, _, horizon_temp = create_temporal_grid(y_train, y_temporal)
    surv_internal = predict_survival_matrix(model, X_internal, grid_int)
    surv_temporal = predict_survival_matrix(model, X_temporal, grid_temp)
    surv_internal_cox = predict_survival_matrix(cox_model, X_internal, grid_int)
    surv_temporal_cox = predict_survival_matrix(cox_model, X_temporal, grid_temp)
    logger.info(f"Evaluation horizon: internal {horizon_int:.2f}, temporal {horizon_temp:.2f} months")

    _write_json(_cohort_characteristics({"training": df_train, "internal_test": df_internal,
                                         "temporal_validation": df_temporal}),
                metrics_dir / "cohort_characteristics.json")

    # ---- Completeness before imputation (Tables S7 and S9) -----------------
    completeness, feature_missingness, centre_missingness, ml_complete_mask = \
        _temporal_missingness_tables(df_ext_raw, model_features)
    if len(ml_complete_mask) != len(df_temporal):
        raise ValueError("Prepared temporal data and cached predictions are not row-aligned.")
    completeness.to_csv(tables_dir / "temporal_patient_completeness.csv", index=False, float_format="%.2f")
    feature_missingness.to_csv(tables_dir / "temporal_feature_missingness.csv", index=False, float_format="%.2f")
    centre_missingness.to_csv(tables_dir / "temporal_centre_feature_missingness.csv", index=False,
                              float_format="%.2f")

    # ---- Point estimates ----------------------------------------------------
    perf_internal = _compute_cohort_performance(model, X_internal, y_train, y_internal,
                                                df_internal["ml_risk_score"].values, "internal, ensemble")
    perf_internal_cox = _compute_cohort_performance(cox_model, X_internal, y_train, y_internal,
                                                    df_internal["cox_risk_score"].values, "internal, Cox")
    _write_json({"ml": perf_internal, "cox": _nan_to_none(perf_internal_cox)},
                metrics_dir / "performance_internal.json")

    cc_mask = df_temporal["seneca_complete_case"] == 1
    y_temporal_cc = make_structured_array(df_temporal.loc[cc_mask, "event"].values,
                                          df_temporal.loc[cc_mask, "tte"].values)
    X_temporal_cc = X_temporal.loc[cc_mask]
    logger.info(f"SENECA complete cases: {int(cc_mask.sum())}/{len(df_temporal)}")

    perf_temporal_ml = _compute_cohort_performance(model, X_temporal, y_train, y_temporal,
                                                   df_temporal["ml_risk_score"].values, "temporal, ensemble")
    perf_temporal_ml_cc = _compute_cohort_performance(model, X_temporal_cc, y_train, y_temporal_cc,
                                                      df_temporal.loc[cc_mask, "ml_risk_score"].values,
                                                      "temporal complete cases, ensemble")
    perf_temporal_ml_cc["n_complete_cases"] = int(cc_mask.sum())
    seneca_scores_cc = df_temporal.loc[cc_mask, "seneca_risk_score"].values
    perf_temporal_seneca = {"c_index": compute_c_index(y_temporal_cc, seneca_scores_cc),
                            "c_index_ipcw": compute_c_index_ipcw(y_train, y_temporal_cc, seneca_scores_cc)}
    perf_temporal_seneca.update(compute_time_dependent_auc(y_train, y_temporal_cc, seneca_scores_cc))
    perf_temporal_seneca["n_complete_cases"] = int(cc_mask.sum())
    perf_temporal_cox = _compute_cohort_performance(cox_model, X_temporal, y_train, y_temporal,
                                                    df_temporal["cox_risk_score"].values, "temporal, Cox")
    perf_temporal_cox_cc = _compute_cohort_performance(cox_model, X_temporal_cc, y_train, y_temporal_cc,
                                                       df_temporal.loc[cc_mask, "cox_risk_score"].values,
                                                       "temporal complete cases, Cox")
    perf_temporal_cox_cc["n_complete_cases"] = int(cc_mask.sum())
    _write_json({"ml_full": perf_temporal_ml, "ml_cc": perf_temporal_ml_cc, "seneca": perf_temporal_seneca,
                 "cox_full": _nan_to_none(perf_temporal_cox), "cox_cc": _nan_to_none(perf_temporal_cox_cc)},
                metrics_dir / "performance_temporal.json")

    # ---- Bootstrap ----------------------------------------------------------
    boot_internal = _run_bootstrap(df_internal, y_train, args.n_bootstrap, surv_matrix=surv_internal,
                                   grid=grid_int, horizon=horizon_int, include_seneca=False,
                                   label="internal", include_cox=True, surv_matrix_cox=surv_internal_cox)
    _write_json(boot_internal, metrics_dir / "bootstrap_internal.json")

    boot_temporal_full = _run_bootstrap(df_temporal, y_train, args.n_bootstrap, surv_matrix=surv_temporal,
                                        grid=grid_temp, horizon=horizon_temp, include_seneca=False,
                                        ml_complete_mask=ml_complete_mask, label="temporal",
                                        include_cox=True, surv_matrix_cox=surv_temporal_cox)
    _write_json(boot_temporal_full, metrics_dir / "bootstrap_temporal_full.json")

    boot_temporal_cc = _run_bootstrap(
        df_temporal[cc_mask].reset_index(drop=True), y_train, args.n_bootstrap,
        surv_matrix=surv_temporal[cc_mask.values] if surv_temporal is not None else None,
        grid=grid_temp, horizon=horizon_temp, include_seneca=True, label="temporal complete cases",
        include_cox=True,
        surv_matrix_cox=surv_temporal_cox[cc_mask.values] if surv_temporal_cox is not None else None)
    _write_json(boot_temporal_cc, metrics_dir / "bootstrap_temporal_cc.json")

    agg_internal = boot_internal["aggregated"]
    agg_full = boot_temporal_full["aggregated"]
    agg_cc = boot_temporal_cc["aggregated"]

    # ---- Discrimination by completeness (Table S8) -------------------------
    subgroup_rows = []
    subgroups = [("full", np.ones(len(df_temporal), dtype=bool), ""),
                 ("ml_complete", ml_complete_mask, ""),
                 ("ml_incomplete_imputed", ~ml_complete_mask, "Descriptive only; potentially underpowered.")]
    for subgroup, mask, note in subgroups:
        n, events = int(mask.sum()), int(df_temporal.loc[mask, "event"].sum())
        for metric, value in _subgroup_c_indices(y_train, df_temporal, mask, horizon_temp).items():
            ci = agg_full.get(f"{metric}_ml" if subgroup == "full" else f"{metric}_{subgroup}", {})
            subgroup_rows.append({"cohort": "temporal_validation", "subgroup": subgroup, "note": note,
                                  "n_patients": n, "n_events": events, "metric": metric,
                                  "point_estimate": value, "ci_lower": ci.get("ci_lower"),
                                  "ci_upper": ci.get("ci_upper"), "n_valid_bootstrap": ci.get("n_valid")})
    for subgroup, mask, note in subgroups:  # Cox benchmark and paired difference, same strata
        n, events = int(mask.sum()), int(df_temporal.loc[mask, "event"].sum())
        point_ml = _subgroup_c_indices(y_train, df_temporal, mask, horizon_temp)
        point_cox = _subgroup_c_indices(y_train, df_temporal, mask, horizon_temp, score_col="cox_risk_score")
        suffix = "" if subgroup == "full" else f"_{subgroup}"
        for metric in ["c_index", "c_index_ipcw"]:
            for row_metric, value, row_note in [
                (f"{metric}_cox", point_cox[metric], "Cox PH benchmark."),
                (f"{metric}_diff_ml_cox", point_ml[metric] - point_cox[metric],
                 "Paired bootstrap difference: ML ensemble minus Cox PH."),
            ]:
                ci = agg_full.get(f"{row_metric}{suffix}") or {}
                subgroup_rows.append({"cohort": "temporal_validation", "subgroup": subgroup,
                                      "note": f"{row_note} {note}".strip(), "n_patients": n, "n_events": events,
                                      "metric": row_metric, "point_estimate": value,
                                      "ci_lower": ci.get("ci_lower"), "ci_upper": ci.get("ci_upper"),
                                      "n_valid_bootstrap": ci.get("n_valid")})
    pd.DataFrame(subgroup_rows).to_csv(tables_dir / "temporal_ml_completeness_c_indices.csv", index=False,
                                       float_format="%.4f")
    _write_json(subgroup_rows, metrics_dir / "temporal_ml_completeness_c_indices.json")

    # ---- Paired differences ensemble - Cox ---------------------------------
    paired_rows = (
        _paired_difference_rows("internal", len(df_internal), perf_internal, perf_internal_cox, boot_internal)
        + _paired_difference_rows("temporal_full", len(df_temporal), perf_temporal_ml, perf_temporal_cox,
                                  boot_temporal_full)
        + _paired_difference_rows("temporal_cc", int(cc_mask.sum()), perf_temporal_ml_cc, perf_temporal_cox_cc,
                                  boot_temporal_cc))
    pd.DataFrame(paired_rows).to_csv(tables_dir / "paired_differences_ml_minus_cox.csv", index=False,
                                     float_format="%.4f")
    with open(metrics_dir / "paired_differences_ml_minus_cox.json", "w") as f:
        json.dump(_nan_to_none(paired_rows), f, indent=2)

    # ---- Risk groups (Figure 3, Table S4) ----------------------------------
    df_temporal_cc = df_temporal[cc_mask].copy()
    agreement, assignments, crosstabs = [], [], []
    agreement_cox, assignments_cox, crosstabs_cox = [], [], []
    for scheme in STRATIFICATION_SCHEMES:
        name = scheme["name"]
        percentiles = dict(low_percentile=scheme["low_percentile"], high_percentile=scheme["high_percentile"])

        # Ensemble: cutoffs from the training risk scores, applied unchanged
        thresholds = define_risk_groups(train_risk_scores, **percentiles)
        with open(tables_dir / f"risk_thresholds_{name}.json", "w") as f:
            json.dump({"scheme": name, **percentiles, **thresholds}, f, indent=2)
        for cohort, df, y in [("internal", df_internal, y_internal), ("temporal", df_temporal, y_temporal),
                              ("temporal_cc", df_temporal_cc, y_temporal_cc)]:
            result = _evaluate_stratification(y, df["ml_risk_score"].values, thresholds, name)
            _write_stratification(result, df, y, f"{cohort}_{name}", name, tables_dir, r_input_dir,
                                  km_figures_dir, km_title_cohort=cohort)
            if cohort == "temporal":
                summary, pairs, table = _risk_group_agreement(df_temporal, result["groups"], name)
                agreement.append(summary)
                assignments.append(pairs)
                crosstabs.append(table)
                logger.info(f"{name}: kappa (linear) ensemble vs SENECA {summary['cohen_kappa_linear']:.4f}")

        # SENECA groups (published cutoffs), complete cases
        seneca_groups = apply_risk_thresholds(df_temporal_cc["seneca_risk_score"].values,
                                              SENECA_RISK_THRESHOLDS["low"], SENECA_RISK_THRESHOLDS["high"])
        compute_km_statistics(y_temporal_cc, seneca_groups).to_csv(
            tables_dir / f"km_statistics_temporal_seneca_{name}.csv", index=False)
        pd.DataFrame({"tte": df_temporal_cc["tte"].values, "event": df_temporal_cc["event"].values.astype(int),
                      "risk_group": seneca_groups, "scheme": f"{name}_seneca", "cohort": "temporal"}).to_csv(
            r_input_dir / f"km_data_temporal_seneca_{name}.csv", index=False)
        _km_figure(seneca_groups, y_temporal_cc, f"{name}_seneca", "temporal", km_figures_dir)

        # Cox benchmark: cutoffs from its own training risk scores, same rule
        thresholds_cox = define_risk_groups(cox_train_risk_scores, **percentiles)
        with open(tables_dir / f"risk_thresholds_cox_{name}.json", "w") as f:
            json.dump({"scheme": name, **percentiles, **thresholds_cox}, f, indent=2)
        for cohort, df, y in [("internal", df_internal, y_internal), ("temporal", df_temporal, y_temporal),
                              ("temporal_cc", df_temporal_cc, y_temporal_cc)]:
            result = _evaluate_stratification(y, df["cox_risk_score"].values, thresholds_cox, name)
            _write_stratification(result, df, y, f"{cohort}_cox_{name}", f"{name}_cox", tables_dir, r_input_dir,
                                  km_figures_dir, km_title_cohort=cohort)
            if cohort == "temporal":
                summary, pairs, table = _risk_group_agreement(df_temporal, result["groups"], name,
                                                              score_col="cox_risk_score",
                                                              group_col="cox_risk_group")
                agreement_cox.append(summary)
                assignments_cox.append(pairs)
                crosstabs_cox.append(table)

    for suffix, rows, assign, cross in [("", agreement, assignments, crosstabs),
                                        ("_cox", agreement_cox, assignments_cox, crosstabs_cox)]:
        with open(metrics_dir / f"risk_group_agreement_temporal{suffix}.json", "w") as f:
            json.dump(_nan_to_none(rows) if suffix else _to_native(rows), f, indent=2)
        pd.DataFrame(rows).to_csv(tables_dir / f"risk_group_agreement_temporal{suffix}.csv", index=False,
                                  float_format="%.4f")
        pd.concat(assign, ignore_index=True).to_csv(tables_dir / f"risk_group_assignments_temporal{suffix}.csv",
                                                    index=False, float_format="%.6f")
        pd.concat(cross, ignore_index=True).to_csv(tables_dir / f"risk_group_crosstab_temporal{suffix}.csv",
                                                   index=False)

    # ---- Inputs of publication_figures.R -----------------------------------
    tte_t, event_t = df_temporal["tte"].values, df_temporal["event"].values
    tte_cc, event_cc = df_temporal_cc["tte"].values, df_temporal_cc["event"].values
    tte_i, event_i = df_internal["tte"].values, df_internal["event"].values
    full_models = [("ML", df_temporal["ml_risk_score"].values), ("Cox", df_temporal["cox_risk_score"].values)]
    cc_models = [("ML", df_temporal_cc["ml_risk_score"].values),
                 ("SENECA", df_temporal_cc["seneca_risk_score"].values),
                 ("Cox", df_temporal_cc["cox_risk_score"].values)]
    internal_models = [("ML", df_internal["ml_risk_score"].values), ("Cox", df_internal["cox_risk_score"].values)]
    for tp in TIMEPOINTS:
        _roc_frame(tte_t, event_t, full_models, tp).to_csv(
            r_input_dir / f"roc_curves_temporal_full_{tp}m.csv", index=False, float_format="%.6f")
        _pr_frame(tte_t, event_t, full_models, tp).to_csv(
            r_input_dir / f"pr_curves_temporal_full_{tp}m.csv", index=False, float_format="%.6f")
        _roc_frame(tte_cc, event_cc, cc_models, tp).to_csv(
            r_input_dir / f"roc_curves_{tp}m_cc.csv", index=False, float_format="%.6f")
        _pr_frame(tte_cc, event_cc, cc_models, tp).to_csv(
            r_input_dir / f"pr_curves_{tp}m_cc.csv", index=False, float_format="%.6f")
        _roc_frame(tte_i, event_i, internal_models, tp).to_csv(
            r_input_dir / f"roc_curves_internal_{tp}m.csv", index=False, float_format="%.6f")

    _calibration_r_data(df_internal, "internal").to_csv(r_input_dir / "calibration_input_internal.csv",
                                                        index=False, float_format="%.6f")
    _calibration_r_data(df_temporal, "temporal").to_csv(r_input_dir / "calibration_input_temporal.csv",
                                                        index=False, float_format="%.6f")
    _risk_density_r_data(df_internal, "internal").to_csv(r_input_dir / "risk_density_internal.csv",
                                                         index=False, float_format="%.6f")
    _risk_density_r_data(df_temporal, "temporal").to_csv(r_input_dir / "risk_density_temporal.csv",
                                                         index=False, float_format="%.6f")
    _predicted_risk_distributions(df_internal, df_temporal).to_csv(
        r_input_dir / "predicted_risk_distributions.csv", index=False, float_format="%.6f")
    for prefix, score_col, out_name in [("", "ml_risk_score", "ipcw_stratified_distributions.csv"),
                                        ("cox_", "cox_risk_score", "ipcw_stratified_distributions_cox.csv")]:
        p_cols = {tp: df_temporal[f"{prefix}P_event_{tp}m"].values
                  for tp in [6, 18] if f"{prefix}P_event_{tp}m" in df_temporal.columns}
        if p_cols:
            _ipcw_stratified_distributions(tte_t, event_t, df_temporal[score_col].values, p_cols).rename(
                columns={f"p_event_{tp}m": f"{prefix}p_event_{tp}m" for tp in [6, 18]}).to_csv(
                r_input_dir / out_name, index=False, float_format="%.6f")

    point = {("internal", "ML"): perf_internal, ("temporal", "ML"): perf_temporal_ml,
             ("temporal", "SENECA"): perf_temporal_seneca, ("internal", "Cox"): perf_internal_cox,
             ("temporal", "Cox"): perf_temporal_cox}
    boot = {("internal", "ML"): agg_internal, ("temporal", "ML"): agg_full, ("temporal", "SENECA"): agg_cc,
            ("internal", "Cox"): agg_internal, ("temporal", "Cox"): agg_full}
    for name, table in _temporal_curves_for_r(point, boot).items():
        table.to_csv(r_input_dir / f"{name}.csv", index=False, float_format="%.6f")

    # ---- Summary tables (Tables 2-4) ---------------------------------------
    comparison = {}
    for suffix, internal, temporal in [("", perf_internal, perf_temporal_ml),
                                       ("_cox", perf_internal_cox, perf_temporal_cox)]:
        for key in ["c_index", "c_index_ipcw", "mean_auc"] + [f"auc_{tp}m" for tp in TIMEPOINTS]:
            if internal.get(key) is not None and temporal.get(key) is not None:
                entry = {"internal": internal[key], "temporal_full": temporal[key],
                         "difference": round(temporal[key] - internal[key], 4)}
                comparison[f"{key}{suffix}"] = _nan_to_none(entry) if suffix else entry
    _write_json(comparison, metrics_dir / "comparison_internal_vs_temporal.json")

    for name, builder, perf_ml, perf_sen, agg, perf_cox in [
        ("overall_internal", _build_overall_summary_table, perf_internal, {}, agg_internal, perf_internal_cox),
        ("overall_temporal_full", _build_overall_summary_table, perf_temporal_ml, {}, agg_full, perf_temporal_cox),
        ("overall_temporal_cc", _build_overall_summary_table, perf_temporal_ml_cc, perf_temporal_seneca, agg_cc,
         perf_temporal_cox_cc),
        ("timedep_internal", _build_timedep_summary_table, perf_internal, {}, agg_internal, perf_internal_cox),
        ("timedep_temporal_full", _build_timedep_summary_table, perf_temporal_ml, {}, agg_full, perf_temporal_cox),
        ("timedep_temporal_cc", _build_timedep_summary_table, perf_temporal_ml_cc, perf_temporal_seneca, agg_cc,
         perf_temporal_cox_cc),
    ]:
        cohort = name.split("_", 1)[1]
        builder(perf_ml, perf_sen, agg, cohort, perf_cox=perf_cox).to_csv(
            tables_dir / f"{name}.csv", index=False, float_format="%.4f")

    n_cc = int(cc_mask.sum())
    _performance_summary([
        ("internal", "ML", len(df_internal), perf_internal, agg_internal, "ml"),
        ("temporal_full", "ML", len(df_temporal), perf_temporal_ml, agg_full, "ml"),
        ("temporal_cc", "ML", n_cc, perf_temporal_ml_cc, agg_cc, "ml"),
        ("temporal_cc", "SENECA", n_cc, perf_temporal_seneca, agg_cc, "seneca"),
        ("internal", "Cox", len(df_internal), perf_internal_cox, agg_internal, "cox"),
        ("temporal_full", "Cox", len(df_temporal), perf_temporal_cox, agg_full, "cox"),
        ("temporal_cc", "Cox", n_cc, perf_temporal_cox_cc, agg_cc, "cox"),
    ]).to_csv(tables_dir / "performance_summary.csv", index=False, float_format="%.4f")

    for directory, name in [(args.output_dir, "metrics"), (args.tables_dir, "tables"),
                            (args.r_input_dir, "r_input"), (args.python_figures_dir, "python_figures")]:
        mlflow.log_artifacts(directory, artifact_path=name)
    mlflow.end_run()


if __name__ == "__main__":
    main()
