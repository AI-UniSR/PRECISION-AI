"""Evaluation helpers of the development pipeline (hold-out comparison, Supplementary Table S2).

The definitions differ from those of the validation pipeline
(5_validation/lib/survival_metrics.py), which produced Tables 2-4:

- time points: 6, 12, 15, 18 and 24 months, if below the shorter maximum
  follow-up of the training and test sets (minus 0.1 month);
- mean AUC: the summary returned by scikit-survival over these time points;
- IBS: trapezoidal integral of the Brier score over a grid of 40 percentiles
  (5th-95th) of the training times plus the time points, divided by 24 months;
- follow-up is not truncated.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from sksurv.metrics import brier_score

CLINICAL_TIMEPOINTS_MONTHS = [6, 12, 15, 18, 24]


def create_temporal_grid(y_train, y_test=None):
    """Evaluation grid and the clinical time points within follow-up (months)."""
    max_time = y_train["time"].max() if y_test is None else min(y_train["time"].max(), y_test["time"].max())
    min_time = y_train["time"].min()

    clinical = np.array(CLINICAL_TIMEPOINTS_MONTHS, dtype=float)
    clinical = clinical[clinical < (max_time - 0.1)]
    valid_times = y_train["time"][(y_train["time"] >= min_time) & (y_train["time"] < max_time)]
    if len(clinical) == 0:  # not reached with the study data
        clinical = (np.percentile(valid_times, [25, 50, 75]) if len(valid_times) > 10
                    else np.array([max_time * 0.5]))

    percentiles = (np.percentile(valid_times, np.linspace(5, 95, 40)) if len(valid_times) > 10
                   else np.percentile(valid_times, [10, 25, 50, 75, 90]))
    grid = np.unique(np.concatenate([percentiles, clinical]))
    grid = np.sort(grid[(grid >= min_time) & (grid < max_time)])
    print(f"Evaluation grid: {len(grid)} points in [{grid.min():.2f}, {grid.max():.2f}] months; "
          f"time points {clinical.tolist()}")
    return grid, clinical


def compute_brier_scores_on_grid(model, X_test, y_train, y_test, grid_months):
    """IPCW Brier score (censoring from y_train) at every grid point; None if unavailable."""
    if not hasattr(model, "predict_survival_function") or len(grid_months) == 0:
        return None
    try:
        surv_funcs = model.predict_survival_function(X_test)
        preds = np.asarray([[fn(t) for t in grid_months] for fn in surv_funcs])
        return np.array([brier_score(y_train, y_test, preds[:, i], t)[1][0]
                         for i, t in enumerate(grid_months)])
    except Exception as e:
        print(f"  Brier scores not computed: {e}")
        return None


def compute_ibs_intervals(grid_months, brier_scores):
    """IBS over 0-6, 6-12, 12-18, 18-24 months and overall (0-24 months, divided by 24)."""
    ibs = {}
    for start, end in [(0, 6), (6, 12), (12, 18), (18, 24)]:
        mask = (grid_months >= start) & (grid_months <= end)
        ibs[f"ibs_{start}_{end}m"] = (trapezoid(brier_scores[mask], grid_months[mask]) / (end - start)
                                      if mask.sum() >= 3 else np.nan)
    mask = (grid_months >= 0) & (grid_months <= 24)
    overall = trapezoid(brier_scores[mask], grid_months[mask]) / 24.0 if mask.sum() >= 3 else np.nan
    return ibs, overall


def extract_feature_names_from_model(model):
    """Features a fitted learner was trained on (used when no selected_features.txt exists)."""
    if hasattr(model, "xgbse_model_"):
        booster = getattr(getattr(model.xgbse_model_, "feature_extractor", None), "bst", None)
        if booster is not None and booster.feature_names:
            return list(booster.feature_names)
    if hasattr(model, "named_steps"):
        first_step = list(model.named_steps.values())[0]
        if hasattr(first_step, "feature_names_in_"):
            return list(first_step.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None


def load_model_and_metrics(model_dir, model_name):
    """Fitted learner, its cross-validated metrics and its feature list."""
    if model_name == "deepsurv" and os.path.exists(os.path.join(model_dir, "deepsurv_net.pt")):
        from deepsurv_wrapper import DeepSurvWrapper
        model = DeepSurvWrapper(model_dir)
        model._load_artifacts()
        cv_metrics = {}
        metrics_path = os.path.join(model_dir, "training_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                training = json.load(f)
            cv_metrics = {"best_cv_mean_ci": training.get("best_cv_mean_ci"),
                          "final_train_ci": training.get("training_history", {}).get("final_train_ci"),
                          "final_val_ci": training.get("training_history", {}).get("final_val_ci")}
        return model, cv_metrics, model.feature_names or []

    model_path = None
    for root, _, files in os.walk(model_dir):
        pickles = [f for f in files if f.endswith(".pkl")]
        if pickles:
            model_path = os.path.join(root, pickles[0])
            break
    if model_path is None:
        raise FileNotFoundError(f"No .pkl file in {model_dir}")
    model = joblib.load(model_path)
    folder = os.path.dirname(model_path)

    cv_metrics = {}
    if os.path.exists(os.path.join(folder, "cv_metrics.json")):
        with open(os.path.join(folder, "cv_metrics.json")) as f:
            cv_metrics = json.load(f)

    features_path = os.path.join(folder, "selected_features.txt")
    if os.path.exists(features_path):  # elastic-net Cox
        with open(features_path) as f:
            features = [line.strip() for line in f if line.strip()]
    else:  # tree-based learners: the features they were fitted on
        features = extract_feature_names_from_model(model) or []
    print(f"Loaded {model_name} from {model_path} ({len(features)} features)")
    return model, cv_metrics, features


def create_summary_table(test_results, comparison_dir):
    """Hold-out metrics of every model (summary_metrics.csv; Supplementary Table S2)."""
    rows = []
    for model_name, m in test_results.items():
        row = {"Model": model_name.upper(), "C-index": m.get("c_index", np.nan),
               "Mean AUC": m.get("mean_auc", np.nan)}
        row.update({f"AUC_{t}m": m.get(f"auc_{t}m", np.nan) for t in CLINICAL_TIMEPOINTS_MONTHS})
        row.update({f"Brier_{t}m": m.get(f"brier_{t}m", np.nan) for t in CLINICAL_TIMEPOINTS_MONTHS})
        row.update({"IBS 0-6m": m.get("ibs_0_6m", np.nan), "IBS 6-12m": m.get("ibs_6_12m", np.nan),
                    "IBS 12-18m": m.get("ibs_12_18m", np.nan), "IBS 18-24m": m.get("ibs_18_24m", np.nan),
                    "IBS Overall": m.get("ibs_overall", np.nan)})
        rows.append(row)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(comparison_dir, "summary_metrics.csv"), index=False)


def create_comparison_table(test_results, cv_results, comparison_dir):
    """Hold-out C-index next to the cross-validated metrics of the tuning (model_comparison.csv)."""
    rows = []
    for model_name, m in test_results.items():
        cv = cv_results.get(model_name, {}).get("cv_metrics_mean", {})
        rows.append({"Model": model_name, "Test C-index": m.get("c_index", np.nan),
                     "CV C-index (mean)": cv.get("c_index", np.nan),
                     "CV IBS (mean)": cv.get("integrated_brier_score", np.nan),
                     "CV Mean AUC": cv.get("mean_auc", np.nan)})
    pd.DataFrame(rows).round(4).to_csv(os.path.join(comparison_dir, "model_comparison.csv"), index=False)
