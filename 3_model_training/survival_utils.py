"""Helpers shared by the training scripts of the base learners.

cv_splits() defines the cross-validation used for tuning (5 x 10 repeated
stratified folds in the paper, seed 42). evaluate_model_cv() refits the tuned
configuration on the same folds and records C-index, IBS and time-dependent
AUC per fold; these cross-validated metrics are saved with each learner
(cv_metrics.json) but are not reported in the paper.
"""

import os

import mlflow
import numpy as np
import pandas as pd
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv


def load_training_data(folder):
    """Feature matrix (numeric columns, indexed by patient id) and survival target."""
    csv = [f for f in os.listdir(folder) if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(folder, csv))
    X = df.drop(columns=["event", "tte"]).select_dtypes(include=[np.number]).set_index("id")
    y = Surv.from_arrays(event=df["event"].astype(bool), time=df["tte"].astype(float))
    return X, y


def load_feature_list(folder):
    """Features selected by the stability selection (selected_features.txt)."""
    with open(os.path.join(folder, "selected_features.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def cv_splits(X, y, n_splits=5, n_repeats=1, stratify_by_event=True):
    """Train/validation indices of the cross-validation (seed 42).

    Stratified folds keep the event rate of y; repeats are available only with
    stratification.
    """
    if not stratify_by_event:
        return KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X)
    if n_repeats > 1:
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                       random_state=42).split(X, y["event"])
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(X, y["event"])


def compute_survival_metrics(y_train, y_test, risk_scores, pipeline, X_test, times=None):
    """C-index, IBS and time-dependent AUC on one validation fold.

    The IBS uses 10 time points between the 5th and 95th percentiles of the
    training times, the AUC the quartiles of the training event times. When
    the IBS cannot be computed (e.g. a time point beyond the fold's follow-up),
    the fold reports the C-index only.
    """
    metrics = {"c_index": concordance_index_censored(y_test["event"], y_test["time"], risk_scores)[0]}
    try:
        surv_funcs = pipeline.predict_survival_function(X_test)
        if times is None:
            times = np.percentile(y_train["time"][y_train["event"]], [25, 50, 75])
        times_brier = np.percentile(y_train["time"], np.linspace(5, 95, 10))
        preds = np.asarray([[fn(t) for t in times_brier] for fn in surv_funcs])
        metrics["integrated_brier_score"] = integrated_brier_score(y_train, y_test, preds, times_brier)
        try:
            auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
            for t, value in zip(times, auc):
                metrics[f"auc_at_{int(t)}"] = value
            metrics["mean_auc"] = mean_auc
        except Exception as e:
            print(f"  AUC not computed: {e}")
    except Exception as e:
        print(f"  IBS and AUC not computed: {e}")
    return metrics


def evaluate_model_cv(X, y, model_factory, n_splits=5, n_repeats=1, stratify_by_event=True,
                      verbose=True):
    """Refit model_factory() in every fold; per-fold metrics, their means and SDs."""
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv_splits(X, y, n_splits, n_repeats, stratify_by_event)):
        model = model_factory()
        model.fit(X.iloc[train_idx], y[train_idx])
        risk = model.predict(X.iloc[test_idx])
        row = compute_survival_metrics(y[train_idx], y[test_idx], risk, model, X.iloc[test_idx])
        row["fold"] = fold + 1
        rows.append(row)
        if verbose:
            print(f"Fold {fold + 1}: C-index {row['c_index']:.4f}")
    metrics_df = pd.DataFrame(rows)
    return (metrics_df, metrics_df.drop(columns=["fold"]).mean().to_dict(),
            metrics_df.drop(columns=["fold"]).std().to_dict())


def save_cv_metrics(metrics_df, mean, std, output_dir, prefix="cv"):
    """Write the per-fold metrics and log their means and SDs to MLflow."""
    for name, value in mean.items():
        mlflow.log_metric(f"{prefix}_{name}_mean", value)
        mlflow.log_metric(f"{prefix}_{name}_std", std.get(name, 0))
        print(f"{name}: {value:.4f} (SD {std.get(name, 0):.4f})")
    metrics_df.to_csv(os.path.join(output_dir, f"{prefix}_survival_metrics.csv"), index=False)


def save_optuna_plots(study, output_dir):
    """Standard Optuna plots of the search, as HTML."""
    for name, plot in [("optimization_history", plot_optimization_history),
                       ("param_importances", plot_param_importances),
                       ("slice_plot", plot_slice),
                       ("parallel_coordinate", plot_parallel_coordinate),
                       ("contour_plot", plot_contour)]:
        try:
            plot(study).write_html(os.path.join(output_dir, f"optuna_{name}.html"))
        except Exception as e:  # e.g. too few completed trials
            print(f"Optuna plot '{name}' not produced: {e}")
