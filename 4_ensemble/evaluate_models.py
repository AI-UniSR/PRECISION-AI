"""Hold-out evaluation of the base learners and construction of the stacking ensemble.

1. Load the training and hold-out test sets (data_prep.py) and the trained
   learners (cox, rsf, xgboost, xgbse and, if given, deepsurv).
2. Evaluate every learner on the hold-out set (C-index, time-dependent AUC,
   Brier score, IBS; definitions in evaluation_utils.py).
3. Build the ensemble from out-of-fold predictions, register it in MLflow and
   evaluate it on the hold-out set (ensemble_model.py).

comparison/summary_metrics.csv is Supplementary Table S2. The hold-out row of
Table 2 comes instead from the validation pipeline (5_validation), which
re-evaluates the registered ensemble with its own metric definitions.
"""

import argparse
import os

import mlflow
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

from ensemble_model import create_and_evaluate_ensemble, generate_oof_predictions, train_meta_learner
from evaluation_utils import (
    CLINICAL_TIMEPOINTS_MONTHS,
    compute_brier_scores_on_grid,
    compute_ibs_intervals,
    create_comparison_table,
    create_summary_table,
    create_temporal_grid,
    load_model_and_metrics,
)
# Classes of the pickled base learners, needed by joblib.load
from xgboost_survival_model import XGBoostSurvival  # noqa: F401
from xgbse_pipeline_wrapper import XGBSEPipeline  # noqa: F401


def _label(t):
    """Nominal time point (months) closest to t."""
    return CLINICAL_TIMEPOINTS_MONTHS[int(np.argmin(np.abs(np.array(CLINICAL_TIMEPOINTS_MONTHS) - t)))]


def evaluate_model_on_test(model, X_test, y_test, y_train, model_name, output_dir,
                           grid_months, clinical_months, risk_scores=None):
    """Hold-out metrics of one model; time points beyond follow-up are NaN."""
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    if risk_scores is None:
        risk_scores = model.predict(X_test)
        if risk_scores.ndim > 1:  # DeepSurv returns shape (n, 1)
            risk_scores = risk_scores.flatten()

    metrics = {"c_index": concordance_index_censored(y_test["event"], y_test["time"], risk_scores)[0]}
    labels = [_label(t) for t in clinical_months]
    not_evaluated = [t for t in CLINICAL_TIMEPOINTS_MONTHS if t not in labels]

    try:
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, clinical_months)
        metrics["mean_auc"] = mean_auc
        metrics.update({f"auc_{label}m": value for label, value in zip(labels, auc)})
        metrics.update({f"auc_{t}m": np.nan for t in not_evaluated})
    except Exception as e:
        print(f"  {model_name}: AUC not computed ({e})")
        metrics.update({f"auc_{t}m": np.nan for t in CLINICAL_TIMEPOINTS_MONTHS})

    brier = compute_brier_scores_on_grid(model, X_test, y_train, y_test, grid_months)
    if brier is None:
        metrics.update({f"brier_{t}m": np.nan for t in CLINICAL_TIMEPOINTS_MONTHS})
    else:
        for label, t in zip(labels, clinical_months):
            metrics[f"brier_{label}m"] = brier[int(np.argmin(np.abs(grid_months - t)))]
        metrics.update({f"brier_{t}m": np.nan for t in not_evaluated})
        ibs_intervals, metrics_ibs = compute_ibs_intervals(grid_months, brier)
        metrics.update(ibs_intervals)
        metrics["ibs_overall"] = metrics_ibs

    pd.DataFrame([metrics]).to_csv(os.path.join(model_dir, "test_metrics.csv"), index=False)
    print(f"{model_name}: C-index {metrics['c_index']:.4f}, mean AUC {metrics.get('mean_auc', np.nan):.4f}, "
          f"IBS {metrics.get('ibs_overall', np.nan):.4f}")
    return metrics


def _load_split(folder):
    csv = [f for f in os.listdir(folder) if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(folder, csv))
    X = df.drop(columns=["event", "tte"]).select_dtypes(include=[np.number]).set_index("id")
    y = Surv.from_arrays(event=df["event"].astype(bool), time=df["tte"].astype(float))
    return X, y


def main(args):
    os.makedirs(args.evaluation_output, exist_ok=True)
    mlflow.start_run()

    X_train, y_train = _load_split(args.training_data)
    X_test, y_test = _load_split(args.test_data)
    X_test = X_test[[c for c in X_train.columns if c in X_test.columns]]

    model_dirs = {"cox": args.cox_model, "rsf": args.rsf_model, "xgboost": args.xgboost_model,
                  "xgbse": args.xgbse_model}
    if args.deepsurv_model:
        model_dirs["deepsurv"] = args.deepsurv_model
    models, cv_metrics, features = {}, {}, {}
    for name, folder in model_dirs.items():
        models[name], cv_metrics[name], features[name] = load_model_and_metrics(folder, name)

    grid_months, clinical_months = create_temporal_grid(y_train, y_test)

    test_results = {}
    for name, model in models.items():
        X_model = X_test[[f for f in features[name] if f in X_test.columns]] if features[name] else X_test
        test_results[name] = evaluate_model_on_test(model, X_model, y_test, y_train, name,
                                                    args.evaluation_output, grid_months, clinical_months)

    oof, stack_models = generate_oof_predictions(models, features, X_train, y_train)
    meta_learner = train_meta_learner(oof, y_train, X_train, args.evaluation_output)
    test_results["ensemble"] = create_and_evaluate_ensemble(
        stack_models, {name: features[name] for name in stack_models}, meta_learner,
        X_test, y_test, y_train, args.evaluation_output, grid_months, clinical_months,
        evaluate_fn=evaluate_model_on_test)
    cv_metrics["ensemble"] = {}

    comparison_dir = os.path.join(args.evaluation_output, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    create_comparison_table(test_results, cv_metrics, comparison_dir)
    create_summary_table(test_results, comparison_dir)

    for name, metrics in test_results.items():
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                mlflow.log_metric(f"test_{name}_{key}", value)
    mlflow.log_artifacts(comparison_dir, artifact_path="comparison")
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training_data", required=True, help="Folder with the training CSV")
    parser.add_argument("--test_data", required=True, help="Folder with the hold-out test CSV")
    parser.add_argument("--cox_model", required=True, help="Output folder of cox_net/model_train.py")
    parser.add_argument("--rsf_model", required=True, help="Output folder of rsf/model_train.py")
    parser.add_argument("--xgboost_model", required=True, help="Output folder of xgboost_surv/model_train.py")
    parser.add_argument("--xgbse_model", required=True, help="Output folder of xgbse_weibull/model_train.py")
    parser.add_argument("--deepsurv_model", default=None, help="Output folder of deepsurv/model_train.py")
    parser.add_argument("--evaluation_output", required=True, help="Output folder")
    main(parser.parse_args())
