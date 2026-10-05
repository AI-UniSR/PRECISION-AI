"""XGBoost accelerated failure time base learner ('xgboost').

Optuna (TPE sampler, seed 42) tunes the booster on the stability-selected
features, maximising the mean Harrell's C over the cross-validation folds.
The pipeline is IterativeImputer -> XGBoostSurvival (xgboost_survival_model.py);
column subsampling is fixed at 1 because there are few features.
"""

import argparse
import json
import os
import sys

import joblib
import mlflow
import numpy as np
import optuna
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (enables IterativeImputer)
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from survival_utils import (  # noqa: E402
    cv_splits,
    evaluate_model_cv,
    load_feature_list,
    load_training_data,
    save_cv_metrics,
    save_optuna_plots,
)
from xgboost_survival_model import XGBoostSurvival  # noqa: E402


def xgboost_pipeline(params):
    return Pipeline([("imputation", IterativeImputer(random_state=42, max_iter=10)),
                     ("model", XGBoostSurvival(**params, colsample_bytree=1.0, random_state=42,
                                               n_jobs=-1, verbosity=0))])


def objective(trial, X, y, n_splits, n_repeats, stratify_by_event):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "aft_loss_distribution": trial.suggest_categorical("aft_loss_distribution",
                                                           ["normal", "logistic", "extreme"]),
        "aft_loss_distribution_scale": trial.suggest_float("aft_loss_distribution_scale", 0.3, 3.0, log=True),
    }
    c_indices = []
    for train_idx, test_idx in cv_splits(X, y, n_splits, n_repeats, stratify_by_event):
        pipeline = xgboost_pipeline(params).fit(X.iloc[train_idx], y[train_idx])
        y_test = y[test_idx]
        c_indices.append(concordance_index_censored(y_test["event"], y_test["time"],
                                                    pipeline.predict(X.iloc[test_idx]))[0])
    return np.mean(c_indices)


def main(args):
    out = args.trained_model
    os.makedirs(out, exist_ok=True)
    mlflow.start_run()
    X, y = load_training_data(args.training_data)
    features = load_feature_list(args.selected_features)
    X = X[features]
    cv = dict(n_splits=args.cv_n_splits, n_repeats=args.cv_n_repeats,
              stratify_by_event=args.cv_stratify_by_event)

    study = optuna.create_study(direction="maximize", study_name="xgboost_survival_tuning",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X, y, **cv), n_trials=args.n_optuna_trials)
    best = study.best_params
    print(f"Best C-index {study.best_value:.4f} with {best}")
    study.trials_dataframe().to_csv(os.path.join(out, "optuna_history.csv"), index=False)
    save_optuna_plots(study, out)

    model = xgboost_pipeline(best).fit(X, y)
    cv_df, cv_mean, cv_std = evaluate_model_cv(X, y, lambda: xgboost_pipeline(best), **cv)
    save_cv_metrics(cv_df, cv_mean, cv_std, out)
    with open(os.path.join(out, "cv_metrics.json"), "w") as f:
        json.dump({"cv_metrics_per_fold": cv_df.to_dict(orient="records"),
                   "cv_metrics_mean": cv_mean, "cv_metrics_std": cv_std}, f, indent=2)

    joblib.dump(model, os.path.join(out, "xgboost_survival_pipeline.pkl"))
    mlflow.log_params({**best, "n_features": len(features)})
    mlflow.log_metric("optuna_best_cindex", study.best_value)
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
    mlflow.log_artifacts(out)
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training_data", required=True, help="Folder with the training CSV")
    parser.add_argument("--selected_features", required=True,
                        help="Output folder of the feature selection (selected_features.txt)")
    parser.add_argument("--trained_model", required=True, help="Output folder")
    parser.add_argument("--cv_n_splits", type=int, default=5)
    parser.add_argument("--cv_n_repeats", type=int, default=10)
    parser.add_argument("--cv_stratify_by_event", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--n_optuna_trials", type=int, default=50)
    main(parser.parse_args())
