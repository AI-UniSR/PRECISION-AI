"""XGBSE stacked Weibull base learner ('xgbse').

Optuna (TPE sampler, seed 42) tunes the model on the stability-selected
features, maximising the mean Harrell's C over the cross-validation folds.
Missing values are not imputed: XGBoost sends them along a learned default
direction at each split. The model class is in xgbse_pipeline_wrapper.py.
"""

import argparse
import json
import os
import sys

import joblib
import mlflow
import numpy as np
import optuna
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
from xgbse_pipeline_wrapper import XGBSEPipeline  # noqa: E402


def objective(trial, X, y, n_splits, n_repeats, stratify_by_event):
    num_boost_round = trial.suggest_int("num_boost_round", 100, 500)
    xgb_params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "seed": 42,
    }
    c_indices = []
    for train_idx, test_idx in cv_splits(X, y, n_splits, n_repeats, stratify_by_event):
        model = XGBSEPipeline(xgb_params=xgb_params, num_boost_round=num_boost_round)
        model.fit(X.iloc[train_idx], y[train_idx])
        y_test = y[test_idx]
        c_indices.append(concordance_index_censored(y_test["event"], y_test["time"],
                                                    model.predict(X.iloc[test_idx]))[0])
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

    study = optuna.create_study(direction="maximize", study_name="xgbse_stacked_weibull_tuning",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X, y, **cv), n_trials=args.n_optuna_trials)
    print(f"Best C-index {study.best_value:.4f} with {study.best_params}")
    study.trials_dataframe().to_csv(os.path.join(out, "optuna_history.csv"), index=False)
    save_optuna_plots(study, out)

    xgb_params = study.best_params.copy()
    num_boost_round = xgb_params.pop("num_boost_round")
    xgb_params["seed"] = 42

    def model_factory():
        return XGBSEPipeline(xgb_params=xgb_params, num_boost_round=num_boost_round)

    model = model_factory().fit(X, y)
    cv_df, cv_mean, cv_std = evaluate_model_cv(X, y, model_factory, **cv)
    save_cv_metrics(cv_df, cv_mean, cv_std, out)
    with open(os.path.join(out, "cv_metrics.json"), "w") as f:
        json.dump({"cv_metrics_per_fold": cv_df.to_dict(orient="records"),
                   "cv_metrics_mean": cv_mean, "cv_metrics_std": cv_std}, f, indent=2)

    joblib.dump(model, os.path.join(out, "xgbse_stacked_weibull_pipeline.pkl"))
    mlflow.log_params({**study.best_params, "n_features": len(features), "imputation": False})
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
