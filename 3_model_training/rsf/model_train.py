import argparse
import json
import os

# Import reusable utility functions
import sys

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# When running in Azure ML, the working directory is model_train/
# When running locally from rsf/, we need to go up one level
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import json

from survival_utils import (
    compute_survival_metrics,
    compute_survival_metrics_cv,
    evaluate_model_cv,
    log_metrics_to_mlflow,
    log_optuna_plots,
)


def objective(trial, X_selected, y, cv_n_splits, cv_n_repeats, cv_stratify_by_event):
    """
    Optuna objective function to tune Random Survival Forest hyperparameters.
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    X_selected : pd.DataFrame
        Feature matrix after feature selection
    y : structured array
        Survival target (event, time)
    cv_n_splits : int
        Number of cross-validation folds
    cv_n_repeats : int
        Number of CV repetitions
    cv_stratify_by_event : bool
        If True, stratify folds by event status
        
    Returns
    -------
    float
        Mean C-index across folds
    """
    # Define search space for Random Survival Forest
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7, 0.9]),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.1),
        'max_samples': trial.suggest_float('max_samples', 0.5, 1.0)
    }
    
    print(f"\n[Trial {trial.number}] Testing parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # Cross-validation with stratification
    if cv_stratify_by_event:
        if cv_n_repeats > 1:
            from sklearn.model_selection import RepeatedStratifiedKFold
            cv = RepeatedStratifiedKFold(n_splits=cv_n_splits, n_repeats=cv_n_repeats, random_state=42)
            splitter = cv.split(X_selected, y["event"])
        else:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
            splitter = cv.split(X_selected, y["event"])
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
        splitter = cv.split(X_selected)
    
    c_indices = []
    
    for fold, (train_idx, test_idx) in enumerate(splitter):
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create pipeline with IterativeImputer and RSF
        pipeline = make_pipeline(
            IterativeImputer(random_state=42, max_iter=10),
            RobustScaler(),
            RandomSurvivalForest(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                max_samples=params['max_samples'],
                random_state=42,
                n_jobs=-1
            )
        )
        
        pipeline.fit(X_train, y_train)
        risk = pipeline.predict(X_test)
        cindex = concordance_index_censored(y_test["event"], y_test["time"], risk)[0]
        c_indices.append(cindex)
    
    mean_cindex = np.mean(c_indices)
    print(f"[Trial {trial.number}] Mean C-index: {mean_cindex:.4f}")
    
    return mean_cindex


def main(args):
    print("=== Starting Random Survival Forest model training with Optuna ===")
    os.makedirs(args.trained_model, exist_ok=True)
    mlflow.start_run()

    # --- Load dataset ---
    csv_files = [f for f in os.listdir(args.training_data) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in training_data folder.")
    csv_path = os.path.join(args.training_data, csv_files[0])
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")

    # --- Prepare data ---
    print("Preparing features and survival target...")
    X = df.drop(columns=["event", "tte"]).select_dtypes(include=[np.number])
    X.set_index("id", inplace=True)
    y = Surv.from_arrays(event=df["event"].astype(bool), time=df["tte"].astype(float))
    print(f"Feature matrix: {X.shape}, Survival vector: {y.shape}")

    # --- Load selected features ---
    print("\n=== Loading selected features from feature selection step ===")
    selected_features_path = os.path.join(args.selected_features, "selected_features.txt")
    if not os.path.exists(selected_features_path):
        raise FileNotFoundError(f"Selected features file not found: {selected_features_path}")
    
    with open(selected_features_path, 'r') as f:
        selected_features = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(selected_features)} selected features")
    print(f"Selected features: {selected_features}")
    
    # Verify all selected features exist in the dataset
    missing_features = set(selected_features) - set(X.columns)
    if missing_features:
        raise ValueError(f"Selected features not found in dataset: {missing_features}")
    
    # Filter dataset to selected features
    X_selected = X[selected_features]
    print(f"Filtered feature matrix shape: {X_selected.shape}")
    
    # Log feature selection info
    mlflow.log_param("n_features_original", X.shape[1])
    mlflow.log_param("n_features_selected", len(selected_features))
    
    # --- Log CV configuration ---
    print(f"\n=== Cross-Validation Configuration ===")
    print(f"CV Strategy: {'Stratified' if args.cv_stratify_by_event else 'Standard'} {args.cv_n_splits}-Fold")
    print(f"CV Repetitions: {args.cv_n_repeats}")
    print(f"Total CV iterations per evaluation: {args.cv_n_splits * args.cv_n_repeats}")
    print(f"Optuna trials: {args.n_optuna_trials}")
    
    mlflow.log_param("cv_n_splits", args.cv_n_splits)
    mlflow.log_param("cv_n_repeats", args.cv_n_repeats)
    mlflow.log_param("cv_stratify_by_event", args.cv_stratify_by_event)
    mlflow.log_param("n_optuna_trials", args.n_optuna_trials)
    mlflow.log_param("total_cv_iterations", args.cv_n_splits * args.cv_n_repeats)

    # --- Step 1: Optuna Hyperparameter Optimization ---
    print("\n=== Step 1: Optuna Hyperparameter Optimization for Random Survival Forest ===")
    study = optuna.create_study(
        direction="maximize",
        study_name="rsf_tuning",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda trial: objective(trial, X_selected, y, args.cv_n_splits, args.cv_n_repeats, args.cv_stratify_by_event),
        n_trials=args.n_optuna_trials,
        show_progress_bar=True
    )
    
    print(f"\n✅ Optimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best C-index: {study.best_value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Log Optuna results to MLflow
    mlflow.log_params(study.best_params)
    mlflow.log_metric("optuna_best_cindex", study.best_value)
    mlflow.log_metric("optuna_n_trials", len(study.trials))
    
    # Save optimization history
    optuna_df = study.trials_dataframe()
    optuna_history_path = os.path.join(args.trained_model, "optuna_history.csv")
    optuna_df.to_csv(optuna_history_path, index=False)
    mlflow.log_artifact(optuna_history_path, artifact_path="optimization")
    print(f"Logged Optuna history with {len(study.trials)} trials")
    
    # Log Optuna visualization plots
    print("\n=== Generating Optuna visualization plots ===")
    log_optuna_plots(study, args.trained_model, artifact_path="optimization")

    # --- Step 2: Train final model with best hyperparameters ---
    print("\n=== Step 2: Training final model with best hyperparameters ===")
    best_params = study.best_params
    
    # Create final pipeline with feature selection, imputation, and tuned RSF
    final_pipeline = Pipeline([
        ('imputation', IterativeImputer(random_state=42, max_iter=10)),
        ('model', RandomSurvivalForest(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            max_features=best_params['max_features'],
            min_weight_fraction_leaf=best_params['min_weight_fraction_leaf'],
            max_samples=best_params['max_samples'],
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("Fitting final pipeline on full dataset...")
    final_pipeline.fit(X_selected, y)
    print("✅ Final model training completed!")
    
    # --- Step 3: Cross-validation evaluation with comprehensive metrics ---
    print("\n=== Step 3: Cross-validation evaluation with comprehensive metrics ===")
    
    def model_factory():
        return Pipeline([
            ('imputation', IterativeImputer(random_state=42, max_iter=10)),
            ('model', RandomSurvivalForest(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                max_features=best_params['max_features'],
                min_weight_fraction_leaf=best_params['min_weight_fraction_leaf'],
                max_samples=best_params['max_samples'],
                random_state=42,
                n_jobs=-1
            ))
        ])
    
    metrics_df, avg_metrics, std_metrics = evaluate_model_cv(
        X_selected, y,
        model_factory=model_factory,
        n_splits=args.cv_n_splits,
        n_repeats=args.cv_n_repeats,
        stratify_by_event=args.cv_stratify_by_event,
        verbose=True
    )
    
    # Log metrics to MLflow
    log_metrics_to_mlflow(metrics_df, avg_metrics, std_metrics, args.trained_model, prefix="cv")
    
    # Save CV metrics to JSON for later evaluation
    cv_metrics_json = {
        "cv_metrics_per_fold": metrics_df.to_dict(orient='records'),
        "cv_metrics_mean": avg_metrics,
        "cv_metrics_std": std_metrics
    }
    cv_metrics_path = os.path.join(args.trained_model, "cv_metrics.json")
    with open(cv_metrics_path, 'w') as f:
        json.dump(cv_metrics_json, f, indent=2)
    mlflow.log_artifact(cv_metrics_path, artifact_path="metrics")
    print(f"Saved CV metrics to {cv_metrics_path}")
    
    # --- Step 4: Generate explainability artifacts ---
    print("\n=== Step 4: Generating explainability artifacts ===")
    explainability_dir = os.path.join(args.trained_model, "explainability")
    os.makedirs(explainability_dir, exist_ok=True)
    
    
    # Per-model SHAP and PFI are computed at ensemble evaluation (see 4_ensemble/evaluate_models.py
    # and 5_validation/shap_analysis.py).


    # --- Step 5: Save final pipeline ---
    print("\n=== Step 5: Saving final pipeline ===")
    
    # Save with joblib (for compatibility)
    pipeline_path = os.path.join(args.trained_model, "rsf_pipeline.pkl")
    joblib.dump(final_pipeline, pipeline_path)
    print(f"Saved pipeline to {pipeline_path}")
    
    # Log model with MLflow
    mlflow.sklearn.log_model(
        sk_model=final_pipeline,
        artifact_path='model',
        registered_model_name=None
    )
    
    print("\n✅ Pipeline saved successfully!")
    print(f"  - Selected features: {len(selected_features)} features")
    print(f"  - Iterative Imputer: max_iter=10")
    print(f"  - Random Survival Forest with optimized hyperparameters")

    mlflow.end_run()
    print("\n=== Training completed successfully. All artifacts logged to MLflow ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, required=True,
                        help="Path to folder containing training CSV")
    parser.add_argument("--selected_features", type=str, required=True,
                        help="Path to folder containing selected features from feature selection step")
    parser.add_argument("--trained_model", type=str, required=True,
                        help="Path to output folder for trained model and artifacts")
    parser.add_argument("--cv_n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--cv_n_repeats", type=int, default=10, help="Number of CV repetitions")
    parser.add_argument("--cv_stratify_by_event", type=lambda x: x.lower() == 'true', default=True,
                        help="Stratify CV folds by event status")
    parser.add_argument("--n_optuna_trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()
    main(args)
