import argparse
import os
import sys

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# XGBSE imports
try:
    from xgbse import XGBSEStackedWeibull
    from xgbse.converters import convert_to_structured
    XGBSE_AVAILABLE = True
except ImportError:
    XGBSE_AVAILABLE = False
    print("Warning: XGBSE not available. Install with: pip install xgbse")

# Import reusable utility functions
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


class XGBSEPipeline(BaseEstimator, RegressorMixin):
    """
    Wrapper for XGBSEStackedWeibull to work in sklearn pipeline.
    IMPORTANT: This does NOT apply imputation - XGBoost handles NaN natively.
    """
    
    def __init__(self, xgb_params=None, num_boost_round=100, random_state=42, enable_categorical=False):
        self.xgb_params = xgb_params or {}
        self.num_boost_round = num_boost_round
        self.random_state = random_state
        self.enable_categorical = enable_categorical
        self.xgbse_model_ = None
        
    def fit(self, X, y):
        """
        Fit XGBSE model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features (can contain NaN - XGBoost will handle them)
        y : structured array
            Survival data (event, time)
        """
        # Convert sksurv structured array to format expected by XGBSE
        # XGBSE expects y as structured array with dtype.names
        # Keep the same format but ensure correct field names
        if hasattr(y, 'dtype') and hasattr(y.dtype, 'names') and y.dtype.names:
            # Already a structured array, just use it
            y_xgbse = y
        else:
            # Convert to structured array
            y_xgbse = np.array(
                list(zip(y['event'].astype(bool), y['time'].astype(float))),
                dtype=[('event', bool), ('time', float)]
            )
        
        # Create XGBSE model with XGBoost parameters dict
        self.xgbse_model_ = XGBSEStackedWeibull(
            xgb_params=self.xgb_params,  # Dict of XGBoost params (no n_estimators)
            enable_categorical=self.enable_categorical
        )
        
        # Fit with y as structured array and num_boost_round
        self.xgbse_model_.fit(
            X,
            y_xgbse,  # Pass structured array with dtype.names
            num_boost_round=self.num_boost_round
        )
        return self
    
    def predict(self, X):
        """
        Predict risk scores (negative median survival time).
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
            
        Returns
        -------
        np.ndarray
            Risk scores (higher = higher risk)
        """
        if self.xgbse_model_ is None:
            raise ValueError("Model not fitted yet")
        
        # Get survival probability curves (DataFrame: rows=samples, cols=timepoints)
        surv_curves = self.xgbse_model_.predict(X)
        
        # Calculate median survival time for each sample
        # Median = time when S(t) = 0.5
        median_times = []
        time_bins = surv_curves.columns.values
        
        # Iterate by position, not by index (surv_curves may have RangeIndex)
        for i in range(len(X)):
            surv_probs = surv_curves.iloc[i].values
            # Find first time where survival probability <= 0.5
            below_half = np.where(surv_probs <= 0.5)[0]
            if len(below_half) > 0:
                median_time = time_bins[below_half[0]]
            else:
                # If survival never drops below 0.5, use max time
                median_time = time_bins[-1]
            median_times.append(median_time)
        
        median_times = np.array(median_times)
        # Return negative values (higher risk = shorter survival)
        return -median_times
    
    def predict_survival_function(self, X):
        """
        Predict survival functions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
            
        Returns
        -------
        list of callables
            Survival functions
        """
        if self.xgbse_model_ is None:
            raise ValueError("Model not fitted yet")
        
        # Get survival curves (DataFrame: rows=samples, cols=timepoints)
        # XGBSE.predict() returns survival probabilities, not predict_survival_function
        surv_curves = self.xgbse_model_.predict(X)
        
        # Convert to list of callables (interpolation functions)
        surv_funcs = []
        times = surv_curves.columns.values  # Time bins
        
        # Iterate by position (surv_curves has RangeIndex)
        for i in range(len(X)):
            surv_probs = surv_curves.iloc[i].values
            
            # Create interpolation function
            from scipy.interpolate import interp1d
            func = interp1d(times, surv_probs, kind='linear', 
                          bounds_error=False, fill_value=(1.0, 0.0))
            surv_funcs.append(func)
        
        return surv_funcs
    
    @property
    def feature_importances_(self):
        """Get feature importances from base XGBoost model."""
        if self.xgbse_model_ is not None and hasattr(self.xgbse_model_, 'feature_importances_'):
            return self.xgbse_model_.feature_importances_
        return None


def plot_feature_importance(feature_names, importances, output_path, top_n=30):
    """
    Plot feature importances from XGBSE.

    Parameters
    ----------
    feature_names : list
        List of feature names
    importances : np.ndarray or dict
        Feature importance scores (array or dict {feature: importance})
    output_path : str
        Path to save the plot
    top_n : int
        Number of top features to display
    """
    # Handle dict format (XGBSE returns dict)
    if isinstance(importances, dict):
        # Align with feature_names order
        importances = [importances.get(f, 0.0) for f in feature_names]
    
    # Ensure arrays are proper lists/arrays to avoid ambiguous ordering
    importances = np.asarray(importances).flatten()
    if len(importances) != len(feature_names):
        raise ValueError(f"Length mismatch: {len(importances)} importances vs {len(feature_names)} features")
    
    # Create DataFrame and sort
    df = pd.DataFrame({
        'feature': list(feature_names),
        'importance': list(importances)
    }).sort_values('importance', ascending=False)

    # Take top N
    df_top = df.head(top_n)

    # Create plot
    fig = go.Figure(go.Bar(
        x=df_top['importance'],
        y=df_top['feature'],
        orientation='h',
        marker=dict(
            color=df_top['importance'],
            colorscale='Viridis',
            showscale=True
        )
    ))

    fig.update_layout(
        title=f"Top {top_n} Feature Importances - XGBSE Stacked Weibull",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=max(600, top_n * 20),
        yaxis=dict(autorange="reversed")
    )

    pio.write_html(fig, file=output_path, include_plotlyjs="cdn", full_html=True)
    return df


def objective(trial, X_selected, y, cv_n_splits=5, cv_n_repeats=1, cv_stratify_by_event=True):
    """
    Optuna objective function to tune XGBSE hyperparameters.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    X_selected : pd.DataFrame
        Feature matrix after feature selection (can contain NaN)
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
    # Define hyperparameter search space for XGBSE
    num_boost_round = trial.suggest_int('num_boost_round', 100, 500)
    
    xgb_params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 30.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
    }

    print(f"\n[Trial {trial.number}] Testing parameters:")
    print(f"  num_boost_round: {num_boost_round}")
    for k, v in xgb_params.items():
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
        kf = KFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
        splitter = kf.split(X_selected)
    
    c_indices = []

    for fold, (train_idx, test_idx) in enumerate(splitter):
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create pipeline with XGBSE (NO imputation - XGBoost handles NaN)
        pipeline = XGBSEPipeline(
            xgb_params=xgb_params,
            num_boost_round=num_boost_round,
            random_state=42,
            enable_categorical=False
        )

        pipeline.fit(X_train, y_train)
        risk = pipeline.predict(X_test)
        cindex = concordance_index_censored(
            y_test["event"],
            y_test["time"],
            risk
        )[0]
        c_indices.append(cindex)

    mean_cindex = np.mean(c_indices)
    print(f"[Trial {trial.number}] Mean C-index: {mean_cindex:.4f}")

    return mean_cindex


def main(args):
    if not XGBSE_AVAILABLE:
        raise ImportError("XGBSE is not available. Please install: pip install xgbse")
    
    print("=== Starting XGBSE Stacked Weibull model training with Optuna ===")
    os.makedirs(args.trained_model, exist_ok=True)
    mlflow.start_run()
    
    # Log CV configuration
    mlflow.log_param("cv_n_splits", args.cv_n_splits)
    mlflow.log_param("cv_n_repeats", args.cv_n_repeats)
    mlflow.log_param("cv_stratify_by_event", args.cv_stratify_by_event)
    mlflow.log_param("n_optuna_trials", args.n_optuna_trials)
    print(f"CV Configuration: {args.cv_n_splits} folds × {args.cv_n_repeats} repeats (stratified={args.cv_stratify_by_event})")

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
    print(f"⚠️  IMPORTANT: Keeping NaN values for XGBoost native handling")
    print(f"   NaN count per feature:\n{X_selected.isna().sum()[X_selected.isna().sum() > 0]}")
    
    # Log feature selection info
    mlflow.log_param("n_features_original", X.shape[1])
    mlflow.log_param("n_features_selected", len(selected_features))
    mlflow.log_param("imputation_applied", False)  # Key difference from other models

    # --- Step 1: Optuna Hyperparameter Optimization ---
    print("\n=== Step 1: Optuna Hyperparameter Optimization for XGBSE ===")
    study = optuna.create_study(
        direction="maximize",
        study_name="xgbse_stacked_weibull_tuning",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda trial: objective(trial, X_selected, y, 
                               cv_n_splits=args.cv_n_splits, 
                               cv_n_repeats=args.cv_n_repeats,
                               cv_stratify_by_event=args.cv_stratify_by_event),
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
    best_params = study.best_params.copy()
    
    # Extract num_boost_round from params
    num_boost_round = best_params.pop('num_boost_round')
    xgb_params = best_params  # Remaining params are XGBoost params

    # Create final pipeline with XGBSE (NO imputation)
    final_pipeline = XGBSEPipeline(
        xgb_params=xgb_params,
        num_boost_round=num_boost_round,
        random_state=42,
        enable_categorical=False
    )

    print("Fitting final pipeline on full dataset (selected features, keeping NaN)...")
    final_pipeline.fit(X_selected, y)
    print("✅ Final model training completed!")

    # Extract feature importances if available
    if hasattr(final_pipeline, 'feature_importances_') and final_pipeline.feature_importances_ is not None:
        feature_importances = final_pipeline.feature_importances_
        
        # XGBSE returns dict {feature_name: importance}
        # Convert to aligned arrays matching selected_features order
        if isinstance(feature_importances, dict):
            importance_values = [feature_importances.get(f, 0.0) for f in selected_features]
        else:
            importance_values = feature_importances
        
        # Plot feature importances
        print("Generating feature importance plot...")
        importance_plot_path = os.path.join(args.trained_model, "feature_importances.html")
        importance_df = plot_feature_importance(
            selected_features,
            importance_values,
            importance_plot_path,
            top_n=min(30, len(selected_features))
        )

        # Save feature importances
        importance_csv_path = os.path.join(args.trained_model, "feature_importances.csv")
        importance_df.to_csv(importance_csv_path, index=False)

    # --- Step 3: Evaluate final model with cross-validation ---
    print("\n=== Step 3: Evaluating final model with survival metrics ===")

    def model_factory():
        """Factory function to create a new pipeline instance for CV"""
        return XGBSEPipeline(
            xgb_params=xgb_params,
            num_boost_round=num_boost_round,
            random_state=42,
            enable_categorical=False
        )

    # Evaluate on X_selected (same feature space as final model, keeping NaN)
    metrics_df, avg_metrics, std_metrics = evaluate_model_cv(
        X_selected,
        y,
        model_factory=model_factory,
        n_splits=args.cv_n_splits,
        n_repeats=args.cv_n_repeats,
        stratify_by_event=args.cv_stratify_by_event,
        verbose=True
    )

    # Log all metrics to MLflow
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
    
    # Move existing feature importance to explainability folder
    existing_importance_csv = os.path.join(args.trained_model, "feature_importances.csv")
    if os.path.exists(existing_importance_csv):
        import shutil
        new_importance_csv = os.path.join(explainability_dir, "feature_importances.csv")
        shutil.move(existing_importance_csv, new_importance_csv)
        mlflow.log_artifact(new_importance_csv, artifact_path="explainability")
    
    existing_importance_html = os.path.join(args.trained_model, "feature_importances.html")
    if os.path.exists(existing_importance_html):
        import shutil
        new_importance_html = os.path.join(explainability_dir, "feature_importances.html")
        shutil.move(existing_importance_html, new_importance_html)
        mlflow.log_artifact(new_importance_html, artifact_path="explainability")
    
    # Per-model SHAP and PFI are computed at ensemble evaluation (see 4_ensemble/evaluate_models.py
    # and 5_validation/shap_analysis.py).

    # --- Step 5: Save final pipeline ---
    print("\n=== Step 5: Saving final pipeline ===")

    # Save with joblib (for compatibility)
    pipeline_path = os.path.join(args.trained_model, "xgbse_stacked_weibull_pipeline.pkl")
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
    print(f"  - NO imputation applied (XGBoost handles NaN natively)")
    print(f"  - XGBSE Stacked Weibull with optimized hyperparameters")

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
    parser.add_argument("--cv_n_splits", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--cv_n_repeats", type=int, default=10,
                        help="Number of cross-validation repetitions")
    parser.add_argument("--n_optuna_trials", type=int, default=50,
                        help="Number of Optuna optimization trials")
    parser.add_argument("--cv_stratify_by_event", type=lambda x: x.lower() == 'true', default=True,
                        help="Whether to stratify CV folds by event status")
    args = parser.parse_args()
    main(args)
