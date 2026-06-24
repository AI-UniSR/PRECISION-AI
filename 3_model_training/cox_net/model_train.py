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
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# When running in Azure ML, the working directory is model_train/
# When running locally from cox-net/, we need to go up one level
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from survival_utils import (
    compute_permutation_importance,
    compute_shap_values,
    compute_survival_metrics,
    compute_survival_metrics_cv,
    evaluate_model_cv,
    log_metrics_to_mlflow,
    log_optuna_plots,
    plot_pfi_importance,
    plot_shap_summary,
)


def plot_coefficients_interactive(coefs, n_highlight=30, height=800, width=1000, best_alpha=None):
    """
    Plot regularization path for coefficients with interactive features.

    Parameters
    ----------
    coefs : pd.DataFrame
        Coefficient matrix: index = feature names, columns = alphas.
    n_highlight : int
        Number of top coefficients (by |coef| at min alpha) to color distinctly.
    height, width : int
        Plot dimensions.
    best_alpha : float, optional
        If provided, a vertical line will mark this alpha and features with nonzero
        coefficients at this alpha will be emphasized in the legend and opacity.
    """
    fig = go.Figure()
    alphas = coefs.columns.astype(float)
    alpha_min = alphas.min()

    # Determine highlight features (top by absolute value at smallest alpha)
    top_coefs = coefs.loc[:, alpha_min].abs().sort_values().tail(n_highlight).index.tolist()
    highlight_colors = px.colors.qualitative.Plotly
    non_highlight_color = "rgba(180, 180, 180, 0.3)"

    # Determine active features at best_alpha, if provided
    active_features = []
    if best_alpha is not None:
        closest_alpha = alphas[np.argmin(np.abs(alphas - best_alpha))]
        active_features = coefs.index[coefs[closest_alpha] != 0].tolist()
    else:
        closest_alpha = None

    # --- Add non-highlighted traces ---
    for idx, row in coefs.iterrows():
        color = non_highlight_color
        width_line = 1.2
        opacity_line = 0.3
        if best_alpha is not None and idx in active_features:
            color = "rgba(100,100,100,0.9)"
            width_line = 2
            opacity_line = 1
        fig.add_trace(go.Scatter(
            x=alphas,
            y=row.values,
            mode="lines",
            name=idx,
            line=dict(color=color, width=width_line),
            opacity=opacity_line,
            hoverinfo="text",
            hovertext=[f"{idx}: {v:.4f}" for v in row.values],
            hoverlabel=dict(namelength=-1),
        ))

    # --- Add highlighted traces (top |coef| ones) ---
    for i, idx in enumerate(top_coefs):
        color_idx = i % len(highlight_colors)
        fig.add_trace(go.Scatter(
            x=alphas,
            y=coefs.loc[idx].values,
            mode="lines+markers",
            name=f"⭐ {idx}" if idx in active_features else idx,
            line=dict(color=highlight_colors[color_idx], width=3),
            marker=dict(size=6),
            hoverinfo="text",
            hovertext=[f"{idx}: {v:.4f}" for v in coefs.loc[idx].values],
            hoverlabel=dict(namelength=-1)
        ))

    # --- Add vertical line for best_alpha ---
    shapes = []
    annotations = []
    if best_alpha is not None:
        shapes.append(dict(
            type="line",
            x0=closest_alpha,
            x1=closest_alpha,
            y0=coefs.min().min(),
            y1=coefs.max().max(),
            line=dict(color="black", width=2.5, dash="dash")
        ))
        annotations.append(dict(
            x=closest_alpha,
            y=coefs.max().max(),
            text=f"Best α = {closest_alpha:.3g}",
            showarrow=False,
            yshift=15,
            font=dict(size=12, color="black")
        ))

    fig.update_layout(
        title=dict(
            text="Regularization Path for Elastic Net Coefficients",
            x=0.5, xanchor="center", yanchor="top"
        ),
        xaxis=dict(
            title="Alpha (log scale)",
            type="log",
            showgrid=True,
            gridcolor="rgba(230,230,230,0.8)"
        ),
        yaxis=dict(
            title="Coefficient Value",
            showgrid=True,
            gridcolor="rgba(230,230,230,0.8)",
            zeroline=True,
            zerolinecolor="rgba(0,0,0,0.2)"
        ),
        height=height,
        width=width,
        hovermode="closest",
        paper_bgcolor="white",
        plot_bgcolor="white",
        shapes=shapes,
        annotations=annotations,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        margin=dict(l=60, r=60, t=80, b=160)
    )

    return fig, active_features


def cross_validate_alpha_path(X, y, n_alphas, l1_ratio, n_splits=5, n_repeats=1, stratify_by_event=True, verbose=False):
    """
    Fit CoxNet model and cross-validate over the alpha regularization path.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : structured array
        Survival target (event, time)
    n_alphas : int
        Number of alphas to generate in regularization path
    l1_ratio : float
        Elastic net mixing parameter (0 = Ridge, 1 = Lasso)
    n_splits : int
        Number of cross-validation folds
    n_repeats : int
        Number of CV repetitions
    stratify_by_event : bool
        If True, stratify folds by event status
    verbose : bool
        Whether to print fold-level details
        
    Returns
    -------
    alphas : np.ndarray
        Array of alpha values in the regularization path
    alpha_summary : pd.DataFrame
        Summary with columns: alpha, mean_cindex, std_cindex
    coefs : pd.DataFrame
        Coefficient matrix (features x alphas)
    """
    # Initial fit to get alpha path
    initial_pipe = make_pipeline(
        IterativeImputer(random_state=42, max_iter=10),
        RobustScaler(),
        CoxnetSurvivalAnalysis(n_alphas=n_alphas, l1_ratio=l1_ratio, max_iter=10000, fit_baseline_model=True)
    )
    initial_pipe.fit(X, y)
    alphas = initial_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    coefs = pd.DataFrame(
        initial_pipe.named_steps["coxnetsurvivalanalysis"].coef_,
        index=X.columns,
        columns=np.round(alphas, 5)
    )
    
    # Cross-validation over alpha path with stratification
    if stratify_by_event:
        if n_repeats > 1:
            from sklearn.model_selection import RepeatedStratifiedKFold
            cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
            splitter = cv.split(X, y["event"])
        else:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splitter = cv.split(X, y["event"])
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splitter = cv.split(X)
    
    alpha_results = {alpha: [] for alpha in alphas}
    pipeline = make_pipeline(
        IterativeImputer(random_state=42, max_iter=10),
        RobustScaler(),
        CoxnetSurvivalAnalysis(alphas=alphas, l1_ratio=l1_ratio, max_iter=10000, fit_baseline_model=True)
    )
    
    total_folds = n_splits * n_repeats
    for fold, (train_idx, test_idx) in enumerate(splitter):
        if verbose:
            print(f"  Fold {fold+1}/{total_folds}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        pipeline.fit(X_train, y_train)
        coefs_fold = pipeline.named_steps["coxnetsurvivalanalysis"].coef_.copy()
        # Ensure we only iterate over the actual number of alphas generated
        n_alphas_actual = coefs_fold.shape[1]
        for i in range(min(len(alphas), n_alphas_actual)):
            alpha = alphas[i]
            pipeline.named_steps["coxnetsurvivalanalysis"].coef_ = coefs_fold[:, i].reshape(-1, 1)
            risk = pipeline.predict(X_test)
            cindex = concordance_index_censored(y_test["event"], y_test["time"], risk)[0]
            alpha_results[alpha].append(cindex)
            if verbose:
                print(f"    Alpha={alpha:.6f} → C-index={cindex:.4f}")
    
    # Summarize results
    alpha_summary = pd.DataFrame([
        {"alpha": alpha, "mean_cindex": np.mean(scores), "std_cindex": np.std(scores)}
        for alpha, scores in alpha_results.items()
    ])
    
    return alphas, alpha_summary, coefs


def objective(trial, X, y, cv_n_splits, cv_n_repeats, cv_stratify_by_event):
    """
    Optuna objective function to tune n_alphas and l1_ratio.
    Uses cross-validation to find the best alpha in the regularization path,
    and returns the best mean C-index across alphas.
    """
    # Suggest hyperparameters
    n_alphas = trial.suggest_int("n_alphas", 5, 50)
    l1_ratio = trial.suggest_float("l1_ratio", 0.1, 1.0)
    
    print(f"\n[Trial {trial.number}] Testing n_alphas={n_alphas}, l1_ratio={l1_ratio:.3f}")
    
    # Perform cross-validation over alpha path
    _, alpha_summary, _ = cross_validate_alpha_path(
        X, y, n_alphas, l1_ratio, 
        n_splits=cv_n_splits, 
        n_repeats=cv_n_repeats,
        stratify_by_event=cv_stratify_by_event,
        verbose=False
    )
    
    # Return best C-index across alphas
    best_cindex = alpha_summary["mean_cindex"].max()
    print(f"[Trial {trial.number}] Best C-index across alphas: {best_cindex:.4f}")
    
    return best_cindex


def main(args):
    print("=== Starting Cox Elastic-Net model training with Optuna ===")
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

    # --- Optuna hyperparameter optimization ---
    print("\n=== Running Optuna optimization for n_alphas and l1_ratio ===")
    study = optuna.create_study(direction="maximize", study_name="cox_elasticnet_tuning")
    study.optimize(
        lambda trial: objective(trial, X, y, args.cv_n_splits, args.cv_n_repeats, args.cv_stratify_by_event), 
        n_trials=args.n_optuna_trials, 
        show_progress_bar=True
    )
    
    print(f"\n✅ Optimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best C-index: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    best_n_alphas = study.best_params["n_alphas"]
    best_l1_ratio = study.best_params["l1_ratio"]
    
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

    # --- Initial fit to get alpha path with best hyperparameters ---
    print(f"\nFitting model with best hyperparameters: n_alphas={best_n_alphas}, l1_ratio={best_l1_ratio:.3f}")

    alphas, alpha_summary, coefs = cross_validate_alpha_path(
        X, y, 
        n_alphas=best_n_alphas, 
        l1_ratio=best_l1_ratio, 
        n_splits=args.cv_n_splits,
        n_repeats=args.cv_n_repeats,
        stratify_by_event=args.cv_stratify_by_event,
        verbose=True
    )
    print(f"Extracted {len(alphas)} alphas from initial fit.")

    # --- Cross-validation results ---
    print("\n=== Cross-validated results ===")
    print(alpha_summary)
    summary_path = os.path.join(args.trained_model, "alpha_summary.csv")
    alpha_summary.to_csv(summary_path, index=False)
    mlflow.log_artifact(summary_path, artifact_path="cv_results")

    # Find best performance model
    best_idx = alpha_summary["mean_cindex"].idxmax()
    best_row = alpha_summary.iloc[best_idx]
    alpha_max_cindex = best_row["alpha"]
    best_cindex = best_row["mean_cindex"]
    best_std = best_row["std_cindex"]
    
    # ========================================================================
    # ALPHA SELECTION STRATEGY: Choose between best vs parsimonious model
    # ========================================================================
    # OPTION 1: Use best alpha directly (maximum C-index, no parsimony)
    # OPTION 2 (CURRENTLY ACTIVE): Use 0.25-SE Rule for parsimony
    # ========================================================================
    USE_BEST_ALPHA = False  # Set to True to disable parsimony rule
    
    if USE_BEST_ALPHA:
        # Use alpha with maximum C-index (no parsimony penalty)
        best_alpha = alpha_max_cindex
        print(f"\n=== Alpha Selection: Best Performance (No Parsimony Rule) ===")
        print(f"Selected alpha: {best_alpha:.6f} (C-index={best_cindex:.4f} ± {best_std:.4f})")
        
        # Log to MLflow
        mlflow.log_param("alpha_selection_rule", "best")
        mlflow.log_param("best_alpha", best_alpha)
        mlflow.log_metric("cv_best_cindex", best_cindex)
        mlflow.log_metric("cv_best_std", best_std)
    
    else:
        # ====================================================================
        # Use 0.25-SE Rule for parsimony (CURRENTLY ACTIVE)
        # ====================================================================
        # Reference: Adapted from Hastie, Tibshirani, Friedman - "The Elements of Statistical Learning" (2009)
        # The 0.25-SE rule selects more parsimonious models, balancing performance and simplicity.
        # It's more stringent than 0.5-SE, providing better regularization and model simplicity.
        
        # Calculate 0.25-SE threshold
        threshold_025se = best_cindex - (0.25 * best_std)
        
        # Find most parsimonious model within 0.25-SE (largest alpha >= threshold)
        # Note: larger alpha = stronger regularization = fewer features = simpler model
        candidates = alpha_summary[alpha_summary["mean_cindex"] >= threshold_025se]
        alpha_025se_row = candidates.loc[candidates["alpha"].idxmax()]
        alpha_025se = alpha_025se_row["alpha"]
        
        print(f"\n=== Alpha Selection via 0.25-SE Rule (Higher Parsimony) ===")
        print(f"Best alpha (max C-index): {alpha_max_cindex:.6f} (C-index={best_cindex:.4f} ± {best_std:.4f})")
        print(f"0.25-SE threshold: {threshold_025se:.4f}")
        print(f"Selected alpha (0.25-SE rule): {alpha_025se:.6f} (C-index={alpha_025se_row['mean_cindex']:.4f} ± {alpha_025se_row['std_cindex']:.4f})")
        
        # Calculate regularization increase
        reg_increase = ((alpha_025se / alpha_max_cindex) - 1) * 100 if alpha_max_cindex > 0 else 0
        print(f"→ Using {reg_increase:+.1f}% regularization vs max C-index model (higher parsimony)")
        
        # Use 0.25-SE alpha for final model
        best_alpha = alpha_025se
        
        # Log to MLflow
        mlflow.log_param("alpha_selection_rule", "0.25SE")
        mlflow.log_param("alpha_max_cindex", alpha_max_cindex)
        mlflow.log_param("alpha_025se", alpha_025se)
        mlflow.log_param("best_alpha", best_alpha)
        mlflow.log_metric("cv_max_cindex", best_cindex)
        mlflow.log_metric("cv_025se_cindex", alpha_025se_row["mean_cindex"])
        mlflow.log_metric("threshold_025se", threshold_025se)
        mlflow.log_metric("regularization_increase_pct", reg_increase)

    # --- Plot coefficient paths with selected alpha highlighted ---
    fig_interactive, active_features = plot_coefficients_interactive(coefs, best_alpha=best_alpha)
    plotly_html_path = os.path.join(args.trained_model, "coef_path_interactive.html")
    pio.write_html(fig_interactive, file=plotly_html_path, include_plotlyjs="cdn", full_html=True)
    mlflow.log_artifact(plotly_html_path, artifact_path="plots")
    print(f"Logged interactive coefficient path: {plotly_html_path}")

    # --- Final fit ---
    final_pipe = make_pipeline(
        IterativeImputer(random_state=42, max_iter=10),
        RobustScaler(),
        CoxnetSurvivalAnalysis(alphas=[best_alpha], l1_ratio=best_l1_ratio, max_iter=1000, fit_baseline_model=True)
    )
    final_pipe.fit(X, y)
    print("\n✅ Final model refit completed on full dataset.")
    
    # --- Evaluate final model with comprehensive metrics ---
    print("\n=== Evaluating final model with survival metrics ===")
    
    # Define model factory for cross-validation
    def model_factory():
        return make_pipeline(
            IterativeImputer(random_state=42, max_iter=10),
            RobustScaler(),
            CoxnetSurvivalAnalysis(alphas=[best_alpha], l1_ratio=best_l1_ratio, max_iter=1000, fit_baseline_model=True)
        )
    
    # Use utility function for comprehensive evaluation
    metrics_df, avg_metrics, std_metrics = evaluate_model_cv(
        X, y, 
        model_factory=model_factory,
        n_splits=args.cv_n_splits,
        n_repeats=args.cv_n_repeats,
        stratify_by_event=args.cv_stratify_by_event,
        verbose=True
    )
    
    # Log all metrics to MLflow
    log_metrics_to_mlflow(metrics_df, avg_metrics, std_metrics, args.trained_model, prefix="cv")
    
    # Note: CV metrics JSON will be saved after retraining on selected features
    # (the retrained model is the final one used for inference)

    # --- Extract and log all features with non-zero coefficients ---
    print("\n=== Extracting features with non-zero coefficients ===")
    final_coefs = final_pipe.named_steps["coxnetsurvivalanalysis"].coef_.flatten()
    
    # Create comprehensive table with all non-zero coefficients
    SELECTION_THRESHOLD = 1e-5  # New threshold for feature selection
    coef_table = pd.DataFrame({
        'feature': X.columns,
        'coefficient': final_coefs,
        'abs_coefficient': np.abs(final_coefs)
    })
    
    # Filter for non-zero coefficients and sort by absolute value (descending)
    coef_table = coef_table[coef_table['coefficient'] != 0].copy()
    coef_table = coef_table.sort_values('abs_coefficient', ascending=False).reset_index(drop=True)
    
    # Add selection flag based on new threshold
    coef_table['selected'] = coef_table['abs_coefficient'] >= SELECTION_THRESHOLD
    
    # Log summary statistics
    n_nonzero = len(coef_table)
    n_selected = coef_table['selected'].sum()
    print(f"Features with non-zero coefficients: {n_nonzero} out of {len(X.columns)}")
    print(f"Features selected (|coef| >= {SELECTION_THRESHOLD}): {n_selected}")
    print(f"\nTop 10 features by |coefficient|:")
    print(coef_table[['feature', 'coefficient', 'selected']].head(10).to_string(index=False))
    
    # Save comprehensive coefficient table
    coef_table_path = os.path.join(args.trained_model, "coefficient_table.csv")
    coef_table.to_csv(coef_table_path, index=False)
    mlflow.log_artifact(coef_table_path, artifact_path="feature_selection")
    print(f"\n✓ Saved coefficient table to {coef_table_path}")
    
    # Extract selected features based on new threshold
    selected_features = coef_table[coef_table['selected']]['feature'].tolist()
    print(f"\nSelected {len(selected_features)} features with |coef| >= {SELECTION_THRESHOLD}")
    
    # Save selected features list (for backward compatibility)
    selected_features_path = os.path.join(args.trained_model, "selected_features.txt")
    with open(selected_features_path, 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    mlflow.log_artifact(selected_features_path, artifact_path="feature_selection")
    
    # Log metrics
    mlflow.log_param("n_features_nonzero", n_nonzero)
    mlflow.log_param("n_features_selected", n_selected)
    mlflow.log_param("selection_threshold", SELECTION_THRESHOLD)
    mlflow.log_metric("feature_selection_rate", n_selected / len(X.columns) if len(X.columns) > 0 else 0)
    
    # --- Retrain Cox model without penalty on selected features ---
    print("\n=== Retraining Cox model without penalty on selected features ===")
    print(f"  Using {len(selected_features)} selected features")
    
    # Create subset of data with only selected features
    X_selected = X[selected_features]
    
    # Build new pipeline with CoxPH (no penalty)
    retrained_pipe = make_pipeline(
        IterativeImputer(random_state=42, max_iter=10),
        RobustScaler(),
        CoxPHSurvivalAnalysis(alpha=0.0)  # No regularization
    )
    
    retrained_pipe.fit(X_selected, y)
    print("  ✓ Retrained CoxPH model (no penalty) on selected features")
    
    # Evaluate retrained model with CV
    print("\n  Evaluating retrained model with CV...")
    def retrained_model_factory():
        return make_pipeline(
            IterativeImputer(random_state=42, max_iter=10),
            RobustScaler(),
            CoxPHSurvivalAnalysis(alpha=0.0)
        )
    
    retrained_metrics_df, retrained_avg_metrics, retrained_std_metrics = evaluate_model_cv(
        X_selected, y,
        model_factory=retrained_model_factory,
        n_splits=args.cv_n_splits,
        n_repeats=args.cv_n_repeats,
        stratify_by_event=args.cv_stratify_by_event,
        verbose=False
    )
    
    print(f"  Retrained model CV C-index: {retrained_avg_metrics['c_index']:.4f} ± {retrained_std_metrics['c_index']:.4f}")
    
    # Log retrained model metrics
    mlflow.log_metric("retrained_cv_cindex_mean", retrained_avg_metrics['c_index'])
    mlflow.log_metric("retrained_cv_cindex_std", retrained_std_metrics['c_index'])
    
    # Save retrained CV metrics
    retrained_cv_metrics_json = {
        "cv_metrics_per_fold": retrained_metrics_df.to_dict(orient='records'),
        "cv_metrics_mean": retrained_avg_metrics,
        "cv_metrics_std": retrained_std_metrics
    }
    retrained_cv_metrics_path = os.path.join(args.trained_model, "cv_metrics.json")
    with open(retrained_cv_metrics_path, 'w') as f:
        json.dump(retrained_cv_metrics_json, f, indent=2)
    mlflow.log_artifact(retrained_cv_metrics_path, artifact_path="metrics")
    print(f"  ✓ Saved retrained model CV metrics to {retrained_cv_metrics_path}")

    # --- Generate explainability artifacts (on retrained model) ---
    print("\n=== Generating explainability artifacts (retrained model) ===")
    explainability_dir = os.path.join(args.trained_model, "explainability")
    os.makedirs(explainability_dir, exist_ok=True)
    
    # Per-model SHAP and PFI are computed at ensemble evaluation (see 4_ensemble/evaluate_models.py
    # and 5_validation/shap_analysis.py).

    # --- Save retrained model (this is the model to use for inference) ---
    print("\n=== Saving retrained pipeline ===")
    
    # Save retrained model with joblib (for compatibility with evaluation step)
    pipeline_path = os.path.join(args.trained_model, "cox_pipeline.pkl")
    joblib.dump(retrained_pipe, pipeline_path)
    print(f"Saved retrained pipeline to {pipeline_path}")
    
    # Log model with MLflow
    mlflow.sklearn.log_model(
        sk_model=retrained_pipe,
        artifact_path='model',
    )

    mlflow.end_run()
    print("\n=== Training completed successfully. Artifacts logged to MLflow ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument("--trained_model", type=str, required=True)
    parser.add_argument("--cv_n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--cv_n_repeats", type=int, default=10, help="Number of CV repetitions")
    parser.add_argument("--cv_stratify_by_event", type=lambda x: x.lower() == 'true', default=True,
                        help="Stratify CV folds by event status")
    parser.add_argument("--n_optuna_trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()
    main(args)
