"""
Utility functions for survival analysis model training.
Provides reusable components for metrics computation, visualization, and optimization.
"""

import os
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.inspection import permutation_importance
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    brier_score,
    cumulative_dynamic_auc,
    integrated_brier_score
)
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_parallel_coordinate,
    plot_contour
)

# Try importing SHAP - it's optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")


def plot_time_dependent_auc(auc_times, auc_scores, auc_mean, output_path, title="Time-Dependent AUC"):
    """
    Plot time-dependent AUC scores across different time points with mean line.
    
    Parameters
    ----------
    auc_times : np.ndarray
        Time points where AUC was evaluated
    auc_scores : np.ndarray
        AUC scores at each time point
    auc_mean : float
        Mean AUC across all time points
    output_path : str
        Path to save HTML plot
    title : str
        Plot title
    """
    fig = go.Figure()
    
    # Plot AUC scores at each time point
    fig.add_trace(go.Scatter(
        x=auc_times,
        y=auc_scores,
        mode='lines+markers',
        name='AUC',
        line=dict(color='steelblue', width=3),
        marker=dict(size=10, color='steelblue', symbol='circle')
    ))
    
    # Add horizontal line for mean AUC
    fig.add_hline(
        y=auc_mean,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Mean AUC = {auc_mean:.3f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (days)",
        yaxis_title="AUC",
        yaxis_range=[0, 1],
        height=500,
        width=800,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    pio.write_html(fig, file=output_path, include_plotlyjs="cdn", full_html=True)
    print(f"Saved time-dependent AUC plot to {output_path}")


def compute_survival_metrics(y_train, y_test, risk_scores, pipeline, X_test, times=None):
    """
    Compute comprehensive survival analysis metrics.
    
    Parameters
    ----------
    y_train : structured array
        Training survival data (event, time)
    y_test : structured array
        Test survival data (event, time)
    risk_scores : np.ndarray
        Predicted risk scores for test set
    pipeline : sklearn.Pipeline
        Fitted survival model pipeline
    X_test : pd.DataFrame
        Test features
    times : np.ndarray, optional
        Time points for time-dependent metrics
        
    Returns
    -------
    dict
        Dictionary of metric names and values
    """
    metrics = {}
    
    # Concordance index (C-index)
    # Measures discrimination - how well the model ranks patients by risk
    cindex_result = concordance_index_censored(y_test["event"], y_test["time"], risk_scores)
    metrics["c_index"] = cindex_result[0]
    
    # Try to compute time-dependent metrics if survival function is available
    try:
        # Get survival functions
        surv_funcs = pipeline.predict_survival_function(X_test)
        
        # Define time points if not provided
        if times is None:
            times = np.percentile(y_train["time"][y_train["event"]], [25, 50, 75])
        
        # Integrated Brier Score (IBS)
        # Measures prediction error over time (lower is better)
        times_brier = np.percentile(y_train["time"], np.linspace(5, 95, 10))
        preds = np.asarray([[fn(t) for t in times_brier] for fn in surv_funcs])
        ibs = integrated_brier_score(y_train, y_test, preds, times_brier)
        metrics["integrated_brier_score"] = ibs
        
        # Time-dependent AUC
        # Discrimination at specific time points
        try:
            auc_times = times
            auc_result, auc_mean = cumulative_dynamic_auc(y_train, y_test, risk_scores, auc_times)
            for i, t in enumerate(auc_times):
                metrics[f"auc_at_{int(t)}"] = auc_result[i]
            metrics["mean_auc"] = auc_mean
        except Exception as e:
            print(f"  Warning: Could not compute AUC: {e}")
            
    except (AttributeError, Exception) as e:
        print(f"  Note: Time-dependent metrics not available: {e}")
    
    return metrics


def log_optuna_plots(study, output_dir, artifact_path="optimization"):
    """
    Generate and log Optuna visualization plots to MLflow.
    
    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study
    output_dir : str
        Directory to save plot files
    artifact_path : str
        MLflow artifact path for plots
        
    Returns
    -------
    list
        List of generated plot file paths
    """
    plot_files = []
    
    # 1. Optimization History - shows objective value progression
    try:
        fig = plot_optimization_history(study)
        filepath = os.path.join(output_dir, "optuna_optimization_history.html")
        fig.write_html(filepath)
        mlflow.log_artifact(filepath, artifact_path=artifact_path)
        plot_files.append(filepath)
        print(f"  ✓ Logged optimization history")
    except Exception as e:
        print(f"  Warning: Could not create optimization history plot: {e}")
    
    # 2. Parameter Importances - shows which hyperparameters matter most
    try:
        fig = plot_param_importances(study)
        filepath = os.path.join(output_dir, "optuna_param_importances.html")
        fig.write_html(filepath)
        mlflow.log_artifact(filepath, artifact_path=artifact_path)
        plot_files.append(filepath)
        print(f"  ✓ Logged parameter importances")
    except Exception as e:
        print(f"  Warning: Could not create param importances plot: {e}")
    
    # 3. Slice Plot - shows how each parameter affects objective
    try:
        fig = plot_slice(study)
        filepath = os.path.join(output_dir, "optuna_slice_plot.html")
        fig.write_html(filepath)
        mlflow.log_artifact(filepath, artifact_path=artifact_path)
        plot_files.append(filepath)
        print(f"  ✓ Logged slice plot")
    except Exception as e:
        print(f"  Warning: Could not create slice plot: {e}")
    
    # 4. Parallel Coordinate Plot - shows parameter combinations
    try:
        fig = plot_parallel_coordinate(study)
        filepath = os.path.join(output_dir, "optuna_parallel_coordinate.html")
        fig.write_html(filepath)
        mlflow.log_artifact(filepath, artifact_path=artifact_path)
        plot_files.append(filepath)
        print(f"  ✓ Logged parallel coordinate plot")
    except Exception as e:
        print(f"  Warning: Could not create parallel coordinate plot: {e}")
    
    # 5. Contour Plot - shows parameter interaction effects
    try:
        fig = plot_contour(study)
        filepath = os.path.join(output_dir, "optuna_contour_plot.html")
        fig.write_html(filepath)
        mlflow.log_artifact(filepath, artifact_path=artifact_path)
        plot_files.append(filepath)
        print(f"  ✓ Logged contour plot")
    except Exception as e:
        print(f"  Warning: Could not create contour plot: {e}")
    
    return plot_files


def evaluate_model_cv(X, y, model_factory, n_splits=5, n_repeats=1, stratify_by_event=True, verbose=True):
    """
    Evaluate a survival model using cross-validation with comprehensive metrics.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : structured array
        Survival data (event, time)
    model_factory : callable
        Function that returns a new fitted model instance
    n_splits : int
        Number of CV folds
    n_repeats : int
        Number of CV repetitions (1 = no repetition)
    stratify_by_event : bool
        If True, stratify folds by event status (balanced event/censored distribution)
    verbose : bool
        Print progress
        
    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for each fold
    dict
        Dictionary of mean metrics
    dict
        Dictionary of std metrics
    """
    # Select appropriate CV strategy
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
    
    all_metrics = []
    total_folds = n_splits * n_repeats
    
    for fold, (train_idx, test_idx) in enumerate(splitter):
        if verbose:
            print(f"Fold {fold+1}/{total_folds}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Get a new model instance and fit
        model = model_factory()
        model.fit(X_train, y_train)
        risk = model.predict(X_test)
        
        # Compute comprehensive metrics
        fold_metrics = compute_survival_metrics(y_train, y_test, risk, model, X_test)
        fold_metrics["fold"] = fold + 1
        all_metrics.append(fold_metrics)
        
        if verbose:
            print(f"  C-index: {fold_metrics['c_index']:.4f}")
            if "integrated_brier_score" in fold_metrics:
                print(f"  IBS: {fold_metrics['integrated_brier_score']:.4f}")
    
    metrics_df = pd.DataFrame(all_metrics)
    avg_metrics = metrics_df.drop(columns=["fold"]).mean().to_dict()
    std_metrics = metrics_df.drop(columns=["fold"]).std().to_dict()
    
    return metrics_df, avg_metrics, std_metrics


def log_metrics_to_mlflow(metrics_df, avg_metrics, std_metrics, output_dir, prefix="cv"):
    """
    Log survival metrics to MLflow.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with metrics per fold
    avg_metrics : dict
        Dictionary of mean metrics
    std_metrics : dict
        Dictionary of std metrics
    output_dir : str
        Directory to save CSV file
    prefix : str
        Prefix for metric names in MLflow
    """
    print(f"\n=== Average {prefix} metrics ===")
    for metric_name, value in avg_metrics.items():
        std_value = std_metrics.get(metric_name, 0)
        print(f"{metric_name}: {value:.4f} ± {std_value:.4f}")
        mlflow.log_metric(f"{prefix}_{metric_name}_mean", value)
        mlflow.log_metric(f"{prefix}_{metric_name}_std", std_value)
    
    # Save metrics to CSV
    metrics_path = os.path.join(output_dir, f"{prefix}_survival_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    mlflow.log_artifact(metrics_path, artifact_path="metrics")
    print(f"Saved metrics to {metrics_path}")


def compute_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    """
    Compute permutation feature importance for survival models.
    
    Parameters
    ----------
    model : fitted model
        Trained survival model with predict method
    X : pd.DataFrame
        Feature matrix
    y : structured array
        Survival data (event, time)
    n_repeats : int
        Number of times to permute each feature
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with features and their importance scores (sorted)
    """
    print(f"Computing permutation importance ({n_repeats} repeats)...")
    
    # Custom scoring function for survival models (C-index)
    def survival_score(model, X, y):
        try:
            risk = model.predict(X)
            return concordance_index_censored(y["event"], y["time"], risk)[0]
        except:
            return 0.0
    
    try:
        perm_importance = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=survival_score,
            n_jobs=-1
        )
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        print(f"✅ Permutation importance computed for {len(importance_df)} features")
        return importance_df
        
    except Exception as e:
        print(f"Warning: Could not compute permutation importance: {e}")
        return pd.DataFrame(columns=['feature', 'importance_mean', 'importance_std'])


def plot_pfi_importance(importance_df, output_path, top_n=30, title="Permutation Feature Importance"):
    """
    Plot permutation feature importance as horizontal bar chart.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with columns: feature, importance_mean, importance_std
    output_path : str
        Path to save HTML plot
    top_n : int
        Number of top features to display
    title : str
        Plot title
    """
    if len(importance_df) == 0:
        print("Warning: No importance data to plot")
        return
    
    # Take top N features
    df_top = importance_df.head(top_n)
    
    # Create interactive bar plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_top['importance_mean'],
        y=df_top['feature'],
        orientation='h',
        error_x=dict(type='data', array=df_top['importance_std']),
        marker=dict(
            color=df_top['importance_mean'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=df_top['importance_mean'].round(4),
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score (C-index decrease)",
        yaxis_title="Feature",
        height=max(600, top_n * 20),
        yaxis=dict(autorange="reversed"),
        showlegend=False
    )
    
    pio.write_html(fig, file=output_path, include_plotlyjs="cdn", full_html=True)
    print(f"Saved PFI plot to {output_path}")


def compute_shap_values(model, X, model_type='tree', background_samples=100, random_state=42):
    """
    Compute SHAP values for model explainability.
    
    Parameters
    ----------
    model : fitted model
        Trained model (pipeline or estimator)
    X : pd.DataFrame
        Feature matrix
    model_type : str
        'tree' for tree-based models (TreeExplainer), 
        'kernel' for other models (KernelExplainer)
    background_samples : int
        Number of background samples for KernelExplainer
    random_state : int
        Random state for sampling
        
    Returns
    -------
    shap.Explanation or None
        SHAP explanation object, or None if SHAP unavailable
    """
    if not SHAP_AVAILABLE:
        print("Warning: SHAP not available, skipping SHAP computation")
        return None
    
    print(f"Computing SHAP values using {model_type} explainer...")
    
    try:
        # Extract the actual model from pipeline if needed
        if hasattr(model, 'named_steps'):
            # It's a pipeline - get the last step
            actual_model = model.steps[-1][1]
            # Transform X through all preprocessing steps
            X_transformed = X.copy()
            for name, step in model.steps[:-1]:
                X_transformed = step.transform(X_transformed)
            X_for_shap = X_transformed
            feature_names = X.columns if hasattr(X, 'columns') else None
        else:
            actual_model = model
            X_for_shap = X
            feature_names = X.columns if hasattr(X, 'columns') else None
        
        # Special handling for RandomSurvivalForest - try to use internal estimators
        if model_type == 'tree' and 'RandomSurvivalForest' in str(type(actual_model)):
            try:
                # RSF has a list of trees in estimators_ attribute
                if hasattr(actual_model, 'estimators_') and len(actual_model.estimators_) > 0:
                    print("  Attempting TreeExplainer with RSF estimators...")
                    # Use a subset of trees for speed (e.g., first 50 trees)
                    n_trees = min(50, len(actual_model.estimators_))
                    # Note: This might still fail, but worth trying
                    explainer = shap.TreeExplainer(actual_model.estimators_[:n_trees])
                    shap_values = explainer.shap_values(X_for_shap)
                    print(f"✅ SHAP values computed using TreeExplainer on RSF estimators")
                    
                    # Convert scalar base_values to array if needed
                    base_values = explainer.expected_value
                    if isinstance(base_values, (int, float)):
                        base_values = np.full(len(X_for_shap), base_values)
                    
                    if feature_names is not None:
                        explanation = shap.Explanation(
                            values=shap_values,
                            base_values=base_values,
                            data=X_for_shap,
                            feature_names=feature_names
                        )
                    else:
                        explanation = shap_values
                    return explanation
            except Exception as e:
                print(f"  RSF TreeExplainer attempt failed: {e}")
        
        # Choose explainer based on model type
        if model_type == 'tree':
            # For tree-based models (RSF, XGBoost)
            try:
                explainer = shap.TreeExplainer(actual_model)
                shap_values = explainer.shap_values(X_for_shap)
                print(f"✅ SHAP values computed using TreeExplainer")
                
                # Create explanation object
                if feature_names is not None:
                    # Convert scalar base_values to array if needed
                    base_values = explainer.expected_value
                    if isinstance(base_values, (int, float)):
                        base_values = np.full(len(X_for_shap), base_values)
                    
                    explanation = shap.Explanation(
                        values=shap_values,
                        base_values=base_values,
                        data=X_for_shap,
                        feature_names=feature_names
                    )
                else:
                    explanation = shap_values
                    
                return explanation
                
            except Exception as e:
                print(f"TreeExplainer failed: {e}, falling back to KernelExplainer")
                model_type = 'kernel'
        
        if model_type == 'kernel':
            # For Cox models or when TreeExplainer fails
            # Sample background data
            np.random.seed(random_state)
            background_idx = np.random.choice(
                len(X_for_shap), 
                size=min(background_samples, len(X_for_shap)), 
                replace=False
            )
            background = X_for_shap.iloc[background_idx] if hasattr(X_for_shap, 'iloc') else X_for_shap[background_idx]
            
            # Create prediction function that handles array→DataFrame conversion
            def predict_fn(x):
                # SHAP passes numpy arrays, but models expect DataFrame with column names
                if not isinstance(x, pd.DataFrame):
                    if feature_names is not None:
                        x = pd.DataFrame(x, columns=feature_names)
                    else:
                        x = pd.DataFrame(x)
                return model.predict(x)
            
            explainer = shap.KernelExplainer(predict_fn, background)
            
            # Compute SHAP for a subset (KernelExplainer is slow)
            # Reduced from 100 to 30 samples for much faster computation
            sample_size = min(30, len(X_for_shap))
            sample_idx = np.random.choice(len(X_for_shap), size=sample_size, replace=False)
            X_sample = X_for_shap.iloc[sample_idx] if hasattr(X_for_shap, 'iloc') else X_for_shap[sample_idx]
            
            print(f"  Computing SHAP for {sample_size} samples (this may take several minutes)...")
            shap_values = explainer.shap_values(X_sample)
            print(f"✅ SHAP values computed using KernelExplainer (on {sample_size} samples)")
            
            # Create explanation object
            if feature_names is not None:
                # Convert scalar base_values to array if needed
                base_values = explainer.expected_value
                if isinstance(base_values, (int, float)):
                    base_values = np.full(len(X_sample), base_values)
                
                explanation = shap.Explanation(
                    values=shap_values,
                    base_values=base_values,
                    data=X_sample,
                    feature_names=feature_names
                )
            else:
                explanation = shap_values
                
            return explanation
            
    except Exception as e:
        print(f"Warning: Could not compute SHAP values: {e}")
        return None


def plot_shap_summary(shap_explanation, output_path, plot_type='beeswarm', max_display=30):
    """
    Create SHAP summary plot (beeswarm or bar).
    
    Parameters
    ----------
    shap_explanation : shap.Explanation
        SHAP explanation object from compute_shap_values
    output_path : str
        Path to save the plot (PNG)
    plot_type : str
        'beeswarm' for detailed summary, 'bar' for mean importance
    max_display : int
        Maximum number of features to display
    """
    if not SHAP_AVAILABLE or shap_explanation is None:
        print("Warning: Cannot create SHAP plot (SHAP not available or no explanation)")
        return
    
    try:
        plt.figure(figsize=(10, max(8, max_display * 0.3)))
        
        if plot_type == 'beeswarm':
            shap.summary_plot(
                shap_explanation, 
                plot_type='dot',
                max_display=max_display,
                show=False
            )
        elif plot_type == 'bar':
            shap.summary_plot(
                shap_explanation,
                plot_type='bar',
                max_display=max_display,
                show=False
            )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP {plot_type} plot to {output_path}")
        
    except Exception as e:
        print(f"Warning: Could not create SHAP plot: {e}")
        plt.close()


def compute_survival_metrics_cv(y_train, y_test, risk_scores, pipeline, X_test, times=None):
    """
    Compute comprehensive survival metrics with robust error handling for CV.
    This is a wrapper around compute_survival_metrics that handles exceptions gracefully.
    
    Parameters
    ----------
    y_train : structured array
        Training survival data
    y_test : structured array
        Test survival data
    risk_scores : np.ndarray
        Predicted risk scores
    pipeline : model
        Fitted survival model
    X_test : pd.DataFrame
        Test features
    times : np.ndarray, optional
        Time points for time-dependent metrics
        
    Returns
    -------
    dict
        Dictionary of metrics (empty values for failed metrics)
    """
    metrics = {}
    
    # Always try C-index (most compatible)
    try:
        cindex_result = concordance_index_censored(y_test["event"], y_test["time"], risk_scores)
        metrics["c_index"] = cindex_result[0]
    except Exception as e:
        print(f"  Warning: Could not compute C-index: {e}")
        metrics["c_index"] = np.nan
    
    # Try Brier score
    try:
        if hasattr(pipeline, 'predict_survival_function'):
            surv_funcs = pipeline.predict_survival_function(X_test)
            times_brier = np.percentile(y_train["time"], np.linspace(5, 95, 10))
            preds = np.asarray([[fn(t) for t in times_brier] for fn in surv_funcs])
            ibs = integrated_brier_score(y_train, y_test, preds, times_brier)
            metrics["integrated_brier_score"] = ibs
    except Exception as e:
        print(f"  Note: Could not compute Brier score: {e}")
    
    # Try dynamic AUC
    try:
        if times is None:
            times = np.percentile(y_train["time"][y_train["event"]], [25, 50, 75])
        
        auc_result, auc_mean = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
        for i, t in enumerate(times):
            metrics[f"auc_at_{int(t)}"] = auc_result[i]
        metrics["mean_auc"] = auc_mean
    except Exception as e:
        print(f"  Note: Could not compute dynamic AUC: {e}")
    
    return metrics

