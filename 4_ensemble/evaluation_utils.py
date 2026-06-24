"""
Utility functions for model evaluation and reporting.

This module contains helper functions for:
- Temporal grid creation and temporal metrics computation
- Model loading and artifact management
- Metric visualization and reporting
- Feature selection comparison
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import mlflow

from sksurv.metrics import brier_score
from scipy.integrate import trapz


# Clinical time points for temporal validation (months)
CLINICAL_TIMEPOINTS_MONTHS = [6, 12, 15, 18, 24]


# ============================================================================
# TEMPORAL GRID AND METRICS
# ============================================================================

def create_temporal_grid(y_train, y_test=None):
    """
    Create unified temporal grid combining percentiles and clinical time points.
    Grid is constrained to valid follow-up range of test set if provided.
    
    IMPORTANT: All time values in y_train and y_test are assumed to be in MONTHS.
    
    Parameters
    ----------
    y_train : structured array
        Training survival data (time in MONTHS)
    y_test : structured array, optional
        Test survival data (time in MONTHS). If provided, grid will be constrained to test set follow-up time.
        
    Returns
    -------
    grid_months : np.ndarray
        Unified temporal grid in MONTHS (sorted, deduplicated)
    clinical_months : np.ndarray
        Clinical time points in MONTHS (filtered to valid range if y_test provided)
    """
    # Determine valid time range - use minimum of train and test max follow-up
    max_time_train = y_train["time"].max()
    min_time = y_train["time"].min()
    
    if y_test is not None:
        max_time_test = y_test["time"].max()
        min_time_test = y_test["time"].min()
        # Use minimum to ensure all timepoints are valid for test set
        max_time = min(max_time_train, max_time_test)
        print(f"  Max follow-up: train={max_time_train:.2f} months, test={max_time_test:.2f} months")
        print(f"  Using constrained max={max_time:.2f} months")
    else:
        max_time = max_time_train
        print(f"  Max follow-up: {max_time:.2f} months")
    
    # Filter clinical time points to valid range
    # Add small epsilon (0.1 month ~= 3 days) to avoid boundary issues
    clinical_months_all = np.array(CLINICAL_TIMEPOINTS_MONTHS, dtype=float)
    clinical_months = clinical_months_all[clinical_months_all < (max_time - 0.1)]
    
    print(f"  Clinical timepoints requested: {CLINICAL_TIMEPOINTS_MONTHS}")
    print(f"  Clinical timepoints available within follow-up: {clinical_months.tolist()} ({len(clinical_months)}/{len(CLINICAL_TIMEPOINTS_MONTHS)})")
    
    if len(clinical_months) == 0:
        print(f"  ⚠ Warning: No clinical timepoints within follow-up range (max={max_time:.1f} months)")
        # Fallback: use quartiles of available times
        valid_times = y_train["time"][(y_train["time"] >= min_time) & (y_train["time"] < max_time)]
        if len(valid_times) > 10:
            clinical_months = np.percentile(valid_times, [25, 50, 75])
            print(f"  → Using quartiles instead: {clinical_months.round(1)} months")
        else:
            clinical_months = np.array([max_time * 0.5])  # Use mid-point
            print(f"  → Using mid-point: {clinical_months[0]:.1f} months")
    
    # Get percentiles 5-95 from training data within valid range
    valid_train_times = y_train["time"][(y_train["time"] >= min_time) & (y_train["time"] < max_time)]
    
    if len(valid_train_times) > 10:
        percentile_months = np.percentile(valid_train_times, np.linspace(5, 95, 40))
    else:
        # Not enough data, use fewer percentiles
        percentile_months = np.percentile(valid_train_times, [10, 25, 50, 75, 90])
        print(f"  ⚠ Limited data for grid: using {len(percentile_months)} percentiles")
    
    # Combine and deduplicate, ensure within valid range
    all_points = np.concatenate([percentile_months, clinical_months])
    grid_months = np.unique(all_points)
    grid_months = grid_months[(grid_months >= min_time) & (grid_months < max_time)]
    grid_months = np.sort(grid_months)
    
    print(f"  Grid summary: {len(grid_months)} time points, range=[{grid_months.min():.2f}, {grid_months.max():.2f}] months")
    
    return grid_months, clinical_months


def compute_brier_scores_on_grid(model, X_test, y_train, y_test, grid_months):
    """
    Compute Brier scores at all grid time points.
    
    Parameters
    ----------
    model : object
        Trained survival model with predict_survival_function method
    X_test : pd.DataFrame
        Test features
    y_train : structured array
        Training survival data (time in MONTHS, for reference)
    y_test : structured array
        Test survival data (time in MONTHS)
    grid_months : np.ndarray
        Time points in MONTHS at which to evaluate Brier score
        
    Returns
    -------
    brier_scores : np.ndarray
        Brier score at each grid point, or None if computation fails
    """
    if not hasattr(model, 'predict_survival_function'):
        return None
    
    if len(grid_months) == 0:
        print("    Warning: Empty grid_months, cannot compute Brier scores")
        return None
    
    try:
        surv_funcs = model.predict_survival_function(X_test)
        preds = np.asarray([[fn(t) for t in grid_months] for fn in surv_funcs])
        
        brier_scores = np.zeros(len(grid_months))
        for i, t in enumerate(grid_months):
            brier_scores[i] = brier_score(y_train, y_test, preds[:, i], t)[1]
        
        return brier_scores
    except Exception as e:
        print(f"    Warning: Could not compute Brier scores: {e}")
        return None


def compute_ibs_intervals(grid_months, brier_scores):
    """
    Compute IBS over clinical intervals using trapezoidal integration.
    
    Parameters
    ----------
    grid_months : np.ndarray
        Temporal grid in MONTHS
    brier_scores : np.ndarray
        Brier scores at grid points
        
    Returns
    -------
    ibs_intervals : dict
        IBS for each clinical interval
    ibs_overall : float
        Overall IBS (0-24 months)
    """
    # Define clinical intervals (in months)
    intervals_months = [(0, 6), (6, 12), (12, 18), (18, 24)]
    ibs_intervals = {}
    
    for start_m, end_m in intervals_months:
        # Find grid points in interval
        mask = (grid_months >= start_m) & (grid_months <= end_m)
        interval_times = grid_months[mask]
        interval_brier = brier_scores[mask]
        
        # Require at least 3 points for reliable integration
        if len(interval_times) >= 3:
            ibs = trapz(interval_brier, interval_times) / (end_m - start_m)
            ibs_intervals[f"ibs_{start_m}_{end_m}m"] = ibs
        else:
            ibs_intervals[f"ibs_{start_m}_{end_m}m"] = np.nan
    
    # Overall IBS (0-24 months)
    mask_overall = (grid_months >= 0) & (grid_months <= 24)
    overall_times = grid_months[mask_overall]
    overall_brier = brier_scores[mask_overall]
    
    if len(overall_times) >= 3:
        ibs_overall = trapz(overall_brier, overall_times) / 24.0
    else:
        ibs_overall = np.nan
    
    return ibs_intervals, ibs_overall


# ============================================================================
# MODEL LOADING AND ARTIFACTS
# ============================================================================

def extract_feature_names_from_model(model):
    """
    Extract feature names from a fitted sklearn model/pipeline.
    
    Tries to get feature_names_in_ from various components:
    - Pipeline: checks first step (usually imputer/scaler)
    - Direct model: checks feature_names_in_ attribute
    - XGBSEPipeline: checks internal xgbse_model_ booster
    
    Parameters
    ----------
    model : sklearn model or pipeline
        Fitted model
        
    Returns
    -------
    list or None
        List of feature names, or None if not found
    """
    try:
        # Special case: XGBSEPipeline custom wrapper
        if hasattr(model, 'xgbse_model_'):
            xgbse = model.xgbse_model_
            # Try to get feature names from internal booster
            if hasattr(xgbse, 'feature_extractor'):
                fe = xgbse.feature_extractor
                if hasattr(fe, 'bst') and hasattr(fe.bst, 'feature_names'):
                    feature_names = fe.bst.feature_names
                    if feature_names:
                        return list(feature_names)
        
        # Check if it's a sklearn Pipeline
        if hasattr(model, 'named_steps'):
            # Try first step (usually imputer or scaler)
            first_step_name = list(model.named_steps.keys())[0]
            first_step = model.named_steps[first_step_name]
            if hasattr(first_step, 'feature_names_in_'):
                return list(first_step.feature_names_in_)
        
        # Check if model itself has feature_names_in_
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
            
    except Exception as e:
        print(f"    Warning: Could not extract feature names from model: {e}")
    
    return None


def load_model_and_metrics(model_dir, model_name):
    """
    Load a trained model and its CV metrics.
    
    Parameters
    ----------
    model_dir : str
        Path to model directory
    model_name : str
        Name of the model (for logging)
        
    Returns
    -------
    model : object
        Loaded model pipeline
    cv_metrics : dict
        CV metrics from training
    selected_features : list
        List of selected features
    """
    print(f"\nLoading {model_name} model...")
    print(f"  Looking in: {model_dir}")
    
    # Special case: DeepSurv models use directory structure with PyTorch artifacts
    if model_name == 'deepsurv':
        deepsurv_net = os.path.join(model_dir, 'deepsurv_net.pt')
        if os.path.exists(deepsurv_net):
            # Import DeepSurvWrapper (wrapper is duplicated in validate/ for portability)
            from deepsurv_wrapper import DeepSurvWrapper
            
            # Create wrapper and load artifacts
            model = DeepSurvWrapper(model_dir)
            model._load_artifacts()  # Explicitly load artifacts
            
            print(f"  ✓ Loaded DeepSurv model with wrapper")
            
            # DeepSurv uses training_metrics.json instead of cv_metrics.json
            cv_metrics_path = os.path.join(model_dir, "training_metrics.json")
            if os.path.exists(cv_metrics_path):
                with open(cv_metrics_path, 'r') as f:
                    training_data = json.load(f)
                # Extract CV metrics from training_metrics
                cv_metrics = {
                    'best_cv_mean_ci': training_data.get('best_cv_mean_ci', None),
                    'final_train_ci': training_data.get('training_history', {}).get('final_train_ci', None),
                    'final_val_ci': training_data.get('training_history', {}).get('final_val_ci', None)
                }
                print(f"  ✓ Loaded training metrics")
            else:
                print(f"  ⚠ No training metrics found")
                cv_metrics = {}
            
            # Load selected features
            features_path = os.path.join(model_dir, "selected_features.txt")
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    selected_features = [line.strip() for line in f if line.strip()]
                print(f"  ✓ Loaded {len(selected_features)} selected features")
            else:
                print(f"  ⚠ No selected features file found")
                selected_features = model.feature_names if model.feature_names else []
            
            return model, cv_metrics, selected_features
    
    # Standard case: scikit-survival models in .pkl files
    model_path = None
    for root, dirs, files in os.walk(model_dir):
        pkl_files = [f for f in files if f.endswith('.pkl')]
        if pkl_files:
            model_path = os.path.join(root, pkl_files[0])
            break
    
    if model_path is None:
        # List directory structure for debugging
        print(f"  ✗ No .pkl file found. Directory structure:")
        for root, dirs, files in os.walk(model_dir):
            level = root.replace(model_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for file in files[:10]:  # Show first 10 files
                print(f"{subindent}{file}")
            if len(files) > 10:
                print(f"{subindent}... and {len(files) - 10} more files")
        raise FileNotFoundError(
            f"No .pkl file found in {model_dir} or its subdirectories.\n"
            f"Expected files: cox_pipeline.pkl, rsf_pipeline.pkl, or xgboost_survival_pipeline.pkl"
        )
    
    model = joblib.load(model_path)
    print(f"  ✓ Loaded model from {model_path}")
    
    # Find the model directory (where pkl was found) for other artifacts
    model_artifact_dir = os.path.dirname(model_path)
    
    # Load CV metrics
    cv_metrics_path = os.path.join(model_artifact_dir, "cv_metrics.json")
    if os.path.exists(cv_metrics_path):
        with open(cv_metrics_path, 'r') as f:
            cv_metrics = json.load(f)
        print(f"  ✓ Loaded CV metrics")
    else:
        print(f"  ⚠ No CV metrics found at {cv_metrics_path}")
        cv_metrics = {}
    
    # Load selected features
    features_path = os.path.join(model_artifact_dir, "selected_features.txt")
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            selected_features = [line.strip() for line in f if line.strip()]
        print(f"  ✓ Loaded {len(selected_features)} selected features from file")
    else:
        print(f"  ⚠ No selected features file found at {features_path}")
        # Fallback: extract feature names from the model itself
        selected_features = extract_feature_names_from_model(model)
        if selected_features:
            print(f"  ✓ Extracted {len(selected_features)} features from model")
        else:
            print(f"  ⚠ Could not extract features from model, will use all features")
            selected_features = []
    
    return model, cv_metrics, selected_features


# ============================================================================
# VISUALIZATION AND REPORTING
# ============================================================================

def create_comparative_plots(test_results, comparison_dir):
    """
    Create comparative AUC and Brier score plots across all models.
    
    Parameters
    ----------
    test_results : dict
        Dict mapping model_name -> test_metrics (with _temporal_auc, _temporal_brier)
    comparison_dir : str
        Directory to save plots
    """
    print("\n=== Creating comparative temporal plots ===")
    
    # Extract temporal data
    models_with_temporal = {}
    for model_name, metrics in test_results.items():
        if '_temporal_auc' in metrics and metrics['_temporal_auc'] is not None:
            models_with_temporal[model_name] = {
                'auc': metrics['_temporal_auc'],
                'brier': metrics.get('_temporal_brier')
            }
    
    if not models_with_temporal:
        print("  Warning: No temporal metrics available for comparative plots")
        return
    
    clinical_months = CLINICAL_TIMEPOINTS_MONTHS
    
    # 1. Comparative AUC plot
    fig_auc = go.Figure()
    
    for model_name, data in models_with_temporal.items():
        fig_auc.add_trace(go.Scatter(
            x=clinical_months,
            y=data['auc'],
            mode='lines+markers',
            name=model_name.upper(),
            marker=dict(size=10),
            line=dict(width=2)
        ))
    
    fig_auc.update_layout(
        title="Comparative Time-Dependent AUC (Clinical Time Points)",
        xaxis_title="Time (months)",
        yaxis_title="AUC",
        yaxis_range=[0.4, 1.0],
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    auc_path = os.path.join(comparison_dir, "comparative_auc.html")
    pio.write_html(fig_auc, file=auc_path, include_plotlyjs="cdn", full_html=True)
    print(f"  ✓ Saved comparative AUC plot to {auc_path}")
    mlflow.log_artifact(auc_path, artifact_path="comparison")
    
    # 2. Comparative Brier score plot
    fig_brier = go.Figure()
    
    for model_name, data in models_with_temporal.items():
        if data['brier'] is not None:
            fig_brier.add_trace(go.Scatter(
                x=clinical_months,
                y=data['brier'],
                mode='lines+markers',
                name=model_name.upper(),
                marker=dict(size=10),
                line=dict(width=2)
            ))
    
    fig_brier.update_layout(
        title="Comparative Brier Score (Clinical Time Points)",
        xaxis_title="Time (months)",
        yaxis_title="Brier Score",
        yaxis_range=[0, 0.35],
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    brier_path = os.path.join(comparison_dir, "comparative_brier.html")
    pio.write_html(fig_brier, file=brier_path, include_plotlyjs="cdn", full_html=True)
    print(f"  ✓ Saved comparative Brier plot to {brier_path}")
    mlflow.log_artifact(brier_path, artifact_path="comparison")


def create_summary_table(test_results, comparison_dir):
    """
    Create comprehensive summary table with all metrics.
    
    Parameters
    ----------
    test_results : dict
        Dict mapping model_name -> test_metrics
    comparison_dir : str
        Directory to save table
    """
    print("\n=== Creating summary metrics table ===")
    
    # Hardcoded clinical timepoints in months
    CLINICAL_TIMEPOINTS = [6, 12, 15, 18, 24]
    
    rows = []
    for model_name, metrics in test_results.items():
        row = {'Model': model_name.upper()}
        
        # Global metrics
        row['C-index'] = metrics.get('c_index', np.nan)
        row['Mean AUC'] = metrics.get('mean_auc', np.nan)
        
        # Time-dependent AUC at clinical timepoints (months)
        for timepoint in CLINICAL_TIMEPOINTS:
            row[f'AUC_{timepoint}m'] = metrics.get(f'auc_{timepoint}m', np.nan)
        
        # Time-dependent Brier scores at clinical timepoints (months)
        for timepoint in CLINICAL_TIMEPOINTS:
            row[f'Brier_{timepoint}m'] = metrics.get(f'brier_{timepoint}m', np.nan)
        
        # IBS intervals
        row['IBS 0-6m'] = metrics.get('ibs_0_6m', np.nan)
        row['IBS 6-12m'] = metrics.get('ibs_6_12m', np.nan)
        row['IBS 12-18m'] = metrics.get('ibs_12_18m', np.nan)
        row['IBS 18-24m'] = metrics.get('ibs_18_24m', np.nan)
        row['IBS Overall'] = metrics.get('ibs_overall', np.nan)
        
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.round(4)
    
    # Save CSV
    csv_path = os.path.join(comparison_dir, "summary_metrics.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved summary CSV to {csv_path}")
    mlflow.log_artifact(csv_path, artifact_path="comparison")
    
    # Create HTML table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(summary_df.columns),
            fill_color='paleturquoise',
            align='center',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[summary_df[col] for col in summary_df.columns],
            fill_color='lavender',
            align='center',
            font=dict(size=11),
            format=[None] + ['.4f'] * (len(summary_df.columns) - 1)
        )
    )])
    
    fig.update_layout(
        title="Summary Metrics: Test Set Performance (Clinical Time Points)",
        height=250
    )
    
    html_path = os.path.join(comparison_dir, "summary_metrics.html")
    pio.write_html(fig, file=html_path, include_plotlyjs="cdn", full_html=True)
    print(f"  ✓ Saved summary HTML to {html_path}")
    mlflow.log_artifact(html_path, artifact_path="comparison")


def create_comparison_table(results_dict, cv_results_dict, output_path):
    """
    Create a comparison table of all models.
    
    Parameters
    ----------
    results_dict : dict
        Dict mapping model_name -> test_metrics
    cv_results_dict : dict
        Dict mapping model_name -> cv_metrics
    output_path : str
        Path to save comparison table
    """
    print("\n=== Creating model comparison table ===")
    
    # Prepare data
    rows = []
    for model_name in results_dict.keys():
        test_metrics = results_dict[model_name]
        cv_metrics = cv_results_dict.get(model_name, {}).get('cv_metrics_mean', {})
        
        row = {
            'Model': model_name,
            'Test C-index': test_metrics.get('c_index', np.nan),
            'CV C-index (mean)': cv_metrics.get('c_index', np.nan),
            'CV IBS (mean)': cv_metrics.get('integrated_brier_score', np.nan),
            'CV Mean AUC': cv_metrics.get('mean_auc', np.nan)
        }
        rows.append(row)
    
    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.round(4)
    
    # Save as CSV
    csv_path = output_path.replace('.html', '.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved comparison CSV to {csv_path}")
    
    # Create interactive table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(comparison_df.columns),
            fill_color='paleturquoise',
            align='left',
            font=dict(size=14, color='black')
        ),
        cells=dict(
            values=[comparison_df[col] for col in comparison_df.columns],
            fill_color='lavender',
            align='left',
            font=dict(size=12)
        )
    )])
    
    fig.update_layout(
        title="Model Comparison: Test and CV Metrics",
        height=300
    )
    
    pio.write_html(fig, file=output_path, include_plotlyjs="cdn", full_html=True)
    print(f"  ✓ Saved comparison table to {output_path}")


def plot_feature_selection_comparison(features_dict, output_path):
    """
    Create Venn diagram and overlap table for feature selection.
    
    Parameters
    ----------
    features_dict : dict
        Dict mapping model_name -> list of selected features
    output_path : str
        Path to save plots
    """
    print("\n=== Comparing feature selection across models ===")
    
    model_names = list(features_dict.keys())
    if len(model_names) < 2:
        print(f"  ⚠ Need at least 2 models for comparison, got {len(model_names)}.")
        return
    
    # Get feature sets
    sets = [set(features_dict[name]) for name in model_names]
    
    # Create bar chart comparing feature counts
    plt.figure(figsize=(10, 6))
    feature_counts = [len(s) for s in sets]
    
    # Add bars for each model and intersection
    x_pos = np.arange(len(model_names))
    plt.bar(x_pos, feature_counts, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Number of Features', fontsize=12)
    plt.title('Feature Count by Model', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, model_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, count in enumerate(feature_counts):
        plt.text(i, count, str(count), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    bar_path = output_path.replace('.html', '_feature_counts.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved feature comparison to {bar_path}")
    
    # Create overlap table
    overlap_data = []
    for i, name1 in enumerate(model_names):
        for name2 in model_names[i+1:]:
            set1, set2 = sets[model_names.index(name1)], sets[model_names.index(name2)]
            overlap = set1 & set2
            union = set1 | set2
            jaccard = len(overlap) / len(union) if len(union) > 0 else 0
            
            overlap_data.append({
                'Model 1': name1,
                'Model 2': name2,
                'Overlap Count': len(overlap),
                'Jaccard Index': round(jaccard, 3),
                'Common Features': ', '.join(sorted(list(overlap))[:10]) + ('...' if len(overlap) > 10 else '')
            })
    
    overlap_df = pd.DataFrame(overlap_data)
    
    # Save as CSV
    csv_path = output_path.replace('.html', '_overlap.csv')
    overlap_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved overlap table to {csv_path}")
    
    # Create HTML table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(overlap_df.columns),
            fill_color='lightblue',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[overlap_df[col] for col in overlap_df.columns],
            fill_color='white',
            align='left',
            font=dict(size=11),
            height=30
        )
    )])
    
    fig.update_layout(
        title="Feature Selection Overlap Analysis",
        height=300
    )
    
    pio.write_html(fig, file=output_path, include_plotlyjs="cdn", full_html=True)
    print(f"  ✓ Saved overlap analysis to {output_path}")
    
    # Summary statistics
    print(f"\n  Feature counts:")
    for name in model_names:
        print(f"    {name}: {len(features_dict[name])} features")
    
    all_features = set()
    for features in features_dict.values():
        all_features.update(features)
    print(f"    Total unique features: {len(all_features)}")
    
    if len(sets) >= 3:
        common_to_all = sets[0] & sets[1] & sets[2]
        print(f"    Features selected by all models: {len(common_to_all)}")
        if len(common_to_all) > 0 and len(common_to_all) <= 20:
            print(f"    Common features: {', '.join(sorted(list(common_to_all)))}")
