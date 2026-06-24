"""
Model Evaluation Pipeline

Main script for evaluating trained survival models on test set.
Handles individual model evaluation and ensemble stacking.
"""

import argparse
import os
import joblib
import mlflow
import numpy as np
import pandas as pd

from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis

# Import utilities
from evaluation_utils import (
    create_temporal_grid,
    compute_brier_scores_on_grid,
    compute_ibs_intervals,
    load_model_and_metrics,
    create_comparative_plots,
    create_summary_table,
    create_comparison_table,
    plot_feature_selection_comparison,
    CLINICAL_TIMEPOINTS_MONTHS
)

# Import ensemble wrapper and functions
from ensemble_model import (
    CoxEnsembleSurvivalPredictor,
    generate_oof_predictions,
    train_meta_learner,
    create_and_evaluate_ensemble
)

# Import XGBoostSurvival and XGBSEPipeline classes for unpickling
try:
    from xgboost_survival_model import XGBoostSurvival
except ImportError:
    import sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_train_dir = os.path.join(parent_dir, 'model_train')
    if model_train_dir not in sys.path:
        sys.path.insert(0, model_train_dir)
    try:
        from xgboost_survival_model import XGBoostSurvival
    except ImportError:
        print("Warning: Could not import XGBoostSurvival - XGBoost model loading may fail")

# Import XGBSEPipeline for unpickling
try:
    from xgbse_pipeline_wrapper import XGBSEPipeline
except ImportError:
    print("Warning: Could not import XGBSEPipeline - XGBSE model loading may fail")

# Import DeepSurvWrapper for DeepSurv model loading
try:
    import sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    deepsurv_dir = os.path.join(parent_dir, 'model_train', 'deepsurv')
    if deepsurv_dir not in sys.path:
        sys.path.insert(0, deepsurv_dir)
    from deepsurv_wrapper import DeepSurvWrapper
except ImportError:
    print("Warning: Could not import DeepSurvWrapper - DeepSurv model loading may fail")


# ============================================================================
# CORE EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model_on_test(model, X_test, y_test, y_train, model_name, output_dir, 
                          grid_months, clinical_months, risk_scores=None):
    """
    Evaluate a model on test set using clinical time-point based temporal validation.
    
    IMPORTANT: All time values are in MONTHS.
    
    Parameters
    ----------
    model : object
        Trained model pipeline
    X_test : pd.DataFrame
        Test features
    y_test : structured array
        Test survival data (time in MONTHS)
    y_train : structured array
        Training survival data (time in MONTHS, needed for time-dependent metrics)
    model_name : str
        Name of the model
    output_dir : str
        Directory to save results
    grid_months : np.ndarray
        Unified temporal grid in MONTHS
    clinical_months : np.ndarray
        Clinical time points in MONTHS
    risk_scores : np.ndarray, optional
        Pre-computed risk scores (for ensemble)
        
    Returns
    -------
    dict
        Test metrics with temporal components
    """
    print(f"\n=== Evaluating {model_name} on test set ===")
    
    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Predict risk scores if not provided
    if risk_scores is None:
        try:
            risk_scores = model.predict(X_test)
            # Flatten to 1D for scikit-survival compatibility (e.g., DeepSurv returns 2D)
            if hasattr(risk_scores, 'ndim') and risk_scores.ndim > 1:
                risk_scores = risk_scores.flatten()
            print(f"  ✓ Generated predictions")
        except Exception as e:
            print(f"  ✗ Prediction failed: {e}")
            return {}
    else:
        print(f"  ✓ Using pre-computed risk scores")
    
    # Compute metrics
    metrics = {}
    
    # C-index (global concordance)
    try:
        cindex_result = concordance_index_censored(y_test["event"], y_test["time"], risk_scores)
        metrics["c_index"] = cindex_result[0]
        print(f"  C-index: {metrics['c_index']:.4f}")
    except Exception as e:
        print(f"  Warning: Could not compute C-index: {e}")
    
    # Time-dependent AUC at clinical time points
    # Always process ALL hardcoded clinical timepoints [6, 12, 15, 18, 24] months
    # Save actual values when available, NaN otherwise
    auc_at_clinical = None
    print(f"  Processing {len(CLINICAL_TIMEPOINTS_MONTHS)} hardcoded clinical timepoints: {CLINICAL_TIMEPOINTS_MONTHS} months")
    
    if len(clinical_months) > 0:
        try:
            print(f"  Computing AUC at {len(clinical_months)} available timepoints: {clinical_months.round(2)} months")
            auc_result, auc_mean = cumulative_dynamic_auc(y_train, y_test, risk_scores, clinical_months)
            metrics["mean_auc"] = auc_mean
            auc_at_clinical = auc_result
            
            # Map available clinical_months to corresponding labels from CLINICAL_TIMEPOINTS_MONTHS
            # Assume clinical_months are ordered and correspond to the first N elements
            print(f"  AUC values at available timepoints:")
            for i, t_months in enumerate(clinical_months):
                # Find closest match in CLINICAL_TIMEPOINTS_MONTHS
                closest_idx = np.argmin(np.abs(np.array(CLINICAL_TIMEPOINTS_MONTHS) - t_months))
                t_label = CLINICAL_TIMEPOINTS_MONTHS[closest_idx]
                metrics[f"auc_{t_label}m"] = auc_result[i]
                print(f"    AUC at {t_label}m ({t_months:.1f} months): {auc_result[i]:.4f}")
            
            # Fill missing timepoints with NaN
            computed_timepoints = set()
            for t_months in clinical_months:
                closest_idx = np.argmin(np.abs(np.array(CLINICAL_TIMEPOINTS_MONTHS) - t_months))
                computed_timepoints.add(CLINICAL_TIMEPOINTS_MONTHS[closest_idx])
            
            missing_timepoints = [t for t in CLINICAL_TIMEPOINTS_MONTHS if t not in computed_timepoints]
            if missing_timepoints:
                print(f"  Missing timepoints {missing_timepoints} months (follow-up too short): setting to NaN")
                for t_months in missing_timepoints:
                    metrics[f"auc_{t_months}m"] = np.nan
            
            print(f"  Mean AUC (over {len(clinical_months)} available timepoints): {auc_mean:.4f}")
        except Exception as e:
            print(f"  \u2717 Error computing time-dependent AUC: {e}")
            import traceback
            traceback.print_exc()
            # Set all timepoints to NaN on error
            for t_months in CLINICAL_TIMEPOINTS_MONTHS:
                metrics[f"auc_{t_months}m"] = np.nan
    else:
        print(f"  \u26a0 No clinical timepoints available within follow-up range")
        # Set all timepoints to NaN
        for t_months in CLINICAL_TIMEPOINTS_MONTHS:
            metrics[f"auc_{t_months}m"] = np.nan
    
    # Brier scores and IBS
    brier_at_clinical = None
    
    if len(grid_months) > 0:
        try:
            print(f"  Computing Brier on grid of {len(grid_months)} points, range=[{grid_months.min():.2f}, {grid_months.max():.2f}] months")
            # Compute Brier scores on full grid
            brier_scores = compute_brier_scores_on_grid(model, X_test, y_train, y_test, grid_months)
            
            if brier_scores is not None:
                if len(clinical_months) > 0:
                    print(f"  Brier values at available timepoints:")
                    # Extract Brier at available clinical points
                    brier_at_clinical = np.zeros(len(clinical_months))
                    for i, t_months in enumerate(clinical_months):
                        idx = np.argmin(np.abs(grid_months - t_months))
                        brier_at_clinical[i] = brier_scores[idx]
                        # Find closest match in CLINICAL_TIMEPOINTS_MONTHS
                        closest_idx = np.argmin(np.abs(np.array(CLINICAL_TIMEPOINTS_MONTHS) - t_months))
                        t_label = CLINICAL_TIMEPOINTS_MONTHS[closest_idx]
                        metrics[f"brier_{t_label}m"] = brier_at_clinical[i]
                        print(f"    Brier at {t_label}m ({t_months:.1f} months): {brier_at_clinical[i]:.4f}")
                    
                    # Fill missing timepoints with NaN
                    computed_timepoints = set()
                    for t_months in clinical_months:
                        closest_idx = np.argmin(np.abs(np.array(CLINICAL_TIMEPOINTS_MONTHS) - t_months))
                        computed_timepoints.add(CLINICAL_TIMEPOINTS_MONTHS[closest_idx])
                    
                    missing_timepoints = [t for t in CLINICAL_TIMEPOINTS_MONTHS if t not in computed_timepoints]
                    if missing_timepoints:
                        print(f"  Missing timepoints {missing_timepoints} months (follow-up too short): setting to NaN")
                        for t_months in missing_timepoints:
                            metrics[f"brier_{t_months}m"] = np.nan
                else:
                    # No clinical timepoints available, set all to NaN
                    print(f"  No clinical timepoints available, setting all Brier values to NaN")
                    for t_months in CLINICAL_TIMEPOINTS_MONTHS:
                        metrics[f"brier_{t_months}m"] = np.nan
                
                # Compute IBS over intervals
                ibs_intervals, ibs_overall = compute_ibs_intervals(grid_months, brier_scores)
                metrics.update(ibs_intervals)
                metrics["ibs_overall"] = ibs_overall
                
                print(f"  IBS (0-{int(grid_months.max())}m): {ibs_overall:.4f}")
            else:
                print(f"  \u26a0 Model does not have predict_survival_function, skipping Brier scores")
                # Set all Brier timepoints to NaN
                for t_months in CLINICAL_TIMEPOINTS_MONTHS:
                    metrics[f"brier_{t_months}m"] = np.nan
        except Exception as e:
            print(f"  \u2717 Error computing Brier/IBS: {e}")
            import traceback
            traceback.print_exc()
            # Set all Brier timepoints to NaN on error
            for t_months in CLINICAL_TIMEPOINTS_MONTHS:
                metrics[f"brier_{t_months}m"] = np.nan
    else:
        print(f"  \u26a0 Empty temporal grid, skipping Brier/IBS computation")
        # Set all Brier timepoints to NaN
        for t_months in CLINICAL_TIMEPOINTS_MONTHS:
            metrics[f"brier_{t_months}m"] = np.nan
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(model_output_dir, "test_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  ✓ Saved test metrics to {metrics_path}")
    
    # Store temporal data for comparative plots
    return {
        **metrics,
        '_temporal_auc': auc_at_clinical,
        '_temporal_brier': brier_at_clinical
    }


def load_data(training_data_dir, test_data_dir):
    """
    Load training and test datasets.
    
    Parameters
    ----------
    training_data_dir : str
        Path to training data folder
    test_data_dir : str
        Path to test data folder
        
    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test)
    """
    # Load test data
    csv_files = [f for f in os.listdir(test_data_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in test_data folder.")
    test_csv_path = os.path.join(test_data_dir, csv_files[0])
    print(f"Loading test dataset: {test_csv_path}")
    df_test = pd.read_csv(test_csv_path)
    print(f"Test set shape: {df_test.shape}")
    
    # Prepare test data
    X_test = df_test.drop(columns=["event", "tte"]).select_dtypes(include=[np.number])
    X_test.set_index("id", inplace=True)
    y_test = Surv.from_arrays(event=df_test["event"].astype(bool), time=df_test["tte"].astype(float))
    print(f"Test features: {X_test.shape}, Test survival: {y_test.shape}")
    
    # Load training data (needed for time-dependent metrics and ensemble stacking)
    train_csv_files = [f for f in os.listdir(training_data_dir) if f.endswith(".csv")]
    if not train_csv_files:
        raise FileNotFoundError("No CSV file found in training_data folder.")
    train_csv_path = os.path.join(training_data_dir, train_csv_files[0])
    print(f"Loading training dataset for time-dependent metrics and ensemble stacking: {train_csv_path}")
    df_train = pd.read_csv(train_csv_path)
    
    # Prepare training features (needed for ensemble stacking)
    X_train = df_train.drop(columns=["event", "tte"]).select_dtypes(include=[np.number])
    X_train.set_index("id", inplace=True)
    y_train = Surv.from_arrays(event=df_train["event"].astype(bool), time=df_train["tte"].astype(float))
    print(f"Training data loaded: {X_train.shape} features, {y_train.shape} samples")
    
    # Align test features with training features (same columns, same order)
    # This handles cases where train/test may have different feature engineering
    print(f"\nAligning test and training features...")
    print(f"  Training features: {X_train.shape[1]}")
    print(f"  Test features (before alignment): {X_test.shape[1]}")
    
    # Keep only common features, in the same order as training
    common_features = [col for col in X_train.columns if col in X_test.columns]
    missing_in_test = set(X_train.columns) - set(X_test.columns)
    extra_in_test = set(X_test.columns) - set(X_train.columns)
    
    if missing_in_test:
        print(f"  ⚠ Warning: {len(missing_in_test)} features in training but not in test")
        print(f"    First 10: {list(missing_in_test)[:10]}")
    if extra_in_test:
        print(f"  ⚠ Warning: {len(extra_in_test)} features in test but not in training (will be dropped)")
        print(f"    First 10: {list(extra_in_test)[:10]}")
    
    # Align test data: select only common features, in training order
    X_test = X_test[common_features]
    print(f"  Test features (after alignment): {X_test.shape[1]}")
    
    return X_train, y_train, X_test, y_test


def load_all_models(cox_dir, rsf_dir, xgboost_dir, xgbse_dir, deepsurv_dir=None):
    """
    Load all trained models and their artifacts.
    
    Parameters
    ----------
    cox_dir : str
        Path to Cox model directory
    rsf_dir : str
        Path to RSF model directory
    xgboost_dir : str
        Path to XGBoost model directory
    xgbse_dir : str
        Path to XGBSE model directory
    deepsurv_dir : str, optional
        Path to DeepSurv model directory
        
    Returns
    -------
    tuple
        (models, cv_metrics_all, selected_features_all)
    """
    models = {}
    cv_metrics_all = {}
    selected_features_all = {}
    
    model_configs = [
        ("cox", cox_dir),
        ("rsf", rsf_dir),
        ("xgboost", xgboost_dir),
        ("xgbse", xgbse_dir)
    ]
    
    # Add DeepSurv if provided
    if deepsurv_dir:
        model_configs.append(("deepsurv", deepsurv_dir))
    
    for model_name, model_dir in model_configs:
        model, cv_metrics, selected_features = load_model_and_metrics(model_dir, model_name)
        models[model_name] = model
        cv_metrics_all[model_name] = cv_metrics
        selected_features_all[model_name] = selected_features
    
    return models, cv_metrics_all, selected_features_all


def evaluate_base_models(models, selected_features_all, X_test, y_test, y_train, 
                        output_dir, grid_months, clinical_months):
    """
    Evaluate all base models on test set.
    
    Parameters
    ----------
    models : dict
        Dict mapping model_name -> trained model
    selected_features_all : dict
        Dict mapping model_name -> list of selected features
    X_test : pd.DataFrame
        Test features
    y_test : structured array
        Test survival data (time in MONTHS)
    y_train : structured array
        Training survival data (time in MONTHS)
    output_dir : str
        Directory to save results
    grid_months : np.ndarray
        Temporal grid in MONTHS
    clinical_months : np.ndarray
        Clinical time points in MONTHS
        
    Returns
    -------
    dict
        Dict mapping model_name -> test metrics
    """
    print("\n" + "="*70)
    print("=== BASE MODELS EVALUATION ===")
    print("="*70)
    
    test_results = {}
    for model_name, model in models.items():
        selected_features = selected_features_all[model_name]
        
        # All models use selected features (if available)
        if selected_features:
            # Filter to selected features that exist in test set
            available_features = [f for f in selected_features if f in X_test.columns]
            X_test_model = X_test[available_features]
            print(f"  Model {model_name}: using {len(available_features)} selected features")
            
            if len(available_features) != len(selected_features):
                missing = set(selected_features) - set(available_features)
                print(f"    ⚠ Warning: {len(missing)} features missing in test set: {list(missing)[:5]}")
        else:
            # Fallback: use all features if no selection file found
            X_test_model = X_test
            print(f"  Model {model_name}: using all {len(X_test.columns)} features (no selection)")
        
        metrics = evaluate_model_on_test(
            model, X_test_model, y_test, y_train, model_name, output_dir,
            grid_months, clinical_months
        )
        test_results[model_name] = metrics
    
    return test_results


def create_comparison_artifacts(test_results, cv_metrics_all, selected_features_all, output_dir):
    """
    Create all comparison artifacts (tables, plots, feature analysis).
    
    Parameters
    ----------
    test_results : dict
        Dict mapping model_name -> test metrics
    cv_metrics_all : dict
        Dict mapping model_name -> cv metrics
    selected_features_all : dict
        Dict mapping model_name -> list of selected features
    output_dir : str
        Directory to save artifacts
    """
    print("\n=== Creating aggregated comparison ===")
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 1. Comparison table (CV metrics)
    comparison_table_path = os.path.join(comparison_dir, "model_comparison.html")
    create_comparison_table(test_results, cv_metrics_all, comparison_table_path)
    
    # 2. Feature selection comparison
    feature_comparison_path = os.path.join(comparison_dir, "feature_selection_comparison.html")
    plot_feature_selection_comparison(selected_features_all, feature_comparison_path)
    
    # 3. Comparative temporal plots (AUC and Brier)
    create_comparative_plots(test_results, comparison_dir)
    
    # 4. Summary metrics table (Test set performance with clinical time points)
    create_summary_table(test_results, comparison_dir)
    
    # Log all comparison artifacts (avoid duplicates by collecting unique files)
    logged_artifacts = set()
    for filename in os.listdir(comparison_dir):
        filepath = os.path.join(comparison_dir, filename)
        if os.path.isfile(filepath) and filepath not in logged_artifacts:
            try:
                mlflow.log_artifact(filepath, artifact_path="comparison")
                logged_artifacts.add(filepath)
                print(f"  ✓ Logged artifact: {filename}")
            except Exception as e:
                print(f"  ⚠ Could not log {filename}: {e}")


def log_test_metrics_to_mlflow(test_results):
    """
    Log test metrics to MLflow.
    
    Parameters
    ----------
    test_results : dict
        Dict mapping model_name -> test metrics
    """
    for model_name, metrics in test_results.items():
        for metric_name, value in metrics.items():
            # Skip internal temporal arrays (start with underscore)
            if not metric_name.startswith('_') and isinstance(value, (int, float, np.number)):
                mlflow.log_metric(f"test_{model_name}_{metric_name}", value)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(args):
    """
    Main evaluation pipeline.
    
    Workflow:
    1. Load data (train and test)
    2. Load all trained models
    3. Create temporal grid for evaluation
    4. Evaluate base models on test set
    5. Generate OOF predictions for ensemble
    6. Train meta-learner
    7. Create and evaluate ensemble
    8. Generate comparison artifacts
    9. Log results to MLflow
    """
    print("=== Starting Model Evaluation on Test Set ===")
    os.makedirs(args.evaluation_output, exist_ok=True)
    mlflow.start_run()
    
    # Step 1: Load data
    X_train, y_train, X_test, y_test = load_data(args.training_data, args.test_data)
    
    # Step 2: Load all trained models
    models, cv_metrics_all, selected_features_all = load_all_models(
        args.cox_model, args.rsf_model, args.xgboost_model, args.xgbse_model,
        deepsurv_dir=getattr(args, 'deepsurv_model', None)
    )
    
    # Step 3: Create unified temporal grid for evaluation
    print("\n=== Creating unified temporal grid for evaluation ===")
    grid_months, clinical_months = create_temporal_grid(y_train, y_test)
    
    # Step 4: Evaluate base models on test set
    test_results = evaluate_base_models(
        models, selected_features_all, X_test, y_test, y_train,
        args.evaluation_output, grid_months, clinical_months
    )
    
    # Step 5-7: Ensemble stacking
    print("\n" + "="*70)
    print("=== ENSEMBLE STACKING: Meta-Learner Training ===")
    print("="*70)
    
    oof_predictions, stack_models = generate_oof_predictions(models, selected_features_all, X_train, y_train)
    meta_learner = train_meta_learner(oof_predictions, y_train, X_train, args.evaluation_output)
    ensemble_metrics = create_and_evaluate_ensemble(
        stack_models, {k: selected_features_all[k] for k in stack_models.keys()}, meta_learner,
        X_test, y_test, y_train, args.evaluation_output, grid_months, clinical_months,
        evaluate_fn=evaluate_model_on_test
    )
    
    # Add ensemble to results
    test_results["ensemble"] = ensemble_metrics
    cv_metrics_all["ensemble"] = {}  # No CV metrics for ensemble (trained on OOF predictions)
    
    print("\n  ✅ Ensemble stacking completed!")
    print("="*70 + "\n")
    
    # Step 8: Create comparison artifacts
    create_comparison_artifacts(test_results, cv_metrics_all, selected_features_all, args.evaluation_output)
    
    # Step 9: Log results to MLflow
    log_test_metrics_to_mlflow(test_results)
    
    mlflow.end_run()
    print("\n=== Evaluation completed successfully! ===")
    print(f"Results saved to: {args.evaluation_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, required=True,
                        help="Path to training data folder (needed for time-dependent metrics)")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to test data folder")
    parser.add_argument("--cox_model", type=str, required=True,
                        help="Path to trained Cox-Net model folder")
    parser.add_argument("--rsf_model", type=str, required=True,
                        help="Path to trained RSF model folder")
    parser.add_argument("--xgboost_model", type=str, required=True,
                        help="Path to trained XGBoost model folder")
    parser.add_argument("--xgbse_model", type=str, required=True,
                        help="Path to trained XGBSE model folder")
    parser.add_argument("--deepsurv_model", type=str, required=False, default=None,
                        help="Path to trained DeepSurv model folder (optional)")
    parser.add_argument("--evaluation_output", type=str, required=True,
                        help="Path to output folder for evaluation results")
    args = parser.parse_args()
    main(args)
