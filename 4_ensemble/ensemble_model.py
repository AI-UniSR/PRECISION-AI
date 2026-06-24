"""
Ensemble Stacking Module for Survival Analysis

This module provides complete ensemble functionality:
- CoxEnsembleSurvivalPredictor: Generic wrapper for ensemble prediction
- generate_oof_predictions(): Out-of-fold prediction generation
- train_meta_learner(): Meta-learner training
- create_and_evaluate_ensemble(): Ensemble evaluation on test set
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis

try:
    import mlflow
except ImportError:
    mlflow = None


class CoxEnsembleSurvivalPredictor:
    """
    Generic wrapper for Cox ensemble that takes full feature matrix X,
    computes base model predictions, and applies meta-learner.
    
    This acts as a black-box predictor: X → risk_score
    
    The wrapper is completely generic:
    - Dynamically reads base model names from dictionary keys
    - Generates meta-features with matching names (e.g., 'cox_pred', 'rsf_pred')
    - Works with any number of base models (not limited to 3)
    - Column order is determined by sorted model names (for reproducibility)
    
    Features:
    - Strict feature validation (raises error if required features missing)
    - Stable serialization for MLflow and joblib
    """
    
    def __init__(self, base_models, selected_features_all, meta_learner):
        """
        Parameters
        ----------
        base_models : dict
            Dict mapping model_name -> fitted model
            Example: {'cox': cox_model, 'rsf': rsf_model, 'xgboost': xgb_model}
        selected_features_all : dict
            Dict mapping model_name -> list of selected features (or empty list)
            Example: {'cox': [], 'rsf': ['feature1', 'feature2'], 'xgboost': [...]}
        meta_learner : CoxPHSurvivalAnalysis
            Fitted meta-learner that expects N features (one per base model)
        """
        # Validate inputs
        if not isinstance(base_models, dict) or len(base_models) == 0:
            raise ValueError("base_models must be a non-empty dictionary")
        
        if not isinstance(selected_features_all, dict):
            raise ValueError("selected_features_all must be a dictionary")
        
        # Check consistency between base_models and selected_features_all
        missing_features = set(base_models.keys()) - set(selected_features_all.keys())
        if missing_features:
            raise ValueError(
                f"Missing selected features for models: {missing_features}\n"
                f"selected_features_all must have entries for all base models"
            )
        
        self.base_models = base_models
        self.selected_features_all = selected_features_all
        self.meta_learner = meta_learner
        
        # Dynamically determine model order (sorted for reproducibility)
        self.model_names = sorted(base_models.keys())
        
        # Generate meta-feature column names dynamically
        # Format: {model_name}_pred (e.g., 'cox_pred', 'rsf_pred', 'xgboost_pred')
        self.meta_feature_names = [f"{name}_pred" for name in self.model_names]
        
        # Store number of base models
        self.n_base_models = len(self.model_names)
    
    def predict(self, X):
        """
        Predict risk scores for samples in X.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (full set of features with feature names as columns)
            
        Returns
        -------
        risk_scores : np.ndarray
            Ensemble risk scores (1D array)
            
        Raises
        ------
        TypeError
            If X is not a pandas DataFrame
        ValueError
            If required features are missing from X for any base model
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")
        
        # Compute base model predictions in sorted order
        base_predictions = {}
        
        for model_name in self.model_names:
            model = self.base_models[model_name]
            selected_features = self.selected_features_all[model_name]
            
            # Debug logging
            print(f"  Predicting with {model_name}: {len(selected_features) if selected_features else 'all'} features")
            
            # Determine which features to use for this model
            if not selected_features:
                # Model uses all features (e.g., Cox with L1 regularization)
                X_model = X
                print(f"    Using all {X.shape[1]} features from input")
            else:
                # Model uses pre-selected features
                # Strict validation: all selected features must be present
                missing_features = [f for f in selected_features if f not in X.columns]
                if missing_features:
                    raise ValueError(
                        f"Model '{model_name}' requires features that are missing from X:\n"
                        f"  Required features: {len(selected_features)}\n"
                        f"  Missing features: {len(missing_features)}\n"
                        f"  First 10 missing: {missing_features[:10]}"
                        + (f"\n  ... and {len(missing_features)-10} more" if len(missing_features) > 10 else "")
                    )
                X_model = X[selected_features]
                print(f"    Filtered to {X_model.shape[1]} selected features")
            
            # Predict with base model
            base_predictions[model_name] = model.predict(X_model)
        
        # Create meta-features DataFrame with dynamic column names in sorted order
        meta_data = {
            f"{name}_pred": base_predictions[name] 
            for name in self.model_names
        }
        X_meta = pd.DataFrame(meta_data, index=X.index)
        
        # Ensure column order matches expected meta-feature names
        X_meta = X_meta[self.meta_feature_names]
        
        # Apply meta-learner
        ensemble_risk_scores = self.meta_learner.predict(X_meta)
        
        return ensemble_risk_scores
    
    def predict_survival_function(self, X):
        """
        Predict survival functions using the meta-learner.
        
        This method creates meta-features from base model predictions,
        then uses the Cox meta-learner's predict_survival_function.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (full set of features with feature names as columns)
            
        Returns
        -------
        survival_functions : list of StepFunction
            Survival function for each sample
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")
        
        # Compute base model predictions (same as predict())
        base_predictions = {}
        
        for model_name in self.model_names:
            model = self.base_models[model_name]
            selected_features = self.selected_features_all[model_name]
            
            # Determine which features to use
            if not selected_features:
                X_model = X
            else:
                missing_features = [f for f in selected_features if f not in X.columns]
                if missing_features:
                    raise ValueError(
                        f"Model '{model_name}' requires features missing from X: {missing_features[:10]}"
                    )
                X_model = X[selected_features]
            
            # Predict with base model
            base_predictions[model_name] = model.predict(X_model)
        
        # Create meta-features
        meta_data = {
            f"{name}_pred": base_predictions[name] 
            for name in self.model_names
        }
        X_meta = pd.DataFrame(meta_data, index=X.index)
        X_meta = X_meta[self.meta_feature_names]
        
        # Use meta-learner's predict_survival_function
        return self.meta_learner.predict_survival_function(X_meta)
    
    def get_base_model_names(self):
        """
        Get list of base model names in the order used for meta-features.
        
        Returns
        -------
        list
            Sorted list of base model names
        """
        return self.model_names.copy()
    
    def get_meta_feature_names(self):
        """
        Get list of meta-feature column names.
        
        Returns
        -------
        list
            List of meta-feature names (e.g., ['cox_pred', 'rsf_pred', 'xgboost_pred'])
        """
        return self.meta_feature_names.copy()
    
    def __repr__(self):
        return (
            f"CoxEnsembleSurvivalPredictor(\n"
            f"  n_base_models={self.n_base_models},\n"
            f"  base_models={self.model_names},\n"
            f"  meta_features={self.meta_feature_names}\n"
            f")"
        )


# ============================================================================
# OOF PREDICTION GENERATION
# ============================================================================

def generate_oof_predictions(models, selected_features_all, X_train, y_train, n_splits=5):
    """
    Generate out-of-fold predictions for ensemble stacking.
    
    Parameters
    ----------
    models : dict
        Dict mapping model_name -> trained model
    selected_features_all : dict
        Dict mapping model_name -> list of selected features
    X_train : pd.DataFrame
        Training features
    y_train : structured array
        Training survival data
    n_splits : int
        Number of CV folds
        
    Returns
    -------
    dict
        Dict mapping model_name -> OOF predictions array
    """
    print("\n[Step 1/4] Generating Out-of-Fold predictions with 5-fold stratified CV...")
    
    # Exclude DeepSurv from ensemble stacking (pre-trained model with internal CV)
    stack_models = {k: v for k, v in models.items() if k != 'deepsurv'}
    print(f"  Models in ensemble stack: {list(stack_models.keys())}")
    if 'deepsurv' in models:
        print(f"  Models excluded from stack: ['deepsurv'] (pre-trained with internal CV)")
    
    n_samples = len(y_train)
    # Initialize OOF predictions only for models in stack
    oof_predictions = {model_name: np.zeros(n_samples) for model_name in stack_models.keys()}
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train["event"])):
        print(f"\n  Fold {fold+1}/{n_splits}:")
        
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        
        for model_name, model in stack_models.items():
            selected_features = selected_features_all[model_name]
            
            # Prepare fold data with correct features
            # All models use selected features (if available)
            if selected_features:
                available_features = [f for f in selected_features if f in X_fold_train.columns]
                X_fold_train_model = X_fold_train[available_features]
                X_fold_val_model = X_fold_val[available_features]
            else:
                # Fallback: use all features if no selection file found
                X_fold_train_model = X_fold_train
                X_fold_val_model = X_fold_val
            
            # Clone and train model on fold
            fold_model = clone(model)
            fold_model.fit(X_fold_train_model, y_fold_train)
            
            # Predict on validation fold
            oof_predictions[model_name][val_idx] = fold_model.predict(X_fold_val_model)
            
        print(f"    ✓ Generated {len(val_idx)} OOF predictions for {len(stack_models)} stacked models")
    
    print("\n  ✅ OOF prediction generation completed!")
    return oof_predictions, stack_models


# ============================================================================
# META-LEARNER TRAINING
# ============================================================================

def train_meta_learner(oof_predictions, y_train, X_train, output_dir):
    """
    Train meta-learner on OOF predictions.
    
    Parameters
    ----------
    oof_predictions : dict
        Dict mapping model_name -> OOF predictions array
    y_train : structured array
        Training survival data
    X_train : pd.DataFrame
        Training features (for index)
    output_dir : str
        Directory to save meta-learner artifacts
        
    Returns
    -------
    meta_learner : CoxPHSurvivalAnalysis
        Trained meta-learner
    """
    print("\n[Step 2/4] Creating meta-features from OOF predictions...")
    
    # Create meta-features dynamically from all models
    # Column names: {model_name}_pred (e.g., 'cox_pred', 'rsf_pred', 'xgboost_pred', 'xgbse_pred')
    meta_features = {f"{model_name}_pred": predictions 
                     for model_name, predictions in sorted(oof_predictions.items())}
    X_meta_train = pd.DataFrame(meta_features, index=X_train.index)
    
    print(f"  Meta-features shape: {X_meta_train.shape}")
    print(f"  Meta-features summary:")
    print(X_meta_train.describe())
    
    # Train meta-learner (CoxPH without regularization)
    print("\n[Step 3/4] Training meta-learner (CoxPH)...")
    
    meta_learner = CoxPHSurvivalAnalysis()
    meta_learner.fit(X_meta_train, y_train)
    
    # Log coefficients dynamically for all models
    coef_df = pd.DataFrame({
        'base_model': sorted(oof_predictions.keys()),
        'meta_coefficient': meta_learner.coef_
    })
    
    print("\n  Meta-Learner Coefficients:")
    print(coef_df.to_string(index=False))
    
    # Save artifacts
    ensemble_dir = os.path.join(output_dir, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    
    coef_path = os.path.join(ensemble_dir, "meta_coefficients.csv")
    coef_df.to_csv(coef_path, index=False)
    if mlflow:
        mlflow.log_artifact(coef_path, artifact_path="ensemble")
    
    meta_path = os.path.join(ensemble_dir, "meta_learner.pkl")
    joblib.dump(meta_learner, meta_path)
    if mlflow:
        mlflow.log_artifact(meta_path, artifact_path="ensemble")
    
    # Save OOF predictions dynamically for all models
    oof_data = {'id': X_train.index}
    for model_name, predictions in sorted(oof_predictions.items()):
        oof_data[f"{model_name}_oof"] = predictions
    oof_df = pd.DataFrame(oof_data)
    oof_path = os.path.join(ensemble_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    if mlflow:
        mlflow.log_artifact(oof_path, artifact_path="ensemble")
    
    print(f"\n  ✅ Meta-learner trained and saved to {ensemble_dir}")
    return meta_learner


# ============================================================================
# ENSEMBLE CREATION AND EVALUATION
# ============================================================================

def create_and_evaluate_ensemble(models, selected_features_all, meta_learner, 
                                 X_test, y_test, y_train, output_dir, grid_months, 
                                 clinical_months, evaluate_fn):
    """
    Create ensemble wrapper and evaluate on test set.
    
    Parameters
    ----------
    models : dict
        Dict mapping model_name -> trained model
    selected_features_all : dict
        Dict mapping model_name -> list of selected features
    meta_learner : CoxPHSurvivalAnalysis
        Trained meta-learner
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
    evaluate_fn : callable
        Function to evaluate model on test set
        Signature: evaluate_fn(model, X_test, y_test, y_train, model_name, 
                               output_dir, grid_months, clinical_months, risk_scores=None)
        
    Returns
    -------
    dict
        Ensemble test metrics
    """
    print("\n[Step 4/4] Creating ensemble wrapper and generating predictions...")
    
    # Create generic ensemble wrapper
    ensemble_model = CoxEnsembleSurvivalPredictor(
        base_models=models,
        selected_features_all=selected_features_all,
        meta_learner=meta_learner
    )
    
    # Display ensemble configuration
    print(f"  Ensemble configuration:")
    print(f"    Base models: {ensemble_model.get_base_model_names()}")
    print(f"    Meta-features: {ensemble_model.get_meta_feature_names()}")
    
    # Get ensemble predictions on test set
    ensemble_risk_scores = ensemble_model.predict(X_test)
    
    print(f"  ✓ Generated {len(ensemble_risk_scores)} ensemble predictions")
    
    # Save ensemble wrapper for future use
    ensemble_dir = os.path.join(output_dir, "ensemble")
    ensemble_wrapper_path = os.path.join(ensemble_dir, "ensemble_model.pkl")
    joblib.dump(ensemble_model, ensemble_wrapper_path)
    if mlflow:
        mlflow.log_artifact(ensemble_wrapper_path, artifact_path="ensemble")
    print(f"  ✓ Saved ensemble wrapper to {ensemble_wrapper_path}")
    
    # =========================================================================
    # REGISTER ENSEMBLE TO MLFLOW MODEL REGISTRY
    # =========================================================================
    
    print("\n" + "="*70)
    print("REGISTERING ENSEMBLE TO MLFLOW MODEL REGISTRY")
    print("="*70)
    
    try:
        # Save base models separately for MLflow artifacts
        print("\n[1/4] Saving base model artifacts...")
        base_model_paths = {}
        for model_name, model_obj in models.items():
            model_path = os.path.join(ensemble_dir, f"{model_name}_model.pkl")
            joblib.dump(model_obj, model_path)
            base_model_paths[model_name] = model_path
            print(f"  ✓ Saved {model_name} model")
        
        # Save selected features mapping
        print("\n[2/4] Saving selected features mapping...")
        features_path = os.path.join(ensemble_dir, "selected_features.json")
        import json
        with open(features_path, 'w') as f:
            # Convert to list for JSON serialization
            features_dict = {k: list(v) if v else [] 
                           for k, v in selected_features_all.items()}
            json.dump(features_dict, f, indent=2)
        print(f"  ✓ Saved selected features")
        
        # Prepare artifacts dictionary
        artifacts = {
            "cox_model": base_model_paths['cox'],
            "rsf_model": base_model_paths['rsf'],
            "xgboost_model": base_model_paths['xgboost'],
            "xgbse_model": base_model_paths['xgbse'],
            "meta_learner": os.path.join(ensemble_dir, "meta_learner.pkl"),
            "selected_features": features_path
        }
        
        # Import PyFunc wrapper
        print("\n[3/4] Preparing MLflow PyFunc model...")
        from ensemble_pyfunc import EnsembleSurvivalPyFunc
        
        # Get current directory for code_path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Register model to MLflow (environment inferred automatically)
        print("\n[4/4] Registering to MLflow Model Registry...")
        model_info = mlflow.pyfunc.log_model(
            artifact_path="ensemble_model_registered",
            python_model=EnsembleSurvivalPyFunc(),
            artifacts=artifacts,
            code_path=[
                # Include files with custom class definitions
                os.path.join(current_dir, "ensemble_pyfunc.py"),
                os.path.join(current_dir, "xgboost_survival_model.py"),
                os.path.join(current_dir, "xgbse_pipeline_wrapper.py"),
            ],
            registered_model_name="ensemble_survival_model"
        )

        print(f"\n✅ Model registered successfully!")
        print(f"   Model URI: {model_info.model_uri}")
        print(f"   Registry: models:/ensemble_survival_model/latest")
        
        # =====================================================================
        # VALIDATION: Verify registered model produces same predictions
        # =====================================================================
        
        print("\n" + "="*70)
        print("VALIDATING REGISTERED MODEL")
        print("="*70)
        
        print("\n[1/2] Loading registered model from MLflow...")
        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        print("  ✓ Model loaded successfully")
        
        print("\n[2/2] Comparing predictions (in-memory vs registered)...")
        
        # Use a subset of test data for validation
        X_test_sample = X_test.head(min(100, len(X_test)))
        
        # Predictions from in-memory model
        preds_memory = ensemble_model.predict(X_test_sample)
        
        # Predictions from registered model
        preds_registered_df = loaded_model.predict(X_test_sample)
        preds_registered = preds_registered_df['risk_score'].values
        
        # Compute difference
        max_diff = np.abs(preds_memory - preds_registered).max()
        mean_diff = np.abs(preds_memory - preds_registered).mean()
        
        print(f"\n  Prediction comparison (n={len(X_test_sample)}):")
        print(f"    Max absolute difference: {max_diff:.2e}")
        print(f"    Mean absolute difference: {mean_diff:.2e}")
        
        # Check if predictions match (with numerical tolerance)
        tolerance = 1e-6
        if max_diff < tolerance:
            print(f"  ✅ VALIDATION PASSED: Predictions match within tolerance ({tolerance})")
        else:
            print(f"  ⚠️  WARNING: Predictions differ by {max_diff:.2e} (tolerance: {tolerance})")
            print(f"      This may indicate serialization/deserialization issues.")
            
            # Show first few predictions for debugging
            comparison_df = pd.DataFrame({
                'in_memory': preds_memory[:5],
                'registered': preds_registered[:5],
                'difference': (preds_memory - preds_registered)[:5]
            })
            print(f"\n  First 5 predictions:")
            print(comparison_df.to_string())
        
        # Log validation metrics
        if mlflow:
            mlflow.log_metric("model_validation_max_diff", float(max_diff))
            mlflow.log_metric("model_validation_mean_diff", float(mean_diff))
        
        # =====================================================================
        # ADDITIONAL VALIDATION: Test predict_survival_function
        # =====================================================================
        
        print("\n[3/3] Validating predict_survival_function method...")
        
        try:
            # Check method exists
            if not hasattr(loaded_model._model_impl.python_model, 'predict_survival_function'):
                print("  ⚠️  WARNING: predict_survival_function method not found on registered model")
                print("      Brier score computation will not work in validation pipeline")
            else:
                # Test survival function prediction on small sample
                surv_funcs = loaded_model._model_impl.python_model.predict_survival_function(X_test_sample)
                
                # Validate output structure
                if not isinstance(surv_funcs, (list, np.ndarray)):
                    print(f"  ⚠️  WARNING: Unexpected return type: {type(surv_funcs)}")
                elif len(surv_funcs) != len(X_test_sample):
                    print(f"  ⚠️  WARNING: Length mismatch: {len(surv_funcs)} vs {len(X_test_sample)}")
                else:
                    # Check that survival functions are callable
                    test_times = np.array([6, 12, 18, 24])
                    try:
                        test_probs = [fn(test_times[0]) for fn in surv_funcs[:3]]
                        print(f"  ✓ predict_survival_function works correctly")
                        print(f"    - Returned {len(surv_funcs)} survival functions")
                        print(f"    - Sample survival probabilities at {test_times[0]} months: {test_probs[:3]}")
                        print(f"  ✅ Brier score computation will be enabled in validation pipeline")
                    except Exception as e:
                        print(f"  ⚠️  WARNING: Survival functions not callable: {e}")
        
        except Exception as e:
            print(f"  ⚠️  WARNING: Could not validate predict_survival_function: {e}")
            import traceback
            traceback.print_exc()
            mlflow.log_metric("model_validation_passed", 1.0 if max_diff < tolerance else 0.0)
        
        print("\n" + "="*70)
        print("MODEL REGISTRATION COMPLETE")
        print("="*70)
        print(f"\nTo use the registered model in another job:")
        print(f"  >>> import mlflow")
        print(f"  >>> model = mlflow.pyfunc.load_model('models:/btc_ensemble_survival_model/latest')")
        print(f"  >>> predictions = model.predict(X_new)")
        print(f"  >>> print(predictions)  # DataFrame with 'risk_score' column")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR during model registration: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("\n⚠️  Continuing with pipeline (registration failed but ensemble still saved)")
    
    # =========================================================================
    # END OF MODEL REGISTRATION
    # =========================================================================
    
    
    # Evaluate ensemble on test set using temporal validation
    print("\n=== Evaluating ENSEMBLE on test set ===")
    
    ensemble_metrics = evaluate_fn(
        ensemble_model, X_test, y_test, y_train, "ensemble", output_dir,
        grid_months, clinical_months, risk_scores=ensemble_risk_scores
    )
    
    return ensemble_metrics
