"""
MLflow PyFunc wrapper for ensemble survival model.

This file is included via code_path in MLflow model registration.
It must be self-contained and importable.
"""

import joblib
import json
import pandas as pd
import numpy as np
from mlflow.pyfunc import PythonModel


class EnsembleSurvivalPyFunc(PythonModel):
    """
    Ensemble survival model for MLflow deployment.
    
    Loads base models and meta-learner from artifacts,
    generates predictions from each base model,
    and combines them using the meta-learner.
    """
    
    def load_context(self, context):
        """
        Load model artifacts.
        
        Called automatically by MLflow when loading the model.
        All artifacts are available in context.artifacts dict.
        """
        # Load base models (sklearn pipelines)
        self.cox_model = joblib.load(context.artifacts["cox_model"])
        self.rsf_model = joblib.load(context.artifacts["rsf_model"])
        self.xgboost_model = joblib.load(context.artifacts["xgboost_model"])
        self.xgbse_model = joblib.load(context.artifacts["xgbse_model"])
        
        # Load meta-learner (CoxPH)
        self.meta_learner = joblib.load(context.artifacts["meta_learner"])
        
        # Load selected features mapping
        with open(context.artifacts["selected_features"], 'r') as f:
            self.selected_features = json.load(f)
        
        # Create model dictionary and order
        self.models = {
            'cox': self.cox_model,
            'rsf': self.rsf_model,
            'xgboost': self.xgboost_model,
            'xgbse': self.xgbse_model
        }
        self.model_order = sorted(self.models.keys())
        
        print(f"Ensemble loaded: {self.model_order}")
    
    def predict(self, context, model_input):
        """
        Generate ensemble risk score predictions.
        
        Parameters
        ----------
        context : MLflow context (not used in prediction)
        model_input : pandas.DataFrame
            Input features with same schema as training data
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with 'risk_score' column (higher = higher risk)
        """
        # Validate input
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError(f"Input must be pandas.DataFrame, got {type(model_input)}")
        
        # Generate predictions from each base model
        base_predictions = {}
        
        for model_name in self.model_order:
            model = self.models[model_name]
            selected_feats = self.selected_features.get(model_name, [])
            
            # Use selected features if available, otherwise all features
            if selected_feats and len(selected_feats) > 0:
                # Filter to available features only
                available_feats = [f for f in selected_feats if f in model_input.columns]
                if len(available_feats) != len(selected_feats):
                    missing = set(selected_feats) - set(available_feats)
                    raise ValueError(
                        f"Model '{model_name}' requires features not in input: {missing}"
                    )
                X_model = model_input[available_feats]
            else:
                X_model = model_input
            
            # Predict with base model
            base_predictions[model_name] = model.predict(X_model)
        
        # Create meta-features (in sorted order for reproducibility)
        meta_features = pd.DataFrame({
            f"{name}_pred": base_predictions[name]
            for name in self.model_order
        }, index=model_input.index)
        
        # Apply meta-learner
        ensemble_risk_scores = self.meta_learner.predict(meta_features)
        
        # Return as DataFrame (MLflow standard)
        return pd.DataFrame({
            'risk_score': ensemble_risk_scores
        }, index=model_input.index)
    
    def predict_survival_function(self, model_input):
        """
        Predict survival functions using the Cox meta-learner.
        
        This method generates meta-features from base model predictions,
        then delegates to the meta-learner's predict_survival_function.
        Required for Brier score computation in validation pipeline.
        
        Parameters
        ----------
        model_input : pandas.DataFrame
            Input features with same schema as training data
        
        Returns
        -------
        list of StepFunction
            Survival function for each sample (from scikit-survival)
        """
        # Validate input
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError(f"Input must be pandas.DataFrame, got {type(model_input)}")
        
        # Generate predictions from each base model (same logic as predict())
        base_predictions = {}
        
        for model_name in self.model_order:
            model = self.models[model_name]
            selected_feats = self.selected_features.get(model_name, [])
            
            # Use selected features if available, otherwise all features
            if selected_feats and len(selected_feats) > 0:
                # Filter to available features only
                available_feats = [f for f in selected_feats if f in model_input.columns]
                if len(available_feats) != len(selected_feats):
                    missing = set(selected_feats) - set(available_feats)
                    raise ValueError(
                        f"Model '{model_name}' requires features not in input: {missing}"
                    )
                X_model = model_input[available_feats]
            else:
                X_model = model_input
            
            # Predict with base model
            base_predictions[model_name] = model.predict(X_model)
        
        # Create meta-features (in sorted order for reproducibility)
        meta_features = pd.DataFrame({
            f"{name}_pred": base_predictions[name]
            for name in self.model_order
        }, index=model_input.index)
        
        # Delegate to meta-learner's predict_survival_function
        # CoxPHSurvivalAnalysis has this method, returns list of StepFunction
        return self.meta_learner.predict_survival_function(meta_features)
