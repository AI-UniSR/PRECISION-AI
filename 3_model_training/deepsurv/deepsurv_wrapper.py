"""
DeepSurvWrapper: Scikit-survival compatible wrapper for pycox DeepSurv (CoxPH) model.

This wrapper provides a unified interface compatible with scikit-survival models,
enabling seamless integration with the multi-model-pipeline evaluation framework.
"""

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import CoxPH
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path


class DeepSurvWrapper(BaseEstimator):
    """
    Wrapper for pycox CoxPH (DeepSurv) model with scikit-survival API compatibility.
    
    This wrapper implements:
    - predict(): Returns risk scores (higher = higher risk)
    - predict_survival_function(): Returns survival functions for Brier score computation
    - fit(): Trains the model (delegates to separate training script)
    
    The wrapper handles:
    - Data type conversion (float64 -> float32 for PyTorch)
    - Preprocessing pipeline (imputation, scaling)
    - Baseline hazards computation and caching
    - Survival function prediction
    
    Parameters
    ----------
    model_path : str or Path
        Path to directory containing trained model artifacts:
        - deepsurv_net.pt: Network weights
        - baseline_hazards.pkl: Baseline hazards
        - hyperparameters.json: Model architecture
        - selected_features.txt: Feature names (optional)
        - scaler.pkl: Fitted RobustScaler
        - imputer.pkl: Fitted IterativeImputer
    """
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.hyperparameters = None
        self.feature_names = None
        self.baseline_hazards = None
        self.scaler = None
        self.imputer = None
        self._is_fitted = False
        
    def _load_artifacts(self):
        """Load all model artifacts from disk."""
        import json
        
        if self._is_fitted:
            return
        
        # Load hyperparameters
        with open(self.model_path / 'hyperparameters.json', 'r') as f:
            self.hyperparameters = json.load(f)
        
        # Load feature names if available
        features_file = self.model_path / 'selected_features.txt'
        if features_file.exists():
            with open(features_file, 'r') as f:
                self.feature_names = [line.strip() for line in f if line.strip()]
        
        # Load preprocessing components
        self.imputer = joblib.load(self.model_path / 'imputer.pkl')
        self.scaler = joblib.load(self.model_path / 'scaler.pkl')
        
        # Reconstruct network architecture
        n_layers = self.hyperparameters['n_layers']
        num_nodes = [self.hyperparameters[f'num_nodes_l{i}'] for i in range(n_layers)]
        batch_norm = self.hyperparameters['batch_norm']
        dropout = self.hyperparameters['dropout']
        input_dim = self.hyperparameters['input_dim']
        
        # Create network
        net = tt.practical.MLPVanilla(
            in_features=input_dim,
            num_nodes=num_nodes,
            out_features=1,
            batch_norm=batch_norm,
            dropout=dropout,
            output_bias=False
        )
        
        # Create model (optimizer doesn't matter for inference)
        self.model = CoxPH(net, tt.optim.Adam())
        
        # Load weights
        self.model.load_net(str(self.model_path / 'deepsurv_net.pt'))
        
        # Set to evaluation mode
        self.model.net.eval()
        
        # Load baseline hazards
        self.baseline_hazards = pd.read_pickle(str(self.model_path / 'baseline_hazards.pkl'))
        self.model.baseline_hazards_ = self.baseline_hazards
        
        self._is_fitted = True
        
    def _preprocess(self, X):
        """
        Apply preprocessing pipeline: imputation -> scaling -> float32 conversion.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features
            
        Returns
        -------
        X_processed : np.ndarray (float32)
            Preprocessed features ready for PyTorch
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            if self.feature_names is not None:
                X = pd.DataFrame(X, columns=self.feature_names)
            else:
                X = pd.DataFrame(X)
        
        # Filter to selected features if specified
        if self.feature_names is not None:
            X = X[self.feature_names]
        
        # Apply imputation
        X_imputed = self.imputer.transform(X)
        
        # Apply scaling
        X_scaled = self.scaler.transform(X_imputed)
        
        # Convert to float32 for PyTorch
        X_processed = X_scaled.astype('float32')
        
        return X_processed
        
    def predict(self, X):
        """
        Predict risk scores (log partial hazard).
        
        Higher values indicate higher risk (worse prognosis).
        Compatible with concordance index computation.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Test features
            
        Returns
        -------
        risk_scores : np.ndarray
            Risk scores (1D array)
        """
        self._load_artifacts()
        
        # Preprocess
        X_processed = self._preprocess(X)
        
        # Predict using pycox model
        # Note: pycox CoxPH.predict() returns log partial hazard
        with torch.no_grad():
            risk_scores = self.model.predict(X_processed)
        
        return risk_scores
    
    def predict_survival_function(self, X, return_array=True):
        """
        Predict survival functions S(t|x) for each sample.
        
        Returns callable survival functions compatible with scikit-survival
        for Brier score and time-dependent AUC computation.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Test features
        return_array : bool, default=True
            If True, returns array of StepFunction objects (scikit-survival format)
            If False, returns pycox DataFrame format
            
        Returns
        -------
        survival_functions : np.ndarray of StepFunction
            Array of survival functions, one per sample
            Each function maps time -> survival probability
        """
        self._load_artifacts()
        
        # Preprocess
        X_processed = self._preprocess(X)
        
        # Predict survival curves
        surv_df = self.model.predict_surv_df(X_processed)
        
        if not return_array:
            return surv_df
        
        # Convert to scikit-survival StepFunction format
        from sksurv.functions import StepFunction
        
        survival_functions = []
        time_points = surv_df.index.values
        
        for col in surv_df.columns:
            surv_probs = surv_df[col].values
            # StepFunction expects (time, probability) pairs
            survival_functions.append(StepFunction(time_points, surv_probs))
        
        return np.array(survival_functions)
    
    def fit(self, X, y):
        """
        Placeholder fit method (not used - training done separately).
        
        Training is performed by the standalone training script with
        Optuna hyperparameter optimization. This method exists for
        scikit-learn API compatibility.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : structured array
            Survival target (event, time)
            
        Returns
        -------
        self
        """
        raise NotImplementedError(
            "DeepSurvWrapper does not support direct training. "
            "Use the standalone training script with Optuna optimization."
        )
    
    def __repr__(self):
        if self._is_fitted:
            return f"DeepSurvWrapper(model_path={self.model_path}, fitted=True, n_features={len(self.feature_names) if self.feature_names else 'unknown'})"
        else:
            return f"DeepSurvWrapper(model_path={self.model_path}, fitted=False)"
    
    def __getstate__(self):
        """Custom pickle support - save only model path."""
        return {'model_path': self.model_path}
    
    def __setstate__(self, state):
        """Custom unpickle support - reload from model path."""
        self.model_path = state['model_path']
        self.model = None
        self.hyperparameters = None
        self.feature_names = None
        self.baseline_hazards = None
        self.scaler = None
        self.imputer = None
        self._is_fitted = False
