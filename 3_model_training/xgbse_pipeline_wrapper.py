"""
XGBSE Pipeline Wrapper for model unpickling in evaluation.
This is a copy of XGBSEPipeline to ensure proper deserialization.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

try:
    from xgbse import XGBSEStackedWeibull
    XGBSE_AVAILABLE = True
except ImportError:
    XGBSE_AVAILABLE = False
    print("Warning: XGBSE not available")


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
