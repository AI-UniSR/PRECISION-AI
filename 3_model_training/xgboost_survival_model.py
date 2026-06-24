"""
XGBoost Survival Model - Scikit-learn compatible wrapper.

This module provides a reusable XGBoostSurvival estimator that can be
imported both during training and evaluation/inference.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

# Import for non-parametric survival function
try:
    from lifelines import KaplanMeierFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("Warning: lifelines not available. Install with: pip install lifelines")

# Check XGBoost availability
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


class XGBoostSurvival(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for XGBoost survival with AFT objective.

    Uses the official XGBoost API (DMatrix + xgb.train) with ranged labels
    (label_lower_bound, label_upper_bound) as required for 'survival:aft'.

    y must be a sksurv-style structured array with fields:
      - 'event' (bool)
      - 'time'  (float)

    predict() returns a risk score:
      risk = -log(predicted survival time)
    suitable for concordance_index_censored (higher = worse prognosis).
    """

    def __init__(
        self,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=1,
        subsample=1.0,
        colsample_bytree=1.0,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        aft_loss_distribution="normal",
        aft_loss_distribution_scale=1.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")

        # sklearn-style hyperparameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.aft_loss_distribution = aft_loss_distribution
        self.aft_loss_distribution_scale = aft_loss_distribution_scale
        self.tree_method = tree_method
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity

        # attributes populated in fit
        self.booster_ = None
        self.feature_importances_ = None
        self.n_features_in_ = None
        self.feature_names_ = None

    def _to_numpy(self, X):
        if hasattr(X, "values"):
            return X.values
        return np.asarray(X)

    def fit(self, X, y):
        """
        X : array-like (n_samples, n_features) or DataFrame
        y : structured array with fields 'event' (bool) and 'time' (float)
            as produced by sksurv.util.Surv
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")

        X_np = self._to_numpy(X)
        self.n_features_in_ = X_np.shape[1]

        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"f{i}" for i in range(self.n_features_in_)]

        # Extract event/time from structured array
        event = y["event"].astype(bool)
        time = y["time"].astype(float)

        # XGBoost survival:aft label convention:
        # - uncensored: [t, t]
        # - right-censored: [t, +inf)
        y_lower = time
        y_upper = np.where(event, time, np.inf)

        dtrain = xgb.DMatrix(X_np)
        dtrain.set_float_info("label_lower_bound", y_lower)
        dtrain.set_float_info("label_upper_bound", y_upper)

        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": self.aft_loss_distribution,
            "aft_loss_distribution_scale": self.aft_loss_distribution_scale,
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "alpha": self.reg_alpha,
            "lambda": self.reg_lambda,
            "tree_method": self.tree_method,
            "verbosity": self.verbosity,
        }

        if self.n_jobs is not None and self.n_jobs != 0:
            params["nthread"] = self.n_jobs
        if self.random_state is not None:
            params["seed"] = self.random_state

        self.booster_ = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
        )

        # Feature importances: map f0, f1, ... to column order
        score_dict = self.booster_.get_score(importance_type="gain")
        importances = np.zeros(self.n_features_in_, dtype=float)
        for i in range(self.n_features_in_):
            importances[i] = score_dict.get(f"f{i}", 0.0)
        self.feature_importances_ = importances

        # Store training data for non-parametric survival function prediction
        self.X_train_ = self._to_numpy(X).copy()
        self.y_train_ = y.copy()

        return self

    def predict(self, X):
        """
        Returns risk score = -log(predicted survival time).
        Use with concordance_index_censored (higher = higher risk).
        """
        if self.booster_ is None:
            raise RuntimeError("Model is not fitted yet.")

        X_np = self._to_numpy(X)
        dtest = xgb.DMatrix(X_np)
        log_time_pred = self.booster_.predict(dtest)
        risk_scores = -log_time_pred
        return risk_scores

    def predict_survival_function(self, X, n_risk_groups=5):
        """
        Non-parametric survival function using stratified Kaplan-Meier.
        
        Uses ML-based approach: stratifies training data by predicted risk scores,
        fits Kaplan-Meier estimator per stratum, assigns test samples to appropriate
        stratum based on their risk scores. Avoids parametric AFT assumptions.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        n_risk_groups : int, default=5
            Number of risk strata for KM estimation (quintiles)
            
        Returns
        -------
        list of callable
            Survival functions (one per sample). Each callable takes time t
            and returns S(t), the probability of surviving beyond time t.
        """
        if not LIFELINES_AVAILABLE:
            raise ImportError("lifelines required for survival function prediction. "
                            "Install with: pip install lifelines")
        
        if self.booster_ is None:
            raise RuntimeError("Model is not fitted yet.")
        
        if not hasattr(self, 'X_train_') or not hasattr(self, 'y_train_'):
            raise RuntimeError("Training data not stored. Model must be fitted with "
                             "current version to support survival function prediction.")
        
        # Get risk scores for test and training data
        risk_scores_test = self.predict(X)
        dtrain = xgb.DMatrix(self.X_train_)
        log_time_train = self.booster_.predict(dtrain)
        risk_scores_train = -log_time_train
        
        # Create risk groups based on training data quantiles
        risk_quantiles = np.percentile(risk_scores_train, 
                                      np.linspace(0, 100, n_risk_groups + 1))
        
        # Fit Kaplan-Meier estimator for each risk group
        km_estimators = []
        for i in range(n_risk_groups):
            # Select samples in this risk group
            if i == n_risk_groups - 1:
                # Last group: include upper boundary
                mask = (risk_scores_train >= risk_quantiles[i]) & \
                       (risk_scores_train <= risk_quantiles[i+1])
            else:
                mask = (risk_scores_train >= risk_quantiles[i]) & \
                       (risk_scores_train < risk_quantiles[i+1])
            
            if mask.sum() > 0:
                # Fit KM estimator on this group
                kmf = KaplanMeierFitter()
                kmf.fit(self.y_train_["time"][mask], 
                       event_observed=self.y_train_["event"][mask])
                km_estimators.append(kmf)
            else:
                # Empty group - use None as placeholder
                km_estimators.append(None)
        
        # Fit overall KM as fallback for empty groups
        kmf_overall = KaplanMeierFitter()
        kmf_overall.fit(self.y_train_["time"], 
                       event_observed=self.y_train_["event"])
        
        # Create survival function for each test sample
        def make_survival_function(risk_score):
            """Create survival function for given risk score."""
            # Find appropriate risk group
            group_idx = np.digitize(risk_score, risk_quantiles) - 1
            group_idx = np.clip(group_idx, 0, n_risk_groups - 1)
            
            # Get KM estimator (use overall if group is empty)
            kmf = km_estimators[group_idx]
            if kmf is None:
                kmf = kmf_overall
            
            # Extract survival function values
            km_times = kmf.survival_function_.index.values
            km_probs = kmf.survival_function_.values.flatten()
            
            # Return callable that interpolates KM curve
            def survival_fn(t):
                """Survival probability at time t."""
                # Linear interpolation with proper boundaries:
                # - left=1.0: S(0) = 1 (everyone alive at time 0)
                # - right=km_probs[-1]: constant extrapolation beyond observed times
                return np.interp(t, km_times, km_probs, 
                               left=1.0, right=km_probs[-1])
            
            return survival_fn
        
        # Return list of survival functions (one per test sample)
        return [make_survival_function(rs) for rs in risk_scores_test]
