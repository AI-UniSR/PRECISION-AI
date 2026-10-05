"""scikit-learn wrapper of an XGBoost accelerated failure time (AFT) model.

The booster is trained with the 'survival:aft' objective on interval labels:
[t, t] for deaths and [t, +inf) for censored patients. For this objective
booster.predict() returns the predicted survival time on the original time
scale; predict() returns minus that time, so that higher means higher risk.

The AFT objective gives no survival function, so predict_survival_function()
builds one from the training data: the training risk scores are cut at their
quintiles, a Kaplan-Meier curve is fitted in each quintile, and each patient
receives the curve of the quintile containing their own score, linearly
interpolated between time points, with S(0) = 1 and the last value carried
forward. The fitted object therefore keeps a copy of the training data.

An identical copy of this file is in 4_ensemble/, where it is needed to load
the pickled learner and is packaged with the MLflow model.
"""

import numpy as np
import xgboost as xgb
from lifelines import KaplanMeierFitter
from sklearn.base import BaseEstimator, RegressorMixin


class XGBoostSurvival(BaseEstimator, RegressorMixin):
    """XGBoost AFT model; y is a structured array with fields 'event' and 'time'."""

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
        self.booster_ = None
        self.feature_importances_ = None
        self.n_features_in_ = None
        self.feature_names_ = None

    @staticmethod
    def _to_numpy(X):
        return X.values if hasattr(X, "values") else np.asarray(X)

    def fit(self, X, y):
        X_np = self._to_numpy(X)
        self.n_features_in_ = X_np.shape[1]
        self.feature_names_ = (list(X.columns) if hasattr(X, "columns")
                               else [f"f{i}" for i in range(self.n_features_in_)])

        time = y["time"].astype(float)
        dtrain = xgb.DMatrix(X_np)
        dtrain.set_float_info("label_lower_bound", time)
        dtrain.set_float_info("label_upper_bound", np.where(y["event"].astype(bool), time, np.inf))

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
        self.booster_ = xgb.train(params=params, dtrain=dtrain, num_boost_round=self.n_estimators)

        gain = self.booster_.get_score(importance_type="gain")
        self.feature_importances_ = np.array([gain.get(f"f{i}", 0.0) for i in range(self.n_features_in_)])

        # Kept for predict_survival_function()
        self.X_train_ = X_np.copy()
        self.y_train_ = y.copy()
        return self

    def predict(self, X):
        """Risk score: minus the predicted survival time."""
        if self.booster_ is None:
            raise RuntimeError("Model is not fitted yet.")
        predicted_time = self.booster_.predict(xgb.DMatrix(self._to_numpy(X)))
        return -predicted_time

    def predict_survival_function(self, X, n_risk_groups=5):
        """One callable S(t) per patient (Kaplan-Meier curve of the risk quintile)."""
        if self.booster_ is None:
            raise RuntimeError("Model is not fitted yet.")
        risk_test = self.predict(X)
        risk_train = -self.booster_.predict(xgb.DMatrix(self.X_train_))
        cuts = np.percentile(risk_train, np.linspace(0, 100, n_risk_groups + 1))

        km_curves = []
        for i in range(n_risk_groups):
            upper = risk_train <= cuts[i + 1] if i == n_risk_groups - 1 else risk_train < cuts[i + 1]
            mask = (risk_train >= cuts[i]) & upper
            if mask.sum() > 0:
                km_curves.append(KaplanMeierFitter().fit(self.y_train_["time"][mask],
                                                         event_observed=self.y_train_["event"][mask]))
            else:
                km_curves.append(None)
        km_all = KaplanMeierFitter().fit(self.y_train_["time"], event_observed=self.y_train_["event"])

        def survival_function(risk_score):
            group = int(np.clip(np.digitize(risk_score, cuts) - 1, 0, n_risk_groups - 1))
            kmf = km_curves[group] if km_curves[group] is not None else km_all
            times = kmf.survival_function_.index.values
            probs = kmf.survival_function_.values.flatten()
            return lambda t: np.interp(t, times, probs, left=1.0, right=probs[-1])

        return [survival_function(score) for score in risk_test]
