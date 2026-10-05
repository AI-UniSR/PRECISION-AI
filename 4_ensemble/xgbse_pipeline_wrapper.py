"""scikit-learn wrapper of the XGBSE stacked Weibull model.

predict() returns minus the predicted median survival time: the first time
bin at which the predicted survival curve is <= 0.5, or the last bin if it
never drops to 0.5. predict_survival_function() interpolates the predicted
curve linearly between time bins, with S = 1 before the first bin and S = 0
after the last.

No imputation is applied; XGBoost handles missing values natively. An
identical copy of this file is in 4_ensemble/, where it is needed to load the
pickled learner and is packaged with the MLflow model.
"""

import numpy as np
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, RegressorMixin
from xgbse import XGBSEStackedWeibull


class XGBSEPipeline(BaseEstimator, RegressorMixin):

    def __init__(self, xgb_params=None, num_boost_round=100, random_state=42, enable_categorical=False):
        self.xgb_params = xgb_params or {}
        self.num_boost_round = num_boost_round
        self.random_state = random_state  # not used: the booster seed is xgb_params["seed"]
        self.enable_categorical = enable_categorical
        self.xgbse_model_ = None

    def fit(self, X, y):
        """X: features (may contain NaN); y: structured array with 'event' and 'time'."""
        if not (hasattr(y, "dtype") and y.dtype.names):
            y = np.array(list(zip(y["event"].astype(bool), y["time"].astype(float))),
                         dtype=[("event", bool), ("time", float)])
        self.xgbse_model_ = XGBSEStackedWeibull(xgb_params=self.xgb_params,
                                                enable_categorical=self.enable_categorical)
        self.xgbse_model_.fit(X, y, num_boost_round=self.num_boost_round)
        return self

    def predict(self, X):
        """Risk score: minus the predicted median survival time."""
        if self.xgbse_model_ is None:
            raise ValueError("Model not fitted yet")
        curves = self.xgbse_model_.predict(X)  # rows: patients, columns: time bins
        time_bins = curves.columns.values
        median_times = []
        for i in range(len(X)):
            below_half = np.where(curves.iloc[i].values <= 0.5)[0]
            median_times.append(time_bins[below_half[0]] if len(below_half) > 0 else time_bins[-1])
        return -np.array(median_times)

    def predict_survival_function(self, X):
        """One callable S(t) per patient."""
        if self.xgbse_model_ is None:
            raise ValueError("Model not fitted yet")
        curves = self.xgbse_model_.predict(X)
        times = curves.columns.values
        return [interp1d(times, curves.iloc[i].values, kind="linear", bounds_error=False,
                         fill_value=(1.0, 0.0)) for i in range(len(X))]

    @property
    def feature_importances_(self):
        if self.xgbse_model_ is not None and hasattr(self.xgbse_model_, "feature_importances_"):
            return self.xgbse_model_.feature_importances_
        return None
