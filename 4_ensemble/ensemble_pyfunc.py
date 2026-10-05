"""MLflow pyfunc model of the stacked ensemble (registered as 'ensemble_survival_model').

load_context() loads the four base learners, the Cox meta-learner and the
feature list of each learner from the model artifacts. predict() returns the
ensemble risk score, i.e. the linear predictor of the meta-learner applied to
the base-learner predictions; predict_survival_function() returns the
meta-learner's Breslow survival curves. The base-learner predictions enter the
meta-learner in alphabetical order of the learner names (cox, rsf, xgboost,
xgbse), as when it was fitted.

This file is stored with the registered model (MLflow code_path).
"""

import json

import joblib
import pandas as pd
from mlflow.pyfunc import PythonModel


class EnsembleSurvivalPyFunc(PythonModel):

    def load_context(self, context):
        self.cox_model = joblib.load(context.artifacts["cox_model"])
        self.rsf_model = joblib.load(context.artifacts["rsf_model"])
        self.xgboost_model = joblib.load(context.artifacts["xgboost_model"])
        self.xgbse_model = joblib.load(context.artifacts["xgbse_model"])
        self.meta_learner = joblib.load(context.artifacts["meta_learner"])
        with open(context.artifacts["selected_features"]) as f:
            self.selected_features = json.load(f)
        self.models = {"cox": self.cox_model, "rsf": self.rsf_model,
                       "xgboost": self.xgboost_model, "xgbse": self.xgbse_model}
        self.model_order = sorted(self.models)

    def _meta_features(self, model_input):
        """Base-learner predictions, one column per learner, each on its own features."""
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(model_input)}")
        predictions = {}
        for name in self.model_order:
            features = self.selected_features.get(name, [])
            if features:
                missing = set(features) - set(model_input.columns)
                if missing:
                    raise ValueError(f"Model '{name}' requires features not in input: {missing}")
                X = model_input[features]
            else:
                X = model_input
            predictions[f"{name}_pred"] = self.models[name].predict(X)
        return pd.DataFrame(predictions, index=model_input.index)

    def predict(self, context, model_input):
        """Ensemble risk score (higher = worse prognosis), column 'risk_score'."""
        risk = self.meta_learner.predict(self._meta_features(model_input))
        return pd.DataFrame({"risk_score": risk}, index=model_input.index)

    def predict_survival_function(self, model_input):
        """Survival curves (sksurv StepFunction), one per patient."""
        return self.meta_learner.predict_survival_function(self._meta_features(model_input))
