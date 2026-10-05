"""Stacking ensemble: out-of-fold predictions, Cox meta-learner, registration.

The four base learners (cox, rsf, xgboost, xgbse) are refitted, with their
tuned hyperparameters and their own features, in each fold of a 5-fold
stratified split of the training set (seed 42), and predict the left-out fold.
An unpenalised Cox model fitted on these out-of-fold predictions is the
meta-learner. To predict, the base learners fitted on the whole training set
feed the meta-learner, whose linear predictor is the ensemble risk score.

The ensemble is registered in MLflow (ensemble_pyfunc.py) and then reloaded
from the registry; its predictions for the first 100 test patients must match
those of the in-memory ensemble (maximum absolute difference < 1e-6).
"""

import json
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis

from ensemble_pyfunc import EnsembleSurvivalPyFunc

REGISTERED_MODEL_NAME = "ensemble_survival_model"


class CoxEnsembleSurvivalPredictor:
    """In-memory ensemble: base learners -> meta-features -> Cox meta-learner."""

    def __init__(self, base_models, selected_features_all, meta_learner):
        missing = set(base_models) - set(selected_features_all)
        if missing:
            raise ValueError(f"No feature list for base learners {missing}")
        self.base_models = base_models
        self.selected_features_all = selected_features_all
        self.meta_learner = meta_learner
        self.model_names = sorted(base_models)
        self.meta_feature_names = [f"{name}_pred" for name in self.model_names]

    def _meta_features(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")
        predictions = {}
        for name in self.model_names:
            features = self.selected_features_all[name]
            if features:
                missing = [f for f in features if f not in X.columns]
                if missing:
                    raise ValueError(f"Model '{name}' requires features missing from X: {missing}")
                X_model = X[features]
            else:
                X_model = X
            predictions[f"{name}_pred"] = self.base_models[name].predict(X_model)
        return pd.DataFrame(predictions, index=X.index)[self.meta_feature_names]

    def predict(self, X):
        return self.meta_learner.predict(self._meta_features(X))

    def predict_survival_function(self, X):
        return self.meta_learner.predict_survival_function(self._meta_features(X))


def generate_oof_predictions(models, selected_features_all, X_train, y_train, n_splits=5):
    """Out-of-fold predictions of each stacked learner (DeepSurv is not stacked)."""
    stack_models = {name: model for name, model in models.items() if name != "deepsurv"}
    oof = {name: np.zeros(len(y_train)) for name in stack_models}
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in folds.split(X_train, y_train["event"]):
        for name, model in stack_models.items():
            features = selected_features_all[name]
            columns = [f for f in features if f in X_train.columns] if features else list(X_train.columns)
            fold_model = clone(model).fit(X_train.iloc[train_idx][columns], y_train[train_idx])
            oof[name][val_idx] = fold_model.predict(X_train.iloc[val_idx][columns])
    return oof, stack_models


def train_meta_learner(oof_predictions, y_train, X_train, output_dir):
    """Unpenalised Cox model on the out-of-fold predictions (columns in name order)."""
    names = sorted(oof_predictions)
    X_meta = pd.DataFrame({f"{name}_pred": oof_predictions[name] for name in names}, index=X_train.index)
    meta_learner = CoxPHSurvivalAnalysis().fit(X_meta, y_train)

    ensemble_dir = os.path.join(output_dir, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    pd.DataFrame({"base_model": names, "meta_coefficient": meta_learner.coef_}).to_csv(
        os.path.join(ensemble_dir, "meta_coefficients.csv"), index=False)
    joblib.dump(meta_learner, os.path.join(ensemble_dir, "meta_learner.pkl"))
    pd.DataFrame({"id": X_train.index, **{f"{name}_oof": oof_predictions[name] for name in names}}).to_csv(
        os.path.join(ensemble_dir, "oof_predictions.csv"), index=False)
    print("Meta-learner coefficients:", dict(zip(names, np.round(meta_learner.coef_, 4))))
    return meta_learner


def register_ensemble(ensemble, selected_features_all, ensemble_dir, X_test, tolerance=1e-6):
    """Log the ensemble as an MLflow pyfunc model, register it and check the round trip."""
    artifacts = {}
    for name, model in ensemble.base_models.items():
        artifacts[f"{name}_model"] = os.path.join(ensemble_dir, f"{name}_model.pkl")
        joblib.dump(model, artifacts[f"{name}_model"])
    artifacts["meta_learner"] = os.path.join(ensemble_dir, "meta_learner.pkl")
    artifacts["selected_features"] = os.path.join(ensemble_dir, "selected_features.json")
    with open(artifacts["selected_features"], "w") as f:
        json.dump({k: list(v) if v else [] for k, v in selected_features_all.items()}, f, indent=2)

    here = os.path.dirname(os.path.abspath(__file__))
    info = mlflow.pyfunc.log_model(
        artifact_path="ensemble_model_registered",
        python_model=EnsembleSurvivalPyFunc(),
        artifacts=artifacts,
        # files defining the classes of the pickled base learners and of the pyfunc model
        code_path=[os.path.join(here, f) for f in
                   ("ensemble_pyfunc.py", "xgboost_survival_model.py", "xgbse_pipeline_wrapper.py")],
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    registered = mlflow.pyfunc.load_model(info.model_uri)
    sample = X_test.head(min(100, len(X_test)))
    diff = np.abs(ensemble.predict(sample) - registered.predict(sample)["risk_score"].values)
    mlflow.log_metric("model_validation_max_diff", float(diff.max()))
    if diff.max() >= tolerance:
        raise RuntimeError(f"Registered model differs from the in-memory ensemble (max |diff| {diff.max():.2e})")
    surv = registered._model_impl.python_model.predict_survival_function(sample)
    if len(surv) != len(sample):
        raise RuntimeError("Registered model returns the wrong number of survival functions")
    print(f"Registered {info.model_uri}; round-trip max |diff| {diff.max():.2e} on {len(sample)} patients")


def create_and_evaluate_ensemble(models, selected_features_all, meta_learner, X_test, y_test, y_train,
                                 output_dir, grid_months, clinical_months, evaluate_fn):
    """Build, save and register the ensemble, then evaluate it on the hold-out set."""
    ensemble = CoxEnsembleSurvivalPredictor(models, selected_features_all, meta_learner)
    risk_scores = ensemble.predict(X_test)
    ensemble_dir = os.path.join(output_dir, "ensemble")
    joblib.dump(ensemble, os.path.join(ensemble_dir, "ensemble_model.pkl"))
    register_ensemble(ensemble, selected_features_all, ensemble_dir, X_test)
    return evaluate_fn(ensemble, X_test, y_test, y_train, "ensemble", output_dir,
                       grid_months, clinical_months, risk_scores=risk_scores)
