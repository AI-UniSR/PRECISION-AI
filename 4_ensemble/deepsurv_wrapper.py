"""scikit-survival style interface to a trained DeepSurv network (pycox CoxPH).

The wrapper reads the artifacts written by model_train.py (network weights,
baseline hazards, architecture, fitted imputer and scaler, feature list) so
that DeepSurv can be evaluated with the same code as the other learners.
predict() returns the network output (log partial hazard); higher means
higher risk. Only the model path is pickled; the artifacts are reloaded on use.

An identical copy of this file is in 4_ensemble/.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import CoxPH
from sklearn.base import BaseEstimator


class DeepSurvWrapper(BaseEstimator):
    """DeepSurv model loaded from the folder model_path."""

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
        if self._is_fitted:
            return
        with open(self.model_path / "hyperparameters.json") as f:
            self.hyperparameters = json.load(f)
        features_file = self.model_path / "selected_features.txt"
        if features_file.exists():
            with open(features_file) as f:
                self.feature_names = [line.strip() for line in f if line.strip()]
        self.imputer = joblib.load(self.model_path / "imputer.pkl")
        self.scaler = joblib.load(self.model_path / "scaler.pkl")

        hp = self.hyperparameters
        net = tt.practical.MLPVanilla(
            in_features=hp["input_dim"],
            num_nodes=[hp[f"num_nodes_l{i}"] for i in range(hp["n_layers"])],
            out_features=1,
            batch_norm=hp["batch_norm"],
            dropout=hp["dropout"],
            output_bias=False,
        )
        self.model = CoxPH(net, tt.optim.Adam())  # the optimiser is not used for prediction
        self.model.load_net(str(self.model_path / "deepsurv_net.pt"))
        self.model.net.eval()
        self.baseline_hazards = pd.read_pickle(str(self.model_path / "baseline_hazards.pkl"))
        self.model.baseline_hazards_ = self.baseline_hazards
        self._is_fitted = True

    def _preprocess(self, X):
        """Imputation and scaling fitted on the training set, then float32 for PyTorch."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        if self.feature_names is not None:
            X = X[self.feature_names]
        return self.scaler.transform(self.imputer.transform(X)).astype("float32")

    def predict(self, X):
        """Log partial hazard, shape (n, 1)."""
        self._load_artifacts()
        with torch.no_grad():
            return self.model.predict(self._preprocess(X))

    def predict_survival_function(self, X, return_array=True):
        """Survival curves as sksurv StepFunctions (or the pycox DataFrame)."""
        self._load_artifacts()
        surv_df = self.model.predict_surv_df(self._preprocess(X))
        if not return_array:
            return surv_df
        from sksurv.functions import StepFunction
        times = surv_df.index.values
        return np.array([StepFunction(times, surv_df[col].values) for col in surv_df.columns])

    def fit(self, X, y):
        raise NotImplementedError("DeepSurv is trained by model_train.py.")

    def __repr__(self):
        return f"DeepSurvWrapper(model_path={self.model_path}, loaded={self._is_fitted})"

    def __getstate__(self):
        return {"model_path": self.model_path}

    def __setstate__(self, state):
        self.__init__(state["model_path"])
