"""Scoring script of the online endpoint behind the PRECISION-AI web application.

Reference only: it runs inside an Azure ML managed online endpoint, and the
deployment package (SHAP explainer, training-set summaries) is built outside
this repository. For one patient it returns the ensemble risk score, its
percentile among the training risk scores, the predicted survival curve and
median survival, and a SHAP explanation.

The predicted median survival is the first time at which the patient's curve
S(t) = S0(t)^exp(eta) falls to 0.5 or below, where S0 is the Breslow baseline
survival of the Cox meta-learner and eta the ensemble risk score; it is None
if the curve stays above 0.5.
"""

import json
import logging
import os

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

training_aggregates: dict = {}


def _get_survival_model(obj):
    """Innermost object exposing predict_survival_function (unwraps the MLflow pyfunc)."""
    if hasattr(obj, "predict_survival_function"):
        return obj
    if hasattr(obj, "_model_impl"):
        impl = obj._model_impl
        if hasattr(impl, "predict_survival_function"):
            return impl
        if hasattr(impl, "sklearn_model"):
            return _get_survival_model(impl.sklearn_model)
        if hasattr(impl, "python_model"):
            return _get_survival_model(impl.python_model)
    if hasattr(obj, "python_model"):
        return _get_survival_model(obj.python_model)
    return None


def _compute_median_survival(baseline_x, baseline_y, risk_score):
    """First time t with S0(t) <= 0.5^exp(-eta), i.e. S(t | eta) <= 0.5; None if never."""
    if baseline_x is None or len(baseline_x) == 0:
        return None
    indices = np.where(baseline_y <= 0.5 ** np.exp(-risk_score))[0]
    return float(baseline_x[indices[0]]) if len(indices) else None


def init():
    """Load the model, the SHAP explainer and the training-set summaries (once per container)."""
    global model, survival_model, explainer, training_aggregates, training_predictions_df
    global survival_training_data, baseline_x, baseline_y, percentile_median_survivals

    model_dir = os.getenv("AZUREML_MODEL_DIR")
    if "MLmodel" not in os.listdir(model_dir):
        subdirs = [d.path for d in os.scandir(model_dir) if d.is_dir()]
        if subdirs:
            model_dir = subdirs[0]

    model = mlflow.pyfunc.load_model(model_dir)
    survival_model = _get_survival_model(model)

    # Breslow baseline survival of the meta-learner: same model that produces the risk score
    baseline_x, baseline_y = np.array([]), np.array([])
    baseline = getattr(getattr(survival_model, "meta_learner", None), "baseline_survival_", None)
    if baseline is not None:
        baseline_x, baseline_y = np.array(baseline.x), np.array(baseline.y)
    else:
        logging.warning("meta_learner.baseline_survival_ not found; median survival unavailable")

    # SHAP sends float arrays; call the pyfunc model directly to bypass the input-schema
    # check, which rejects floats for integer-typed columns such as 'ps ecog'
    explainer = mlflow.shap.load_explainer(os.path.join(model_dir, "shap_explainer"))
    schema = model.metadata.get_input_schema() if model.metadata else None
    schema_names = schema.input_names() if schema else None

    def predict_wrapper(X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=schema_names)
        result = survival_model.predict(None, X)
        return result.values.ravel() if isinstance(result, pd.DataFrame) else result

    explainer.model = predict_wrapper

    def read_json(*parts):
        path = os.path.join(model_dir, *parts)
        if not os.path.exists(path):
            logging.warning(f"{path} not found")
            return None
        with open(path) as f:
            return json.load(f)

    training_aggregates = read_json("extra_artifacts/training_data_aggregates", "training_data_aggregates.json") or {}
    predictions = read_json("extra_artifacts/training_predictions", "training_predictions.json")
    if predictions is not None:
        training_predictions_df = pd.DataFrame.from_dict(predictions, orient="index")
    survival_training_data = read_json("extra_artifacts/survival_training_data", "survival_training_data.json") or {}

    # Median survival at selected percentiles of the training risk scores (gauge of the web app)
    percentile_median_survivals = {}
    try:
        training_risk = training_predictions_df["predicted_risk"].values
        for p in [1] + list(range(10, 100, 10)) + [99]:
            percentile_median_survivals[str(p)] = _compute_median_survival(
                baseline_x, baseline_y, float(np.percentile(training_risk, p)))
    except Exception as e:
        logging.warning(f"Percentile median survivals not computed: {e}")


def run(raw_data):
    """Risk score, percentile, survival curve, median survival and SHAP values of one patient."""
    payload = json.loads(raw_data)
    df = pd.DataFrame(payload["input_data"]["data"], columns=payload["input_data"]["columns"])
    # JSON has no int/float distinction; cast each column to the dtype of the model signature
    schema = model.metadata.get_input_schema() if model.metadata else None
    if schema is not None:
        for name, dtype in zip(schema.input_names(), schema.pandas_types()):
            if name in df.columns:
                df[name] = df[name].astype(dtype)
    else:
        df = df.astype(float)

    preds_df = model.predict(df)
    risk_score = float(preds_df["risk_score"].iloc[0])
    predicted_median_survival = _compute_median_survival(baseline_x, baseline_y, risk_score)
    percentile = percentileofscore(training_predictions_df["predicted_risk"], risk_score)

    explanation = explainer(df.values)
    shap_vals = explanation.values[0, :, 0] if explanation.values.ndim == 3 else explanation.values[0]
    base_vals = explanation.base_values

    survival_x, survival_y, survival_at_horizons = [], [], {}
    try:
        if survival_model is not None:
            sf = survival_model.predict_survival_function(df)[0]
            if hasattr(sf, "x") and hasattr(sf, "y"):  # sksurv StepFunction
                survival_x, survival_y = [float(t) for t in sf.x], [float(s) for s in sf.y]
            elif callable(sf):
                times = np.linspace(1, 60, 60)
                survival_x, survival_y = [float(t) for t in times], [float(sf(t)) for t in times]
            if callable(sf):
                survival_at_horizons = {"6m": float(sf(6)), "18m": float(sf(18))}
    except Exception as e:
        logging.warning(f"Survival function not computed: {e}")

    return {
        "prediction": preds_df.to_dict(orient="records"),
        "risk_score": risk_score,
        "percentile": percentile,
        "shap_values": shap_vals.tolist(),
        "base_values": base_vals.tolist() if hasattr(base_vals, "tolist") else [float(base_vals)],
        "feature_names": payload["input_data"]["columns"],
        "training_aggregates": training_aggregates,
        "predicted_survival_x": survival_x,
        "predicted_survival_y": survival_y,
        "predicted_survival_probabilities": survival_at_horizons,
        "predicted_survival_a": 1.0,  # fixed fields expected by the web client
        "predicted_survival_b": 0.0,
        "predicted_median_survival": predicted_median_survival,
        "percentile_median_survivals": percentile_median_survivals,
        "events": {"tte": survival_training_data.get("tte", []), "event": survival_training_data.get("event", [])},
    }
