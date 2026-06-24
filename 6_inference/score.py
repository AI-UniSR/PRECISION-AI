import os
import logging
import json
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

# Will hold the training-data aggregates
training_aggregates: dict = {}


def _get_survival_model(obj):
    """Recursively unwrap MLflow wrappers to find the model with predict_survival_function."""
    if hasattr(obj, 'predict_survival_function'):
        return obj
    # Try unwrapping PyFuncModel._model_impl
    if hasattr(obj, '_model_impl'):
        impl = obj._model_impl
        if hasattr(impl, 'predict_survival_function'):
            return impl
        if hasattr(impl, 'sklearn_model'):
            return _get_survival_model(impl.sklearn_model)
        if hasattr(impl, 'python_model'):
            return _get_survival_model(impl.python_model)
    # Try Python wrapper
    if hasattr(obj, 'python_model'):
        return _get_survival_model(obj.python_model)
    return None


def _compute_median_survival(baseline_x, baseline_y, risk_score):
    """
    Return the predicted median survival time for a Cox risk score.

    Uses the Cox PH relationship S(t|x) = S0(t)^exp(η), so the patient-specific
    survival crosses 0.5 at the first time where S0(t) <= 0.5^exp(-η).

    Parameters
    ----------
    baseline_x : np.ndarray
        Ordered event times from the Cox baseline survival step function.
    baseline_y : np.ndarray
        Baseline survival probabilities S0(t) at each time in baseline_x.
    risk_score : float
        Cox linear predictor η = x'β (raw output of CoxPHSurvivalAnalysis.predict()).

    Returns
    -------
    float or None
        First time t where S0(t) <= threshold, or None if the baseline never
        crosses the threshold within the observed follow-up.
    """
    if baseline_x is None or len(baseline_x) == 0:
        return None
    threshold = 0.5 ** np.exp(-risk_score)
    indices = np.where(baseline_y <= threshold)[0]
    if len(indices) == 0:
        return None
    return float(baseline_x[indices[0]])


def init():
    """
    Called once at container startup: loads the pyfunc model,
    SHAP explainer, and training-data aggregates into memory.
    """
    global model, survival_model, explainer, training_aggregates, training_predictions_df, survival_training_data
    global baseline_x, baseline_y, percentile_median_survivals

    model_dir = os.getenv("AZUREML_MODEL_DIR")

    # If MLmodel isn't at the root, descend into the first subdirectory
    if "MLmodel" not in os.listdir(model_dir):
        subdirs = [d.path for d in os.scandir(model_dir) if d.is_dir()]
        if subdirs:
            model_dir = subdirs[0]

    logging.info(f"Loading artifacts from {model_dir}")

    # 1) Load model as pyfunc (consistent with how model was registered)
    model = mlflow.pyfunc.load_model(model_dir)

    # 2) Unwrap inner sklearn model for survival function prediction
    survival_model = _get_survival_model(model)
    if survival_model is None:
        logging.warning("No inner model with predict_survival_function found")

    # 2a) Cache the Cox meta-learner baseline survival step function.
    #     baseline_survival_ is a sksurv StepFunction whose .x are the ordered
    #     event times and .y are the corresponding S0(t) values.  The baseline
    #     must come from the same meta_learner that produced the stored risk
    #     scores so the Cox formula S(t|x) = S0(t)^exp(η) is internally
    #     consistent.
    baseline_x = np.array([])
    baseline_y = np.array([])
    try:
        meta = getattr(survival_model, 'meta_learner', None)
        bsf = getattr(meta, 'baseline_survival_', None)
        if bsf is not None:
            baseline_x = np.array(bsf.x)
            baseline_y = np.array(bsf.y)
            logging.info(f"Cached Cox baseline survival: {len(baseline_x)} time points")
        else:
            logging.warning("meta_learner.baseline_survival_ not found; median survival will be unavailable")
    except Exception as e:
        logging.warning(f"Could not cache baseline survival: {e}")

    # 3) Load the SHAP explainer and attach wrapped predict function
    #    (wrapping to return numpy arrays instead of DataFrames for numba compatibility)
    explainer = mlflow.shap.load_explainer(os.path.join(model_dir, "shap_explainer"))

    # Cache schema dtypes for use in predict_wrapper
    _schema = model.metadata.get_input_schema() if model.metadata else None
    _schema_names = _schema.input_names() if _schema else None
    _schema_dtypes = _schema.pandas_types() if _schema else None

    def predict_wrapper(X):
        """Wrapper for SHAP: bypasses MLflow schema enforcement by calling the
        unwrapped sklearn model directly, since SHAP sends numpy arrays with
        float64 dtype for all columns (including integer-typed ones like ps ecog)."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=_schema_names)
        result = survival_model.predict(None, X)
        if isinstance(result, pd.DataFrame):
            return result.values.ravel()
        return result
    
    explainer.model = predict_wrapper

    # 4) Load the training-data aggregates JSON
    agg_path = os.path.join(model_dir, "extra_artifacts/training_data_aggregates", "training_data_aggregates.json")
    if not os.path.exists(agg_path):
        logging.warning(f"Aggregates file not found at {agg_path}")
        training_aggregates = {}
    else:
        with open(agg_path, "r") as f:
            training_aggregates = json.load(f)
        logging.info("Loaded training_data_aggregates.json")

    logging.info("Model and explainer loaded successfully")

    # 5) Load the training-data predictions DataFrame
    predictions_path = os.path.join(model_dir, "extra_artifacts/training_predictions", "training_predictions.json")
    if not os.path.exists(predictions_path):
        logging.warning(f"Predictions file not found at {predictions_path}")
    else:
        with open(predictions_path, "r") as f:
            data = json.load(f)

        training_predictions_df = pd.DataFrame.from_dict(data, orient="index")
        logging.info("Loaded training_predictions.json")

    # 5a) Precompute median survival for each decile of the training risk-score
    #     distribution.  For each target percentile p we find the corresponding
    #     training risk-score quantile and map it to a median survival time via
    #     the Cox baseline survival cached above.  This gives a reference curve
    #     percentile → median survival that the client can use without sending
    #     additional requests.
    percentile_median_survivals = {}
    try:
        training_risk_scores = training_predictions_df['predicted_risk'].values
        for p in [1] + list(range(10, 100, 10)) + [99]:
            risk_at_p = float(np.percentile(training_risk_scores, p))
            percentile_median_survivals[str(p)] = _compute_median_survival(baseline_x, baseline_y, risk_at_p)
        logging.info("Precomputed percentile median survivals")
    except Exception as e:
        logging.warning(f"Could not precompute percentile median survivals: {e}")

    # 6) Load the survival training data (tte + event)
    survival_training_data = {}
    surv_path = os.path.join(model_dir, "extra_artifacts/survival_training_data", "survival_training_data.json")
    if not os.path.exists(surv_path):
        logging.warning(f"Survival training data file not found at {surv_path}")
    else:
        with open(surv_path, "r") as f:
            survival_training_data = json.load(f)
        logging.info("Loaded survival_training_data.json")


def run(raw_data):
    """
    For each request:
      - parse incoming DataFrame
      - compute risk score, survival curve, SHAP values
      - return them plus the training-data aggregates
    """
    logging.info("Request received")
    payload = json.loads(raw_data)
    df = pd.DataFrame(
        payload["input_data"]["data"],
        columns=payload["input_data"]["columns"]
    )
    # Align column dtypes with the model's input signature. MLflow schema
    # enforcement refuses lossy casts (e.g. float64 -> int64), so we coerce
    # each column to the dtype declared in the signature before predicting.
    # Required because JSON has no int/float distinction: the client may send
    # 1.0 for an int-typed feature (e.g. "ps ecog") or 2 for a double-typed
    # one (e.g. "number of met sites").
    schema = model.metadata.get_input_schema() if model.metadata else None
    if schema is not None:
        for name, dtype in zip(schema.input_names(), schema.pandas_types()):
            if name in df.columns:
                df[name] = df[name].astype(dtype)
    else:
        df = df.astype(float)

    # Risk score prediction (returns DataFrame with 'risk_score' column)
    preds_df = model.predict(df)
    risk_score = float(preds_df['risk_score'].iloc[0])

    # Predicted median survival for this patient using the Cox baseline.
    # risk_score is the raw linear predictor η = x'β from CoxPHSurvivalAnalysis,
    # the same scale as the training risk scores used in precomputation.
    predicted_median_survival = _compute_median_survival(baseline_x, baseline_y, risk_score)

    # Calculate percentile of the score among the training prediction distribution
    percentile = percentileofscore(training_predictions_df['predicted_risk'], risk_score)

    # SHAP explanation (single-output model -> values are 2D: samples x features)
    # Convert DataFrame to numpy array for SHAP compatibility with numba
    explanation = explainer(df.values)
    if explanation.values.ndim == 3:
        shap_vals = explanation.values[0, :, 0]
    else:
        shap_vals = explanation.values[0]
    base_vals = explanation.base_values

    # Survival curve prediction
    survival_x = []
    survival_y = []
    predicted_survival_probabilities = {}
    try:
        if survival_model is not None:
            surv_funcs = survival_model.predict_survival_function(df)
            sf = surv_funcs[0]
            if hasattr(sf, 'x') and hasattr(sf, 'y'):
                # sksurv StepFunction
                survival_x = [float(t) for t in sf.x]
                survival_y = [float(s) for s in sf.y]
            elif callable(sf):
                # Interpolation function (e.g. from XGBSE)
                times = np.linspace(1, 60, 60)
                survival_x = [float(t) for t in times]
                survival_y = [float(sf(t)) for t in times]
            # Evaluate survival at fixed clinical horizons
            if callable(sf):
                predicted_survival_probabilities = {
                    "6m": float(sf(6)),
                    "18m": float(sf(18)),
                }
    except Exception as e:
        logging.warning(f"Could not compute survival function: {e}")

    response = {
        "prediction": preds_df.to_dict(orient="records"),
        "risk_score": risk_score,
        "percentile": percentile,
        "shap_values": shap_vals.tolist(),
        "base_values": base_vals.tolist() if hasattr(base_vals, 'tolist') else [float(base_vals)],
        "feature_names": payload["input_data"]["columns"],
        "training_aggregates": training_aggregates,
        "predicted_survival_x": survival_x,
        "predicted_survival_y": survival_y,
        "predicted_survival_probabilities": predicted_survival_probabilities,
        "predicted_survival_a": 1.0,
        "predicted_survival_b": 0.0,
        "predicted_median_survival": predicted_median_survival,
        "percentile_median_survivals": percentile_median_survivals,
        "events": {
            "tte": survival_training_data.get("tte", []),
            "event": survival_training_data.get("event", []),
        },
    }

    logging.info("Request processed")
    return response
