"""Kernel SHAP and permutation importance of the ensemble risk score (Figure 4).

- SHAP: model-agnostic Kernel SHAP of the ensemble risk score. The background
  is a k-means summary of the training set (20 centroids; shap.kmeans
  mean-imputes missing values before clustering and snaps each centroid
  coordinate to an observed value). SHAP values are computed for 200 patients
  drawn at random from the temporal cohort (seed 42); missing inputs of these
  patients are imputed inside the model, as at prediction time.
- Permutation importance: decrease of Harrell's C in the whole temporal cohort
  when one feature is permuted, 50 permutations per feature; mean, SD and
  2.5-97.5 percentiles over the permutations.

The ensemble is loaded from the MLflow registry (models:/<name>/<version>).
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sksurv.metrics import concordance_index_censored  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _predict(model, feature_names, X):
    """Ensemble risk score for a DataFrame or an array with columns feature_names."""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)
    preds = model.predict(X)
    return preds.iloc[:, 0].values if isinstance(preds, pd.DataFrame) else np.asarray(preds)


def _load_data(path, features):
    """Model features (in the order given) and survival target of a CSV/Parquet file or folder."""
    p = Path(path)
    if p.is_dir():
        p = (list(p.glob("*.parquet")) + list(p.glob("*.csv")))[0]
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df = df.dropna(subset=["event", "tte"])
    X = df[[f for f in features if f in df.columns]]
    y = np.array([(bool(e), t) for e, t in zip(df["event"], df["tte"])],
                 dtype=[("event", bool), ("time", float)])
    return X, y


def _load_selected_features(model_uri):
    """Union of the base learners' features (selected_features.json of the registered model)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(artifact_uri=f"{model_uri}/artifacts/selected_features.json",
                                                   dst_path=tmp)
        with open(path) as f:
            return sorted({feature for features in json.load(f).values() for feature in features})


def compute_shap(model, feature_names, X_background, X_val, k, n_explained, seed=42):
    """Kernel SHAP values of n_explained random patients of X_val."""
    np.random.seed(seed)
    background = shap.kmeans(X_background, k)
    if n_explained < X_val.shape[0]:
        X_sample = X_val.iloc[np.random.choice(X_val.shape[0], size=n_explained, replace=False)].copy()
    else:
        X_sample = X_val.copy()
    explainer = shap.KernelExplainer(lambda X: _predict(model, feature_names, X), background)
    values = explainer.shap_values(X_sample)
    return shap.Explanation(values=values, base_values=explainer.expected_value, data=X_sample.values,
                            feature_names=list(feature_names))


def save_beeswarm(explanation, png_dir, pdf_dir, max_display=20):
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    for path in (Path(png_dir) / "shap_beeswarm.png", Path(pdf_dir) / "shap_beeswarm.pdf"):
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def compute_pfi(model, feature_names, X, y, n_repeats=50, seed=42):
    """Decrease of Harrell's C when each feature is permuted (one generator for all features)."""
    predict = lambda Xv: _predict(model, feature_names, Xv)  # noqa: E731
    baseline = concordance_index_censored(y["event"], y["time"], predict(X.values))[0]
    rng = np.random.RandomState(seed)
    rows = []
    for feature in X.columns:
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[feature] = rng.permutation(X_perm[feature].values)
            drops.append(baseline - concordance_index_censored(y["event"], y["time"], predict(X_perm.values))[0])
        rows.append({"feature": feature, "importance_mean": np.mean(drops), "importance_std": np.std(drops),
                     "importance_ci_lower": float(np.percentile(drops, 2.5)),
                     "importance_ci_upper": float(np.percentile(drops, 97.5))})
    logger.info(f"Permutation importance done (baseline C-index {baseline:.4f})")
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def save_pfi_plot(df, png_dir, pdf_dir, top_n=30):
    if df.empty:
        return
    top = df.head(top_n).iloc[::-1]
    err_lo = (top["importance_mean"] - top["importance_ci_lower"]).clip(lower=0).values
    err_hi = (top["importance_ci_upper"] - top["importance_mean"]).clip(lower=0).values
    norm = mcolors.Normalize(top["importance_mean"].min(), top["importance_mean"].max())
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(top["feature"], top["importance_mean"], xerr=[err_lo, err_hi],
            color=cm.viridis(norm(top["importance_mean"].values)), edgecolor="none", capsize=3)
    mappable = cm.ScalarMappable(cmap="viridis", norm=norm)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, label="Importance")
    ax.set_xlabel("C-index decrease")
    ax.set_title("Permutation feature importance, temporal validation", fontweight="bold")
    plt.tight_layout()
    for path in (Path(png_dir) / "pfi_importance.png", Path(pdf_dir) / "pfi_importance.pdf"):
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_version", default="latest")
    parser.add_argument("--training_data", required=True, help="Training set (SHAP background)")
    parser.add_argument("--validation_data", required=True, help="Temporal cohort (prepare_external.py)")
    parser.add_argument("--background_samples", type=int, default=20, help="k-means centroids of the background")
    parser.add_argument("--display_samples", type=int, default=200, help="Patients explained by SHAP")
    parser.add_argument("--max_display_features", type=int, default=20)
    parser.add_argument("--pfi_repeats", type=int, default=50)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--shap_output", required=True)
    args = parser.parse_args()

    out = Path(args.shap_output)
    png_dir, pdf_dir = out / "figures" / "png", out / "figures" / "pdf"
    for d in (png_dir, pdf_dir):
        d.mkdir(parents=True, exist_ok=True)

    model_uri = f"models:/{args.model_name}/{args.model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    features = _load_selected_features(model_uri)
    X_train, _ = _load_data(args.training_data, features)
    X_val, y_val = _load_data(args.validation_data, features)

    explanation = compute_shap(model, X_train.columns, X_train, X_val, k=args.background_samples,
                               n_explained=args.display_samples, seed=args.random_state)
    save_beeswarm(explanation, png_dir, pdf_dir, max_display=args.max_display_features)
    pfi = compute_pfi(model, X_train.columns, X_val, y_val, n_repeats=args.pfi_repeats, seed=args.random_state)
    save_pfi_plot(pfi, png_dir, pdf_dir)
    pfi.to_csv(out / "pfi_importance.csv", index=False)

    mlflow.start_run()
    mlflow.log_artifacts(str(out))
    mlflow.end_run()


if __name__ == "__main__":
    main()
