"""
SHAP & PFI Analysis for Ensemble Survival Model

Produces SHAP beeswarm (PNG+PDF) and PFI plot (PNG+PDF) for the registered
ensemble model on the temporal validation cohort, using a k-means subsample
of the training cohort as SHAP background.

Key parameters (§8 Methods):
  - SHAP background: k-means with k=20 centroids from training cohort
  - SHAP evaluation: 200 patients randomly sampled from temporal validation (seed=42)
  - PFI: 50 permutations per feature on the full temporal validation cohort

Note on model loading: the script loads from an MLflow Model Registry URI.
For local use, pass a local run URI (e.g. runs:/<run_id>/model) as --model_name.
"""

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mlflow
import shap
from sksurv.metrics import concordance_index_censored

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _model_uri(model_name, version):
    return f"models:/{model_name}/{'latest' if version == 'latest' else version}"


def _predict(model, feature_names, X):
    """Unified predict: accepts ndarray or DataFrame, returns 1-D ndarray."""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)
    preds = model.predict(X)
    if isinstance(preds, pd.DataFrame):
        return preds.iloc[:, 0].values
    return np.asarray(preds)


def _load_data(path, selected_features):
    """Load CSV or Parquet from *path* (file or folder), return (X, y)."""
    p = Path(path)
    if p.is_dir():
        files = list(p.glob("*.parquet")) + list(p.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV/Parquet files in {p}")
        p = files[0]

    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    logger.info(f"Loaded {p.name}: {len(df)} rows")

    df = df.dropna(subset=["event", "tte"])
    available = [f for f in selected_features if f in df.columns]
    X = df[available]
    y = np.array(
        [(bool(e), t) for e, t in zip(df["event"], df["tte"])],
        dtype=[("event", bool), ("time", float)],
    )
    logger.info(f"  Features used: {len(available)}/{len(selected_features)}, samples: {len(X)}")
    return X, y


def _load_selected_features(model_name, version):
    """Union of all features from selected_features.json model artifact."""
    uri = _model_uri(model_name, version)
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{uri}/artifacts/selected_features.json", dst_path=tmp
        )
        with open(local) as f:
            feat_dict = json.load(f)
    all_feats = sorted({f for feats in feat_dict.values() for f in feats})
    logger.info(f"Selected features ({len(all_feats)}): {all_feats}")
    return all_feats


# ── SHAP ─────────────────────────────────────────────────────────────────────

def compute_shap(model, feature_names, X_bg, X_val, bg_k, n_display, seed=42):
    """Return shap.Explanation on (subsample of) X_val."""
    np.random.seed(seed)
    background = shap.kmeans(X_bg, bg_k)

    if n_display < X_val.shape[0]:
        idx = np.random.choice(X_val.shape[0], size=n_display, replace=False)
        X_sample = X_val.iloc[idx].copy()
    else:
        X_sample = X_val.copy()

    wrapper = lambda X: _predict(model, feature_names, X)  # noqa: E731
    explainer = shap.KernelExplainer(wrapper, background)
    vals = explainer.shap_values(X_sample)

    logger.info(f"SHAP values computed: {vals.shape}")
    return shap.Explanation(
        values=vals,
        base_values=explainer.expected_value,
        data=X_sample.values,
        feature_names=list(feature_names),
    )


def save_beeswarm(explanation, png_dir, pdf_dir, max_display=20):
    """Save beeswarm as PNG + PDF into the respective subdirs."""
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    for path in (Path(png_dir) / "shap_beeswarm.png", Path(pdf_dir) / "shap_beeswarm.pdf"):
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"  Saved {path}")
    plt.close()


# ── PFI ──────────────────────────────────────────────────────────────────────

def compute_pfi(model, feature_names, X, y, n_repeats=50, seed=42):
    """Return DataFrame with permutation feature importance (C-index based)."""
    predict = lambda Xv: _predict(model, feature_names, Xv)  # noqa: E731
    baseline = concordance_index_censored(y["event"], y["time"], predict(X.values))[0]
    logger.info(f"PFI baseline C-index: {baseline:.4f}")
    rng = np.random.RandomState(seed)
    rows = []
    for i, feat in enumerate(X.columns):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[feat] = rng.permutation(Xp[feat].values)
            ci = concordance_index_censored(y["event"], y["time"], predict(Xp.values))[0]
            drops.append(baseline - ci)
        rows.append(dict(
            feature=feat,
            importance_mean=np.mean(drops),
            importance_std=np.std(drops),
            importance_ci_lower=float(np.percentile(drops, 2.5)),
            importance_ci_upper=float(np.percentile(drops, 97.5)),
        ))
        if (i + 1) % 5 == 0 or i == len(X.columns) - 1:
            logger.info(f"  PFI {i+1}/{len(X.columns)} features done")
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def save_pfi_plot(df, png_dir, pdf_dir, top_n=30):
    """Save PFI bar chart as PNG + PDF into the respective subdirs."""
    if df.empty:
        return
    df_top = df.head(top_n).iloc[::-1]
    err_lo = (df_top["importance_mean"] - df_top["importance_ci_lower"]).clip(lower=0).values
    err_hi = (df_top["importance_ci_upper"] - df_top["importance_mean"]).clip(lower=0).values
    norm = mcolors.Normalize(df_top["importance_mean"].min(), df_top["importance_mean"].max())
    colors = cm.viridis(norm(df_top["importance_mean"].values))

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(df_top["feature"], df_top["importance_mean"],
            xerr=[err_lo, err_hi], color=colors, edgecolor="none", capsize=3)
    sm = cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Importance")
    ax.set_xlabel("C-index decrease")
    ax.set_title("Permutation Feature Importance – Temporal Validation", fontweight="bold")
    plt.tight_layout()
    for path in (Path(png_dir) / "pfi_importance.png", Path(pdf_dir) / "pfi_importance.pdf"):
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"  Saved {path}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_version", default="latest")
    parser.add_argument("--training_data", required=True)
    parser.add_argument("--validation_data", required=True)
    parser.add_argument("--background_samples", type=int, default=20)
    parser.add_argument("--display_samples", type=int, default=200)
    parser.add_argument("--max_display_features", type=int, default=20)
    parser.add_argument("--pfi_repeats", type=int, default=50)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--shap_output", required=True)
    args = parser.parse_args()

    out = Path(args.shap_output)
    png_dir = out / "figures" / "png"
    pdf_dir = out / "figures" / "pdf"
    for d in (png_dir, pdf_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load model + features
    model = mlflow.pyfunc.load_model(_model_uri(args.model_name, args.model_version))
    features = _load_selected_features(args.model_name, args.model_version)

    # Load data
    X_train, _ = _load_data(args.training_data, features)
    X_val, y_val = _load_data(args.validation_data, features)

    # SHAP beeswarm (background=training, estimation=temporal validation)
    explanation = compute_shap(
        model, X_train.columns, X_train, X_val,
        bg_k=args.background_samples,
        n_display=args.display_samples,
        seed=args.random_state,
    )
    save_beeswarm(explanation, png_dir, pdf_dir, max_display=args.max_display_features)

    # PFI on temporal validation
    pfi_df = compute_pfi(
        model, X_train.columns, X_val, y_val,
        n_repeats=args.pfi_repeats, seed=args.random_state,
    )
    save_pfi_plot(pfi_df, png_dir, pdf_dir, top_n=30)

    # Log to MLflow
    mlflow.start_run()
    try:
        mlflow.log_artifacts(str(out))
    finally:
        mlflow.end_run()

    logger.info("Done.")


if __name__ == "__main__":
    main()
