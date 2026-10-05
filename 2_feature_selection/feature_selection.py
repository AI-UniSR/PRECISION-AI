"""RSF stability selection on the training set (non-linear selection path).

Settings: 100 subsamples of 60% of the training set, forest hyperparameters
drawn from the ranges below, permutation importance with 5 repeats, features
marked at the 70th percentile of the normalised importances, selection
threshold 0.6. The selected features are the inputs of the RSF, XGBoost-AFT,
XGBSE and DeepSurv learners; stability_scores.csv holds the scores of Fig. S2.
"""

import argparse
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sksurv.util import Surv

from rsf_bootstrap_selector import StabilitySelectionRSF


def plot_stability_scores(scores, threshold, output_path, top_n=30):
    """Horizontal bar chart of the highest stability scores (HTML)."""
    scores = scores.sort_values(ascending=False).head(top_n)
    fig = go.Figure(go.Bar(x=scores.values, y=scores.index, orientation="h",
                           marker=dict(color=scores.values, colorscale="Plasma", showscale=True)))
    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold = {threshold}", annotation_position="top right")
    fig.update_layout(title=f"Top {top_n} stability scores", xaxis_title="Stability score",
                      yaxis_title="Feature", height=max(600, top_n * 20),
                      yaxis=dict(autorange="reversed"))
    pio.write_html(fig, file=output_path, include_plotlyjs="cdn", full_html=True)


def main(args):
    out = args.selected_features_output
    os.makedirs(out, exist_ok=True)
    mlflow.start_run()

    csv = [f for f in os.listdir(args.training_data) if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(args.training_data, csv))
    X = df.drop(columns=["event", "tte"]).select_dtypes(include=[np.number]).set_index("id")
    y = Surv.from_arrays(event=df["event"].astype(bool), time=df["tte"].astype(float))
    print(f"Training set: {X.shape[0]} patients, {X.shape[1]} candidate predictors")

    selector = StabilitySelectionRSF(
        n_iterations=args.n_iterations,
        subsample_fraction=0.6,
        pfi_repeats=5,
        selection_threshold=0.6,
        normalize_pfi=True,
        binarization_quantile=0.70,
        n_estimators_range=(150, 350),
        max_features_str=("sqrt", "log2"),
        max_features_float=(0.5, 1.0),
        min_samples_split_range=(2, 25),
        min_samples_leaf_range=(2, 15),
        max_depth_range=(3, 14),
        random_state=42,
        n_jobs=-1,
    )
    selector.fit(X, y)
    selected = selector.get_support()
    print(f"Selected {len(selected)} features: {selected}")

    with open(os.path.join(out, "selected_features.txt"), "w") as f:
        f.writelines(f"{feature}\n" for feature in selected)
    scores = selector.get_stability_scores()
    pd.DataFrame({"feature": scores.index, "stability_score": scores.values}).sort_values(
        "stability_score", ascending=False).to_csv(os.path.join(out, "stability_scores.csv"), index=False)
    plot_stability_scores(scores, selector.selection_threshold,
                          os.path.join(out, "stability_scores.html"), top_n=min(30, X.shape[1]))
    joblib.dump(selector, os.path.join(out, "stability_selector.pkl"))

    mlflow.log_params({"n_features_candidate": X.shape[1], "n_features_selected": len(selected),
                       "n_iterations": args.n_iterations, "subsample_fraction": 0.6,
                       "selection_threshold": selector.selection_threshold})
    mlflow.log_artifacts(out, artifact_path="feature_selection")
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training_data", required=True, help="Folder with the training CSV")
    parser.add_argument("--selected_features_output", required=True, help="Output folder")
    parser.add_argument("--n_iterations", type=int, default=100, help="Number of subsamples")
    main(parser.parse_args())
