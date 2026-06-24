import argparse
import os
import joblib
import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from sksurv.util import Surv

# Import StabilitySelectionRSF from same directory
from rsf_bootstrap_selector import StabilitySelectionRSF


def plot_stability_scores(selector, output_path, top_n=30):
    """
    Plot stability scores from StabilitySelectionRSF.
    
    Parameters
    ----------
    selector : StabilitySelectionRSF
        Fitted selector
    output_path : str
        Path to save the plot
    top_n : int
        Number of top features to display
    """
    scores = selector.get_stability_scores().sort_values(ascending=False).head(top_n)
    
    fig = go.Figure(go.Bar(
        x=scores.values,
        y=scores.index,
        orientation='h',
        marker=dict(
            color=scores.values,
            colorscale='Plasma',
            showscale=True
        )
    ))
    
    fig.add_vline(
        x=selector.selection_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold = {selector.selection_threshold}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"Top {top_n} Stability Scores - Feature Selection",
        xaxis_title="Stability Score",
        yaxis_title="Feature",
        height=max(600, top_n * 20),
        yaxis=dict(autorange="reversed")
    )
    
    pio.write_html(fig, file=output_path, include_plotlyjs="cdn", full_html=True)


def main(args):
    print("=== Starting Feature Selection with StabilitySelectionRSF ===")
    os.makedirs(args.selected_features_output, exist_ok=True)
    mlflow.start_run()

    # --- Load dataset ---
    csv_files = [f for f in os.listdir(args.training_data) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in training_data folder.")
    csv_path = os.path.join(args.training_data, csv_files[0])
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")

    # --- Prepare data ---
    print("Preparing features and survival target...")
    X = df.drop(columns=["event", "tte"]).select_dtypes(include=[np.number])
    X.set_index("id", inplace=True)
    y = Surv.from_arrays(event=df["event"].astype(bool), time=df["tte"].astype(float))
    print(f"Feature matrix: {X.shape}, Survival vector: {y.shape}")

    # --- Feature Selection with StabilitySelectionRSF ---
    print("\n=== Feature Selection with StabilitySelectionRSF ===")
    print(f"Using {args.n_iterations} bootstrap iterations")
    selector = StabilitySelectionRSF(
        n_iterations=args.n_iterations,
        subsample_fraction=0.6,
        pfi_repeats=5,
        selection_threshold=0.6,
        normalize_pfi=True,
        binarization_quantile=0.70,
        n_estimators_range=(150, 350),
        max_features_str=("sqrt", "log2"),
        max_features_float=(0.5, 1),
        min_samples_split_range=(2, 25),
        min_samples_leaf_range=(2, 15),
        max_depth_range=(3, 14),
        random_state=42,
        n_jobs=-1
    )
    
    print("Fitting feature selector...")
    selector.fit(X, y)
    selected_features = selector.get_support()
    print(f"\n✅ Feature selection completed!")
    print(f"Selected {len(selected_features)} features out of {X.shape[1]}")
    print(f"Selected features: {selected_features}")
    
    # Log feature selection results
    mlflow.log_param("n_features_original", X.shape[1])
    mlflow.log_param("n_features_selected", len(selected_features))
    mlflow.log_param("feature_selection_threshold", selector.selection_threshold)
    mlflow.log_param("stability_n_iterations", args.n_iterations)
    mlflow.log_param("stability_subsample_fraction", 0.6)
    
    # Save selected features list
    selected_features_path = os.path.join(args.selected_features_output, "selected_features.txt")
    with open(selected_features_path, 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    mlflow.log_artifact(selected_features_path, artifact_path="feature_selection")
    
    # Save stability scores
    stability_scores = selector.get_stability_scores()
    stability_df = pd.DataFrame({
        'feature': stability_scores.index,
        'stability_score': stability_scores.values
    }).sort_values('stability_score', ascending=False)
    stability_path = os.path.join(args.selected_features_output, "stability_scores.csv")
    stability_df.to_csv(stability_path, index=False)
    mlflow.log_artifact(stability_path, artifact_path="feature_selection")
    
    # Plot stability scores
    print("Generating stability scores plot...")
    stability_plot_path = os.path.join(args.selected_features_output, "stability_scores.html")
    plot_stability_scores(selector, stability_plot_path, top_n=min(30, len(X.columns)))
    mlflow.log_artifact(stability_plot_path, artifact_path="plots")
    
    # Save the selector object (for reproducibility)
    selector_path = os.path.join(args.selected_features_output, "stability_selector.pkl")
    joblib.dump(selector, selector_path)
    mlflow.log_artifact(selector_path, artifact_path="feature_selection")
    
    # Save selected data (optional, for debugging)
    X_selected = X[selected_features]
    selected_data_path = os.path.join(args.selected_features_output, "selected_data.csv")
    df_selected = pd.DataFrame(X_selected)
    df_selected['event'] = df['event'].values
    df_selected['tte'] = df['tte'].values
    df_selected.to_csv(selected_data_path, index=True)
    
    print(f"\n✅ Feature selection artifacts saved to {args.selected_features_output}")
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Selection with Stability Selection RSF")
    parser.add_argument("--training_data", type=str, required=True, help="Path to training data folder")
    parser.add_argument("--selected_features_output", type=str, required=True, help="Path to output folder for selected features")
    parser.add_argument("--n_iterations", type=int, default=100, help="Number of bootstrap iterations for stability selection")
    
    args = parser.parse_args()
    main(args)
