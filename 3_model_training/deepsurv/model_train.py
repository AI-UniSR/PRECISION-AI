"""DeepSurv (Cox proportional hazards neural network, pycox), benchmark only.

DeepSurv is compared with the other learners on the hold-out set
(Supplementary Table S2) but is not part of the ensemble. Its protocol differs
from that of the other learners:

- imputation (IterativeImputer) and scaling (RobustScaler) are fitted once on
  the whole training set, before the hyperparameter search;
- Optuna (TPE sampler, seed 42) maximises the mean time-dependent concordance
  of pycox (EvalSurv.concordance_td) over the repeated stratified folds, with
  up to 100 epochs per fit and early stopping (patience 10) on the validation
  fold; a fold that fails scores 0;
- the final network is trained on a stratified 80/20 split of the training set
  (up to 200 epochs, patience 20), and the baseline hazards are estimated on
  the 80% part.

In the paper DeepSurv used the stability-selected features (--selected_features).
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
import torchtuples as tt
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (enables IterativeImputer)
from sklearn.impute import IterativeImputer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sksurv.util import Surv

# pycox calls Series.is_monotonic, which pandas 2.0 removed
if not hasattr(pd.Series, "is_monotonic"):
    pd.Series.is_monotonic = property(lambda self: self.is_monotonic_increasing)

from pycox.evaluation import EvalSurv  # noqa: E402
from pycox.models import CoxPH  # noqa: E402


def objective_with_cv(trial, X, y_time, y_event, cv, input_dim, n_epochs, early_stop_patience):
    """Mean time-dependent concordance over the folds of cv."""
    n_layers = trial.suggest_int("n_layers", 1, 3)
    num_nodes = [trial.suggest_int(f"num_nodes_l{i}", 16, 128, step=16) for i in range(n_layers)]
    batch_norm = trial.suggest_categorical("batch_norm", [True, False])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_event)):
        x_train = X[train_idx].astype("float32")
        y_train = (y_time[train_idx].astype("float32"), y_event[train_idx].astype("float32"))
        x_val = X[val_idx].astype("float32")
        y_val = (y_time[val_idx].astype("float32"), y_event[val_idx].astype("float32"))

        net = tt.practical.MLPVanilla(in_features=input_dim, num_nodes=num_nodes, out_features=1,
                                      batch_norm=batch_norm, dropout=dropout, output_bias=False)
        model = CoxPH(net, tt.optim.Adam(lr=learning_rate))
        callbacks = [tt.callbacks.EarlyStopping(patience=early_stop_patience)]
        try:
            model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, callbacks=callbacks,
                      verbose=False, val_data=(x_val, y_val), val_batch_size=batch_size)
            model.compute_baseline_hazards()
            val_ci = EvalSurv(model.predict_surv_df(x_val), y_val[0], y_val[1],
                              censor_surv="km").concordance_td()
            fold_scores.append(val_ci)
            mlflow.log_metric(f"cv_fold_{fold_idx}_ci", val_ci, step=trial.number)
        except Exception as e:
            print(f"Fold {fold_idx} failed: {e}")
            fold_scores.append(0.0)

    mean_ci = np.mean(fold_scores)
    mlflow.log_metric("cv_mean_ci", mean_ci, step=trial.number)
    return mean_ci


def train_final_model(x_train, y_train, x_val, y_val, best_params, n_epochs):
    """Train the network with the best hyperparameters; early stopping on (x_val, y_val)."""
    num_nodes = [best_params[f"num_nodes_l{i}"] for i in range(best_params["n_layers"])]
    net = tt.practical.MLPVanilla(in_features=x_train.shape[1], num_nodes=num_nodes, out_features=1,
                                  batch_norm=best_params["batch_norm"], dropout=best_params["dropout"],
                                  output_bias=False)
    model = CoxPH(net, tt.optim.Adam(lr=best_params["learning_rate"]))
    log = model.fit(x_train, y_train, batch_size=best_params["batch_size"], epochs=n_epochs,
                    callbacks=[tt.callbacks.EarlyStopping(patience=20)], verbose=True,
                    val_data=(x_val, y_val), val_batch_size=best_params["batch_size"])
    model.compute_baseline_hazards()

    train_ci = EvalSurv(model.predict_surv_df(x_train), y_train[0], y_train[1],
                        censor_surv="km").concordance_td()
    val_ci = EvalSurv(model.predict_surv_df(x_val), y_val[0], y_val[1], censor_surv="km").concordance_td()
    print(f"Final network: concordance {train_ci:.4f} (training part), {val_ci:.4f} (validation part)")

    train_loss = log.monitors["train_"].scores["loss"]["score"]
    history = {
        "train_loss": train_loss,
        "val_loss": log.monitors["val_"].scores["loss"]["score"],
        "epochs": list(range(1, len(train_loss) + 1)),
        "final_train_ci": float(train_ci),
        "final_val_ci": float(val_ci),
    }
    return model, history


def save_model(model, output_path, best_params, training_metrics, baseline_hazards, imputer, scaler,
               feature_cols=None):
    """Write the artifacts read by DeepSurvWrapper (network weights, baseline
    hazards, architecture, preprocessing, feature list) and the wrapper itself."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_net(str(output_path / "deepsurv_net.pt"))
    baseline_hazards.to_pickle(str(output_path / "baseline_hazards.pkl"))
    with open(output_path / "hyperparameters.json", "w") as f:
        json.dump({**best_params, "input_dim": len(feature_cols) if feature_cols else model.net.in_features},
                  f, indent=2)
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)
    joblib.dump(imputer, output_path / "imputer.pkl")
    joblib.dump(scaler, output_path / "scaler.pkl")
    if feature_cols is not None:
        with open(output_path / "selected_features.txt", "w") as f:
            f.writelines(f"{feature}\n" for feature in feature_cols)

    from deepsurv_wrapper import DeepSurvWrapper
    joblib.dump(DeepSurvWrapper(output_path), output_path / "deepsurv_model.pkl")


def main(args):
    mlflow.start_run()
    np.random.seed(42)
    torch.manual_seed(42)

    csv = [f for f in os.listdir(args.training_data) if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(args.training_data, csv))
    X = df.drop(columns=["event", "tte"]).select_dtypes(include=[np.number])
    X.set_index("id", inplace=True)
    y = Surv.from_arrays(event=df["event"].astype(bool), time=df["tte"].astype(float))

    if args.selected_features:
        with open(os.path.join(args.selected_features, "selected_features.txt")) as f:
            feature_cols = [line.strip() for line in f if line.strip()]
        X = X[feature_cols]
    else:
        feature_cols = X.columns.tolist()
    print(f"DeepSurv on {X.shape[1]} features, {X.shape[0]} patients")

    # Preprocessing fitted once on the whole training set (see module docstring)
    imputer = IterativeImputer(random_state=42, max_iter=10)
    scaler = RobustScaler()
    X_processed = scaler.fit_transform(imputer.fit_transform(X)).astype("float64")
    y_time = y["time"].astype("float64")
    y_event = y["event"].astype("int32")
    input_dim = X_processed.shape[1]

    cv = RepeatedStratifiedKFold(n_splits=args.cv_n_splits, n_repeats=args.cv_n_repeats, random_state=42)
    mlflow.log_params({"n_features": input_dim, "n_samples": len(X_processed),
                       "n_events": int(y_event.sum()), "cv_n_splits": args.cv_n_splits,
                       "cv_n_repeats": args.cv_n_repeats, "n_optuna_trials": args.n_optuna_trials,
                       "n_epochs": args.n_epochs})

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective_with_cv(trial, X_processed, y_time, y_event, cv, input_dim,
                                                   args.n_epochs, args.early_stop_patience),
                   n_trials=args.n_optuna_trials)
    best_params, best_ci = study.best_trial.params, study.best_trial.value
    print(f"Best cross-validated concordance {best_ci:.4f} with {best_params}")
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_cv_mean_ci", best_ci)

    train_idx, val_idx = train_test_split(np.arange(len(X_processed)), test_size=0.2, random_state=42,
                                          stratify=y_event)
    x_train = X_processed[train_idx].astype("float32")
    y_train = (y_time[train_idx].astype("float32"), y_event[train_idx].astype("float32"))
    x_val = X_processed[val_idx].astype("float32")
    y_val = (y_time[val_idx].astype("float32"), y_event[val_idx].astype("float32"))
    model, history = train_final_model(x_train, y_train, x_val, y_val, best_params, args.n_epochs * 2)
    mlflow.log_metric("final_train_ci", history["final_train_ci"])
    mlflow.log_metric("final_val_ci", history["final_val_ci"])

    training_metrics = {
        "best_hyperparameters": best_params,
        "best_cv_mean_ci": float(best_ci),
        "training_history": history,
        "cv_config": {"n_splits": args.cv_n_splits, "n_repeats": args.cv_n_repeats,
                      "stratify_by_event": args.cv_stratify_by_event.lower() == "true"},
    }
    save_model(model, Path(args.trained_model), best_params, training_metrics,
               model.compute_baseline_hazards(), imputer, scaler, feature_cols=feature_cols)
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training_data", required=True, help="Folder with the training CSV")
    parser.add_argument("--trained_model", required=True, help="Output folder")
    parser.add_argument("--selected_features", default=None,
                        help="Output folder of the feature selection; all features if omitted")
    parser.add_argument("--cv_n_splits", type=int, default=5)
    parser.add_argument("--cv_n_repeats", type=int, default=10)
    parser.add_argument("--cv_stratify_by_event", default="true",
                        help="Recorded only: the folds are always stratified by event")
    parser.add_argument("--n_optuna_trials", type=int, default=50)
    parser.add_argument("--n_epochs", type=int, default=100, help="Maximum epochs per fit in the search")
    parser.add_argument("--early_stop_patience", type=int, default=10)
    main(parser.parse_args())
