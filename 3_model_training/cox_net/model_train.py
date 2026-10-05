"""Elastic-net Cox model: linear selection path and the 'cox' base learner.

1. Optuna (TPE sampler, seed 42) tunes the length of the regularisation path
   (n_alphas, 5-50) and the elastic-net mixing parameter (l1_ratio, 0.1-1).
   A candidate is scored by computing, for every alpha on its path, the mean
   Harrell's C over the cross-validation folds and taking the best mean.
2. For the best candidate, alpha is chosen with a parsimony rule: the largest
   alpha whose mean C-index is at least the best mean minus 0.25 times the
   standard deviation of the fold C-indices at the best alpha. With 50 folds
   this tolerance is about 1.8 standard errors, wider than the one-standard-
   error rule, so the model is sparser.
3. The elastic net is refitted on the whole training set at that alpha; the
   features with |coefficient| >= 1e-5 are kept and refitted with an
   unpenalised Cox model. This refit is the 'cox' learner of the ensemble.

Imputation (IterativeImputer) and scaling (RobustScaler) are pipeline steps
and are refitted in every fold.
"""

import argparse
import json
import os
import sys

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (enables IterativeImputer)
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from survival_utils import (  # noqa: E402
    cv_splits,
    evaluate_model_cv,
    load_training_data,
    save_optuna_plots,
)

PARSIMONY_SD_FRACTION = 0.25
COEFFICIENT_THRESHOLD = 1e-5


def coxnet_pipeline(**coxnet_params):
    return make_pipeline(IterativeImputer(random_state=42, max_iter=10), RobustScaler(),
                         CoxnetSurvivalAnalysis(fit_baseline_model=True, **coxnet_params))


def cox_pipeline():
    return make_pipeline(IterativeImputer(random_state=42, max_iter=10), RobustScaler(),
                         CoxPHSurvivalAnalysis(alpha=0.0))


def cross_validate_alpha_path(X, y, n_alphas, l1_ratio, n_splits=5, n_repeats=1, stratify_by_event=True):
    """Mean and SD of the fold C-indices at every alpha of the elastic-net path.

    The path is computed once on the whole training set and the same alphas are
    used in every fold. Returns the alphas, a table (alpha, mean_cindex,
    std_cindex) and the coefficient path of the whole-data fit.
    """
    path_fit = coxnet_pipeline(n_alphas=n_alphas, l1_ratio=l1_ratio, max_iter=10000).fit(X, y)
    alphas = path_fit.named_steps["coxnetsurvivalanalysis"].alphas_
    coefs = pd.DataFrame(path_fit.named_steps["coxnetsurvivalanalysis"].coef_,
                         index=X.columns, columns=np.round(alphas, 5))

    scores = {alpha: [] for alpha in alphas}
    pipeline = coxnet_pipeline(alphas=alphas, l1_ratio=l1_ratio, max_iter=10000)
    for train_idx, test_idx in cv_splits(X, y, n_splits, n_repeats, stratify_by_event):
        pipeline.fit(X.iloc[train_idx], y[train_idx])
        net = pipeline.named_steps["coxnetsurvivalanalysis"]
        path = net.coef_.copy()
        y_test = y[test_idx]
        # predict() uses the last column of coef_, so each alpha is scored by
        # placing its coefficients there (the intercept offset does not change
        # the C-index). The path may stop early if a fit does not converge.
        for i in range(min(len(alphas), path.shape[1])):
            net.coef_ = path[:, i].reshape(-1, 1)
            risk = pipeline.predict(X.iloc[test_idx])
            scores[alphas[i]].append(concordance_index_censored(y_test["event"], y_test["time"], risk)[0])

    summary = pd.DataFrame([{"alpha": alpha, "mean_cindex": np.mean(s), "std_cindex": np.std(s)}
                            for alpha, s in scores.items()])
    return alphas, summary, coefs


def objective(trial, X, y, n_splits, n_repeats, stratify_by_event):
    n_alphas = trial.suggest_int("n_alphas", 5, 50)
    l1_ratio = trial.suggest_float("l1_ratio", 0.1, 1.0)
    _, summary, _ = cross_validate_alpha_path(X, y, n_alphas, l1_ratio, n_splits, n_repeats, stratify_by_event)
    return summary["mean_cindex"].max()


def main(args):
    out = args.trained_model
    os.makedirs(out, exist_ok=True)
    mlflow.start_run()
    X, y = load_training_data(args.training_data)
    cv = dict(n_splits=args.cv_n_splits, n_repeats=args.cv_n_repeats,
              stratify_by_event=args.cv_stratify_by_event)

    study = optuna.create_study(direction="maximize", study_name="cox_elasticnet_tuning",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X, y, **cv), n_trials=args.n_optuna_trials)
    n_alphas, l1_ratio = study.best_params["n_alphas"], study.best_params["l1_ratio"]
    print(f"Best path: n_alphas={n_alphas}, l1_ratio={l1_ratio:.3f} (C-index {study.best_value:.4f})")
    study.trials_dataframe().to_csv(os.path.join(out, "optuna_history.csv"), index=False)
    save_optuna_plots(study, out)

    _, alpha_summary, _ = cross_validate_alpha_path(X, y, n_alphas, l1_ratio, **cv)
    alpha_summary.to_csv(os.path.join(out, "alpha_summary.csv"), index=False)

    # Parsimony rule (step 2 of the module docstring)
    best = alpha_summary.loc[alpha_summary["mean_cindex"].idxmax()]
    threshold = best["mean_cindex"] - PARSIMONY_SD_FRACTION * best["std_cindex"]
    candidates = alpha_summary[alpha_summary["mean_cindex"] >= threshold]
    chosen = candidates.loc[candidates["alpha"].idxmax()]
    alpha = chosen["alpha"]
    print(f"alpha: best {best['alpha']:.6f} (C {best['mean_cindex']:.4f}, SD {best['std_cindex']:.4f}); "
          f"chosen {alpha:.6f} (C {chosen['mean_cindex']:.4f}, threshold {threshold:.4f})")

    penalised = coxnet_pipeline(alphas=[alpha], l1_ratio=l1_ratio, max_iter=1000).fit(X, y)
    coef = penalised.named_steps["coxnetsurvivalanalysis"].coef_.flatten()
    table = pd.DataFrame({"feature": X.columns, "coefficient": coef, "abs_coefficient": np.abs(coef)})
    table = table[table["coefficient"] != 0].sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    table["selected"] = table["abs_coefficient"] >= COEFFICIENT_THRESHOLD
    table.to_csv(os.path.join(out, "coefficient_table.csv"), index=False)
    selected = table.loc[table["selected"], "feature"].tolist()
    with open(os.path.join(out, "selected_features.txt"), "w") as f:
        f.writelines(f"{feature}\n" for feature in selected)
    print(f"Selected {len(selected)} of {X.shape[1]} features: {selected}")

    model = cox_pipeline().fit(X[selected], y)
    cv_df, cv_mean, cv_std = evaluate_model_cv(X[selected], y, cox_pipeline, verbose=False, **cv)
    with open(os.path.join(out, "cv_metrics.json"), "w") as f:
        json.dump({"cv_metrics_per_fold": cv_df.to_dict(orient="records"),
                   "cv_metrics_mean": cv_mean, "cv_metrics_std": cv_std}, f, indent=2)
    print(f"Unpenalised refit, cross-validated C-index {cv_mean['c_index']:.4f}")

    joblib.dump(model, os.path.join(out, "cox_pipeline.pkl"))
    mlflow.log_params({**study.best_params, "alpha": alpha, "alpha_selection_rule": "0.25 SD",
                       "n_features_selected": len(selected)})
    mlflow.log_metrics({"optuna_best_cindex": study.best_value, "cv_cindex_mean": cv_mean["c_index"]})
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
    mlflow.log_artifacts(out)
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training_data", required=True, help="Folder with the training CSV")
    parser.add_argument("--trained_model", required=True, help="Output folder")
    parser.add_argument("--cv_n_splits", type=int, default=5)
    parser.add_argument("--cv_n_repeats", type=int, default=10)
    parser.add_argument("--cv_stratify_by_event", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--n_optuna_trials", type=int, default=50)
    main(parser.parse_args())
