"""Stability selection with random survival forests.

In each iteration a random survival forest with randomly drawn hyperparameters
is fitted on a random subsample of the training set, drawn without
replacement. Permutation importance is computed on the same subsample with a
C-index scorer, min-max normalised, and the features at or above its 70th
percentile are marked. The stability score of a feature is the fraction of
iterations in which it was marked. Features with a score of at least the
selection threshold are selected (the three highest-scoring features if none
reaches it).

The forests are fitted on data with missing values, which scikit-survival
supports from version 0.23. Despite the file name, the subsamples are drawn
without replacement; no bootstrap samples are used.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm


def rsf_cindex_scorer(estimator, X, y):
    """Harrell's C of the forest's risk score (scorer for permutation_importance)."""
    return concordance_index_censored(y["event"], y["time"], estimator.predict(X))[0]


class StabilitySelectionRSF(BaseEstimator, TransformerMixin):
    """Stability selection with randomised random survival forests.

    Parameters
    ----------
    n_iterations : int
        Number of subsamples.
    subsample_fraction : float
        Fraction of the training set drawn, without replacement, per iteration.
    pfi_repeats : int
        Permutations per feature in the permutation importance.
    selection_threshold : float
        Minimum stability score of a selected feature.
    normalize_pfi : bool
        Min-max normalise the importances before marking.
    binarization_quantile : float
        Features at or above this quantile of the importances are marked.
    n_estimators_range, min_samples_split_range, min_samples_leaf_range, max_depth_range : tuple
        Inclusive ranges from which the forest hyperparameters are drawn.
    max_features_str, max_features_float : tuple
        Candidate values of max_features; each family is chosen with probability 1/2.
    random_state : int
        Seed of the generator that draws the subsamples, the hyperparameters and
        the seeds of each forest and permutation.
    """

    def __init__(
        self,
        n_iterations=100,
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
    ):
        self.n_iterations = n_iterations
        self.subsample_fraction = subsample_fraction
        self.pfi_repeats = pfi_repeats
        self.selection_threshold = selection_threshold
        self.normalize_pfi = normalize_pfi
        self.binarization_quantile = binarization_quantile
        self.n_estimators_range = n_estimators_range
        self.max_features_str = max_features_str
        self.max_features_float = max_features_float
        self.min_samples_split_range = min_samples_split_range
        self.min_samples_leaf_range = min_samples_leaf_range
        self.max_depth_range = max_depth_range
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        self.feature_names_ = list(X.columns)
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        subsample_size = int(self.subsample_fraction * n_samples)

        counts = np.zeros(X.shape[1], dtype=float)
        importances = []
        for _ in tqdm(range(self.n_iterations), desc="Subsamples"):
            idx = rng.choice(n_samples, size=subsample_size, replace=False)
            X_i, y_i = X.iloc[idx], y[idx]

            # The order of the draws below fixes the random sequence; do not reorder.
            n_estimators = rng.randint(self.n_estimators_range[0], self.n_estimators_range[1] + 1)
            if rng.rand() < 0.5:
                max_features = rng.choice(self.max_features_str)
            else:
                max_features = float(rng.choice(self.max_features_float))
            min_samples_split = rng.randint(self.min_samples_split_range[0],
                                            self.min_samples_split_range[1] + 1)
            min_samples_leaf = rng.randint(self.min_samples_leaf_range[0],
                                           self.min_samples_leaf_range[1] + 1)
            max_depth = rng.randint(self.max_depth_range[0], self.max_depth_range[1] + 1)

            rsf = RandomSurvivalForest(
                n_estimators=n_estimators,
                max_features=max_features,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                n_jobs=self.n_jobs,
                random_state=rng.randint(0, 10**9),
            )
            rsf.fit(X_i, y_i)

            pfi = permutation_importance(
                rsf, X_i, y_i,
                n_repeats=self.pfi_repeats,
                scoring=rsf_cindex_scorer,
                random_state=rng.randint(0, 10**9),
                n_jobs=self.n_jobs,
            ).importances_mean.astype(float)

            if self.normalize_pfi:
                spread = pfi.max() - pfi.min()
                pfi = np.zeros_like(pfi) if spread < 1e-12 else (pfi - pfi.min()) / (spread + 1e-12)
            importances.append(pfi)
            counts += (pfi >= np.quantile(pfi, self.binarization_quantile)).astype(float)

        self.stability_scores_ = pd.Series(counts / float(self.n_iterations), index=self.feature_names_)
        self.pfi_matrix_ = pd.DataFrame(np.vstack(importances), columns=self.feature_names_)
        self.rankings_ = self.stability_scores_.sort_values(ascending=False)

        selected = self.stability_scores_[self.stability_scores_ >= self.selection_threshold]
        if selected.empty:
            selected = self.rankings_.head(3)
        self.selected_features_ = list(selected.index)
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.selected_features_].copy()

    def get_support(self):
        return self.selected_features_

    def get_stability_scores(self):
        return self.stability_scores_
