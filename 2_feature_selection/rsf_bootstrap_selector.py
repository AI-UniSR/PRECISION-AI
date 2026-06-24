import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm

# ============================================================
# C-INDEX SCORER FOR RSF (handles structured array output)
# ============================================================

def rsf_cindex_scorer(estimator, X, y):
    pred = estimator.predict(X)
    event = y["event"]
    time = y["time"]
    res = concordance_index_censored(event, time, pred)
    try:
        c = res[0]
    except Exception:
        c = res
    return c


# ============================================================
# CLASS: Stability Selection + PFI + RSF with random hyperparameters
# ============================================================

class StabilitySelectionRSF(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        n_iterations=100,
        subsample_fraction=0.7,
        pfi_repeats=5,
        selection_threshold=0.6,
        normalize_pfi=True,
        binarization_quantile=0.75,

        # hyperparameter ranges
        n_estimators_range=(80, 120),
        max_features_str=("sqrt", "log2"),
        max_features_float=(0.5, 1),
        min_samples_split_range=(2, 10),
        min_samples_leaf_range=(1, 5),
        max_depth_range=(2, 10),

        random_state=42,
        n_jobs=-1,
        plot_style=None,
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
        self.plot_style = plot_style

        # attributes filled after fit
        self.stability_scores_ = None
        self.pfi_matrix_ = None
        self.selected_features_ = None
        self.rankings_ = None
        self.feature_names_ = None


    # ============================================================
    # FIT METHOD
    # ============================================================

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        self.feature_names_ = list(X.columns)
        n_features = X.shape[1]

        rng = np.random.RandomState(self.random_state)

        stability_counts = np.zeros(n_features, dtype=float)
        pfi_storage = []

        n_samples = X.shape[0]
        subsample_size = int(self.subsample_fraction * n_samples)

        for it in tqdm(range(self.n_iterations), desc="Bootstrap iterations"):

            # -------------------------
            # 1) Subsample without replacement
            # -------------------------
            idx = rng.choice(n_samples, size=subsample_size, replace=False)
            X_i = X.iloc[idx]
            y_i = y[idx]

            # -------------------------
            # 2) Draw random hyperparameters for this iteration
            # -------------------------
            n_estimators = rng.randint(
                self.n_estimators_range[0], self.n_estimators_range[1] + 1
            )

            # max_features: randomly choose a string specifier or a float fraction
            if rng.rand() < 0.5:
                max_features = rng.choice(self.max_features_str)
            else:
                max_features = float(rng.choice(self.max_features_float))

            min_samples_split = rng.randint(
                self.min_samples_split_range[0],
                self.min_samples_split_range[1] + 1,
            )
            min_samples_leaf = rng.randint(
                self.min_samples_leaf_range[0],
                self.min_samples_leaf_range[1] + 1,
            )

            max_depth = rng.randint(
                self.max_depth_range[0],
                self.max_depth_range[1] + 1,
            )

            # -------------------------
            # 3) Fit RSF
            # -------------------------
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

            # -------------------------
            # 4) Permutation Feature Importance
            # -------------------------
            pfi_res = permutation_importance(
                rsf,
                X_i,
                y_i,
                n_repeats=self.pfi_repeats,
                scoring=rsf_cindex_scorer,
                random_state=rng.randint(0, 10**9),
                n_jobs=self.n_jobs,
            )
            pfi = pfi_res.importances_mean.astype(float)

            # -------------------------
            # 5) Optional min-max normalisation of PFI scores
            # -------------------------
            if self.normalize_pfi:
                pmin = pfi.min()
                pmax = pfi.max()
                denom = pmax - pmin
                if denom < 1e-12:
                    pfi_norm = np.zeros_like(pfi)
                else:
                    pfi_norm = (pfi - pmin) / (denom + 1e-12)
            else:
                pfi_norm = pfi

            pfi_storage.append(pfi_norm)

            # -------------------------
            # 6) Binarize at the specified quantile threshold
            # -------------------------
            thr = np.quantile(pfi_norm, self.binarization_quantile)
            mask_selected = pfi_norm >= thr

            # -------------------------
            # 7) Accumulate stability counts
            # -------------------------
            stability_counts += mask_selected.astype(float)

        # -----------------------------
        # 8) Compute final stability scores
        # -----------------------------
        stability_scores = stability_counts / float(self.n_iterations)
        self.stability_scores_ = pd.Series(stability_scores, index=self.feature_names_)

        self.pfi_matrix_ = pd.DataFrame(
            np.vstack(pfi_storage), columns=self.feature_names_
        )

        # rank features by stability score
        self.rankings_ = self.stability_scores_.sort_values(ascending=False)

        # apply selection threshold; fall back to top-3 if none qualify
        selected = self.stability_scores_[self.stability_scores_ >= self.selection_threshold]

        if selected.empty:
            selected = self.rankings_.head(3)

        self.selected_features_ = list(selected.index)

        return self


    # ============================================================
    # TRANSFORM METHOD
    # ============================================================

    def transform(self, X):
        X = pd.DataFrame(X)
        return X[self.selected_features_].copy()

    def get_support(self):
        return self.selected_features_


    # ============================================================
    # INSPECTION METHODS
    # ============================================================

    def get_stability_scores(self):
        return self.stability_scores_

    def get_pfi_matrix(self):
        return self.pfi_matrix_

    def summary(self):
        print("=== Stability Selection RSF (PFI-based) ===")
        print(f"Iterations:          {self.n_iterations}")
        print(f"Subsample fraction:  {self.subsample_fraction}")
        print(f"Selection threshold: {self.selection_threshold}")
        print(f"Selected features:   {len(self.selected_features_)}")
        print()
        print("Top features by stability:")
        print(self.rankings_.head(20))
        print()
        return


    # ============================================================
    # PLOTTING METHODS
    # ============================================================

    def plot_stability_bar(self, top_n=None):
        if self.plot_style:
            plt.style.use(self.plot_style)

        scores = self.rankings_
        if top_n is not None:
            scores = scores.head(top_n)

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(scores)), scores.values)
        plt.xticks(range(len(scores)), scores.index, rotation=90)
        plt.axhline(self.selection_threshold, linestyle="--", color="red")
        plt.ylabel("Stability score")
        plt.title("Stability Selection – Feature Scores")
        plt.tight_layout()
        plt.show()

    def plot_feature_distribution(self, feature_name):
        if feature_name not in self.pfi_matrix_.columns:
            raise ValueError(f"Feature {feature_name} not found")

        data = self.pfi_matrix_[feature_name]

        if self.plot_style:
            plt.style.use(self.plot_style)

        plt.figure(figsize=(8, 4))
        plt.hist(data, bins=20, alpha=0.7)
        plt.xlabel("Normalized PFI")
        plt.ylabel("Frequency")
        plt.title(f"PFI distribution across iterations – {feature_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
