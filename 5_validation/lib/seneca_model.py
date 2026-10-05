"""SENECA prognostic score with the published coefficients and cutoffs.

score = 0.1757 x neutrophils (10^9/L) + 0.5055 x ECOG PS (0 vs >=1)
        + 1.4037 x stage (0 locally advanced, 1 metastatic)
        + 0.1108 x [CA19-9 > 1051 U/mL] + 0.2126 x [CEA > 3.80 ng/mL]

Risk groups: low <= 2.14, intermediate 2.14-2.89, high > 2.89. The score is
missing (NaN) unless all five inputs are observed; analyses comparing SENECA
with the ensemble use these complete cases only.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SENECA_COEFFICIENTS = {
    "neutrophils": 0.1757,
    "ecog_ps": 0.5055,
    "stage": 1.4037,
    "ca199_bin": 0.1108,
    "cea_bin": 0.2126,
}
SENECA_FEATURE_THRESHOLDS = {"ca199": 1051.0, "cea": 3.80}
SENECA_RISK_THRESHOLDS = {"low": 2.14, "high": 2.89}
SENECA_FEATURES = list(SENECA_COEFFICIENTS)


def _binary(source: pd.Series, condition: pd.Series) -> pd.Series:
    """0/1 indicator that stays missing where the source value is missing."""
    return condition.astype(float).where(source.notna())


class SENECAModel:
    """Published SENECA score; inputs are the study's raw columns."""

    def __init__(self):
        self.coefficients = SENECA_COEFFICIENTS
        self.feature_thresholds = SENECA_FEATURE_THRESHOLDS
        self.risk_thresholds = SENECA_RISK_THRESHOLDS

    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """The five SENECA inputs, coded as in the publication."""
        out = pd.DataFrame(index=df.index)
        out["neutrophils"] = df["neutrophils"] if "neutrophils" in df.columns else np.nan
        if "ecog bin" in df.columns:
            out["ecog_ps"] = df["ecog bin"]
        elif "ps ecog" in df.columns:  # binary in the prepared data, 0-3 in the raw data
            out["ecog_ps"] = _binary(df["ps ecog"], df["ps ecog"] > 0)
        else:
            out["ecog_ps"] = np.nan
        if "locally advanced/metastatic" in df.columns:  # 1 locally advanced, 2 metastatic
            stage = df["locally advanced/metastatic"]
            out["stage"] = _binary(stage, stage == 2)
        else:
            out["stage"] = np.nan
        for name in ["ca199", "cea"]:
            out[f"{name}_bin"] = (_binary(df[name], df[name] > self.feature_thresholds[name])
                                  if name in df.columns else np.nan)
        missing_columns = [c for c in ["neutrophils", "ca199", "cea"] if c not in df.columns]
        if missing_columns:
            logger.warning(f"SENECA inputs not found: {missing_columns}")
        return out

    def compute_risk_score(self, df: pd.DataFrame) -> np.ndarray:
        """Linear predictor; NaN if any input is missing."""
        features = self._preprocess_features(df)
        return sum(self.coefficients[f] * features[f] for f in self.coefficients).values

    def assign_risk_groups(self, risk_scores: np.ndarray) -> np.ndarray:
        """0 low (<= 2.14), 1 intermediate, 2 high (> 2.89). Missing scores fall in
        group 1 and must be excluded with complete_case_mask()."""
        groups = np.ones(len(risk_scores), dtype=int)
        groups[risk_scores <= self.risk_thresholds["low"]] = 0
        groups[risk_scores > self.risk_thresholds["high"]] = 2
        return groups

    def complete_case_mask(self, df: pd.DataFrame) -> np.ndarray:
        """True where all five inputs are observed."""
        return self._preprocess_features(df).notna().all(axis=1).values

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score, group and complete-case flag of every patient."""
        scores = self.compute_risk_score(df)
        complete = self.complete_case_mask(df)
        logger.info(f"SENECA: {int(complete.sum())}/{len(df)} patients with all five inputs")
        return pd.DataFrame({"seneca_risk_score": scores,
                             "seneca_group": self.assign_risk_groups(scores),
                             "complete_case": complete.astype(int)}, index=df.index)
