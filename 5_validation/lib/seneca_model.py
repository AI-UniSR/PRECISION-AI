"""
SENECA Prognostic Model for Biliary Tract Cancer

Canonical implementation of the published clinical Cox proportional hazards model.
Consolidates the SENECAModel class (seneca_predictions.py) and the inline formula
(validate_external.py) into a single source of truth.

Published coefficients, feature thresholds, and risk group thresholds are
immutable constants — they must match the original publication exactly.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# PUBLISHED SENECA CONSTANTS (immutable)
# ============================================================================

SENECA_COEFFICIENTS = {
    "neutrophils": 0.1757,
    "ecog_ps": 0.5055,
    "stage": 1.4037,
    "ca199_bin": 0.1108,
    "cea_bin": 0.2126,
}

SENECA_FEATURE_THRESHOLDS = {
    "ca199": 1051.0,   # U/mL — binary: >1051
    "cea": 3.80,       # ng/mL — binary: >3.80
}

SENECA_RISK_THRESHOLDS = {
    "low": 2.14,       # score ≤ 2.14 → Low risk
    "high": 2.89,      # score > 2.89 → High risk
}

SENECA_FEATURES = list(SENECA_COEFFICIENTS.keys())


class SENECAModel:
    """Published SENECA Cox PH model for BTC prognostication."""

    def __init__(self):
        self.coefficients = SENECA_COEFFICIENTS
        self.feature_thresholds = SENECA_FEATURE_THRESHOLDS
        self.risk_thresholds = SENECA_RISK_THRESHOLDS

    # ------------------------------------------------------------------
    # Feature preprocessing
    # ------------------------------------------------------------------
    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map raw clinical columns to the 5 SENECA features."""
        seneca_df = pd.DataFrame(index=df.index)

        # 1. Neutrophils (continuous, cells × 10⁹/L)
        if "neutrophils" in df.columns:
            seneca_df["neutrophils"] = df["neutrophils"]
        else:
            logger.warning("neutrophils column not found")
            seneca_df["neutrophils"] = np.nan

        # 2. ECOG PS: binary (0 if ECOG=0, 1 if ECOG>0)
        if "ecog bin" in df.columns:
            seneca_df["ecog_ps"] = df["ecog bin"]
        elif "ps ecog" in df.columns:
            seneca_df["ecog_ps"] = (df["ps ecog"] > 0).astype(float)
        else:
            logger.warning("ECOG column not found")
            seneca_df["ecog_ps"] = np.nan

        # 3. Stage: binary (0=locally advanced, 1=metastatic)
        if "locally advanced/metastatic" in df.columns:
            seneca_df["stage"] = (df["locally advanced/metastatic"] == 2).astype(float)
        else:
            logger.warning("locally advanced/metastatic column not found")
            seneca_df["stage"] = np.nan

        # 4. CA19-9: binary, threshold >1051 U/mL
        if "ca199" in df.columns:
            seneca_df["ca199_bin"] = (
                df["ca199"] > self.feature_thresholds["ca199"]
            ).astype(float)
        else:
            logger.warning("ca199 column not found")
            seneca_df["ca199_bin"] = np.nan

        # 5. CEA: binary, threshold >3.80 ng/mL
        if "cea" in df.columns:
            seneca_df["cea_bin"] = (
                df["cea"] > self.feature_thresholds["cea"]
            ).astype(float)
        else:
            logger.warning("cea column not found")
            seneca_df["cea_bin"] = np.nan

        return seneca_df

    # ------------------------------------------------------------------
    # Risk score
    # ------------------------------------------------------------------
    def compute_risk_score(self, df: pd.DataFrame) -> np.ndarray:
        """Compute the linear predictor (risk score) for each patient."""
        seneca_features = self._preprocess_features(df)
        score = sum(
            self.coefficients[feat] * seneca_features[feat]
            for feat in self.coefficients
        )
        return score.values

    # ------------------------------------------------------------------
    # Risk groups
    # ------------------------------------------------------------------
    def assign_risk_groups(self, risk_scores: np.ndarray) -> np.ndarray:
        """
        Assign SENECA risk groups using published thresholds.

        Returns
        -------
        groups : np.ndarray of int
            0 = Low (score ≤ 2.14),
            1 = Intermediate (2.14 < score ≤ 2.89),
            2 = High (score > 2.89).
        """
        groups = np.ones(len(risk_scores), dtype=int)  # default: intermediate
        groups[risk_scores <= self.risk_thresholds["low"]] = 0
        groups[risk_scores > self.risk_thresholds["high"]] = 2
        return groups

    # ------------------------------------------------------------------
    # Complete-case mask
    # ------------------------------------------------------------------
    def complete_case_mask(self, df: pd.DataFrame) -> np.ndarray:
        """Return boolean mask: True where all 5 SENECA features are non-null."""
        seneca_features = self._preprocess_features(df)
        return seneca_features.notna().all(axis=1).values

    # ------------------------------------------------------------------
    # Convenience: predict everything at once
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate SENECA predictions for a cohort.

        Returns DataFrame with columns:
            seneca_risk_score, seneca_group, complete_case
        """
        risk_scores = self.compute_risk_score(df)
        groups = self.assign_risk_groups(risk_scores)
        complete = self.complete_case_mask(df)

        n_complete = int(complete.sum())
        logger.info(
            f"SENECA predictions: {len(df)} patients, "
            f"{n_complete} complete cases ({100 * n_complete / len(df):.1f}%)"
        )

        return pd.DataFrame(
            {
                "seneca_risk_score": risk_scores,
                "seneca_group": groups,
                "complete_case": complete.astype(int),
            },
            index=df.index,
        )
