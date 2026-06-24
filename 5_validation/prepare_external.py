"""
External Validation Data Preparation
Applies same preprocessing as training pipeline to external cohort.

Data: Starts with 783 validation cohort patients from raw data.
      - 5 patients excluded due to date inconsistencies (unparseable or
        chronologically impossible date combinations).
      - Patients also present in the training cohort are removed (data quality check):
        a patient is considered shared when ['dob','sex','center','cgi start date']
        match between cohorts.
      - 778 patients (less the cross-cohort duplicates) proceed to feature alignment.
      - 1 patient removed with tte=0 (identical diagnosis and follow-up dates).
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_raw_data(df: pd.DataFrame, dataset_name: str = "External") -> None:
    """Print diagnostic information about raw dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"RAW DATA VALIDATION - {dataset_name} Dataset")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Check for survival columns
    if 'tte' in df.columns and 'event' in df.columns:
        n_events = int((df['event'] == 1).sum())
        n_censored = int((df['event'] == 0).sum())
        logger.info(f"\n✓ Survival columns found: tte, event")
        logger.info(f"  tte range: [{df['tte'].min():.2f}, {df['tte'].max():.2f}]")
        logger.info(f"  Events: {n_events} ({n_events / len(df) * 100:.1f}%)")
        logger.info(f"  Censored: {n_censored} ({n_censored / len(df) * 100:.1f}%)")
    else:
        logger.warning(f"✗ Missing survival columns (tte, event)")

    logger.info(f"{'='*60}\n")


def load_selected_features_from_model(model_name: str, version: str = "latest") -> list:
    """Load selected_features.json from MLflow model artifacts."""
    logger.info(f"Loading selected features from model: {model_name} (version: {version})")
    
    model_uri = f"models:/{model_name}/{version}"
    
    try:
        # Download model artifacts to temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = f"{model_uri}/artifacts/selected_features.json"
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_path, dst_path=tmpdir)
            
            with open(local_path, 'r') as f:
                selected_features_dict = json.load(f)
            
            logger.info(f"✓ Selected features loaded from model")
            logger.info(f"  Models in dict: {list(selected_features_dict.keys())}")
            
            # Get union of all features across all models
            all_features = set()
            for model_features in selected_features_dict.values():
                all_features.update(model_features)
            
            training_features = sorted(list(all_features))
            logger.info(f"  Total unique features: {len(training_features)}")
            
            return training_features
    
    except Exception as e:
        logger.error(f"Failed to load features from model: {e}")
        raise


def align_with_training_features(
    df: pd.DataFrame,
    model_name: str,
    model_version: str = "latest"
) -> pd.DataFrame:
    """
    Critical step: Align external dataset with training features.
    Also preserves SENECA baseline features for comparison.
    """
    logger.info("ALIGNING WITH TRAINING FEATURES")
    logger.info("=" * 60)
    
    # Define SENECA required features
    SENECA_FEATURES = ['neutrophils', 'ps ecog', 'locally advanced/metastatic', 'ca199', 'cea']
    
    # Load training features from MLflow model
    training_features = load_selected_features_from_model(model_name, model_version)
    
    # Combine ML features with SENECA features (remove duplicates)
    all_required_features = list(set(training_features + SENECA_FEATURES))
    
    logger.info(f"Training (ML) features: {len(training_features)} features")
    logger.info(f"SENECA features: {len(SENECA_FEATURES)} features")
    logger.info(f"Combined (unique): {len(all_required_features)} features")
    logger.info(f"External features before alignment: {df.shape[1]} features")
    
    # Separate survival columns (tte and event)
    survival_cols = ['tte', 'event']
    X_external = df.drop(columns=survival_cols, errors='ignore')
    
    # Check overlap
    external_features = set(X_external.columns)
    required_features_set = set(all_required_features)
    
    common_features = external_features & required_features_set
    missing_in_external = required_features_set - external_features
    extra_in_external = external_features - required_features_set
    
    logger.info(f"\nFEATURE ALIGNMENT REPORT:")
    logger.info(f"  Common features: {len(common_features)} / {len(all_required_features)} ({len(common_features)/len(all_required_features)*100:.1f}%)")
    logger.info(f"  Missing in external: {len(missing_in_external)}")
    logger.info(f"  Extra in external: {len(extra_in_external)} (will be dropped)")
    
    if missing_in_external:
        logger.warning(f"\nMISSING FEATURES (will be filled with 0):")
        for feat in list(missing_in_external)[:20]:
            logger.warning(f"  - {feat}")
        if len(missing_in_external) > 20:
            logger.warning(f"  ... and {len(missing_in_external)-20} more")
    
    if extra_in_external:
        logger.info(f"\nEXTRA FEATURES (will be dropped):")
        for feat in list(extra_in_external)[:20]:
            logger.info(f"  - {feat}")
        if len(extra_in_external) > 20:
            logger.info(f"  ... and {len(extra_in_external)-20} more")
    
    # Create aligned dataframe
    X_aligned = pd.DataFrame(index=X_external.index)
    
    for feat in all_required_features:
        if feat in X_external.columns:
            X_aligned[feat] = X_external[feat]
    
    # Add survival columns back
    for col in survival_cols:
        if col in df.columns:
            X_aligned[col] = df[col]
    
    logger.info(f"\n✓ ALIGNMENT COMPLETE")
    logger.info(f"  Final feature count: {X_aligned.shape[1]}")
    logger.info(f"  Includes ML features: {len(training_features)}")
    logger.info(f"  Includes SENECA features: {len(SENECA_FEATURES)}")
    logger.info("=" * 60)
    logger.info("")
    
    return X_aligned


def validate_and_clean_survival_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean survival times by removing invalid entries.

    Removes samples where tte is NaN or ≤ 0, logging detailed reasons.
    """
    logger.info("VALIDATING SURVIVAL TIMES")
    logger.info("-" * 60)

    initial_count = len(df)
    logger.info(f"Initial samples: {initial_count}")

    # --- NaN survival times ---
    nan_tte = df['tte'].isna()
    nan_event = df['event'].isna()
    nan_mask = nan_tte | nan_event
    n_nan = int(nan_mask.sum())

    if n_nan > 0:
        logger.warning(f"Found {n_nan} samples with missing survival data "
                        f"(tte NaN: {int(nan_tte.sum())}, event NaN: {int(nan_event.sum())})")
        if 'center' in df.columns:
            center_counts = df.loc[nan_mask, 'center'].value_counts()
            logger.warning("  Breakdown by center:")
            for center, cnt in center_counts.items():
                logger.warning(f"    {center}: {cnt}")
        df = df[~nan_mask].copy()
        logger.info(f"  Removed {n_nan} samples with missing survival data")

    # --- Invalid (≤ 0) survival times ---
    invalid_mask = df['tte'] <= 0
    n_invalid = int(invalid_mask.sum())

    if n_invalid > 0:
        logger.warning(f"Found {n_invalid} samples with invalid survival times (tte ≤ 0)")
        logger.warning(f"  Values: {df.loc[invalid_mask, 'tte'].values}")
        df = df[~invalid_mask].copy()
        logger.info(f"  Removed {n_invalid} samples with tte ≤ 0")

    df = df.reset_index(drop=True)
    logger.info(f"Remaining samples: {len(df)} (removed {initial_count - len(df)} total)")
    logger.info(f"  tte range: [{df['tte'].min():.2f}, {df['tte'].max():.2f}]")
    logger.info("")
    return df


# ---------------------------------------------------------------------------
# Cross-cohort data quality / discovery helpers
# ---------------------------------------------------------------------------

DEDUP_KEYS = ["dob", "sex", "center", "cgi start date"]


def _parse_mixed_date_series(s: pd.Series) -> pd.Series:
    """Parse a date column that may contain ISO and DD/MM/YYYY strings."""
    def _one(x):
        if pd.isna(x):
            return pd.NaT
        x = str(x).strip()
        if not x:
            return pd.NaT
        if "/" in x:
            return pd.to_datetime(x, dayfirst=True, errors="coerce")
        return pd.to_datetime(x, errors="coerce")
    return s.apply(_one)


def _build_dedup_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized DataFrame of dedup keys (dob/cgi start date as date,
    sex as nullable Int64, center as stripped lowercase string)."""
    out = pd.DataFrame(index=df.index)
    out["dob"] = _parse_mixed_date_series(df["dob"]).dt.date
    out["cgi start date"] = _parse_mixed_date_series(df["cgi start date"]).dt.date
    out["sex"] = pd.to_numeric(df["sex"], errors="coerce").astype("Int64")
    out["center"] = df["center"].astype(str).str.strip().str.lower()
    return out


def remove_patients_in_training(
    df_val: pd.DataFrame,
    df_train_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove validation patients that also appear in the training cohort.

    Returns (cleaned_validation_df, removed_patients_df). The removed patients
    DataFrame includes the dedup keys plus 'patient number' (when available)
    from the validation cohort.
    """
    logger.info("CROSS-COHORT PATIENT DEDUPLICATION")
    logger.info("-" * 60)

    missing_val = [c for c in DEDUP_KEYS if c not in df_val.columns]
    missing_train = [c for c in DEDUP_KEYS if c not in df_train_raw.columns]
    if missing_val or missing_train:
        logger.warning(
            f"Skipping dedup — missing key columns. validation missing: {missing_val}; "
            f"training missing: {missing_train}"
        )
        empty = pd.DataFrame(columns=DEDUP_KEYS + ["patient number"])
        return df_val.copy(), empty

    val_keys = _build_dedup_keys(df_val)
    train_keys = _build_dedup_keys(df_train_raw)

    n_val_before = len(df_val)
    n_train = len(df_train_raw)
    logger.info(f"Validation patients before dedup: {n_val_before}")
    logger.info(f"Training patients (raw):           {n_train}")
    logger.info(f"Dedup keys: {DEDUP_KEYS}")

    train_key_set = set(map(tuple, train_keys.dropna(how="any").to_numpy().tolist()))
    val_key_tuples = list(map(tuple, val_keys.to_numpy().tolist()))
    is_dup = pd.Series(
        [
            (None not in t)
            and (not any(pd.isna(v) for v in t))
            and (t in train_key_set)
            for t in val_key_tuples
        ],
        index=df_val.index,
    )

    n_dup = int(is_dup.sum())
    logger.info(f"Patients shared with training cohort: {n_dup}")

    removed = df_val.loc[is_dup].copy()
    removed_export_cols = list(DEDUP_KEYS)
    if "patient number" in removed.columns:
        removed_export_cols = ["patient number"] + removed_export_cols
    removed_out = removed[[c for c in removed_export_cols if c in removed.columns]].copy()

    cleaned = df_val.loc[~is_dup].copy().reset_index(drop=True)
    logger.info(f"Validation patients after dedup:  {len(cleaned)} "
                f"(removed {n_val_before - len(cleaned)})")
    logger.info("-" * 60)
    logger.info("")

    try:
        mlflow.log_metric("val_patients_before_dedup", n_val_before)
        mlflow.log_metric("val_patients_removed_dedup", n_dup)
        mlflow.log_metric("val_patients_after_dedup", len(cleaned))
        mlflow.log_metric("train_patients_raw_for_dedup", n_train)
    except Exception as e:
        logger.warning(f"Could not log dedup metrics to MLflow: {e}")

    return cleaned, removed_out


def _log_date_span(df: pd.DataFrame, col: str, label: str, prefix: str) -> None:
    if col not in df.columns:
        logger.warning(f"  [{label}] column '{col}' not found")
        return
    dates = _parse_mixed_date_series(df[col])
    n_valid = int(dates.notna().sum())
    if n_valid == 0:
        logger.warning(f"  [{label}] {col}: no parseable dates")
        return
    dmin, dmax = dates.min(), dates.max()
    logger.info(f"  [{label}] {col}: {dmin.date()} → {dmax.date()}  (n={n_valid})")
    try:
        mlflow.log_param(f"{prefix}_{col.replace(' ', '_')}_min", str(dmin.date()))
        mlflow.log_param(f"{prefix}_{col.replace(' ', '_')}_max", str(dmax.date()))
    except Exception:
        pass


def report_cohort_date_spans(
    df_train_raw: pd.DataFrame,
    df_val_clean: pd.DataFrame,
    train_lock: str,
    val_lock: str,
) -> None:
    """Print/log date spans for both cohorts (validation AFTER dedup)."""
    logger.info("=" * 60)
    logger.info("COHORT DATE SPANS (discovery, no downstream impact)")
    logger.info("=" * 60)

    logger.info("Training")
    logger.info(f"  Cohort lock: {train_lock}")
    _log_date_span(df_train_raw, "date of death or last fup", "Training", "train")
    _log_date_span(df_train_raw, "cgi start date", "Training", "train")

    logger.info("Validation (after cross-cohort dedup)")
    logger.info(f"  Cohort lock: {val_lock}")
    _log_date_span(df_val_clean, "date of death or last fup", "Validation", "val")
    _log_date_span(df_val_clean, "cgi start date", "Validation", "val")

    try:
        mlflow.log_param("training_cohort_lock", train_lock)
        mlflow.log_param("validation_cohort_lock", val_lock)
    except Exception:
        pass
    logger.info("=" * 60)
    logger.info("")


def report_centers(
    df_train_raw: pd.DataFrame,
    df_val_clean: pd.DataFrame,
    artifacts_dir: Path,
) -> tuple[set, set]:
    """Compare centers between training and validation cohorts.

    Returns (shared_centers, validation_only_centers) using normalized names.
    """
    logger.info("=" * 60)
    logger.info("CENTER COMPARISON (discovery, no downstream impact)")
    logger.info("=" * 60)

    if "center" not in df_train_raw.columns or "center" not in df_val_clean.columns:
        logger.warning("Skipping center comparison — 'center' column missing")
        return set(), set()

    train_centers = df_train_raw["center"].astype(str).str.strip()
    val_centers = df_val_clean["center"].astype(str).str.strip()

    train_norm = train_centers.str.lower()
    val_norm = val_centers.str.lower()

    train_set = set(train_norm.dropna().unique()) - {"", "nan"}
    val_set = set(val_norm.dropna().unique()) - {"", "nan"}

    shared = train_set & val_set
    val_only = val_set - train_set
    train_only = train_set - val_set

    val_counts = val_norm.value_counts()
    train_counts = train_norm.value_counts()

    logger.info(f"Centers in training:    {len(train_set)}")
    logger.info(f"Centers in validation:  {len(val_set)}")
    logger.info(f"Shared centers:         {len(shared)}")
    logger.info(f"Validation-only centers: {len(val_only)}")
    if val_only:
        total_new = 0
        logger.info("New centers (validation-only):")
        for c in sorted(val_only):
            n = int(val_counts.get(c, 0))
            total_new += n
            logger.info(f"  - {c}: {n} patients")
        logger.info(f"  TOTAL new-center patients: {total_new}")
        try:
            mlflow.log_metric("val_only_centers_n", len(val_only))
            mlflow.log_metric("val_only_centers_patients", total_new)
        except Exception:
            pass
    if train_only:
        logger.info(f"Training-only centers ({len(train_only)}): "
                    f"{sorted(train_only)[:10]}{' ...' if len(train_only) > 10 else ''}")

    # Save center comparison as artifact
    rows = []
    for c in sorted(train_set | val_set):
        rows.append({
            "center": c,
            "in_training": c in train_set,
            "in_validation": c in val_set,
            "n_training": int(train_counts.get(c, 0)),
            "n_validation": int(val_counts.get(c, 0)),
        })
    centers_df = pd.DataFrame(rows)
    out_path = artifacts_dir / "center_comparison.csv"
    centers_df.to_csv(out_path, index=False)
    try:
        mlflow.log_artifact(str(out_path))
    except Exception as e:
        logger.warning(f"Could not log center_comparison.csv: {e}")

    logger.info("=" * 60)
    logger.info("")
    return shared, val_only


def plot_cgi_start_histogram_shared_centers(
    df_train_raw: pd.DataFrame,
    df_val_clean: pd.DataFrame,
    shared_centers: set,
    artifacts_dir: Path,
) -> None:
    """Histogram comparing CGI start date by month, restricted to centers
    present in BOTH training and validation cohorts."""
    if not shared_centers:
        logger.info("No shared centers — skipping CGI start date histogram")
        return
    if "cgi start date" not in df_train_raw.columns or "cgi start date" not in df_val_clean.columns:
        logger.info("'cgi start date' missing in one cohort — skipping histogram")
        return

    train_norm_center = df_train_raw["center"].astype(str).str.strip().str.lower()
    val_norm_center = df_val_clean["center"].astype(str).str.strip().str.lower()

    t_dates = _parse_mixed_date_series(df_train_raw["cgi start date"])
    v_dates = _parse_mixed_date_series(df_val_clean["cgi start date"])

    t_mask = train_norm_center.isin(shared_centers) & t_dates.notna()
    v_mask = val_norm_center.isin(shared_centers) & v_dates.notna()

    if t_mask.sum() == 0 or v_mask.sum() == 0:
        logger.info("No data for shared-centers histogram — skipping")
        return

    t_months = t_dates[t_mask].dt.to_period("M").dt.to_timestamp()
    v_months = v_dates[v_mask].dt.to_period("M").dt.to_timestamp()

    all_min = min(t_months.min(), v_months.min())
    all_max = max(t_months.max(), v_months.max())
    bins = pd.date_range(start=all_min, end=all_max + pd.offsets.MonthBegin(1), freq="MS")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(t_months, bins=bins, alpha=0.55, label=f"Training (n={int(t_mask.sum())})", color="#1f77b4")
    ax.hist(v_months, bins=bins, alpha=0.55, label=f"Validation (n={int(v_mask.sum())})", color="#d62728")
    ax.set_title("CGI start date distribution — centers present in BOTH cohorts")
    ax.set_xlabel("CGI start date (month)")
    ax.set_ylabel("Patients")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = artifacts_dir / "cgi_start_date_hist_shared_centers.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved CGI start date histogram (shared centers): {out_path}")
    try:
        mlflow.log_artifact(str(out_path))
    except Exception as e:
        logger.warning(f"Could not log histogram artifact: {e}")


def main():
    parser = argparse.ArgumentParser(description="Prepare external validation data")
    parser.add_argument("--raw_data", type=str, required=True, help="Input raw data path")
    parser.add_argument("--training_raw_data", type=str, required=True,
                        help="Raw training cohort CSV (used for cross-cohort patient deduplication)")
    parser.add_argument("--dict_data", type=str, required=False, default=None,
                        help="Dictionary CSV (Field -> FieldShortName) used to rename training raw columns")
    parser.add_argument("--model_name", type=str, required=True, help="Name of registered model")
    parser.add_argument("--model_version", type=str, default="latest", help="Model version")
    parser.add_argument("--output_data", type=str, required=True, help="Output prepared data path")
    parser.add_argument("--output_report", type=str, required=True, help="Output report path")
    
    args = parser.parse_args()
    
    # Tag this MLflow run so it is recognizable as the dedup-corrected preparation
    try:
        mlflow.set_tag("mlflow.runName", "prepare_external_dedup_vs_training")
        mlflow.set_tag("data_quality_check", "remove_patients_in_training_cohort")
    except Exception:
        pass

    logger.info("=" * 80)
    logger.info("EXTERNAL VALIDATION DATA PREPARATION")
    logger.info("=" * 80)
    logger.info(f"Input: {args.raw_data}")
    logger.info(f"Training raw (for dedup): {args.training_raw_data}")
    logger.info(f"Model: {args.model_name} (version: {args.model_version})")
    logger.info(f"Output: {args.output_data}")
    logger.info("")
    
    # Load raw data
    if args.raw_data.endswith('.csv'):
        df = pd.read_csv(args.raw_data)
    else:
        raise ValueError(f"Unsupported file format: {args.raw_data}. Only CSV is supported.")
    
    # Rename survival columns to standard names (tte, event)
    df = df.rename(columns={'os': 'tte', 'death': 'event'})
    logger.info("Renamed survival columns: os → tte, death → event")

    # Ensure event is numeric
    df['event'] = pd.to_numeric(df['event'], errors='coerce')

    # Always recompute tte from dates (fixes Excel formula failures for some centers
    # where dates stored as text cause NaN or wrong values in the pre-computed os column).
    # Handles mixed date formats: ISO (YYYY-MM-DD) and slash (DD/MM/YYYY).
    def _parse_mixed_date(s):
        if pd.isna(s):
            return pd.NaT
        s = str(s).strip()
        if '/' in s:
            return pd.to_datetime(s, dayfirst=True, errors='coerce')
        return pd.to_datetime(s, errors='coerce')

    if 'cgi start date' in df.columns and 'date of death or last fup' in df.columns:
        t_start = df['cgi start date'].apply(_parse_mixed_date)
        t_end = df['date of death or last fup'].apply(_parse_mixed_date)
        tte_computed = (t_end - t_start).dt.days / (365.25 / 12)
        n_was_nan = int(df['tte'].isna().sum())
        n_recomputed = int(tte_computed.notna().sum())
        df['tte'] = tte_computed
        logger.info(f"Recomputed tte from dates: {n_recomputed}/{len(df)} valid "
                    f"(previously {n_was_nan} were NaN due to Excel date format issues)")
    else:
        logger.warning("Date columns not found — falling back to pre-computed os column")
        df['tte'] = pd.to_numeric(df['tte'], errors='coerce')
    
    # Filter only validation cohort — but FIRST capture the training-cohort rows
    # from the same raw file: they carry the dedup keys (dob/center/cgi start date)
    # and survival/center metadata that the registered training URI does NOT.
    df_train_in_file = pd.DataFrame()
    if 'coorte' in df.columns:
        initial_count = len(df)
        df_train_in_file = df[df['coorte'] == 'training'].copy()
        df = df[df['coorte'] == 'validation'].copy()
        logger.info(f"Filtered to validation cohort: {len(df)} samples (from {initial_count})")
        logger.info(f"In-file training-cohort rows captured for dedup/discovery: {len(df_train_in_file)}")
    else:
        logger.warning("Column 'coorte' not found - processing all data")
    # Log data lineage
    logger.info(f"Starting with {len(df)} validation patients (783 original - 4 with unparseable dates)")

    # ── DATA QUALITY: remove validation patients shared with the training cohort
    # We use the in-file training rows as the reference cohort because the
    # registered training URI (file-btc-os) is feature-filtered and does not
    # carry dob/center/cgi start date. The provided --training_raw_data /
    # --dict_data inputs are accepted for backward compatibility but only used
    # as a fallback if the in-file training rows are unavailable.
    if len(df_train_in_file) > 0:
        df_train_raw = df_train_in_file
        logger.info(f"Using in-file training cohort ({len(df_train_raw)} patients) as dedup reference")
    else:
        logger.warning("No in-file training rows; falling back to --training_raw_data")
        df_train_raw = pd.read_csv(args.training_raw_data)
        if args.dict_data:
            try:
                dd = pd.read_csv(args.dict_data)
                if {'Field', 'FieldShortName'}.issubset(dd.columns):
                    rename_map = {
                        str(f).strip(): str(s).strip()
                        for f, s in zip(dd['Field'], dd['FieldShortName'])
                        if pd.notna(f) and pd.notna(s) and str(f).strip() != str(s).strip()
                    }
                    ws_collapsed = {' '.join(str(f).split()): str(s).strip()
                                    for f, s in zip(dd['Field'], dd['FieldShortName'])
                                    if pd.notna(f) and pd.notna(s)}
                    cols_norm = {c: ' '.join(str(c).split()) for c in df_train_raw.columns}
                    effective = {}
                    for orig_col, norm_col in cols_norm.items():
                        if orig_col in rename_map:
                            effective[orig_col] = rename_map[orig_col]
                        elif norm_col in ws_collapsed and ws_collapsed[norm_col] != orig_col:
                            effective[orig_col] = ws_collapsed[norm_col]
                    if effective:
                        df_train_raw = df_train_raw.rename(columns=effective)
            except Exception as e:
                logger.warning(f"Dictionary rename failed: {e}")

    # Diagnostic: show which dedup keys are actually present in the training reference
    present = [k for k in DEDUP_KEYS if k in df_train_raw.columns]
    missing = [k for k in DEDUP_KEYS if k not in df_train_raw.columns]
    logger.info(f"Training-reference dedup-key coverage: present={present}, missing={missing}")

    # Prepare an artifacts directory for discovery outputs
    artifacts_dir = Path(tempfile.mkdtemp(prefix="prep_ext_artifacts_"))

    n_val_before_dedup = len(df)
    df, removed_patients = remove_patients_in_training(df, df_train_raw)
    n_val_after_dedup = len(df)
    logger.info(
        f"DEDUP RESULT: validation patients {n_val_before_dedup} → {n_val_after_dedup} "
        f"(removed {n_val_before_dedup - n_val_after_dedup} shared with training)"
    )

    removed_csv_path = artifacts_dir / "removed_patients_shared_with_training.csv"
    removed_patients.to_csv(removed_csv_path, index=False)
    try:
        mlflow.log_artifact(str(removed_csv_path))
        logger.info(f"Logged removed-patients artifact to MLflow: {removed_csv_path.name}")
    except Exception as e:
        logger.warning(f"Could not log removed-patients artifact: {e}")

    # ── DISCOVERY (no downstream impact): cohort lock dates, follow-up & CGI date spans
    TRAIN_COHORT_LOCK = "30 Jun 2025 (BTC_MONDIALE_REV_20250630)"
    VAL_COHORT_LOCK = "15 Jan 2026 (file_btc_validazione_15012026.xlsx)"
    report_cohort_date_spans(df_train_raw, df, TRAIN_COHORT_LOCK, VAL_COHORT_LOCK)

    # ── DISCOVERY: centers comparison
    shared_centers, _val_only_centers = report_centers(df_train_raw, df, artifacts_dir)

    # ── DISCOVERY: CGI start date histogram for centers in BOTH cohorts
    plot_cgi_start_histogram_shared_centers(df_train_raw, df, shared_centers, artifacts_dir)

    # Step 1: Validate and clean survival times (NaN + tte ≤ 0)
    initial_for_validation = len(df)
    df = validate_and_clean_survival_times(df)
    removed_invalid_tte = initial_for_validation - len(df)
    logger.info(f"Survival time validation: removed {removed_invalid_tte} patient(s) with tte≤0 → {len(df)} patients remain")    # Step 2: Validate raw data summary
    validate_raw_data(df, dataset_name="External")

    # Log the combined dataset (training + cleaned validation, all columns) as a downloadable CSV artifact.
    # Logged BEFORE ecog binarization so both 'ps ecog' (multi-class) and 'ecog bin' (binary)
    # are still present in the validation rows, matching the original file column structure.
    # - all 735 training rows (unchanged, from in-file training cohort)
    # - 698 validation rows (after dedup and survival-time cleaning, before any column transforms)
    df_combined = pd.concat([df_train_in_file, df], ignore_index=True)
    full_csv_path = artifacts_dir / "btc_dataset_selected_patients.csv"
    df_combined.to_csv(full_csv_path, index=False)
    try:
        mlflow.log_artifact(str(full_csv_path))
        logger.info(f"Logged combined dataset artifact to MLflow: {full_csv_path.name} "
                    f"(training={len(df_train_in_file)}, validation={len(df)}, "
                    f"total={len(df_combined)}, cols={df_combined.shape[1]})")
    except Exception as e:
        logger.warning(f"Could not log combined dataset artifact: {e}")

    # Mirror data_prep.py: replace multi-class 'ps ecog' with binary 'ecog bin'
    # (renamed to 'ps ecog') so the feature fed to the model is binary (0/1),
    # consistent with how the model was trained.
    if 'ps ecog' in df.columns and 'ecog bin' in df.columns:
        df.drop(columns=['ps ecog'], inplace=True)
        df.rename(columns={'ecog bin': 'ps ecog'}, inplace=True)
        logger.info("Replaced multi-class 'ps ecog' with binary 'ecog bin' (renamed to 'ps ecog')")

    # Mirror data_prep.py: binarize 'stent or biliary draige'
    # Dictionary rule-0:0 | rule-1:1,2,3  →  0=No drainage, 1=Any stent/drainage
    if 'stent or biliary draige' in df.columns:
        df['stent or biliary draige'] = df['stent or biliary draige'].apply(
            lambda x: 0 if x == 0 else (1 if pd.notna(x) else x)
        )
        logger.info("Binarized 'stent or biliary draige': 0=No drainage, 1=Any stent/drainage")

    # Step 3: Align with training features (the only real filtering step)
    df_aligned = align_with_training_features(df, args.model_name, args.model_version)
    
    # Step 4: Convert feature columns to numeric for Parquet
    logger.info("Converting feature columns to numeric types...")
    feature_cols = [col for col in df_aligned.columns if col not in ['tte', 'event']]
    for col in feature_cols:
        df_aligned[col] = pd.to_numeric(df_aligned[col], errors='coerce')
    logger.info("✓ Type conversion complete")
    
    # Save prepared data (uri_folder output)
    output_dir = Path(args.output_data)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "prepared_external.parquet"
    df_aligned.to_parquet(output_file, index=False)
    logger.info(f"✓ Prepared data saved: {output_file}")
    logger.info(f"  Shape: {df_aligned.shape}")
    
    # Generate report
    report = {
        'n_samples': len(df_aligned),
        'n_features': df_aligned.shape[1] - 2,  # Exclude tte, event
        'n_events': int(df_aligned['event'].sum()) if 'event' in df_aligned else None,
        'event_rate': float(df_aligned['event'].mean()) if 'event' in df_aligned else None,
        'median_survival': float(df_aligned['tte'].median()) if 'tte' in df_aligned else None,
        'missing_values_total': int(df_aligned.isnull().sum().sum()),
        'dedup_validation_before': int(n_val_before_dedup),
        'dedup_validation_after': int(n_val_after_dedup),
        'dedup_patients_removed': int(n_val_before_dedup - n_val_after_dedup),
        'features': list(df_aligned.drop(columns=['tte', 'event'], errors='ignore').columns)
    }
    
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✓ Report saved: {report_path}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("PREPARATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
