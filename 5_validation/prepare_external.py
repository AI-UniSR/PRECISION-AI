"""Temporal validation cohort: deduplication and alignment with the model.

The input file holds the development-cohort rows (coorte = 'training') and
779 temporal-cohort rows (coorte = 'validation': 783 records minus 4 with
unparseable dates, removed before this step).

1. Follow-up is recomputed from the dates (CGI start to death or last
   follow-up, months of 365.25/12 days) because the stored 'os' value was
   wrong for some centres whose dates Excel had read as text.
2. Temporal-cohort patients also in the development cohort (same date of
   birth, sex, centre and CGI start date) are removed: 80 patients.
3. Patients with missing or non-positive follow-up are removed: 1 patient.
   698 patients remain.
4. ECOG and biliary stent/drainage are recoded as in data_prep.py, and the
   columns are restricted to the predictors of the registered ensemble, the
   SENECA inputs, follow-up and centre.

Also logged to MLflow: the removed patients, the combined analysis dataset of
both cohorts (735 + 698 patients), the date ranges of the two cohorts and the
centres of each. The report also counts the temporal patients whose treatment
started before, within or after the development cohort's treatment-start window
and those from centres absent from the development cohort (Methods, Study cohort).
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEDUP_KEYS = ["dob", "sex", "center", "cgi start date"]
SENECA_INPUTS = ["neutrophils", "ps ecog", "locally advanced/metastatic", "ca199", "cea"]
DAYS_PER_MONTH = 365.25 / 12


def parse_dates(s: pd.Series) -> pd.Series:
    """Dates stored either as ISO strings or as DD/MM/YYYY."""
    def parse(x):
        if pd.isna(x) or not str(x).strip():
            return pd.NaT
        x = str(x).strip()
        return pd.to_datetime(x, dayfirst=True, errors="coerce") if "/" in x else pd.to_datetime(x, errors="coerce")
    return s.apply(parse)


def dedup_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalised matching keys: dates as dates, sex as integer, centre lower case."""
    return pd.DataFrame({
        "dob": parse_dates(df["dob"]).dt.date,
        "cgi start date": parse_dates(df["cgi start date"]).dt.date,
        "sex": pd.to_numeric(df["sex"], errors="coerce").astype("Int64"),
        "center": df["center"].astype(str).str.strip().str.lower(),
    }, index=df.index)


def remove_patients_in_training(df_val: pd.DataFrame, df_train: pd.DataFrame):
    """Drop validation patients whose four keys (all observed) match a development patient."""
    for name, df in [("validation", df_val), ("development", df_train)]:
        missing = [k for k in DEDUP_KEYS if k not in df.columns]
        if missing:
            raise ValueError(f"{name} rows lack the matching keys {missing}")
    train_keys = set(map(tuple, dedup_keys(df_train).dropna(how="any").to_numpy().tolist()))
    is_dup = pd.Series([(None not in key) and not any(pd.isna(v) for v in key) and key in train_keys
                        for key in map(tuple, dedup_keys(df_val).to_numpy().tolist())], index=df_val.index)
    removed = df_val.loc[is_dup, [c for c in ["patient number"] + DEDUP_KEYS if c in df_val.columns]]
    logger.info(f"Shared with the development cohort: {int(is_dup.sum())} of {len(df_val)} patients")
    return df_val.loc[~is_dup].reset_index(drop=True), removed


def log_cohort_dates(df: pd.DataFrame, prefix: str) -> None:
    """Range of the CGI start and last-contact dates of a cohort."""
    for col in ["cgi start date", "date of death or last fup"]:
        if col not in df.columns:
            continue
        dates = parse_dates(df[col]).dropna()
        if len(dates):
            logger.info(f"{prefix} {col}: {dates.min().date()} to {dates.max().date()} (n={len(dates)})")
            mlflow.log_param(f"{prefix}_{col.replace(' ', '_')}_min", str(dates.min().date()))
            mlflow.log_param(f"{prefix}_{col.replace(' ', '_')}_max", str(dates.max().date()))


def centre_comparison(df_train: pd.DataFrame, df_val: pd.DataFrame) -> pd.DataFrame:
    """Patients per centre in each cohort (centre names normalised to lower case)."""
    train = df_train["center"].astype(str).str.strip().str.lower()
    val = df_val["center"].astype(str).str.strip().str.lower()
    centres = sorted((set(train) | set(val)) - {"", "nan"})
    table = pd.DataFrame({"center": centres,
                          "n_training": [int((train == c).sum()) for c in centres],
                          "n_validation": [int((val == c).sum()) for c in centres]})
    table["in_training"], table["in_validation"] = table["n_training"] > 0, table["n_validation"] > 0
    logger.info(f"Centres: {int(table['in_training'].sum())} development, "
                f"{int(table['in_validation'].sum())} temporal, "
                f"{int((table['in_validation'] & ~table['in_training']).sum())} temporal only")
    return table


def temporal_overlap_counts(df_train: pd.DataFrame, df_val: pd.DataFrame) -> dict:
    """Temporal patients by CGI start date relative to the development cohort's treatment-start
    window (before / within / after) and by centre (label also in the development cohort vs
    new); counts only, no centre names (Methods, Study cohort)."""
    dev_start = parse_dates(df_train["cgi start date"])
    val_start = parse_dates(df_val["cgi start date"])
    lo, hi = dev_start.min(), dev_start.max()
    before, after = val_start < lo, val_start > hi
    within = val_start.notna() & ~before & ~after
    dev_centres = set(df_train["center"].astype(str).str.strip().str.lower()) - {"", "nan"}
    val_centres = df_val["center"].astype(str).str.strip().str.lower()
    from_dev = val_centres.isin(dev_centres)
    out = {
        "n_temporal_final": int(len(df_val)),
        "development_cgi_start_min": str(lo.date()),
        "development_cgi_start_max": str(hi.date()),
        "temporal_cgi_start_min": str(val_start.min().date()),
        "temporal_cgi_start_max": str(val_start.max().date()),
        "temporal_start_before_development_window": int(before.sum()),
        "temporal_start_within_development_window": int(within.sum()),
        "temporal_start_after_development_window": int(after.sum()),
        "temporal_start_unparseable": int(val_start.isna().sum()),
        "temporal_patients_development_centres": int(from_dev.sum()),
        "temporal_patients_new_centres": int((~from_dev).sum()),
        "temporal_centre_labels_new": int(len(set(val_centres[~from_dev]) - {"", "nan"})),
    }
    logger.info(f"Temporal overlap counts: {out}")
    return out


def model_features(model_name: str, model_version: str) -> list:
    """Union of the base learners' features (selected_features.json of the registered model)."""
    path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{model_name}/{model_version}/artifacts/selected_features.json",
        dst_path=tempfile.mkdtemp())
    with open(path) as f:
        return sorted({feature for features in json.load(f).values() for feature in features})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw_data", required=True, help="CSV with both cohorts (column 'coorte')")
    parser.add_argument("--model_name", required=True, help="Registered ensemble")
    parser.add_argument("--model_version", default="latest")
    parser.add_argument("--output_data", required=True, help="Output folder (prepared_external.parquet)")
    parser.add_argument("--output_report", required=True, help="Output JSON summary")
    args = parser.parse_args()

    df = pd.read_csv(args.raw_data).rename(columns={"os": "tte", "death": "event"})
    df["event"] = pd.to_numeric(df["event"], errors="coerce")
    df["tte"] = ((parse_dates(df["date of death or last fup"]) - parse_dates(df["cgi start date"])).dt.days
                 / DAYS_PER_MONTH)
    df_train = df[df["coorte"] == "training"].copy()
    df = df[df["coorte"] == "validation"].copy()
    logger.info(f"Input: {len(df)} temporal-cohort and {len(df_train)} development-cohort rows")

    artifacts = Path(tempfile.mkdtemp(prefix="prepare_external_"))
    n_before = len(df)
    df, removed = remove_patients_in_training(df, df_train)
    n_after_dedup = len(df)
    removed.to_csv(artifacts / "removed_patients_shared_with_training.csv", index=False)
    log_cohort_dates(df_train, "train")
    log_cohort_dates(df, "val")
    centre_comparison(df_train, df).to_csv(artifacts / "center_comparison.csv", index=False)

    invalid = df["tte"].isna() | df["event"].isna()
    invalid |= df["tte"] <= 0
    logger.info(f"Missing or non-positive follow-up: {int(invalid.sum())} patients removed")
    df = df[~invalid].reset_index(drop=True)
    overlap = temporal_overlap_counts(df_train, df)

    # Analysis dataset of both cohorts, before any recoding (used for Table 1)
    pd.concat([df_train, df], ignore_index=True).to_csv(artifacts / "btc_dataset_selected_patients.csv",
                                                       index=False)
    mlflow.log_artifacts(str(artifacts))

    # Same recoding as data_prep.py
    if "ps ecog" in df.columns and "ecog bin" in df.columns:
        df = df.drop(columns=["ps ecog"]).rename(columns={"ecog bin": "ps ecog"})
    if "stent or biliary draige" in df.columns:
        df["stent or biliary draige"] = df["stent or biliary draige"].apply(
            lambda x: 0 if x == 0 else (1 if pd.notna(x) else x))

    features = model_features(args.model_name, args.model_version)
    required = features + [f for f in SENECA_INPUTS if f not in features]
    absent = [f for f in required if f not in df.columns]
    if absent:
        logger.warning(f"Not in the temporal data: {absent}")
    prepared = df[[f for f in required if f in df.columns] + ["tte", "event", "center"]].copy()
    feature_cols = [c for c in prepared.columns if c not in ("tte", "event", "center")]
    for col in feature_cols:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    out = Path(args.output_data)
    out.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(out / "prepared_external.parquet", index=False)
    logger.info(f"Prepared temporal cohort: {len(prepared)} patients, {int(prepared['event'].sum())} deaths")

    report = {
        "n_samples": len(prepared),
        "n_features": len(feature_cols),
        "n_events": int(prepared["event"].sum()),
        "event_rate": float(prepared["event"].mean()),
        "median_survival": float(prepared["tte"].median()),
        "missing_values_total": int(prepared.isnull().sum().sum()),
        "dedup_validation_before": n_before,
        "dedup_validation_after": n_after_dedup,
        "dedup_patients_removed": n_before - n_after_dedup,
        "features": feature_cols,
        "temporal_overlap": overlap,
    }
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
