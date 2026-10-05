"""Candidate predictors and training/test split of the development cohort.

1. Keep the variables recorded at baseline according to the study data
   dictionary, except genetic tests, comorbidities, concomitant drugs and the
   centre identifier.
2. Recode location (intrahepatic vs other), timing of surgery relative to CGI
   start (metastatic ab initio vs other), ECOG performance status (0 vs >=1)
   and biliary stent/drainage (any vs none). Drop the single-site metastasis
   indicators and hbv/hcv.
3. Drop variables with more than 30% missing values, then keep one variable of
   each group with Spearman |rho| > 0.8, the one with the largest variance.
4. Split 70/30 into a training and a test set, stratified on the event.

The two filters of step 3 are computed on the whole development cohort before
the split; neither uses the outcome. The patient number is numeric and passes
through both filters too; it has no missing values and is never removed.

The same script runs again in the validation pipeline, where it rebuilds the
same split (fixed seed).
"""

import argparse
import os

import pandas as pd
from feature_engine.selection import SmartCorrelatedSelection
from sklearn.model_selection import StratifiedShuffleSplit

EXCLUDED_CATEGORIES = ["genetic_tests", "comorbidity_type", "drug_type", "comorbidity_info"]
SINGLE_SITE_METASTASIS = ["lung mets only", "lymph mets only", "peri mets only",
                          "liver mets only", "bone mets only"]
MAX_MISSING_FRACTION = 0.3
MAX_CORRELATION = 0.8


def main(args):
    dictionary = pd.read_csv(args.dict_data)
    raw = pd.read_csv(args.btc_os_data)

    baseline = dictionary[(dictionary["RecordedAtBaseline"] == "Yes")
                          & (dictionary["FieldShortName"] != "center")
                          & ~dictionary["Category"].isin(EXCLUDED_CATEGORIES)]
    columns = [c for c in baseline["FieldShortName"] if c in raw.columns]
    df = raw[columns].copy()
    df = df.drop(columns=[c for c in SINGLE_SITE_METASTASIS if c in df.columns])

    # Location: 0 gallbladder, 1 intrahepatic, 2 distal, 3 proximal extrahepatic
    df["location - Intrahepatic"] = df["location"].apply(lambda x: 1 if x == 1 else 0)
    # Surgery and start of CGI: 0 <6 months, 1 >6 months, 2 metastatic ab initio
    df["surgery and start of cgi - Metastatic ab initio"] = (
        df["surgery and start of cgi"].apply(lambda x: 1 if x == 2 else 0))
    df = df.drop(columns=["location", "surgery and start of cgi", "hbv/hcv"])

    # ECOG enters the models as 0 vs >=1 (the dictionary's 'ecog bin')
    if "ps ecog" in df.columns and "ecog bin" in df.columns:
        df = df.drop(columns=["ps ecog"]).rename(columns={"ecog bin": "ps ecog"})

    # Stent or biliary drainage: 0 none; 1 stent, 2 external drainage, 3 both
    if "stent or biliary draige" in df.columns:
        df["stent or biliary draige"] = df["stent or biliary draige"].apply(
            lambda x: 0 if x == 0 else (1 if pd.notna(x) else x))

    missing = df.isnull().mean()
    too_sparse = missing[missing > MAX_MISSING_FRACTION].index.tolist()
    df = df.drop(columns=too_sparse)
    print(f"Dropped for >30% missing values: {too_sparse}")

    selector = SmartCorrelatedSelection(method="spearman", threshold=MAX_CORRELATION,
                                        selection_method="variance")
    df = selector.fit_transform(df)
    print(f"Dropped for correlation: {selector.features_to_drop_}")

    df = df.merge(raw[["patient number", "os", "death"]], on="patient number", how="left")
    df = df.rename(columns={"os": "tte", "death": "event", "patient number": "id"})

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df["event"]))
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    for name, part in [("training", train), ("test", test)]:
        print(f"{name}: n={len(part)}, events={int(part['event'].sum())}")

    os.makedirs(args.training_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    train.to_csv(os.path.join(args.training_data, "file-os-ml-ready-train.csv"), index=False)
    test.to_csv(os.path.join(args.test_data, "file-os-ml-ready-test.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dict_data", required=True,
                        help="Data dictionary (FieldShortName, Category, RecordedAtBaseline)")
    parser.add_argument("--btc_os_data", required=True, help="Cleaned development cohort (CSV)")
    parser.add_argument("--training_data", required=True, help="Output folder, training set")
    parser.add_argument("--test_data", required=True, help="Output folder, test set")
    main(parser.parse_args())
