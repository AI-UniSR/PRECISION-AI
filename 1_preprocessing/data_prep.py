import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def main(args):
    print("Starting data preparation...")

    # Load dictionary and OS data
    dict_data = pd.read_csv(args.dict_data)
    btc_data = pd.read_csv(args.btc_os_data)
    raw_df = btc_data.copy()

    # --- baseline feature filtering ---
    baseline_features = dict_data[dict_data['RecordedAtBaseline'] == 'Yes'].copy()
    identifiers_to_remove = ['center']
    baseline_features = baseline_features[
        ~baseline_features['FieldShortName'].isin(identifiers_to_remove)
    ]

    # Exclude some categories
    exclude_cats = ['genetic_tests', 'comorbidity_type', 'drug_type', 'comorbidity_info']
    baseline_features = baseline_features[
        ~baseline_features['Category'].isin(exclude_cats)
    ]

    # Select columns
    baseline_columns = baseline_features['FieldShortName'].tolist()
    baseline_columns.append('Patient number_ordinal')
    available_columns = [col for col in baseline_columns if col in btc_data.columns]
    df = btc_data[available_columns].copy()
    df.rename(columns={'Patient number_ordinal': 'id'}, inplace=True)

    # Remove metastasis-only features
    metastis_cols = ['lung mets only', 'lymph mets only',
                     'peri mets only', 'liver mets only', 'bone mets only']
    df.drop(columns=[c for c in metastis_cols if c in df.columns], inplace=True, errors='ignore')

    # Transform location into binary: 1=intrahepatic, 0=other
    # ['Location: gallbladder 0, intrahepatic 1, distal extrahepatic 2, proximal extrahepatic 3,']
    df['location - Intrahepatic'] = df['location'].apply(lambda x: 1 if x == 1 else 0)
    df.drop(columns=['location'], inplace=True)

    # Transform 'surgery and start of cgi' into binary: 1=metastatic ab initio, 0=other
    # ['Surgery and start of Cis-Gem-Immuno: <6 months = 0, >6 months = 1, metastatic ab initio = 2,']
    df['surgery and start of cgi - Metastatic ab initio'] = df['surgery and start of cgi'].apply(lambda x: 1 if x == 2 else 0)
    df.drop(columns=['surgery and start of cgi'], inplace=True)


    # drop "hbv/hcv"
    df.drop(columns=['hbv/hcv'], inplace=True)

    # Use binary ECOG (ecog bin: 0 if ECOG=0, 1 if ECOG>0) instead of
    # the multi-class ps ecog (0,1,2,3).  Drop the original and rename
    # 'ecog bin' → 'ps ecog' so the schema stays backward-compatible.
    if 'ps ecog' in df.columns and 'ecog bin' in df.columns:
        df.drop(columns=['ps ecog'], inplace=True)
        df.rename(columns={'ecog bin': 'ps ecog'}, inplace=True)
        print("Replaced multi-class 'ps ecog' with binary 'ecog bin' (renamed to 'ps ecog')")

    # Binarize 'stent or biliary draige': dictionary rule-0:0 | rule-1:1,2,3
    # 0=No drainage → 0;  1=Stent, 2=External drainage, 3=Both → 1
    if 'stent or biliary draige' in df.columns:
        df['stent or biliary draige'] = df['stent or biliary draige'].apply(
            lambda x: 0 if x == 0 else (1 if pd.notna(x) else x)
        )
        print("Binarized 'stent or biliary draige': 0=No drainage, 1=Any stent/drainage")

    # Drop features with >30% missing values
    missing_ratio = df.isnull().mean()
    
    features_to_drop = missing_ratio[missing_ratio > 0.3].index.tolist()

    if features_to_drop:
        print(f"Dropping {len(features_to_drop)} features with >30% missing values: {features_to_drop}")
        df.drop(columns=features_to_drop, inplace=True)
    else:
        print("No features to drop based on missing value threshold")


    # drop highly correlated features (>0.8) with SmartCorrelatedSelection
    from feature_engine.selection import SmartCorrelatedSelection
    selector = SmartCorrelatedSelection(method='spearman', threshold=0.8, selection_method='variance')
    df = selector.fit_transform(df)
    print(f"Dropped correlated features: {selector.features_to_drop_}")

    # Merge with survival data
    df_final = df.merge(raw_df[['patient number', 'os', 'death']], on='patient number', how='left')
    df_final.rename(columns={'os': 'tte', 'death': 'event', 'patient number': 'id'}, inplace=True)

    # --- Stratified Train/Test Split ---
    print("\n=== Performing stratified train/test split (70/30) ===")
    
    # Create stratified split based on event status
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(splitter.split(df_final, df_final['event']))
    
    train_df = df_final.iloc[train_idx].copy()
    test_df = df_final.iloc[test_idx].copy()
    
    print(f"Training set: {train_df.shape[0]} samples")
    print(f"  Events: {train_df['event'].sum()} ({100*train_df['event'].mean():.1f}%)")
    print(f"Test set: {test_df.shape[0]} samples")
    print(f"  Events: {test_df['event'].sum()} ({100*test_df['event'].mean():.1f}%)")

    # Save training data
    os.makedirs(args.training_data, exist_ok=True)
    train_output_csv = os.path.join(args.training_data, "file-os-ml-ready-train.csv")
    train_df.to_csv(train_output_csv, index=False)
    print(f"\nSaved training dataset to: {train_output_csv}")
    
    # Save test data
    os.makedirs(args.test_data, exist_ok=True)
    test_output_csv = os.path.join(args.test_data, "file-os-ml-ready-test.csv")
    test_df.to_csv(test_output_csv, index=False)
    print(f"Saved test dataset to: {test_output_csv}")
    
    print(f"\n✅ Data preparation completed successfully!")
    print(f"Total rows: {df_final.shape[0]}, Columns: {df_final.shape[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_data", type=str, required=True)
    parser.add_argument("--btc_os_data", type=str, required=True)
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()
    main(args)
