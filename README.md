# PRECISION-AI

Analysis code for the paper *Development and Temporal Validation of an Artificial Intelligence–Based Ensemble Model for Overall Survival Stratification in Advanced Biliary Tract Cancer Treated with First-Line Cisplatin, Gemcitabine, and Durvalumab*.

**DOI:** [10.5281/zenodo.23336728](https://doi.org/10.5281/zenodo.23336728)

The analyses ran as two Azure Machine Learning pipelines: one for model development and internal testing, one for temporal validation. This repository holds the scripts of both pipelines, so that each step of the Methods can be checked against the code. Patient-level data cannot be shared, so the scripts cannot be run as they are; the format of the input is described in [Input data format](#input-data-format), the order of the steps and their settings below and in each script's header. Exploratory steps that produced no reported result, the Azure job definitions and the checks that compared each run with our reference outputs are not included.

## Input data format

The pipelines read a data dictionary and two CSV files of patients. This section describes them, so that the code can be run on other data.

### Files

| File | Read by | Content |
|---|---|---|
| Data dictionary (CSV) | `1_preprocessing/data_prep.py --dict_data` | one row per study variable |
| Development cohort (CSV) | `1_preprocessing/data_prep.py --btc_os_data` | one row per development patient |
| Both cohorts (CSV) | `5_validation/prepare_external.py --raw_data` | development patients (`coorte` = `training`) and temporal patients (`coorte` = `validation`) |

The later steps read the files written by these two scripts (`data_prep.py`: the training and test CSV, with the columns `id`, the predictors, `tte` and `event`; `prepare_external.py`: `prepared_external.parquet`) and need no further input.

### General rules

- One row per patient, one column per variable, comma-separated, with a header. Column names are case-sensitive and are the ones listed below.
- Missing values are empty cells. Do not impute: the models impute inside each fold (Methods).
- Every predictor is numeric: binary variables are 0/1 and categories integer codes. Text columns (dates, `center`) are not predictors.
- Follow-up is in months; the outcome is overall survival from the start of first-line treatment (CGI).
- Units matter for the SENECA score: neutrophils in 10^9/L, CA19-9 in U/mL (cut-off 1051) and CEA in ng/mL (cut-off 3.80). The study used albumin in g/dL, BMI in kg/m², AST and ALT in U/L.

### Data dictionary

One row per variable, with the columns `FieldShortName` (name of the column in the patient files), `Category` (free text) and `RecordedAtBaseline` (`Yes` or `No`).

| FieldShortName | Category | RecordedAtBaseline |
|---|---|---|
| patient number | identifier | Yes |
| albumin | laboratory | Yes |
| fgfr2 fusion | genetic_tests | Yes |
| date of death or last fup | dates | No |

The candidate predictors are the variables with `RecordedAtBaseline` = `Yes`, except `center` and the categories `genetic_tests`, `comorbidity_type`, `drug_type` and `comorbidity_info`. Names in the dictionary that are not columns of the patient file are ignored. `patient number` must be listed as a baseline variable.

### Development cohort file

| Column | Content | Notes |
|---|---|---|
| `patient number` | numeric, unique | required; becomes `id` |
| `os` | follow-up in months, > 0 | required; becomes `tte` |
| `death` | 1 death, 0 censored | required; becomes `event` |
| `location` | 0 gallbladder, 1 intrahepatic, 2 distal, 3 proximal extrahepatic | required; recoded to `location - Intrahepatic` |
| `surgery and start of cgi` | 0 <6 months, 1 >6 months, 2 metastatic ab initio | required; recoded to `surgery and start of cgi - Metastatic ab initio` |
| `hbv/hcv` | any | required; dropped |
| `ps ecog`, `ecog bin` | ECOG 0-3; 0 vs ≥1 | if both are present, `ecog bin` replaces `ps ecog` |
| `stent or biliary draige` | 0 none, 1-3 stent, drainage or both | optional; recoded to 0/1 |
| `lung mets only`, `lymph mets only`, `peri mets only`, `liver mets only`, `bone mets only` | 0/1 | optional; dropped |
| every other baseline variable of the dictionary | numeric | e.g. `neutrophils`, `nlr`, `albumin`, `cea`, `ca199`, `bmi`, `astGOT`, `altGPT`, `number of met sites`, `locally advanced/metastatic` |

`data_prep.py` then drops the variables with more than 30% missing values and, from each group with Spearman |ρ| > 0.8, all but the variable with the largest variance, and splits 70/30 on the event.

### File with both cohorts

The columns of the development file, for both cohorts, with `coorte` (`training` or `validation`) and, for every patient:

| Column | Content | Notes |
|---|---|---|
| `coorte` | `training` or `validation` | required |
| `dob` | date of birth | `YYYY-MM-DD` or `DD/MM/YYYY` |
| `sex` | numeric code | |
| `center` | text | upper/lower case and spaces are ignored |
| `cgi start date` | start of first-line treatment | `YYYY-MM-DD` or `DD/MM/YYYY` |
| `date of death or last fup` | date | `YYYY-MM-DD` or `DD/MM/YYYY` |
| `death` | 1 death, 0 censored | |

The follow-up of the temporal patients is recomputed from the two dates (months of 365.25/12 days); `os` is not used for them. A temporal patient is removed if date of birth, sex, centre and start date equal those of a development patient, or if the follow-up is missing or not positive. The columns of the ensemble's predictors and the SENECA inputs (`neutrophils`, `ps ecog`, `locally advanced/metastatic`, `ca199`, `cea`; stage 1 locally advanced, 2 metastatic) must have the same names and codes as in the development data.

### Requirements of the code

These are not visible from the file format:
- The number of predictors of the ensemble, the union of the features of the four base learners, is the result of the selection, not an input: 10 in the paper, and any number from 1 upwards is accepted (tested with 10 and 24). `prepare_external.py` and the Cox benchmark use whatever list the registered ensemble holds.
- The temporal cohort is recoded only for ECOG and stent/drainage, so the predictors of the ensemble must be columns that `data_prep.py` passes through unchanged; the derived `location - Intrahepatic` and `surgery and start of cgi - Metastatic ab initio` cannot be among them.
- The five SENECA inputs must also be candidate predictors of the development data: `threshold_sensitivity.py` reads them from the training set.
- The R scripts evaluate the calibration at 6 and 18 months within deciles of predicted risk, and stop ("no points selected for one or more curves") if a decile has no patient followed beyond 18 months; the temporal cohort needs enough follow-up and patients.
- `threshold_sensitivity.py` compares its rows at k = 15 and k = 33 with the main analysis and stops if they differ; this also happens when a risk group has no deaths, because the hazard ratio is then infinite (small cohorts).
- MLflow creates its store (`MLFLOW_TRACKING_URI=file:<folder>`) itself; a folder created beforehand makes it stop with "Could not find experiment with ID 0".

## Contents

```
1_preprocessing/        candidate predictors, missingness and correlation filters, 70/30 split
2_feature_selection/    RSF stability selection
3_model_training/       base learners (elastic-net Cox, RSF, XGBoost-AFT, XGBSE) and the DeepSurv benchmark
4_ensemble/             stacking ensemble, MLflow model, hold-out comparison of all models
5_validation/           temporal validation: cohort, predictions, metrics, Cox benchmark,
                        cutoff sensitivity, SHAP; lib/ holds the shared metric code
6_inference/            scoring script of the web application (reference only)
7_publication_figures/  R scripts for figures and Word tables; R_env/ defines their environments
```

## Development pipeline

1. `1_preprocessing/data_prep.py` keeps the baseline variables of the study data dictionary, recodes location, timing of surgery, ECOG and biliary drainage, drops variables with more than 30% missing values and, from each group with Spearman |ρ| > 0.8, keeps the variable with the largest variance. Both filters are computed on the whole development cohort, without the outcome, before the 70/30 split stratified on the event (seed 42).
2. `2_feature_selection/feature_selection.py` runs RSF stability selection (`rsf_bootstrap_selector.py`): 100 random 60% subsamples drawn without replacement, permutation importance with a C-index scorer, threshold 0.6 on the selection frequency. The forests run on non-imputed data (scikit-survival ≥ 0.23).
3. `3_model_training/*/model_train.py` tune each learner with Optuna (TPE, seed 42, 50 trials) on 5×10 repeated stratified cross-validation, maximising the mean Harrell C-index (DeepSurv: pycox time-dependent concordance). Imputation and scaling are pipeline steps refitted in every fold, except for DeepSurv, whose imputer and scaler are fitted once on the training set.
   - `cox_net`: elastic-net Cox on all candidate predictors; alpha is the largest value whose mean CV C-index is within 0.25 × the standard deviation of the fold C-indices of the best alpha. The features with |coefficient| ≥ 1e-5 are refitted without penalty.
   - `rsf`, `xgboost_surv`, `xgbse_weibull`: on the stability-selected features. XGBSE is not imputed (native handling of missing values). XGBoost-AFT gets a survival function from Kaplan–Meier curves within quintiles of the training risk score (`xgboost_survival_model.py`).
   - `deepsurv`: benchmark only, not in the ensemble; trained on the stability-selected features (`--selected_features`).
4. `4_ensemble/evaluate_models.py` builds out-of-fold predictions of the four learners (5-fold, stratified, seed 42), fits an unpenalised Cox meta-learner on them (`ensemble_model.py`), registers the ensemble as an MLflow model (`ensemble_pyfunc.py`), checks that the registered model reproduces the in-memory predictions, and evaluates every model on the 30% hold-out set (Supplementary Table S2).

The helper modules `xgboost_survival_model.py`, `xgbse_pipeline_wrapper.py` and `deepsurv_wrapper.py` appear in both `3_model_training/` and `4_ensemble/` because each pipeline step ran from its own folder; the copies are identical.

## Temporal validation pipeline

`data_prep.py` is run again to rebuild the same training and test sets.

1. `prepare_external.py`: from the 779 temporal-cohort patients, removes the 80 also present in the development cohort (same date of birth, sex, centre and treatment start date) and 1 with zero follow-up, leaving 698, and keeps the ensemble's predictors and the SENECA inputs. Its report also counts the temporal patients whose treatment started before, within or after the development cohort's treatment-start window and those from centres absent from the development cohort (Methods, Study cohort).
2. `fit_cox_benchmark.py`: an unpenalised Cox model on the ensemble's predictors (ten in the paper), fitted on the training set with the same preprocessing as the ensemble's Cox learner. It differs from the ensemble only in the modelling method.
3. `generate_predictions.py`: risk scores and survival probabilities of the ensemble and of the Cox benchmark for the internal test set and the temporal cohort, the SENECA score, and the training risk scores used for the risk-group cutoffs.
4. `compute_metrics.py`: Harrell's C, Uno's C (τ = 24 months), time-dependent AUC and Brier score at 6, 12, 18 and 24 months, IBS, with the censoring distribution of the IPCW weights estimated by Kaplan–Meier in the cohort or bootstrap resample being evaluated (definitions in `lib/survival_metrics.py`); 95% CIs from 1000 event-stratified bootstrap resamples, shared by all models compared on a cohort so that differences are paired (`lib/bootstrap.py`); the SENECA comparison on the 579 patients with all five SENECA inputs; risk groups at the 15th/85th (and 33rd/67th) percentiles of the training risk scores with Kaplan–Meier summaries, log-rank test, pairwise hazard ratios, PPV at 6 months and NPV at 18 months; completeness of the inputs and C-indices by completeness.
5. `threshold_sensitivity.py`: the risk-group analysis at other percentile cutoffs (k/100 − k, k = 5 to 33), on the SENECA complete cases (ensemble vs SENECA) and on the full cohort (ensemble). It reuses the resamples of `compute_metrics.py` (same `--n_bootstrap`) and stops unless k = 15 and k = 33 reproduce the main analysis.
6. `shap_analysis.py`: Kernel SHAP of the ensemble risk score (background: 20 k-means centroids of the training set; 200 temporal-cohort patients, seed 42) and permutation importance (50 permutations per feature).
7. `7_publication_figures/publication_figures.R` and `publication_tables.R` draw the figures and write the Word tables. The calibration summary of the Results (6 and 18 months) is computed in `publication_figures.R` on all patients of the temporal cohort: the O:E ratio is the Kaplan–Meier observed risk at the horizon over the mean predicted risk (Greenwood CI), and the calibration slope is the coefficient of the predicted log-odds in a logistic regression of the status at the horizon weighted by inverse probability of censoring weights (percentile CI from 1000 event-stratified bootstrap resamples).

## Paper items

| Paper | Produced by | Output |
|---|---|---|
| Table 2 | `compute_metrics.py` → `publication_tables.R` | `overall_internal.csv`, `overall_temporal_full.csv` |
| Table 3 | `compute_metrics.py` → `publication_tables.R` | `timedep_temporal_full.csv` |
| Table 4 | `compute_metrics.py` → `publication_tables.R` | `overall_temporal_cc.csv` |
| Figure 2 | `publication_figures.R` | `dca_main_highlighted` (A–B, ensemble only; `dca_supplementary_highlighted` adds the Cox benchmark), `calibration_combined_temporal` (C–D) |
| Calibration (Results) | `publication_figures.R` | `calibration_metrics_{6,18}m_temporal.csv` (`observed_km`, `mean_predicted`, `oe_km`, `slope` and their 95% CIs) |
| Figure 3, Table S4 | `compute_metrics.py` | `km_15-85_temporal`, `km_statistics_temporal_15-85.csv`, `hazard_ratios_temporal_15-85.csv`, `logrank_temporal_15-85.csv`, `ppv_npv_temporal_15-85.csv` |
| Figure 4 | `shap_analysis.py` | `shap_beeswarm`, `shap_mean_abs.csv` |
| Table S2 | `4_ensemble/evaluate_models.py` | `comparison/summary_metrics.csv` |
| Tables S7, S8, S9 | `compute_metrics.py` | `temporal_patient_completeness.csv`, `temporal_ml_completeness_c_indices.csv`, `temporal_centre_feature_missingness.csv` |
| Fig. S2 | `2_feature_selection/feature_selection.py` | `stability_scores.csv` |
| Fig. S3 | `publication_figures.R` | `ipcw_risk_stratified_*`, `predicted_risk_*` |
| Tables S10, S11 | `fit_cox_benchmark.py`, `compute_metrics.py`, `threshold_sensitivity.py` → `publication_tables.R` | `ensemble_vs_cox` (S10, Cox benchmark), `cutoff_sensitivity_temporal_full` (S11) |

Table 1, Tables S1, S3, S5 and S6, Figs. S1, S4 and S5 and Figures 1 and 5 were prepared outside these pipelines; the published figures were assembled from the panels above. Table S9 was de-identified by hand: pipeline outputs contain patient-level rows and real centre names. Tables 2–4 and S4 as printed show the ensemble (and SENECA) rows only: the Cox benchmark rows that `publication_tables.R` adds to them were removed by hand, and the benchmark is reported in Table S10 (temporal validation cohort). The table of the benchmark's hazard ratios (`cox_benchmark_coefficients`) is written by `publication_tables.R` but is not part of the manuscript. The summary CSVs are written with four decimals, as in the run behind the paper; the published three-decimal values are rounded from them.

## Running

Each step is a command-line program (`--help` lists the arguments); the numeric defaults are the settings used for the paper. To reproduce the paper's runs, also pass `--selected_features` to the DeepSurv script, `--deepsurv_model` to `evaluate_models.py` and `--shap_figures_dir` to `publication_figures.R`. `data_prep.py` needs the study data dictionary (columns `FieldShortName`, `Category`, `RecordedAtBaseline`), which is not included.

The ensemble is passed between steps through an MLflow model registry as `models:/<name>/<version>`. Outside Azure ML, set `MLFLOW_TRACKING_URI` to a local store (e.g. `file:./mlruns`), where `evaluate_models.py` registers it as `ensemble_survival_model`.

When the published model was trained, the Optuna searches of `cox_net` and `deepsurv` and DeepSurv's network weights were not seeded, and the XGBSE booster used XGBoost's default seed (0). Seed 42 was set in these scripts afterwards, so re-running them will not give exactly the published elastic-net Cox, XGBSE and DeepSurv models.

## Software

Python 3.9. The registered model was logged with scikit-learn 1.5.1, scikit-survival 0.23.0, xgbse 0.3.3, mlflow 2.22.2, numpy 1.26.4, pandas 2.3.3, scipy 1.13.1 and cloudpickle 2.2.1; other packages used: lifelines, xgboost, optuna, feature-engine, shap, plotly, matplotlib. DeepSurv also needs torch, pycox and torchtuples. The R scripts use R 4.4.3; their environments are defined in `7_publication_figures/R_env/figures` and `R_env/tables`.

## Archive and citation

The repository is archived on Zenodo. The DOI [10.5281/zenodo.20832284](https://doi.org/10.5281/zenodo.20832284) refers to all versions and resolves to the latest one. The code of the submitted manuscript is release v1.1.0, DOI [10.5281/zenodo.23336728](https://doi.org/10.5281/zenodo.23336728). Release v1.0.0 (DOI 10.5281/zenodo.20832285) predates the revision of the code for the manuscript and does not reproduce its numbers.
