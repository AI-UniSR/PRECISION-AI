# PRECISION-AI

Analysis code for the paper *Development and Temporal Validation of an Artificial Intelligence–Based Ensemble Model for Overall Survival Stratification in Advanced Biliary Tract Cancer Treated with First-Line Cisplatin, Gemcitabine, and Durvalumab*.

[![DOI](https://zenodo.org/badge/1279256693.svg)](https://doi.org/10.5281/zenodo.20832284)

The analyses ran as two Azure Machine Learning pipelines: one for model development and internal testing, one for temporal validation. This repository holds the scripts of both pipelines, so that each step of the Methods can be checked against the code. Patient-level data cannot be shared, so the scripts cannot be run as they are; the order of the steps and their settings are described below and in each script's header. Exploratory steps that produced no reported result, the Azure job definitions and the checks that compared each run with our reference outputs are not included.

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
2. `fit_cox_benchmark.py`: an unpenalised Cox model on the ensemble's ten predictors, fitted on the training set with the same preprocessing as the ensemble's Cox learner. It differs from the ensemble only in the modelling method.
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
| Cox benchmark, cutoff sensitivity | `fit_cox_benchmark.py`, `compute_metrics.py`, `threshold_sensitivity.py` → `publication_tables.R` | `ensemble_vs_cox`, `cox_benchmark_coefficients`, `cutoff_sensitivity_temporal_full` |

Table 1, Tables S1, S3, S5 and S6, Figs. S1, S4 and S5 and Figures 1 and 5 were prepared outside these pipelines; the published figures were assembled from the panels above. Table S9 was de-identified by hand: pipeline outputs contain patient-level rows and real centre names. Tables 2–4 and S4 as printed show the ensemble (and SENECA) rows only: the Cox benchmark rows that `publication_tables.R` adds to them were removed by hand, and the benchmark is reported in Table S11 (temporal validation cohort). The summary CSVs are written with four decimals, as in the run behind the paper; the published three-decimal values are rounded from them.

## Running

Each step is a command-line program (`--help` lists the arguments); the numeric defaults are the settings used for the paper. To reproduce the paper's runs, also pass `--selected_features` to the DeepSurv script, `--deepsurv_model` to `evaluate_models.py` and `--shap_figures_dir` to `publication_figures.R`. `data_prep.py` needs the study data dictionary (columns `FieldShortName`, `Category`, `RecordedAtBaseline`), which is not included.

The ensemble is passed between steps through an MLflow model registry as `models:/<name>/<version>`. Outside Azure ML, set `MLFLOW_TRACKING_URI` to a local store (e.g. `file:./mlruns`), where `evaluate_models.py` registers it as `ensemble_survival_model`.

When the published model was trained, the Optuna searches of `cox_net` and `deepsurv` and DeepSurv's network weights were not seeded, and the XGBSE booster used XGBoost's default seed (0). Seed 42 was set in these scripts afterwards, so re-running them will not give exactly the published elastic-net Cox, XGBSE and DeepSurv models.

## Software

Python 3.9. The registered model was logged with scikit-learn 1.5.1, scikit-survival 0.23.0, xgbse 0.3.3, mlflow 2.22.2, numpy 1.26.4, pandas 2.3.3, scipy 1.13.1 and cloudpickle 2.2.1; other packages used: lifelines, xgboost, optuna, feature-engine, shap, plotly, matplotlib. DeepSurv also needs torch, pycox and torchtuples. The R scripts use R 4.4.3; their environments are defined in `7_publication_figures/R_env/figures` and `R_env/tables`.

## Archive and citation

The repository is archived on Zenodo. The DOI [10.5281/zenodo.20832284](https://doi.org/10.5281/zenodo.20832284) refers to all versions and resolves to the latest one. Release v1.0.0 (DOI 10.5281/zenodo.20832285) predates the revision of the code for the manuscript and does not reproduce its numbers; use release v1.1.0 or later.
