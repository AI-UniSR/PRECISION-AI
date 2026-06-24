# PRECISION-AI — Reproducibility Code

Code companion for the manuscript:
**"Machine-learning ensemble for overall survival prediction in biliary tract cancer patients receiving cisplatin–gemcitabine–durvalumab (PRECISION-AI)"**

Reviewers should read this file alongside the *Methods & Results Codebook* to cross-check every analytical step against the corresponding script and parameter.

---

## Repository Structure

```
paper_code/
│
├── README.md                     ← this file
├── Dictionary_v08.csv            ← variable dictionary (features, encodings,
│                                    missingness rates, binarization rules)
│
├── 1_preprocessing/              ← §1 Data pipeline
│   └── data_prep.py              ← Missing-value filter (>30%), Spearman
│                                    collinearity removal (|ρ|≥0.80), three
│                                    pre-specified re-encodings, stratified
│                                    70/30 train/test split (seed=42)
│
├── 2_feature_selection/          ← §2 Feature selection
│   ├── rsf_bootstrap_selector.py ← RSF Stability Selection class:
│   │                                100 iterations, 60% subsample, PFI
│   │                                (5 repeats), binarized at 70th percentile,
│   │                                selection threshold=0.60, top-3 fallback
│   └── feature_selection.py      ← Driver: fits selector on training set,
│                                    saves stability scores and selected features
│
├── 3_model_training/             ← §3 Base learners
│   ├── survival_utils.py         ← Shared helpers: CV evaluation, Optuna
│   │                                plotting, permutation importance, metrics
│   ├── xgboost_survival_model.py ← XGBoostSurvival wrapper (AFT objective;
│   │                                risk = −log(predicted survival time);
│   │                                non-parametric S(t) via 5-quantile KM bins)
│   ├── xgbse_pipeline_wrapper.py ← XGBSEPipeline wrapper (StackedWeibull,
│   │                                NaN-native — no imputation step)
│   ├── cox_net/
│   │   └── model_train.py        ← Elastic-net CoxPH: Optuna tunes n_alphas
│   │                                [5,50] and l1_ratio [0.1,1.0]; α chosen by
│   │                                0.25-SE parsimony rule; surviving features
│   │                                refit as unpenalized CoxPH
│   ├── rsf/
│   │   └── model_train.py        ← Random Survival Forest: Optuna TPE (seed=42,
│   │                                50 trials), 5×10 RepeatedStratifiedKFold,
│   │                                IterativeImputer + RobustScaler in pipeline
│   ├── xgboost_surv/
│   │   └── model_train.py        ← XGBoost-AFT: Optuna TPE (seed=42, 50 trials),
│   │                                IterativeImputer (no scaling), AFT distribution
│   │                                ∈ {normal, logistic, extreme}
│   ├── xgbse_weibull/
│   │   └── model_train.py        ← XGBSE StackedWeibull: Optuna TPE (seed=42,
│   │                                50 trials), NaN-native (no imputation)
│   └── deepsurv/
│       ├── model_train.py        ← DeepSurv CoxPH neural network (pycox/torchtuples):
│       │                            Optuna default sampler, 1–3 layers, early stopping;
│       │                            trained for comparison only — NOT included in ensemble
│       └── deepsurv_wrapper.py   ← scikit-learn compatible wrapper for DeepSurv,
│                                    exposes predict() and predict_survival_function()
│
├── 4_ensemble/                   ← §4 Stacking ensemble
│   ├── ensemble_model.py         ← OOF generation: StratifiedKFold k=5 (seed=42);
│   │                                meta-features sorted alphabetically:
│   │                                cox_pred, rsf_pred, xgboost_pred, xgbse_pred;
│   │                                meta-learner: unpenalized CoxPH; round-trip
│   │                                identity check after MLflow registration
│   ├── ensemble_pyfunc.py        ← MLflow PyFunc wrapper (load_context → predict
│   │                                → DataFrame with risk_score column)
│   ├── evaluate_models.py        ← Per-model and ensemble evaluation on held-out
│   │                                test set (C-index, AUC, Brier, IBS)
│   ├── evaluation_utils.py       ← Temporal grid construction, metric aggregation,
│   │                                comparative AUC/Brier plots
│   └── xgboost_survival_model.py ← XGBoostSurvival copy required for joblib
│                                    unpickling of the trained XGBoost pipeline
│
├── 5_validation/                 ← §5–8 Full evaluation pipeline
│   ├── prepare_external.py       ← External cohort: 783 screened → 698 analysed
│   │                                (5 date-inconsistencies + 80 overlapping with
│   │                                development cohort removed; 1 with tte=0)
│   ├── generate_predictions.py   ← Upfront predictions for both cohorts
│   │                                (ML ensemble + SENECA), decoupled from
│   │                                evaluation; training risk scores cached for
│   │                                threshold derivation (no leakage)
│   ├── compute_metrics.py        ← All metrics: C-index (Harrell + Uno IPCW),
│   │                                time-dependent AUC at 6/12/18/24 months,
│   │                                IPCW Brier scores, IBS, calibration slope/
│   │                                intercept/O:E; bootstrap CIs (n=200,
│   │                                event-stratified); risk stratification P15/P85;
│   │                                Cohen's κ (ML vs SENECA risk groups)
│   └── shap_analysis.py          ← Kernel SHAP: k-means background (k=20),
│                                    200 validation patients (seed=42);
│                                    PFI: 50 permutations/feature (seed=42)
│
├── 5_validation/lib/             ← Shared validation library
│   ├── survival_metrics.py       ← Harrell C-index, Uno IPCW C (τ=24 months,
│   │                                training-set censoring distribution), time-
│   │                                dependent AUC, IPCW Brier, IBS, temporal
│   │                                grid (5th–95th percentile training events
│   │                                + 4 clinical timepoints)
│   ├── bootstrap.py              ← Event-stratified bootstrap (200 iterations
│   │                                on cached predictions); per-metric CI
│   │                                aggregation; paired ML vs SENECA comparison
│   │                                (same bootstrap iterations)
│   ├── risk_stratification.py    ← P15/P85 thresholds derived on training set,
│   │                                frozen; KM curves; omnibus log-rank; pairwise
│   │                                Cox HR; censoring-aware PPV and NPV
│   ├── roc_pr.py                 ← IPCW ROC and precision-recall at 6/12/18/24 m
│   └── seneca_model.py           ← Published SENECA formula: 5 fixed coefficients,
│                                    binary encoding rules, risk group thresholds
│
└── 6_inference/                  ← Scoring endpoint (informational)
    ├── score.py                  ← Online inference: risk score, survival curve,
    │                                risk group, median survival, risk percentile,
    │                                SHAP values per patient
    └── requirements.txt          ← Pinned inference-environment dependencies

7_publication_figures/            ← R scripts for all manuscript figures and tables
├── publication_figures.R         ← All publication-quality plots (PNG 300 DPI + PDF):
│                                    calibration curves (6 m + 18 m), DCA (6 m + 18 m),
│                                    Kaplan-Meier with numbers-at-risk, ROC and PR curves
│                                    at 4 timepoints, temporal AUC/Brier line plots,
│                                    risk-score density overlays
├── publication_tables.R          ← .docx tables via flextable: performance summary,
│                                    bootstrap CIs, risk stratification, SENECA comparison
└── R_env/
    ├── Dockerfile                ← R 4.4.3 image (rocker/r-ver) with all dependencies
    └── environment.yml           ← R package list (survminer, dcurves, ggplot2,
                                     patchwork, flextable, officer, riskRegression)
```

---

## Codebook Cross-Reference

| Codebook section | Key constraint to verify | Script |
|---|---|---|
| §1.3 Pre-processing | >30% missingness drop; Spearman collinearity (threshold=0.80); 3 re-encodings | `1_preprocessing/data_prep.py` |
| §1.5 Train/test split | `StratifiedShuffleSplit`, 70/30, `random_state=42` | `1_preprocessing/data_prep.py` |
| §2.1 RSF Stability Selection | 100 iterations; subsample 60%; PFI 5 repeats; binarize at 70th pct; threshold=0.60 | `2_feature_selection/rsf_bootstrap_selector.py` |
| §2.2 Cox Elastic-Net | Optuna n_alphas [5,50], l1_ratio [0.1,1.0]; 0.25-SE rule; unpenalized refit | `3_model_training/cox_net/model_train.py` |
| §3.1 Preprocessing in pipelines | IterativeImputer (`max_iter=10`, `seed=42`) for Cox/RSF/XGB-AFT; no imputation for XGBSE | per-learner `model_train.py` |
| §3.2 Hyperparameter tuning | Optuna TPE (`seed=42`), 50 trials, 5×10 `RepeatedStratifiedKFold` | per-learner `model_train.py` |
| §3.3 RSF search space | n_estimators [100,500], max_depth [3,15], max_features ∈ {sqrt,log2,0.5,0.7,0.9} | `3_model_training/rsf/model_train.py` |
| §3.3 XGBoost-AFT search space | n_estimators [300,800], AFT distribution ∈ {normal,logistic,extreme} | `3_model_training/xgboost_surv/model_train.py` |
| §3.3 XGBSE search space | num_boost_round [100,500], reg_lambda/reg_alpha log[0.01,10] | `3_model_training/xgbse_weibull/model_train.py` |
| §3.3 DeepSurv (benchmark) | n_layers [1,3], num_nodes [16,128], dropout [0,0.5]; excluded from ensemble | `3_model_training/deepsurv/model_train.py` |
| §4.1 OOF generation | `StratifiedKFold` k=5 seed=42; alphabetical column order | `4_ensemble/ensemble_model.py` |
| §4.2 Meta-learner | Unpenalized `CoxPHSurvivalAnalysis` on 4 meta-features | `4_ensemble/ensemble_model.py` |
| §4.3 Serialization + identity check | MLflow PyFunc registration + round-trip prediction comparison | `4_ensemble/ensemble_pyfunc.py`, `ensemble_model.py` |
| §5.1 Discrimination | Harrell C, Uno IPCW C (τ=24 m, training-set censoring), time-dep AUC | `5_validation/lib/survival_metrics.py` |
| §5.2 Calibration | IPCW Brier at 6/12/18/24 m; IBS on 40-point grid clipped to 24 m | `5_validation/lib/survival_metrics.py` |
| §5.4 Bootstrap | n=200, event-stratified, on cached predictions; paired test | `5_validation/lib/bootstrap.py` |
| §6 Risk stratification | P15/P85 thresholds from training set only; KM, log-rank, HR, PPV/NPV | `5_validation/lib/risk_stratification.py`, `compute_metrics.py` |
| §7 SENECA comparator | Fixed coefficients, binary thresholds, complete-case comparison | `5_validation/lib/seneca_model.py` |
| §8.1 Kernel SHAP | k-means k=20 background; 200 patients; `seed=42` | `5_validation/shap_analysis.py` |
| §8.2 Permutation importance | 50 permutations/feature; Harrell C-index degradation; `seed=42` | `5_validation/shap_analysis.py` |
| External cohort exclusions | 5 date-inconsistent + 80 overlapping patients removed; final n=698 | `5_validation/prepare_external.py` |
| Manuscript figures | Calibration, DCA, KM, ROC, PR, AUC/Brier line plots, risk densities | `7_publication_figures/publication_figures.R` |
| Manuscript tables | Performance summary, bootstrap CIs, risk stratification, SENECA comparison | `7_publication_figures/publication_tables.R` |

---

## Key Implementation Details to Verify

### Global random seed
`random_state=42` is used for: data split, CV folds, imputation, Optuna TPE sampler, SHAP background k-means, SHAP patient sampling, bootstrap iterations, permutation importance.

### Pre-processing (`1_preprocessing/data_prep.py`)
- Features dropped for >30% missingness: FT3 (67.8%), FT4 (60.7%), CRP (55.5%), FT3/FT4 ratio (67.9%), TSH (37.1%)
- Collinearity removal via `SmartCorrelatedSelection(method='spearman', threshold=0.80, selection_method='variance')` — drops the lower-variance member of correlated pairs; removes 7 binary encodings: `alp`, `alt_bin`, `ast_bin`, `ca199_bin`, `cea_bin`, `nlr_bin`, `surgery_and_start_of_cgi_metastatic_ab_initio`
- Re-encodings: BTC location → intrahepatic binary; disease status → metastatic binary; ECOG PS → binary 0 vs ≥1 (column kept as `ps ecog`)
- Split: `StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)`

### RSF Stability Selection (`2_feature_selection/rsf_bootstrap_selector.py`)
```python
StabilitySelectionRSF(
    n_iterations=100, subsample_fraction=0.6,
    pfi_repeats=5, selection_threshold=0.6,
    normalize_pfi=True, binarization_quantile=0.70,
    n_estimators_range=(150, 350), max_depth_range=(3, 14),
    min_samples_split_range=(2, 25), min_samples_leaf_range=(2, 15),
    max_features_str=("sqrt", "log2"), max_features_float=(0.5, 1.0),
    random_state=42
)
```

### Cox Elastic-Net (`3_model_training/cox_net/model_train.py`)
- Optuna default sampler (not TPE, per §3.3): tunes `n_alphas` ∈ [5,50] and `l1_ratio` ∈ [0.1,1.0]
- Alpha selection: `USE_BEST_ALPHA = False` activates the 0.25-SE parsimony rule — picks the largest α whose mean CV C-index is within 0.25·SE of the path maximum
- Feature selection threshold: `|coefficient| < 1e-5` → features dropped
- Final model: `CoxPHSurvivalAnalysis(alpha=0.0)` fitted on surviving features only (unpenalized)

### Base learner CV configuration
All four learners use `--cv_n_splits 5 --cv_n_repeats 10` (default in each script's `argparse`), giving 5×10 = 50 fits per Optuna trial via `RepeatedStratifiedKFold`. The `--n_optuna_trials` default is `50`.

### Ensemble (`4_ensemble/ensemble_model.py`)
- OOF generation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Column order: `sorted(base_models.keys())` → `['cox', 'rsf', 'xgboost', 'xgbse']` → meta-features `cox_pred`, `rsf_pred`, `xgboost_pred`, `xgbse_pred`
- Meta-learner: `CoxPHSurvivalAnalysis()` with no regularization

### Bootstrap (`5_validation/lib/bootstrap.py` + `compute_metrics.py`)
- `--n_bootstrap 200` (default)
- `stratified_bootstrap_sample()` uses `sklearn.utils.resample(..., stratify=event, replace=True)`
- Metrics computed on `internal_predictions.csv` / `temporal_predictions.csv` (no re-inference)
- ML vs SENECA comparison: paired differences from the same 200 bootstrap draws

### Risk stratification (`5_validation/compute_metrics.py`)
- Both schemes defined in `STRATIFICATION_SCHEMES`: 33-66 and **15-85** (paper-reported scheme)
- Thresholds derived from `training_risk_scores.json` (written by `generate_predictions.py` from training-set predictions only)
- `apply_risk_thresholds()` in `lib/risk_stratification.py` maps continuous scores to groups 0/1/2

### DeepSurv (`3_model_training/deepsurv/model_train.py`)
- CoxPH deep neural network implemented via `pycox` / `torchtuples`; wrapper in `deepsurv_wrapper.py`
- Optuna default sampler (not TPE); search: `n_layers` [1,3], `num_nodes` [16,128 step 16], `batch_norm` {True/False}, `dropout` [0, 0.5], `learning_rate` log[1e-5, 1e-2], `batch_size` ∈ {32,64,128,256}
- Objective: time-dependent concordance via `pycox.evaluation.EvalSurv` (unlike C-index used for other learners)
- Imputation and scaling fitted **once** on the full training set (not within folds — intentional deviation documented in §3.3)
- Final model: 80/20 stratified split, up to 200 epochs, early-stop patience=20
- **Not included in the stacking ensemble** — reported as a benchmark only; excluded by the `generate_oof_predictions()` filter in `4_ensemble/ensemble_model.py`

### R figures and tables (`7_publication_figures/`)
- `publication_figures.R` reads the CSVs produced by `compute_metrics.py` (`r_input/` subfolder) and generates all manuscript figures: calibration curves (6 m + 18 m), DCA panels, Kaplan-Meier with numbers-at-risk, ROC/PR at 4 timepoints, temporal AUC/Brier line plots, risk-score density overlays
- `publication_tables.R` generates `.docx` Word tables from the `metrics/` and `tables/` JSON/CSV outputs
- Run inside the Docker image defined in `R_env/Dockerfile` (R 4.4.3 + all dependencies)


```
--background_samples 20   (k-means centroids from training cohort)
--display_samples    200  (patients sampled from temporal validation)
--pfi_repeats         50  (permutations per feature)
--random_state        42
```

---

## Software Environment

| Package | Version |
|---|---|
| Python | 3.9.23 |
| scikit-learn | 1.5.1 |
| scikit-survival | 0.23.0 |
| xgbse | 0.3.3 |
| mlflow | 2.22.2 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| scipy | 1.13.1 |
| cloudpickle | 2.2.1 |

Publication figures and tables are produced by the R scripts in `7_publication_figures/`. They consume the CSV outputs of `compute_metrics.py` and produce all figures in the manuscript and supplement (calibration, DCA, KM, ROC, PR, temporal AUC/Brier, risk-score densities). Run the scripts inside the provided Docker image (`R_env/Dockerfile`) which pre-installs all dependencies (R 4.4.3, `survminer`, `dcurves`, `ggplot2`, `patchwork`, `flextable`, `riskRegression`).

---

## What this repository does NOT contain

- **Patient-level data** — cannot be shared
- **Data quality-check scripts** (§1.2) — duplicate detection, date-range validation, and per-centre missingness flags were performed against the source database system and are not reproducible without the raw data
- **Azure ML pipeline YAML definitions** — infrastructure artefacts (compute targets, job orchestration); no analytical logic
- **Web application code** — Streamlit UI, authentication, AKS configuration, SHAP waterfall rendering

---

## Repository Structure

```
paper_code/
│
├── README.md                     ← this file
├── Dictionary_v08.csv            ← variable dictionary (features, encodings, categories)
│
├── 1_preprocessing/              ← Data pipeline (§1 Methods)
│   └── data_prep.py              ← Missing-value filter (>30%), collinearity removal,
│                                    three pre-specified re-encodings, 70/30 stratified split
│
├── 2_feature_selection/          ← Feature selection (§2 Methods)
│   ├── rsf_bootstrap_selector.py ← RSF Stability Selection class (100 bootstrap iterations,
│   │                                PFI binarized at 70th percentile, threshold=0.60)
│   └── feature_selection.py      ← Driver script: fits selector, logs stability scores
│
├── 3_model_training/             ← Base learners & hyperparameter tuning (§3 Methods)
│   ├── survival_utils.py         ← Shared metrics, CV helpers, Optuna plotting utilities
│   ├── xgboost_survival_model.py ← XGBoostSurvival sklearn wrapper (AFT objective;
│   │                                KM-stratified non-parametric survival function)
│   ├── xgbse_pipeline_wrapper.py ← XGBSEPipeline wrapper (XGBSE StackedWeibull;
│   │                                NaN-native — no imputation)
│   ├── cox_net/
│   │   └── model_train.py        ← CoxNet elastic-net + 0.25-SE parsimony rule,
│   │                                unpenalized refit, Optuna joint (n_alphas, l1_ratio)
│   ├── rsf/
│   │   └── model_train.py        ← RSF with RepeatedStratifiedKFold, 50 Optuna trials
│   ├── xgboost_surv/
│   │   └── model_train.py        ← XGBoost-AFT, Optuna (50 trials), IterativeImputer
│   └── xgbse_weibull/
│       └── model_train.py        ← XGBSE StackedWeibull, Optuna (50 trials)
│
├── 4_ensemble/                   ← Stacking ensemble (§4 Methods)
│   ├── ensemble_model.py         ← OOF generation (StratifiedKFold k=5, seed=42),
│   │                                unpenalized CoxPH meta-learner, risk-score output
│   ├── ensemble_pyfunc.py        ← MLflow PyFunc wrapper for serialization
│   │                                (load_context → predict → risk_score column)
│   ├── evaluate_models.py        ← Per-model and ensemble test-set evaluation driver
│   ├── evaluation_utils.py       ← Temporal grid construction, metric aggregation,
│   │                                comparative plotting utilities
│   └── xgboost_survival_model.py ← XGBoostSurvival class copy (required for
│                                    joblib unpickling of trained XGBoost pipeline)
│
├── 5_validation/                 ← Full evaluation pipeline (§5 Methods)
│   ├── prepare_external.py       ← External cohort alignment: 783 raw → 698 final
│   │                                (5 date-inconsistent + 80 overlapping with dev cohort)
│   ├── generate_predictions.py   ← Upfront prediction generation for both cohorts
│   │                                (internal test + temporal validation) + SENECA baseline
│   ├── compute_metrics.py        ← All survival metrics, bootstrap CIs (n=200,
│   │                                event-stratified), risk stratification (P15/P85)
│   └── shap_analysis.py          ← Kernel SHAP (k-means background k=20, n=200 patients,
│                                    seed=42) + permutation importance (50 repeats)
│
├── 5_validation/lib/             ← Shared validation utilities
│   ├── survival_metrics.py       ← C-index, Uno IPCW C, time-dep AUC, IPCW Brier,
│   │                                IBS, temporal grid construction
│   ├── bootstrap.py              ← Event-stratified bootstrap, paired ML vs SENECA tests
│   ├── risk_stratification.py    ← Thresholds (P15/P85 on training), KM, log-rank, HR
│   ├── roc_pr.py                 ← IPCW ROC/PR curves at clinical timepoints
│   └── seneca_model.py           ← SENECA published formula (5 features, fixed coefficients)
│
└── 6_inference/                  ← Deployment inference code (informational, §App Methods)
    ├── score.py                  ← Scoring endpoint: risk score, survival curve,
    │                                risk group, median survival, percentile
    └── requirements.txt          ← Pinned inference dependencies
```

---

## Codebook Cross-Reference

| Methods section | Folder | Key script |
|---|---|---|
| §1 Data pipeline (pre-processing) | `1_preprocessing/` | `data_prep.py` |
| §2.1 RSF Stability Selection | `2_feature_selection/` | `rsf_bootstrap_selector.py` |
| §2.2 Cox Elastic-Net selection | `3_model_training/cox_net/` | `model_train.py` |
| §3 Base learners (tuning) | `3_model_training/` | per-learner `model_train.py` |
| §4 Stacking ensemble | `4_ensemble/` | `ensemble_model.py` |
| §4.3 MLflow serialization | `4_ensemble/` | `ensemble_pyfunc.py` |
| §5 Metrics & bootstrap CIs | `5_validation/` | `compute_metrics.py` |
| §5.1 Discrimination | `5_validation/lib/` | `survival_metrics.py` |
| §5.4 Bootstrap protocol | `5_validation/lib/` | `bootstrap.py` |
| §6 Risk stratification | `5_validation/lib/` | `risk_stratification.py` |
| §7 SENECA comparator | `5_validation/lib/` | `seneca_model.py` |
| §8 SHAP & PFI | `5_validation/` | `shap_analysis.py` |

---

## Key Implementation Constraints to Verify

The items below are directly checkable in the code and map to the numerical results in the manuscript.

**Random seed:** `random_state=42` throughout — data split, CV folds, bootstrap, imputation,
Optuna samplers, SHAP background sampling.

**Pre-processing** (`1_preprocessing/data_prep.py`):
- Features with >30% missing dropped (FT3 67.8%, FT4 60.7%, CRP 55.5%, FT3/FT4 67.9%, TSH 37.1%)
- Spearman |ρ| ≥ 0.80 collinearity removal via `SmartCorrelatedSelection` → 7 redundant binaries dropped
- Three re-encodings: location → intrahepatic binary; disease status → metastatic binary; ECOG → binary 0 vs ≥1
- 70/30 stratified split (`StratifiedShuffleSplit`, seed=42)

**RSF Stability Selection** (`2_feature_selection/rsf_bootstrap_selector.py`):
- 100 bootstrap iterations, 60% subsample without replacement
- Permutation importance (5 repeats), min-max normalized, binarized at 70th percentile
- Selection threshold = 0.60 → 7 features retained; top-3 fallback if none qualify

**Cox Elastic-Net** (`3_model_training/cox_net/model_train.py`):
- Optuna joint tuning of `n_alphas` [5,50] and `l1_ratio` [0.1, 1.0]
- 0.25-SE parsimony rule for alpha selection (more conservative than 1-SE)
- Surviving features refit as **unpenalized** CoxPH (not the regularized model)
- 9 features retained

**Ensemble** (`4_ensemble/ensemble_model.py`):
- OOF meta-features: StratifiedKFold k=5, seed=42
- Meta-feature columns sorted **alphabetically**: `cox_pred`, `rsf_pred`, `xgboost_pred`, `xgbse_pred`
- Meta-learner: unpenalized CoxPH

**Bootstrap** (`5_validation/lib/bootstrap.py`):
- 200 event-stratified bootstrap iterations on **cached prediction tables** (no re-inference)
- Paired ML vs SENECA differences from same iterations

**Risk stratification** (`5_validation/lib/risk_stratification.py`):
- Thresholds: 15th and 85th percentiles of **training-set** risk scores only (frozen before evaluation)

**SHAP** (`5_validation/shap_analysis.py`):
- Kernel SHAP, k-means background k=20, 200 validation patients sampled with seed=42
- Permutation importance: 50 permutations per feature on temporal validation cohort

---

## Software Environment

| Package | Version |
|---|---|
| Python | 3.9.23 |
| scikit-learn | 1.5.1 |
| scikit-survival | 0.23.0 |
| xgbse | 0.3.3 |
| mlflow | 2.22.2 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| scipy | 1.13.1 |
| cloudpickle | 2.2.1 |

Publication figures and tables are produced by the R scripts in `7_publication_figures/`. They consume the CSV outputs of `compute_metrics.py` and produce all figures in the manuscript and supplement (calibration, DCA, KM, ROC, PR, temporal AUC/Brier, risk-score densities). Run the scripts inside the provided Docker image (`R_env/Dockerfile`) which pre-installs all dependencies (R 4.4.3, `survminer`, `dcurves`, `ggplot2`, `patchwork`, `flextable`, `officer`, `riskRegression`).

---

## What this repository does NOT contain

- **Patient-level data** — cannot be shared
- **Data quality-check scripts** (§1.2) — duplicate detection, date-range validation, and per-centre missingness flags were performed against the source database system and are not reproducible without the raw data
- **Azure ML pipeline YAML definitions** — infrastructure artefacts (compute targets, job orchestration); no analytical logic
- **Web application code** — Streamlit UI, authentication, AKS configuration, SHAP waterfall rendering
