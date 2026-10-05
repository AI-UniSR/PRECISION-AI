#!/usr/bin/env Rscript
# Word tables of the temporal validation, from the outputs of compute_metrics.py,
# publication_figures.R and threshold_sensitivity.py.
#
#   main/table2_overall_performance.docx       Table 2
#   main/table3_time_dependent_temporal.docx   Table 3
#   main/table4_ensemble_vs_seneca.docx        Table 4
#   supplementary/table_s4_risk_groups_15_85.docx         Table S4
#   supplementary/calibration_summary.docx                calibration in the Results (Figure 2C-D)
#   supplementary/ensemble_vs_cox.docx                    ensemble vs unpenalised Cox benchmark
#   supplementary/cox_benchmark_coefficients.docx         hazard ratios of the Cox benchmark
#   supplementary/cutoff_sensitivity_temporal_full.docx   risk groups at other percentile cutoffs
#
# Tables 2-4 and S4 also list the Cox benchmark, which the last three tables
# describe; the benchmark and cutoff analyses answer reviewer requests.

suppressPackageStartupMessages({
  library(argparse)
  library(readr)
  library(jsonlite)
  library(dplyr)
  library(flextable)
  library(officer)
})

parser <- ArgumentParser(description = "Word tables of the temporal validation")
parser$add_argument("--metrics_dir", required = TRUE, help = "metrics/ of compute_metrics.py")
parser$add_argument("--tables_dir", required = TRUE, help = "tables/ of compute_metrics.py")
parser$add_argument("--figures_csv_dir", required = TRUE, help = "Output of publication_figures.R (csv/)")
parser$add_argument("--threshold_sensitivity_dir", required = TRUE, help = "Output of threshold_sensitivity.py")
parser$add_argument("--output_dir", required = TRUE)
args <- parser$parse_args()

figures_csv_dir <- args$figures_csv_dir
if (dir.exists(file.path(figures_csv_dir, "csv"))) figures_csv_dir <- file.path(figures_csv_dir, "csv")
main_dir <- file.path(args$output_dir, "tables", "main")
supp_dir <- file.path(args$output_dir, "tables", "supplementary")
for (d in c(main_dir, supp_dir)) dir.create(d, recursive = TRUE, showWarnings = FALSE)

# ---- Helpers --------------------------------------------------------------------

require_file <- function(path) {
  if (!file.exists(path)) stop(sprintf("Missing input: %s", path))
  path
}
read_table <- function(dir, name) read_csv(require_file(file.path(dir, name)), show_col_types = FALSE)
read_json_input <- function(dir, name) fromJSON(require_file(file.path(dir, name)))

fmt_val <- function(x, digits = 2) {
  if (is.null(x) || is.na(x)) return("—")
  formatC(x, format = "f", digits = digits)
}
fmt_ci <- function(val, lo, hi, digits = 2) {
  if (is.null(val) || is.na(val)) return("—")
  if (is.null(lo) || is.na(lo) || is.null(hi) || is.na(hi)) return(fmt_val(val, digits))
  sprintf("%s (%s–%s)", fmt_val(val, digits), fmt_val(lo, digits), fmt_val(hi, digits))
}
first_or_na <- function(x) if (length(x) == 0) NA else x[[1]]
# Value of column value_col of a one-row table, with the CI of boot_agg[[boot_key]]
cell_ci <- function(row, value_col, boot_agg, boot_key, digits = 3) {
  val <- if (value_col %in% names(row)) first_or_na(row[[value_col]]) else NA
  fmt_ci(val, boot_agg[[boot_key]]$ci_lower, boot_agg[[boot_key]]$ci_upper, digits)
}
minus_sign <- function(x) gsub("-", "−", x, fixed = TRUE)

style_table <- function(ft) {
  ft %>%
    bold(part = "header") %>%
    align(j = 1, align = "left", part = "all") %>%
    align(j = -1, align = "center", part = "all") %>%
    border_remove() %>%
    hline_top(part = "header", border = fp_border(width = 1.5)) %>%
    hline_bottom(part = "header", border = fp_border(width = 0.75)) %>%
    hline_bottom(part = "body", border = fp_border(width = 1.5)) %>%
    fontsize(size = 9, part = "all") %>%
    font(fontname = "Times New Roman", part = "all") %>%
    padding(padding = 2, part = "all") %>%
    autofit()
}
# Wide tables: let Word fit the columns to the page width
fit_page <- function(ft) set_table_properties(ft, layout = "autofit", width = 1)

save_docx <- function(ft, path, title, footnote = NULL, landscape = FALSE) {
  doc <- read_docx()
  if (landscape) doc <- body_set_default_section(doc, prop_section(page_size = page_size(orient = "landscape")))
  doc <- body_add_par(doc, title, style = "heading 2")
  doc <- body_add_flextable(doc, ft)
  if (!is.null(footnote)) doc <- body_add_par(doc, footnote, style = "Normal")
  print(doc, target = path)
}

ml_label <- "ML ensemble"
cox_label <- "Cox PH (benchmark)"
diff_cox_label <- "Difference (ML − Cox PH)"
cox_def_note <- paste(
  "Cox PH = unpenalised Cox proportional hazards model fitted on the same training patients, the same",
  "10 predictors and the same preprocessing (iterative imputation and robust scaling fitted on the",
  "training set) as the ML ensemble.")
cox_note <- paste(cox_def_note, "Differences are ML ensemble minus Cox PH with 95% CIs from the paired",
                  "bootstrap (same resamples).")
boot_note <- function(boot) sprintf("95%% CIs from stratified bootstrap (%s iterations).", boot$n_iterations)

# ---- Inputs ---------------------------------------------------------------------

overall_internal <- read_table(args$tables_dir, "overall_internal.csv")
overall_temporal <- read_table(args$tables_dir, "overall_temporal_full.csv")
overall_cc <- read_table(args$tables_dir, "overall_temporal_cc.csv")
timedep_temporal <- read_table(args$tables_dir, "timedep_temporal_full.csv")
boot_internal <- read_json_input(args$metrics_dir, "bootstrap_internal.json")
boot_temporal <- read_json_input(args$metrics_dir, "bootstrap_temporal_full.json")
boot_cc <- read_json_input(args$metrics_dir, "bootstrap_temporal_cc.json")
perf_temporal <- read_json_input(args$metrics_dir, "performance_temporal.json")

# ---- Table 2: overall performance, internal test set and temporal cohort --------

overall_rows <- function(tbl, boot_agg, cohort_label) {
  cell <- function(metric, value_col, boot_key) cell_ci(tbl %>% filter(metric == !!metric), value_col, boot_agg, boot_key)
  data.frame(
    Cohort = cohort_label,
    Model = c(ml_label, cox_label, diff_cox_label),
    `C-index (95% CI)` = c(cell("harrell_c_index", "ml", "c_index_ml"),
                           cell("harrell_c_index", "cox", "c_index_cox"),
                           cell("harrell_c_index", "delta_ml_cox", "c_index_diff_ml_cox")),
    `Mean time-dependent AUC (95% CI)` = c(cell("mean_auc", "ml", "mean_auc_ml"),
                                           cell("mean_auc", "cox", "mean_auc_cox"),
                                           cell("mean_auc", "delta_ml_cox", "mean_auc_diff_ml_cox")),
    `Integrated Brier score (95% CI)` = c(cell("ibs", "ml", "ibs_overall_ml"),
                                          cell("ibs", "cox", "ibs_overall_cox"),
                                          cell("ibs", "delta_ml_cox", "ibs_overall_diff_ml_cox")),
    check.names = FALSE, stringsAsFactors = FALSE)
}
t2 <- bind_rows(overall_rows(overall_internal, boot_internal$aggregated, "Internal hold-out test"),
                overall_rows(overall_temporal, boot_temporal$aggregated, "Temporal validation"))
save_docx(flextable(t2) %>% merge_v(j = 1) %>% style_table(),
          file.path(main_dir, "table2_overall_performance.docx"),
          "Table 2. Overall performance of the final ML ensemble and the Cox PH benchmark in the internal hold-out test set and temporal validation cohort",
          paste("CI = confidence interval. C-index = Harrell’s concordance index. AUC = area under the",
                "time-dependent cumulative/dynamic ROC curve. IBS = integrated Brier score.",
                boot_note(boot_temporal), cox_note))

# ---- Table 3: time-dependent AUC and Brier score, temporal cohort ---------------

timedep_rows <- function(tbl, boot_agg) {
  bind_rows(lapply(c(6, 12, 18, 24), function(tp) {
    auc <- tbl %>% filter(metric == "auc", timepoint == tp)
    brier <- tbl %>% filter(metric == "brier", timepoint == tp)
    data.frame(
      `Time horizon` = sprintf("%d months", tp),
      Model = c(ml_label, cox_label, diff_cox_label),
      `Time-dependent AUC (95% CI)` = c(cell_ci(auc, "ml", boot_agg, sprintf("auc_%dm_ml", tp)),
                                        cell_ci(auc, "cox", boot_agg, sprintf("auc_%dm_cox", tp)),
                                        cell_ci(auc, "delta_ml_cox", boot_agg, sprintf("auc_%dm_diff_ml_cox", tp))),
      `Brier score (95% CI)` = c(cell_ci(brier, "ml", boot_agg, sprintf("brier_%dm_ml", tp)),
                                 cell_ci(brier, "cox", boot_agg, sprintf("brier_%dm_cox", tp)),
                                 cell_ci(brier, "delta_ml_cox", boot_agg, sprintf("brier_%dm_diff_ml_cox", tp))),
      check.names = FALSE, stringsAsFactors = FALSE)
  }))
}
save_docx(flextable(timedep_rows(timedep_temporal, boot_temporal$aggregated)) %>% merge_v(j = 1) %>% style_table(),
          file.path(main_dir, "table3_time_dependent_temporal.docx"),
          "Table 3. Time-dependent discrimination and prediction error of the final ML ensemble and the Cox PH benchmark in the temporal validation cohort",
          paste("AUC = time-dependent cumulative/dynamic AUC. Brier score = IPCW Brier score (lower is better).",
                boot_note(boot_temporal), cox_note))

# ---- Table 4: ensemble vs SENECA, temporal SENECA complete cases ----------------

agg_cc <- boot_cc$aggregated
cc_row <- function(metric) overall_cc %>% filter(metric == !!metric)
model_row <- function(label, value_col, suffix) data.frame(
  Model = label,
  `C-index (95% CI)` = cell_ci(cc_row("harrell_c_index"), value_col, agg_cc, paste0("c_index_", suffix)),
  `Mean time-dependent AUC (95% CI)` = cell_ci(cc_row("mean_auc"), value_col, agg_cc, paste0("mean_auc_", suffix)),
  `Integrated Brier score (95% CI)` = if (value_col == "seneca" || value_col == "delta") "—" else
    cell_ci(cc_row("ibs"), value_col, agg_cc, paste0("ibs_overall_", suffix)),
  check.names = FALSE, stringsAsFactors = FALSE)
t4 <- bind_rows(model_row(ml_label, "ml", "ml"),
                model_row("SENECA", "seneca", "seneca"),
                model_row("Difference (ML − SENECA)", "delta", "diff"),
                model_row(cox_label, "cox", "cox"),
                model_row(diff_cox_label, "delta_ml_cox", "diff_ml_cox"))
save_docx(flextable(t4) %>% style_table(), file.path(main_dir, "table4_ensemble_vs_seneca.docx"),
          "Table 4. Comparative overall performance of the final ML ensemble, the SENECA score and the Cox PH benchmark in temporal validation",
          paste("Difference rows show bootstrap-based 95% CIs for the paired differences (ML minus SENECA; ML",
                "minus Cox PH). IBS is not computed for SENECA (no survival function). Analysis restricted to",
                "SENECA complete cases.", boot_note(boot_cc), cox_note))

# ---- Calibration at 6 and 18 months (Results, Figure 2C-D) ----------------------

calibration <- bind_rows(lapply(c(6, 18), function(tp) {
  bind_rows(lapply(list(c(ml_label, ""), c(cox_label, "_cox")), function(spec) {
    cm <- read_table(figures_csv_dir, sprintf("calibration_metrics_%dm_temporal%s.csv", tp, spec[2]))
    data.frame(Horizon = sprintf("%d months", tp), Model = spec[1],
               `Calibration slope (95% CI)` = fmt_ci(cm$slope[1], cm$slope_lower_95[1], cm$slope_upper_95[1], 2),
               `O:E ratio (95% CI)` = fmt_ci(cm$oe_km[1], cm$oe_km_lower_95[1], cm$oe_km_upper_95[1], 2),
               check.names = FALSE, stringsAsFactors = FALSE)
  }))
}))
save_docx(flextable(calibration) %>% merge_v(j = 1) %>% style_table(),
          file.path(supp_dir, "calibration_summary.docx"),
          "Calibration of the final ML ensemble and the Cox PH benchmark in temporal validation",
          paste("O:E = observed-to-expected ratio: Kaplan-Meier observed risk at the horizon (all patients) over",
                "the mean predicted risk; 95% CI from the Greenwood standard error. Calibration slope from",
                "logistic recalibration on patients with known status at the horizon (Wald CI).", cox_def_note))

# ---- Table S4: risk groups at the 15th/85th percentiles -------------------------

fmt_median <- function(est, lo, hi) {
  fmt_one <- function(x) if (is.na(x) || x == "NR") "NR" else fmt_val(as.numeric(x), 1)
  if (is.na(est) || est == "NR") return("NR")
  sprintf("%s (%s–%s)", fmt_one(est), fmt_one(lo), fmt_one(hi))
}
risk_group_rows <- function(file_tag, label) {
  km <- read_table(args$tables_dir, sprintf("km_statistics_%s_15-85.csv", file_tag))
  bind_rows(lapply(seq_len(nrow(km)), function(i) {
    r <- km[i, ]
    data.frame(Model = label, `Risk group` = r$group, n = r$n,
               `Deaths, n (%)` = sprintf("%d (%s%%)", r$events, fmt_val(round(r$event_rate * 100, 1), 1)),
               `Median OS, months (95% CI)` = fmt_median(r$median_survival, r$median_ci_lower, r$median_ci_upper),
               `6-month OS probability` = fmt_val(r$surv_6m, 2),
               `18-month OS probability` = fmt_val(r$surv_18m, 2),
               check.names = FALSE, stringsAsFactors = FALSE)
  }))
}
s4 <- bind_rows(risk_group_rows("temporal", ml_label), risk_group_rows("temporal_cox", cox_label))
save_docx(flextable(s4) %>% merge_v(j = 1) %>% style_table(),
          file.path(supp_dir, "table_s4_risk_groups_15_85.docx"),
          "Supplementary Table S4. Survival outcomes by ML-derived and Cox PH-derived risk group in temporal validation using 15th/85th percentile thresholds",
          paste("Risk groups defined by percentiles of each model’s own risk score in the training set, applied",
                "unchanged to temporal validation. OS = overall survival. Median OS from the Kaplan-Meier estimator",
                "(months), 95% CI from the log(-log) pointwise confidence bands; NR = not reached. Survival",
                "probabilities are Kaplan-Meier estimates.", cox_def_note))

# ---- Cox benchmark: hazard ratios (training set) --------------------------------

cox_coefs <- read_table(args$tables_dir, "cox_benchmark_coefficients.csv")
coef_table <- data.frame(
  Predictor = cox_coefs$feature,
  `IQR (training set)` = trimws(formatC(cox_coefs$scaler_scale_iqr, format = "fg", digits = 3)),
  `HR per IQR increase (95% CI)` = unname(mapply(function(v, lo, hi) fmt_ci(v, lo, hi, 2), cox_coefs$hr_per_iqr,
                                                 cox_coefs$hr_per_iqr_ci_lower, cox_coefs$hr_per_iqr_ci_upper)),
  `p-value` = format.pval(cox_coefs$p_value, digits = 2, eps = 0.001),
  check.names = FALSE, stringsAsFactors = FALSE)
save_docx(flextable(coef_table) %>% style_table(), file.path(supp_dir, "cox_benchmark_coefficients.docx"),
          "Unpenalised Cox proportional hazards benchmark: hazard ratios estimated in the training set",
          paste("HR = hazard ratio per interquartile-range (IQR) increase of the predictor, i.e. per unit of the",
                "robust-scaled predictor (IQR computed in the training set; for predictors with IQR = 1, per",
                "one-unit increase). 95% CIs are Wald intervals from the Breslow partial-likelihood information;",
                "they treat the single imputation as fixed. HRs per original unit are in",
                "cox_benchmark_coefficients.csv.", cox_def_note))

# ---- Ensemble vs Cox benchmark, full temporal cohort ----------------------------

cohort_chars <- read_json_input(args$metrics_dir, "cohort_characteristics.json")$temporal_validation
paired_ml_cox <- read_json_input(args$metrics_dir, "paired_differences_ml_minus_cox.json")
cox_fit_report <- read_json_input(args$metrics_dir, "cox_benchmark_fit_report.json")
metric_labels <- c(c_index = "Harrell’s C", c_index_ipcw = "Uno’s C", mean_auc = "Mean time-dependent AUC",
                   auc_6m = "AUC at 6 months", auc_12m = "AUC at 12 months", auc_18m = "AUC at 18 months",
                   auc_24m = "AUC at 24 months", ibs_overall = "Integrated Brier score")

# Consistency checks: bootstrap on the whole cohort; same numbers as paired_differences_ml_minus_cox.json
if (boot_temporal$n_samples != cohort_chars$n) stop("Full-cohort bootstrap does not cover the whole cohort")
agg <- boot_temporal$aggregated
cmp_values <- bind_rows(lapply(names(metric_labels), function(key) {
  ml <- perf_temporal$ml_full[[key]]
  cox <- perf_temporal$cox_full[[key]]
  diff <- agg[[paste0(key, "_diff_ml_cox")]]
  ref <- paired_ml_cox[paired_ml_cox$cohort == "temporal_full" & paired_ml_cox$metric == key, ]
  if (nrow(ref) != 1 || max(abs(c(ref$ml - ml, ref$cox - cox, ref$diff_ci_lower - diff$ci_lower,
                                  ref$diff_ci_upper - diff$ci_upper))) > 1e-12) {
    stop(sprintf("%s differs from paired_differences_ml_minus_cox.json", key))
  }
  data.frame(metric = key, metric_label = unname(metric_labels[key]),
             ml = ml, ml_ci_lower = agg[[paste0(key, "_ml")]]$ci_lower, ml_ci_upper = agg[[paste0(key, "_ml")]]$ci_upper,
             cox = cox, cox_ci_lower = agg[[paste0(key, "_cox")]]$ci_lower,
             cox_ci_upper = agg[[paste0(key, "_cox")]]$ci_upper,
             diff_ml_minus_cox = ml - cox, diff_ci_lower = diff$ci_lower, diff_ci_upper = diff$ci_upper,
             diff_ci_excludes_0 = diff$ci_lower > 0 || diff$ci_upper < 0, n_bootstrap = boot_temporal$n_iterations,
             stringsAsFactors = FALSE)
}))
fmt_diff_ci <- function(val, lo, hi, digits = 3) {
  sprintf("%s (%s to %s)", minus_sign(fmt_val(val, digits)), minus_sign(fmt_val(lo, digits)),
          minus_sign(fmt_val(hi, digits)))
}
cmp_table <- data.frame(
  Metric = cmp_values$metric_label,
  `ML ensemble (95% CI)` = unname(mapply(fmt_ci, cmp_values$ml, cmp_values$ml_ci_lower, cmp_values$ml_ci_upper,
                                         MoreArgs = list(digits = 3))),
  `Cox PH (95% CI)` = unname(mapply(fmt_ci, cmp_values$cox, cmp_values$cox_ci_lower, cmp_values$cox_ci_upper,
                                    MoreArgs = list(digits = 3))),
  delta = paste0(unname(mapply(fmt_diff_ci, cmp_values$diff_ml_minus_cox, cmp_values$diff_ci_lower,
                               cmp_values$diff_ci_upper)), ifelse(cmp_values$diff_ci_excludes_0, "*", "")),
  check.names = FALSE, stringsAsFactors = FALSE)
names(cmp_table)[names(cmp_table) == "delta"] <- "Δ ML − Cox (95% CI)"
n_param <- length(cox_fit_report$predictors)
events_train <- cox_fit_report$derivation$events
cmp_footnote <- paste(
  sprintf(paste("Temporal validation cohort, full cohort (n = %d, %d deaths); missing predictor values imputed with",
                "the imputers fitted on the training set, not refitted."), cohort_chars$n, cohort_chars$events),
  sprintf(paste("Cox PH: unpenalised Cox proportional hazards model (Breslow ties) fitted on the same %d training",
                "patients (%d deaths) as the ML ensemble, with the same %d predictors and the same preprocessing",
                "(iterative imputation and robust scaling fitted on the training set). All predictors are entered as",
                "linear main effects, without splines, transformations or interactions; ECOG performance status is",
                "binary (0 vs ≥1) and the number of metastatic sites is a count. %d parameters, %.1f events per",
                "parameter."), cox_fit_report$derivation$n, events_train, n_param, n_param, events_train / n_param),
  "Harrell’s C and Uno’s C (inverse probability of censoring weighted, truncated at 24 months) are computed",
  "on the risk score (ML ensemble) and on the linear predictor (Cox PH). AUC: cumulative/dynamic IPCW AUC at 6, 12,",
  "18 and 24 months; mean time-dependent AUC: its survival-weighted mean over these four time points. Integrated",
  "Brier score: IPCW Brier score of the predicted survival curves integrated over the evaluation time grid up to",
  "24 months; lower is better. Censoring weights are estimated on the training set.",
  sprintf(paste("95%% CIs: percentile intervals from %s event-stratified bootstrap resamples of the cohort.",
                "Δ = ML ensemble − Cox PH; its 95%% CI comes from the paired differences, both models",
                "being evaluated on the same resamples. * 95%% CI of Δ excludes 0."), boot_temporal$n_iterations),
  "For Harrell’s C, Uno’s C and AUC a positive Δ favours the ML ensemble; for the integrated Brier",
  "score a negative Δ favours the ML ensemble.")
save_docx(flextable(cmp_table) %>% style_table() %>% fit_page(), file.path(supp_dir, "ensemble_vs_cox.docx"),
          sprintf(paste("ML ensemble vs unpenalised Cox proportional hazards model on the same predictors:",
                        "discrimination and overall accuracy in the temporal validation cohort (n = %d, %d deaths)"),
                  cohort_chars$n, cohort_chars$events),
          cmp_footnote)
write_excel_csv(cmp_table, file.path(supp_dir, "ensemble_vs_cox.csv"))
write_csv(cmp_values, file.path(supp_dir, "ensemble_vs_cox_values.csv"))

# ---- Risk groups at other cutoffs, full temporal cohort -------------------------

ts_json <- read_json_input(args$threshold_sensitivity_dir, "threshold_sensitivity_temporal_full.json")
ts <- read_csv(require_file(file.path(args$threshold_sensitivity_dir, "threshold_sensitivity_temporal_full.csv")),
               col_types = cols(.default = col_character()))
if (ts_json$reproduction_check$status != "passed" || ts_json$bootstrap_resample_check$status != "passed") {
  stop("threshold_sensitivity.py checks did not pass")
}
num <- function(x) as.numeric(x)
fmt_os <- function(x) if (is.na(x)) "NE" else if (x == "NR") "NR" else fmt_val(as.numeric(x), 1)
fmt_median_os <- function(est, lo, hi) sprintf("%s (%s–%s)", fmt_os(est), fmt_os(lo), fmt_os(hi))
fmt_reported <- function(reported, val, lo, hi) if (!identical(reported, "True")) "NE" else fmt_ci(num(val), num(lo), num(hi), 3)
cutoff_table <- bind_rows(lapply(seq_len(nrow(ts)), function(i) {
  r <- ts[i, ]
  k <- as.integer(num(r$k))
  data.frame(
    Cutoffs = sprintf("%d/%d", k, 100L - k),
    `Cutoff values (low / high)` = sprintf("%s / %s", minus_sign(fmt_val(num(r$cutoff_low), 3)),
                                           minus_sign(fmt_val(num(r$cutoff_high), 3))),
    `Low risk, n (%)` = sprintf("%d (%.1f%%)", as.integer(num(r$n_low)), 100 * num(r$prop_low)),
    `Intermediate risk, n (%)` = sprintf("%d (%.1f%%)", as.integer(num(r$n_intermediate)), 100 * num(r$prop_intermediate)),
    `High risk, n (%)` = sprintf("%d (%.1f%%)", as.integer(num(r$n_high)), 100 * num(r$prop_high)),
    `Median OS low, months (95% CI)` = fmt_median_os(r$median_survival_low, r$median_ci_lower_low, r$median_ci_upper_low),
    `Median OS intermediate, months (95% CI)` = fmt_median_os(r$median_survival_intermediate,
                                                              r$median_ci_lower_intermediate,
                                                              r$median_ci_upper_intermediate),
    `Median OS high, months (95% CI)` = fmt_median_os(r$median_survival_high, r$median_ci_lower_high,
                                                      r$median_ci_upper_high),
    `HR high vs low (95% CI)` = fmt_ci(num(r$hr_high_vs_low), num(r$hr_high_vs_low_ci_lower),
                                       num(r$hr_high_vs_low_ci_upper), 2),
    ppv = fmt_reported(r$ppv_6m_reported, r$ppv_6m, r$ppv_6m_ci_lower, r$ppv_6m_ci_upper),
    `High risk at risk at 6 m, n` = as.integer(num(r$at_risk_high_6m)),
    `NPV, alive at 18 m (95% CI)` = fmt_reported(r$npv_18m_reported, r$npv_18m, r$npv_18m_ci_lower, r$npv_18m_ci_upper),
    `Low risk at risk at 18 m, n` = as.integer(num(r$at_risk_low_18m)),
    check.names = FALSE, stringsAsFactors = FALSE)
}))
names(cutoff_table)[names(cutoff_table) == "ppv"] <- "PPV, death ≤6 m (95% CI)"
settings <- ts_json$settings
cutoff_footnote <- paste(
  sprintf(paste("Temporal validation cohort, full cohort (n = %d, %d deaths; missing predictor values imputed as in",
                "the main analysis); ML ensemble only; the same patients for every row."),
          ts_json$cohort$n, ts_json$cohort$events),
  sprintf(paste("Cutoffs k/(100 − k) are the k-th and (100 − k)-th percentiles of the ensemble risk score in",
                "the %d training patients (linear interpolation), fixed before evaluation and not re-estimated in",
                "the validation cohort or in bootstrap resamples. Score ≤ lower cutoff: low risk; score > upper",
                "cutoff: high risk; otherwise intermediate. 15/85 and 33/67 are the cutoffs of the main analysis",
                "(33/67 is labelled 33-66 in the pipeline outputs)."), settings$training_cutoff_source$n),
  "Because the cutoffs are training-set percentiles, the proportions of validation patients in each group differ",
  "from the nominal k.",
  "Median OS: Kaplan–Meier, 95% CI from the log(-log) pointwise confidence bands; NR = not reached. HR:",
  "univariable Cox model, high vs low risk group, Wald 95% CI.",
  sprintf(paste("PPV: 1 − Kaplan–Meier survival at %g months in the high-risk group. NPV: Kaplan–Meier",
                "survival at %g months in the low-risk group. 95%% CIs: percentile intervals from %d event-stratified",
                "bootstrap resamples (the same resamples as the main temporal full-cohort analysis), with fixed cutoffs."),
          settings$ppv_timepoint_months, settings$npv_timepoint_months, settings$n_bootstrap),
  sprintf(paste("NE = not estimated: fewer than %d patients at risk at the horizon, or horizon beyond the group’s",
                "last observed follow-up."), settings$min_at_risk_reported))
save_docx(flextable(cutoff_table) %>% style_table() %>% fit_page(),
          file.path(supp_dir, "cutoff_sensitivity_temporal_full.docx"),
          "Sensitivity of the ML ensemble risk groups to the cutoff choice in the temporal validation cohort (full cohort)",
          cutoff_footnote, landscape = TRUE)
write_excel_csv(cutoff_table, file.path(supp_dir, "cutoff_sensitivity_temporal_full.csv"))
cat("Tables written to", args$output_dir, "\n")
