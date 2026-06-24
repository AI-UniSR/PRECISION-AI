#!/usr/bin/env Rscript
# =============================================================================
# Publication-Ready Word Tables
# =============================================================================
#
# Reads pre-computed metrics (JSON, CSV) from compute_metrics and
# publication_figures steps and produces .docx tables using flextable.
# Each .docx is logged to MLflow.
#
# Input directories:
#   --metrics_dir    metrics/ subfolder from compute_metrics
#   --tables_dir     tables/ subfolder from compute_metrics
#   --figures_csv_dir csv/ subfolder from publication_figures (calibration metrics)
#
# Output:
#   --output_dir     root for tables/main/ and tables/supplementary/
# =============================================================================

suppressPackageStartupMessages({
  library(argparse)
  library(readr)
  library(jsonlite)
  library(dplyr)
  library(flextable)
  library(officer)
})

# ── CLI ──────────────────────────────────────────────────────────────────────
parser <- ArgumentParser(description = "Publication-ready Word tables for BTC validation")
parser$add_argument("--metrics_dir",     required = TRUE, help = "metrics/ subfolder from compute_metrics")
parser$add_argument("--tables_dir",      required = TRUE, help = "tables/ subfolder from compute_metrics")
parser$add_argument("--figures_csv_dir", required = TRUE, help = "figures output dir from publication_figures (csv/ subfolder will be used)")
parser$add_argument("--output_dir",      required = TRUE, help = "Output folder for .docx tables")
args <- parser$parse_args()

# The publication_figures step writes CSVs to {output}/csv/
# Accept either the parent dir (with csv/ subfolder) or the csv/ dir directly.
figures_csv_dir <- args$figures_csv_dir
if (dir.exists(file.path(figures_csv_dir, "csv"))) {
  figures_csv_dir <- file.path(figures_csv_dir, "csv")
}

# Create output directories
main_dir <- file.path(args$output_dir, "tables", "main")
supp_dir <- file.path(args$output_dir, "tables", "supplementary")
dir.create(main_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(supp_dir, recursive = TRUE, showWarnings = FALSE)

# ── Helpers ──────────────────────────────────────────────────────────────────

require_file <- function(path, label) {
  if (!file.exists(path)) {
    stop(sprintf("REQUIRED FILE MISSING: %s (%s)", path, label))
  }
  path
}

fmt_val <- function(x, digits = 2) {
  if (is.null(x) || is.na(x)) return("\u2014")
  formatC(x, format = "f", digits = digits)
}

fmt_ci <- function(val, lo, hi, digits = 2) {
  if (is.null(val) || is.na(val)) return("\u2014")
  if (is.null(lo) || is.na(lo) || is.null(hi) || is.na(hi)) {
    return(fmt_val(val, digits))
  }
  sprintf("%s (%s\u2013%s)", fmt_val(val, digits), fmt_val(lo, digits), fmt_val(hi, digits))
}

save_docx <- function(ft, path, title, footnote = NULL) {
  # Add title as header paragraph and footnote as footer
  doc <- read_docx()
  doc <- body_add_par(doc, title, style = "heading 2")
  doc <- body_add_flextable(doc, ft)
  if (!is.null(footnote)) {
    doc <- body_add_par(doc, footnote, style = "Normal")
  }
  print(doc, target = path)
  cat(sprintf("  saved %s\n", basename(path)))
}

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

# ── Load shared data ────────────────────────────────────────────────────────

cat("=== Loading input data ===\n")

# Overall summary tables (from compute_metrics)
tbl1_file <- require_file(file.path(args$tables_dir, "table1_overall_internal.csv"),
                          "Table 1 Overall Internal")
# Table 2 (standalone ML on full temporal cohort)
tbl2_file <- require_file(file.path(args$tables_dir, "table2a_overall_temporal_full.csv"),
                          "Table 2 Overall Temporal Full")
# Table 2 CC (ML vs SENECA on complete cases, for Table 4 in paper)
tbl2_cc_file <- require_file(file.path(args$tables_dir, "table2b_overall_temporal_cc.csv"),
                             "Table 2 Overall Temporal CC")

tbl1_raw    <- read_csv(tbl1_file,    show_col_types = FALSE)
tbl2_raw    <- read_csv(tbl2_file,    show_col_types = FALSE)
tbl2_cc_raw <- read_csv(tbl2_cc_file, show_col_types = FALSE)

# Time-dependent tables
tbl3_file <- require_file(file.path(args$tables_dir, "table3_timedep_internal.csv"),
                          "Table 3 Time-dep Internal")
# Table 3 paper (standalone ML time-dep on full temporal cohort)
tbl4_file <- require_file(file.path(args$tables_dir, "table4a_timedep_temporal_full.csv"),
                          "Table 4a Time-dep Temporal Full")

tbl3_raw <- read_csv(tbl3_file, show_col_types = FALSE)
tbl4_raw <- read_csv(tbl4_file, show_col_types = FALSE)

# Bootstrap results
# Full-cohort bootstrap (for ML CIs in Tables 2 & 3)
boot_temporal_file <- require_file(file.path(args$metrics_dir, "bootstrap_temporal.json"),
                                   "Bootstrap Temporal")
boot_temporal <- fromJSON(boot_temporal_file)
# CC bootstrap (for ML vs SENECA comparison in Table 4)
boot_temporal_cc_file <- require_file(file.path(args$metrics_dir, "bootstrap_temporal_cc.json"),
                                      "Bootstrap Temporal CC")
boot_temporal_cc <- fromJSON(boot_temporal_cc_file)

# Performance summary (all models, both cohorts)
perf_summary_file <- require_file(file.path(args$tables_dir, "performance_summary.csv"),
                                  "Performance Summary")
perf_summary <- read_csv(perf_summary_file, show_col_types = FALSE)

# Performance JSONs
perf_internal_file <- require_file(file.path(args$metrics_dir, "performance_internal.json"),
                                   "Performance Internal")
perf_temporal_file <- require_file(file.path(args$metrics_dir, "performance_temporal.json"),
                                   "Performance Temporal")
perf_internal <- fromJSON(perf_internal_file)
perf_temporal <- fromJSON(perf_temporal_file)

# Bootstrap internal
boot_internal_file <- require_file(file.path(args$metrics_dir, "bootstrap_internal.json"),
                                   "Bootstrap Internal")
boot_internal <- fromJSON(boot_internal_file)

cat("  All input files loaded successfully\n")

# ==========================================================================
# TABLE 2 – Overall ML performance (internal + temporal)
# ==========================================================================
cat("=== Table 2: Overall ML ===\n")

make_overall_row <- function(tbl_raw, boot_agg, cohort_label) {
  # Extract ML values for c_index, mean_auc, ibs
  get_ml <- function(metric_name, boot_key) {
    row <- tbl_raw %>% filter(metric == metric_name)
    val <- row$ml[1]
    lo  <- boot_agg[[boot_key]]$ci_lower
    hi  <- boot_agg[[boot_key]]$ci_upper
    fmt_ci(val, lo, hi, 3)
  }

  data.frame(
    Cohort                            = cohort_label,
    `C-index (95% CI)`                = get_ml("harrell_c_index", "c_index_ml"),
    `Mean time-dependent AUC (95% CI)` = get_ml("mean_auc", "mean_auc_ml"),
    `Integrated Brier score (95% CI)` = get_ml("ibs", "ibs_overall_ml"),
    check.names = FALSE, stringsAsFactors = FALSE
  )
}

t2_int  <- make_overall_row(tbl1_raw, boot_internal$aggregated, "Internal hold-out test")
t2_temp <- make_overall_row(tbl2_raw, boot_temporal$aggregated, "Temporal validation")
t2_df   <- bind_rows(t2_int, t2_temp)

ft2 <- flextable(t2_df) %>% style_table()
save_docx(ft2, file.path(main_dir, "table_2_overall_ml.docx"),
          "Table 2. Overall performance of the final ML ensemble in the internal hold-out test set and temporal validation cohort",
          "CI = confidence interval. C-index = Harrell\u2019s concordance index. AUC = area under the time-dependent cumulative/dynamic ROC curve. IBS = integrated Brier score. 95% CIs from stratified bootstrap (200 iterations).")

# ==========================================================================
# TABLE 3 – Time-dependent ML (temporal)
# ==========================================================================
cat("=== Table 3: Temporal time-dependent ML ===\n")

t3_auc   <- tbl4_raw %>% filter(metric == "auc")
t3_brier <- tbl4_raw %>% filter(metric == "brier")
boot_agg_t <- boot_temporal$aggregated

t3_rows <- lapply(c(6, 12, 18, 24), function(tp) {
  auc_row   <- t3_auc   %>% filter(timepoint == tp)
  brier_row <- t3_brier  %>% filter(timepoint == tp)

  auc_boot   <- boot_agg_t[[sprintf("auc_%dm_ml", tp)]]
  brier_boot <- boot_agg_t[[sprintf("brier_%dm_ml", tp)]]

  data.frame(
    `Time horizon`                    = sprintf("%d months", tp),
    `Time-dependent AUC (95% CI)`     = fmt_ci(auc_row$ml[1],
                                               auc_boot$ci_lower, auc_boot$ci_upper, 3),
    `Brier score (95% CI)`            = fmt_ci(brier_row$ml[1],
                                               brier_boot$ci_lower, brier_boot$ci_upper, 3),
    check.names = FALSE, stringsAsFactors = FALSE
  )
})
t3_df <- bind_rows(t3_rows)

ft3 <- flextable(t3_df) %>% style_table()
save_docx(ft3, file.path(main_dir, "table_3_temporal_time_dependent_ml.docx"),
          "Table 3. Time-dependent discrimination and prediction error of the final ML ensemble in the temporal validation cohort",
          "AUC = time-dependent cumulative/dynamic AUC. Brier score = IPCW Brier score (lower is better). 95% CIs from stratified bootstrap (200 iterations).")

# ==========================================================================
# TABLE 4 – ML vs SENECA (temporal)
# ==========================================================================
cat("=== Table 4: ML vs SENECA ===\n")

make_model_row <- function(tbl_raw, boot_agg, model_label, key_prefix) {
  get_val <- function(metric_name, boot_key) {
    row <- tbl_raw %>% filter(metric == metric_name)
    if (key_prefix == "ml") {
      val <- row$ml[1]
    } else if (key_prefix == "seneca") {
      val <- row$seneca[1]
    } else {
      val <- row$delta[1]
    }
    lo <- boot_agg[[boot_key]]$ci_lower
    hi <- boot_agg[[boot_key]]$ci_upper
    fmt_ci(val, lo, hi, 3)
  }

  c_key   <- paste0("c_index_", key_prefix)
  auc_key <- paste0("mean_auc_", key_prefix)
  ibs_key <- paste0("ibs_overall_", key_prefix)

  data.frame(
    Model                              = model_label,
    `C-index (95% CI)`                 = get_val("harrell_c_index", c_key),
    `Mean time-dependent AUC (95% CI)` = get_val("mean_auc", auc_key),
    `Integrated Brier score (95% CI)`  = get_val("ibs", ibs_key),
    check.names = FALSE, stringsAsFactors = FALSE
  )
}

# Table 4 uses the CC comparison: tbl2_cc_raw + boot_temporal_cc
t4_ml <- make_model_row(tbl2_cc_raw, boot_temporal_cc$aggregated, "ML ensemble", "ml")

# SENECA row
t4_seneca <- make_model_row(tbl2_cc_raw, boot_temporal_cc$aggregated, "SENECA", "seneca")

# Difference row: use comparison results from CC bootstrap
comp_c     <- boot_temporal_cc$comparison_c_index
comp_auc   <- boot_temporal_cc$comparison_mean_auc
diff_agg   <- boot_temporal_cc$aggregated

# Delta values from the CC raw table
delta_c_row   <- tbl2_cc_raw %>% filter(metric == "harrell_c_index")
delta_auc_row <- tbl2_cc_raw %>% filter(metric == "mean_auc")
delta_ibs_row <- tbl2_cc_raw %>% filter(metric == "ibs")

t4_diff <- data.frame(
  Model                              = "Difference (ML \u2212 SENECA)",
  `C-index (95% CI)`                 = fmt_ci(delta_c_row$delta[1],
                                              diff_agg$c_index_diff$ci_lower,
                                              diff_agg$c_index_diff$ci_upper, 3),
  `Mean time-dependent AUC (95% CI)` = fmt_ci(delta_auc_row$delta[1],
                                              diff_agg$mean_auc_diff$ci_lower,
                                              diff_agg$mean_auc_diff$ci_upper, 3),
  `Integrated Brier score (95% CI)`  = "\u2014",
  check.names = FALSE, stringsAsFactors = FALSE
)

t4_df <- bind_rows(t4_ml, t4_seneca, t4_diff)

ft4 <- flextable(t4_df) %>% style_table()
save_docx(ft4, file.path(main_dir, "table_4_temporal_ml_vs_seneca.docx"),
          "Table 4. Comparative overall performance of the final ML ensemble and the SENECA score in temporal validation",
          "Difference row shows bootstrap-based 95% CI for the paired difference (ML minus SENECA). IBS is not computed for SENECA (no survival function). Analysis restricted to SENECA complete cases.")

# ==========================================================================
# TABLE S1 – Candidate models (internal)
# ==========================================================================
cat("=== Table S1: Candidate models ===\n")

# The performance_summary.csv has: cohort, model, metric, value, ci_lower, ci_upper
ps_int <- perf_summary %>% filter(cohort == "internal", model == "ML")

# Model-level results may also be available from the multi-model evaluate step.
# For now, use what is available in the saved tables.
# Candidate model names in order
candidate_names <- c("Penalized Cox" = "cox",
                     "RSF"           = "rsf",
                     "XGBoost-AFT"   = "xgboost",
                     "XGBSE"         = "xgbse",
                     "DeepSurv"      = "deepsurv",
                     "Stacking ensemble" = "ensemble")

# Look for a candidate_models / summary_metrics file in the tables dir
candidate_file <- file.path(args$tables_dir, "summary_metrics.csv")
has_candidates <- file.exists(candidate_file)

if (has_candidates) {
  cand_raw <- read_csv(candidate_file, show_col_types = FALSE)
  # Columns: Model, C-index, Mean AUC, AUC_6m ... IBS Overall
  s1_rows <- lapply(seq_len(nrow(cand_raw)), function(i) {
    r <- cand_raw[i, ]
    data.frame(
      Model                       = r$Model,
      `C-index`                   = fmt_val(r$`C-index`, 3),
      `Mean time-dependent AUC`   = fmt_val(r$`Mean AUC`, 3),
      `Integrated Brier score`    = fmt_val(r$`IBS Overall`, 3),
      check.names = FALSE, stringsAsFactors = FALSE
    )
  })
  s1_df <- bind_rows(s1_rows)
  s1_footnote <- "Point estimates from internal hold-out test set. CIs not available for individual candidate models (bootstrap was performed on ensemble only)."
} else {
  # Fallback: use the ensemble-only data from performance_summary
  s1_df <- data.frame(
    Model                       = "Stacking ensemble",
    `C-index`                   = fmt_val(ps_int %>% filter(metric == "c_index") %>% pull(value), 3),
    `Mean time-dependent AUC`   = fmt_val(ps_int %>% filter(metric == "mean_auc") %>% pull(value), 3),
    `Integrated Brier score`    = fmt_val(ps_int %>% filter(metric == "ibs_overall") %>% pull(value), 3),
    check.names = FALSE, stringsAsFactors = FALSE
  )
  s1_footnote <- "Only ensemble results available from this pipeline run. Individual candidate model results were not found in the tables directory."
}

ft_s1 <- flextable(s1_df) %>% style_table()
save_docx(ft_s1, file.path(supp_dir, "table_s1_candidate_models.docx"),
          "Supplementary Table S1. Overall performance of candidate survival models in the internal hold-out test set",
          s1_footnote)

# ==========================================================================
# TABLE S2 – Time-dependent ML (internal)
# ==========================================================================
cat("=== Table S2: Internal time-dependent ML ===\n")

s2_auc   <- tbl3_raw %>% filter(metric == "auc")
s2_brier <- tbl3_raw %>% filter(metric == "brier")
boot_agg_i <- boot_internal$aggregated

s2_rows <- lapply(c(6, 12, 18, 24), function(tp) {
  auc_row   <- s2_auc   %>% filter(timepoint == tp)
  brier_row <- s2_brier  %>% filter(timepoint == tp)

  auc_boot   <- boot_agg_i[[sprintf("auc_%dm_ml", tp)]]
  brier_boot <- boot_agg_i[[sprintf("brier_%dm_ml", tp)]]

  data.frame(
    `Time horizon`                    = sprintf("%d months", tp),
    `Time-dependent AUC (95% CI)`     = fmt_ci(auc_row$ml[1],
                                               auc_boot$ci_lower, auc_boot$ci_upper, 3),
    `Brier score (95% CI)`            = fmt_ci(brier_row$ml[1],
                                               brier_boot$ci_lower, brier_boot$ci_upper, 3),
    check.names = FALSE, stringsAsFactors = FALSE
  )
})
s2_df <- bind_rows(s2_rows)

ft_s2 <- flextable(s2_df) %>% style_table()
save_docx(ft_s2, file.path(supp_dir, "table_s2_internal_time_dependent_ml.docx"),
          "Supplementary Table S2. Time-dependent discrimination and prediction error of the final ML ensemble in the internal hold-out test set",
          "AUC = time-dependent cumulative/dynamic AUC. Brier score = IPCW Brier score. 95% CIs from stratified bootstrap (200 iterations).")

# ==========================================================================
# TABLE S3 – Calibration summary (temporal)
# ==========================================================================
cat("=== Table S3: Calibration summary ===\n")

s3_rows <- lapply(c(6, 18), function(tp) {
  cal_file <- file.path(figures_csv_dir,
                        sprintf("calibration_metrics_%dm_temporal.csv", tp))
  require_file(cal_file, sprintf("Calibration metrics %dm", tp))
  cm <- read_csv(cal_file, show_col_types = FALSE)

  data.frame(
    Horizon                            = sprintf("%d months", tp),
    `Calibration slope (95% CI)`       = fmt_ci(cm$slope[1],
                                                cm$slope_lower_95[1],
                                                cm$slope_upper_95[1], 2),
    `Calibration intercept (95% CI)`   = fmt_ci(cm$intercept[1],
                                                cm$intercept_lower_95[1],
                                                cm$intercept_upper_95[1], 3),
    `O:E ratio (95% CI)`              = fmt_ci(cm$oe_ratio[1],
                                                cm$oe_ratio_lower_95[1],
                                                cm$oe_ratio_upper_95[1], 2),
    check.names = FALSE, stringsAsFactors = FALSE
  )
})
s3_df <- bind_rows(s3_rows)

ft_s3 <- flextable(s3_df) %>% style_table()
save_docx(ft_s3, file.path(supp_dir, "table_s3_temporal_calibration_summary.docx"),
          "Supplementary Table S3. Calibration summary of the final ML ensemble in temporal validation",
          "Calibration slope and intercept from logistic recalibration. O:E = observed-to-expected ratio. CIs from Wald/Poisson methods on labelable patients.")

# ==========================================================================
# TABLE S4a – Risk groups 15/85 (temporal)
# ==========================================================================
cat("=== Table S4a: Risk groups 15-85 ===\n")

make_risk_group_table <- function(scheme_name) {
  km_file <- file.path(args$tables_dir,
                       sprintf("km_statistics_temporal_%s.csv", scheme_name))
  require_file(km_file, sprintf("KM statistics temporal %s", scheme_name))
  km <- read_csv(km_file, show_col_types = FALSE)

  lapply(seq_len(nrow(km)), function(i) {
    r <- km[i, ]
    n_deaths <- r$events
    pct_deaths <- round(r$event_rate * 100, 1)

    # Median OS
    med_os <- if (r$median_survival == "NR" || is.na(r$median_survival)) {
      "NR"
    } else {
      med_val <- as.numeric(r$median_survival)
      med_lo  <- r$median_ci_lower
      med_hi  <- r$median_ci_upper
      # Handle NR in CI bounds
      lo_str <- if (is.na(med_lo) || med_lo == "NR") "NR" else fmt_val(as.numeric(med_lo), 1)
      hi_str <- if (is.na(med_hi) || med_hi == "NR") "NR" else fmt_val(as.numeric(med_hi), 1)
      sprintf("%s (%s\u2013%s)", fmt_val(med_val, 1), lo_str, hi_str)
    }

    # Survival probabilities at 6 and 18 months
    surv_6m  <- if ("surv_6m"  %in% names(r) && !is.na(r$surv_6m))  fmt_val(r$surv_6m,  2) else "\u2014"
    surv_18m <- if ("surv_18m" %in% names(r) && !is.na(r$surv_18m)) fmt_val(r$surv_18m, 2) else "\u2014"

    data.frame(
      `Risk group`                          = r$group,
      n                                     = r$n,
      `Deaths, n (%)`                       = sprintf("%d (%s%%)", n_deaths, fmt_val(pct_deaths, 1)),
      `Median OS, months (95% CI)`          = med_os,
      `6-month OS probability`              = surv_6m,
      `18-month OS probability`             = surv_18m,
      check.names = FALSE, stringsAsFactors = FALSE
    )
  }) %>% bind_rows()
}

s4a_df <- make_risk_group_table("15-85")
ft_s4a <- flextable(s4a_df) %>% style_table()
save_docx(ft_s4a, file.path(supp_dir, "table_s4a_temporal_risk_groups_15_85.docx"),
          "Supplementary Table S4a. Survival outcomes by ML-derived risk group in temporal validation using 15th/85th percentile thresholds",
          "Risk groups defined by training-set percentiles applied to temporal validation. OS = overall survival. Median OS from Kaplan-Meier estimator (months). Survival probabilities are Kaplan-Meier estimates.")

# ==========================================================================
# TABLE S4b – Risk groups 33/66 (temporal)
# ==========================================================================
cat("=== Table S4b: Risk groups 33-66 ===\n")

s4b_df <- make_risk_group_table("33-66")
ft_s4b <- flextable(s4b_df) %>% style_table()
save_docx(ft_s4b, file.path(supp_dir, "table_s4b_temporal_risk_groups_33_66.docx"),
          "Supplementary Table S4b. Survival outcomes by ML-derived risk group in temporal validation using 33rd/66th percentile thresholds",
          "Risk groups defined by training-set percentiles applied to temporal validation. OS = overall survival. Median OS from Kaplan-Meier estimator (months). Survival probabilities are Kaplan-Meier estimates.")

# ==========================================================================
# MLflow Logging
# ==========================================================================
cat("=== MLflow Logging ===\n")

tryCatch({
  suppressPackageStartupMessages(library(reticulate))
  use_python("/opt/pyenv/bin/python", required = TRUE)
  mlflow <- import("mlflow")
  cat("  MLflow client loaded\n")

  docx_files <- list.files(args$output_dir, pattern = "\\.docx$",
                           recursive = TRUE, full.names = TRUE)
  for (f in docx_files) {
    rel_path <- sub(paste0(args$output_dir, "/?"), "", f)
    artifact_dir <- dirname(rel_path)
    mlflow$log_artifact(f, artifact_path = artifact_dir)
    cat(sprintf("  logged %s\n", rel_path))
  }
  cat(sprintf("  %d .docx files logged to MLflow\n", length(docx_files)))

}, error = function(e) {
  cat("  MLflow logging skipped:", conditionMessage(e), "\n")
})

cat("=== Publication Tables Complete ===\n")
