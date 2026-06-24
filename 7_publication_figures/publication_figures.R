#!/usr/bin/env Rscript
# =============================================================================
# Publication-Quality Figures — Unified R Step
# =============================================================================
#
# Reads pre-computed metrics and prediction CSVs from the compute_metrics step
# and produces all publication-quality figures (PNG 300 DPI + PDF vector).
#
# Output structure:
#   output_dir/
#     png/          — raster figures (300 DPI)
#     pdf/          — vector figures (cairo_pdf)
#     csv/          — numeric tables
#
# Figures produced:
#   1. Calibration (6 m + 18 m, temporal only, combined panel)
#   2. DCA (6 m + 18 m, temporal only, combined panel)
#   3. Kaplan-Meier  — per risk-stratification scheme, numbers at risk
#   4. ROC curves (6 m + 18 m combined panels, per cohort)
#   5. PR curves  (6 m + 18 m combined panels; 18 m uses inverted positive class)
#   6. Temporal AUC with CI — line plot with bootstrap bands
#   7. Temporal Brier scores — line plot
#   8. Risk-score distributions:
#      a) Prognostic score: internal vs temporal overlay
#      b) P(T <= t) at 6 m and 18 m
#      c) IPCW-stratified P(T <= t) by event status at t
# =============================================================================

suppressPackageStartupMessages({
  library(argparse)
  library(readr)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(survival)
  library(survminer)
  library(dcurves)
  library(ggplot2)
  library(patchwork)
  library(riskRegression)
  library(gridExtra)
})

# ── Theme ────────────────────────────────────────────────────────────────────
theme_pub <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(
      axis.title         = element_text(face = "bold"),
      axis.text          = element_text(color = "black"),
      plot.title         = element_text(face = "bold", size = base_size + 2),
      plot.subtitle      = element_text(color = "grey40"),
      legend.position    = "bottom",
      legend.title       = element_blank(),
      strip.text         = element_text(face = "bold"),
      strip.background   = element_blank()
    )
}

col_ml        <- "#0072B2"
col_seneca    <- "#D55E00"
col_low       <- "#4DAF4A"
col_medium    <- "#FF7F00"
col_high      <- "#E41A1C"
col_reference <- "grey50"
col_internal  <- "#0072B2"
col_temporal  <- "#D55E00"

# ── CLI ──────────────────────────────────────────────────────────────────────
parser <- ArgumentParser(description = "Publication-quality figures for BTC validation")
parser$add_argument("--metrics_dir",  required = TRUE, help = "metrics/ subfolder from compute_metrics")
parser$add_argument("--r_input_dir",  required = TRUE, help = "r_input/ subfolder from compute_metrics")
parser$add_argument("--tables_dir",   required = TRUE, help = "tables/ subfolder from compute_metrics")
parser$add_argument("--python_figures_dir", required = TRUE, help = "Python-generated figures from compute_metrics")
parser$add_argument("--shap_figures_dir",  default = NULL, help = "SHAP figures directory (optional)")
parser$add_argument("--output_dir",   required = TRUE, help = "Output folder for all figures")
args <- parser$parse_args()

# Create subdirectories
png_dir <- file.path(args$output_dir, "png")
pdf_dir <- file.path(args$output_dir, "pdf")
csv_dir <- file.path(args$output_dir, "csv")
dir.create(png_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(pdf_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(csv_dir, recursive = TRUE, showWarnings = FALSE)

save_fig <- function(plot_obj, name, w = 7, h = 6) {
  png_path <- file.path(png_dir, paste0(name, ".png"))
  pdf_path <- file.path(pdf_dir, paste0(name, ".pdf"))
  ggsave(png_path, plot_obj, width = w, height = h, dpi = 300)
  ggsave(pdf_path, plot_obj, width = w, height = h, device = cairo_pdf)
  cat(sprintf("  saved %s (.png + .pdf)\n", name))
}

save_table <- function(tbl, name) {
  csv_path <- file.path(csv_dir, paste0(name, ".csv"))
  write_csv(tbl, csv_path)
  cat(sprintf("  saved %s.csv\n", name))
}

# ==========================================================================
# 1. CALIBRATION
# ==========================================================================
cat("=== Calibration ===\n")

# ------------ helpers (from original dca_analysis.R) ----------------------
km_risk_at_time <- function(data_subset, t_horizon) {
  if (nrow(data_subset) == 0) return(NA_real_)
  km_fit <- survfit(Surv(tte, event) ~ 1, data = data_subset)
  s <- summary(km_fit, times = t_horizon)
  if (length(s$surv) == 1 && !is.na(s$surv)) return(1 - s$surv)
  if (max(data_subset$tte, na.rm = TRUE) < t_horizon) {
    last_surv <- tail(summary(km_fit)$surv, 1)
    if (length(last_surv) == 1 && !is.na(last_surv)) return(1 - last_surv)
  }
  NA_real_
}

build_calibration_table <- function(df, pred_col, t_horizon, n_groups = 10) {
  tmp <- df %>% mutate(cal_group = ntile(.data[[pred_col]], n_groups))
  tmp %>%
    group_by(cal_group) %>%
    summarise(
      n          = n(),
      mean_predicted = mean(.data[[pred_col]], na.rm = TRUE),
      observed   = km_risk_at_time(pick(everything()), t_horizon),
      .groups    = "drop"
    ) %>%
    mutate(difference = observed - mean_predicted,
           timepoint_months = t_horizon)
}

make_horizon_labelable_data <- function(df, horizon, pred_col, eps = 1e-6) {
  df %>%
    transmute(
      tte     = tte,
      event   = event,
      pred    = pmin(pmax(.data[[pred_col]], eps), 1 - eps),
      outcome = case_when(
        event == 1 & tte <= horizon ~ 1,
        tte > horizon               ~ 0,
        TRUE                        ~ NA_real_
      )
    ) %>%
    filter(!is.na(outcome)) %>%
    mutate(lp = qlogis(pred))
}

compute_calibration_metrics <- function(df, horizon, pred_col) {
  dat <- make_horizon_labelable_data(df, horizon, pred_col)
  if (nrow(dat) < 5) {
    return(data.frame(
      timepoint_months = horizon,
      n_labelable      = nrow(dat),
      intercept        = NA_real_, intercept_lower_95 = NA_real_, intercept_upper_95 = NA_real_,
      slope            = NA_real_, slope_lower_95     = NA_real_, slope_upper_95     = NA_real_,
      oe_ratio         = NA_real_, oe_ratio_lower_95  = NA_real_, oe_ratio_upper_95  = NA_real_
    ))
  }

  z <- qnorm(0.975)

  # Calibration-in-the-large (intercept)
  fit_int <- glm(outcome ~ 1 + offset(lp), family = binomial(), data = dat)
  int_est <- unname(coef(fit_int)["(Intercept)"])
  int_se  <- sqrt(diag(vcov(fit_int)))["(Intercept)"]

  # Calibration slope
  fit_slp <- glm(outcome ~ lp, family = binomial(), data = dat)
  slp_est <- unname(coef(fit_slp)["lp"])
  slp_se  <- sqrt(diag(vcov(fit_slp)))["lp"]

  # O:E ratio (observed / expected)
  o <- sum(dat$outcome)
  e_sum <- sum(dat$pred)
  oe <- o / e_sum
  oe_se <- sqrt(o) / e_sum  # Poisson-based SE
  oe_lo <- oe * exp(-z * oe_se / oe)
  oe_hi <- oe * exp( z * oe_se / oe)

  data.frame(
    timepoint_months  = horizon,
    n_labelable       = nrow(dat),
    intercept         = round(int_est, 4),
    intercept_lower_95 = round(int_est - z * int_se, 4),
    intercept_upper_95 = round(int_est + z * int_se, 4),
    slope             = round(slp_est, 4),
    slope_lower_95    = round(slp_est - z * slp_se, 4),
    slope_upper_95    = round(slp_est + z * slp_se, 4),
    oe_ratio          = round(oe, 4),
    oe_ratio_lower_95 = round(oe_lo, 4),
    oe_ratio_upper_95 = round(oe_hi, 4)
  )
}

build_calibration_plot <- function(cal_tbl, cal_metrics, title) {
  # Smoothed calibration curve + decile points + numeric annotation
  p <- ggplot(cal_tbl, aes(x = mean_predicted, y = observed)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = col_reference) +
    geom_smooth(method = "loess", se = TRUE, color = col_ml, fill = col_ml, alpha = 0.15) +
    geom_point(shape = 21, size = 3, fill = col_ml, color = "black", stroke = 0.3) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    scale_x_continuous(breaks = seq(0, 1, 0.1)) +
    scale_y_continuous(breaks = seq(0, 1, 0.1)) +
    labs(title = title,
         x = "Predicted probability",
         y = "Observed probability (KM)") +
    theme_pub()

  if (nrow(cal_metrics) > 0 && !is.na(cal_metrics$slope[1])) {
    lbl <- sprintf(
      "Slope = %.2f [%.2f, %.2f]\nIntercept = %.3f [%.3f, %.3f]\nO:E = %.2f [%.2f, %.2f]\nn = %d",
      cal_metrics$slope[1], cal_metrics$slope_lower_95[1], cal_metrics$slope_upper_95[1],
      cal_metrics$intercept[1], cal_metrics$intercept_lower_95[1], cal_metrics$intercept_upper_95[1],
      cal_metrics$oe_ratio[1], cal_metrics$oe_ratio_lower_95[1], cal_metrics$oe_ratio_upper_95[1],
      cal_metrics$n_labelable[1]
    )
    p <- p + annotate("text", x = 0.02, y = 0.95, label = lbl,
                      hjust = 0, vjust = 1, size = 3, family = "mono",
                      color = "black")
  }
  p
}

# -- Temporal cohort only, 6 m + 18 m --
cal_file <- file.path(args$r_input_dir, "calibration_input_temporal.csv")
if (file.exists(cal_file)) {
  cal_df <- read_csv(cal_file, show_col_types = FALSE)

  cal_panel <- list()
  for (tp in c(6, 18)) {
    pred_col <- paste0("p_event_", tp, "m")
    if (!(pred_col %in% names(cal_df))) next

    ct <- build_calibration_table(cal_df, pred_col, tp)
    cm <- compute_calibration_metrics(cal_df, tp, pred_col)

    save_table(cm, sprintf("calibration_metrics_%dm_temporal", tp))
    save_table(ct, sprintf("calibration_deciles_%dm_temporal", tp))

    cal_panel[[as.character(tp)]] <- build_calibration_plot(
      ct, cm, sprintf("%d Months", tp)
    )
  }
  if (length(cal_panel) == 2) {
    p_comb <- (cal_panel[["6"]] | cal_panel[["18"]]) +
      plot_annotation(
        title = "Calibration — Temporal Validation",
        theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
      )
    save_fig(p_comb, "calibration_combined_temporal", w = 14, h = 6)
  }
} else {
  cat("  skipping calibration (file not found)\n")
}

# ==========================================================================
# 2. DECISION CURVE ANALYSIS (temporal only, 6 m + 18 m combined)
# ==========================================================================
cat("=== DCA ===\n")

cal_file_dca <- file.path(args$r_input_dir, "calibration_input_temporal.csv")
if (file.exists(cal_file_dca)) {
  dca_df <- read_csv(cal_file_dca, show_col_types = FALSE)

  dca_plots <- list()
  for (tp in c(6, 18)) {
    pred_col <- paste0("p_event_", tp, "m")
    if (!(pred_col %in% names(dca_df))) next

    formula_str <- sprintf("Surv(tte, event) ~ %s", pred_col)
    lbl_list    <- setNames(list("ML Ensemble"), pred_col)

    dca_obj <- dca(
      as.formula(formula_str),
      data       = dca_df,
      time       = tp,
      thresholds = seq(0.01, 0.99, 0.01),
      label      = lbl_list
    )

    p <- plot(dca_obj, smooth = TRUE) +
      labs(
        title = sprintf("%d Months", tp),
        x     = "Threshold Probability",
        y     = "Net Benefit"
      ) +
      coord_cartesian(ylim = c(-0.05, 0.50)) +
      scale_x_continuous(breaks = seq(0, 1, 0.2)) +
      theme_pub()

    dca_plots[[as.character(tp)]] <- p

    nb_tbl <- as.data.frame(dca_obj$dca)
    save_table(nb_tbl, sprintf("net_benefit_%dm_temporal", tp))
  }

  if (length(dca_plots) == 2) {
    p_comb <- (dca_plots[["6"]] | dca_plots[["18"]]) +
      plot_annotation(
        title = "Decision Curve Analysis — Temporal Validation",
        theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
      )
    save_fig(p_comb, "dca_combined_temporal", w = 14, h = 6)
  }
} else {
  cat("  skipping DCA (file not found)\n")
}

# ==========================================================================
# 2b. SUPPLEMENTARY DCA — highlighted clinical relevance ranges
# ==========================================================================
cat("=== Supplementary DCA (highlighted ranges) ===\n")

if (file.exists(cal_file_dca)) {
  # Reuse dca_df already loaded above
  dca_supp_plots <- list()

  # Per-timepoint highlight ranges (threshold probability)
  highlight_ranges <- list(
    "6"  = c(0.60, 0.95),
    "18" = c(0.35, 0.80)
  )

  for (tp in c(6, 18)) {
    pred_col <- paste0("p_event_", tp, "m")
    if (!(pred_col %in% names(dca_df))) next

    formula_str <- sprintf("Surv(tte, event) ~ %s", pred_col)
    lbl_list    <- setNames(list("ML Ensemble"), pred_col)

    dca_obj <- dca(
      as.formula(formula_str),
      data       = dca_df,
      time       = tp,
      thresholds = seq(0.01, 0.99, 0.01),
      label      = lbl_list
    )

    hr <- highlight_ranges[[as.character(tp)]]

    p <- plot(dca_obj, smooth = TRUE) +
      annotate(
        "rect",
        xmin = hr[1], xmax = hr[2],
        ymin = -Inf,  ymax = Inf,
        fill = "#87CEEB", alpha = 0.25
      ) +
      labs(
        title    = sprintf("%d Months", tp),
        subtitle = sprintf("Highlighted range: %.0f%%–%.0f%%", hr[1] * 100, hr[2] * 100),
        x        = "Threshold Probability",
        y        = "Net Benefit"
      ) +
      coord_cartesian(ylim = c(-0.05, 0.50)) +
      scale_x_continuous(breaks = seq(0, 1, 0.1)) +
      theme_pub()

    dca_supp_plots[[as.character(tp)]] <- p
  }

  if (length(dca_supp_plots) == 2) {
    p_supp <- (dca_supp_plots[["6"]] | dca_supp_plots[["18"]]) +
      plot_annotation(
        title = "Supplementary DCA — Clinical Relevance Ranges (Temporal Validation)",
        theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
      )
    save_fig(p_supp, "dca_supplementary_highlighted", w = 14, h = 6)
  }
} else {
  cat("  skipping supplementary DCA (file not found)\n")
}

# ==========================================================================
# 3. KAPLAN-MEIER CURVES (generated in Python — copied below)
# ==========================================================================
cat("=== Kaplan-Meier (generated in Python, will be copied to output) ===\n")

# ==========================================================================
# 4. ROC CURVES (6 m + 18 m combined panels, per cohort)
# ==========================================================================
cat("=== ROC Curves ===\n")

roc_files <- list.files(args$r_input_dir, pattern = "^roc_curves_.*m\\.csv$", full.names = TRUE)

# Group ROC files by cohort and timepoint
roc_groups <- list()
for (roc_file in roc_files) {
  bn <- basename(roc_file)
  roc_df_tmp <- read_csv(roc_file, show_col_types = FALSE)
  tp_val <- unique(roc_df_tmp$timepoint)[1]
  if (!(tp_val %in% c(6, 18))) next

  cohort_key <- ifelse(grepl("internal", bn), "internal", "temporal")
  key <- paste0(cohort_key, "_", tp_val)
  roc_groups[[key]] <- list(df = roc_df_tmp, tp = tp_val, cohort = cohort_key)
}

build_roc_panel <- function(roc_df, tp_val) {
  auc_tbl <- roc_df %>%
    group_by(model) %>%
    summarise(auc_val = first(auc), .groups = "drop")
  roc_df <- roc_df %>%
    left_join(auc_tbl, by = "model") %>%
    mutate(label = sprintf("%s (AUC = %.3f)", model, auc_val))

  model_names <- unique(roc_df$model)
  roc_colors  <- if (length(model_names) == 1) c(col_ml) else c(col_ml, col_seneca)

  ggplot(roc_df, aes(x = fpr, y = tpr, color = label)) +
    geom_line(linewidth = 0.8) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
    scale_color_manual(values = roc_colors) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = sprintf("%d Months", tp_val),
         x = "1 − Specificity", y = "Sensitivity") +
    theme_pub()
}

for (cohort_name in c("internal", "temporal")) {
  k6  <- paste0(cohort_name, "_6")
  k18 <- paste0(cohort_name, "_18")
  panels <- list()
  if (!is.null(roc_groups[[k6]]))  panels[["6"]]  <- build_roc_panel(roc_groups[[k6]]$df, 6)
  if (!is.null(roc_groups[[k18]])) panels[["18"]] <- build_roc_panel(roc_groups[[k18]]$df, 18)

  if (length(panels) == 2) {
    cohort_label <- ifelse(cohort_name == "internal", "Internal Test", "Temporal Validation")
    p_comb <- (panels[["6"]] | panels[["18"]]) +
      plot_annotation(
        title = sprintf("ROC Curves (IPCW) — %s", cohort_label),
        theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
      )
    save_fig(p_comb, sprintf("roc_combined_%s", cohort_name), w = 14, h = 6)
  } else if (length(panels) == 1) {
    save_fig(panels[[1]], sprintf("roc_%s", cohort_name))
  }
}

# ==========================================================================
# 5. PR CURVES (6 m + 18 m combined panels;
#    18 m uses inverted positive class: "no event by 18 m")
# ==========================================================================
cat("=== PR Curves ===\n")

pr_files <- list.files(args$r_input_dir, pattern = "^pr_curves_.*m\\.csv$", full.names = TRUE)

pr_groups <- list()
for (pr_file in pr_files) {
  pr_df_tmp <- read_csv(pr_file, show_col_types = FALSE)
  tp_val <- unique(pr_df_tmp$timepoint)[1]
  if (!(tp_val %in% c(6, 18))) next
  pr_groups[[as.character(tp_val)]] <- pr_df_tmp
}

build_pr_panel <- function(pr_df, tp_val) {
  ap_tbl <- pr_df %>%
    group_by(model) %>%
    summarise(ap_val = first(average_precision), .groups = "drop")
  pr_df <- pr_df %>%
    left_join(ap_tbl, by = "model") %>%
    mutate(label = sprintf("%s (AP = %.3f)", model, ap_val))

  model_names <- unique(pr_df$model)
  pr_colors   <- if (length(model_names) == 1) c(col_ml) else c(col_ml, col_seneca)

  # Axis labels depend on whether positive class is event or survival
  is_inverted <- ("inverted" %in% names(pr_df)) && any(pr_df$inverted == TRUE, na.rm = TRUE)
  if (is_inverted) {
    ylab <- "Precision (NPV)"
    xlab <- "Recall (Specificity)"
    title_str <- sprintf("P(survival > %d mo)", tp_val)
  } else {
    ylab <- "Precision (PPV)"
    xlab <- "Recall (Sensitivity)"
    title_str <- sprintf("P(event ≤ %d mo)", tp_val)
  }

  ggplot(pr_df, aes(x = recall, y = precision, color = label)) +
    geom_line(linewidth = 0.8) +
    scale_color_manual(values = pr_colors) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = title_str, x = xlab, y = ylab) +
    theme_pub()
}

pr_panels <- list()
if (!is.null(pr_groups[["6"]]))  pr_panels[["6"]]  <- build_pr_panel(pr_groups[["6"]], 6)
if (!is.null(pr_groups[["18"]])) pr_panels[["18"]] <- build_pr_panel(pr_groups[["18"]], 18)

if (length(pr_panels) == 2) {
  p_comb <- (pr_panels[["6"]] | pr_panels[["18"]]) +
    plot_annotation(
      title = "Precision-Recall (IPCW) — Temporal Validation",
      theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
    )
  save_fig(p_comb, "pr_combined_temporal", w = 14, h = 6)
} else if (length(pr_panels) == 1) {
  save_fig(pr_panels[[1]], "pr_temporal")
}

# ==========================================================================
# 6. TEMPORAL AUC — per-cohort plots
# ==========================================================================
cat("=== Temporal AUC ===\n")

auc_file <- file.path(args$r_input_dir, "temporal_auc.csv")
if (file.exists(auc_file)) {
  auc_df <- read_csv(auc_file, show_col_types = FALSE)

  # ---- 6a. Internal (ML only) -------------------------------------------
  auc_int <- auc_df %>% filter(cohort == "internal")
  if (nrow(auc_int) > 0) {
    p_int <- ggplot(auc_int, aes(x = timepoint, y = auc, color = model)) +
      geom_line(linewidth = 0.9) +
      geom_point(size = 2.5) +
      {if (all(c("ci_lower", "ci_upper") %in% names(auc_int)) &&
           any(!is.na(auc_int$ci_lower)))
        geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper, fill = model),
                    alpha = 0.15, color = NA)
      } +
      scale_color_manual(values = c("ML" = col_ml, "SENECA" = col_seneca)) +
      scale_fill_manual(values  = c("ML" = col_ml, "SENECA" = col_seneca)) +
      scale_x_continuous(breaks = c(6, 12, 18, 24)) +
      coord_cartesian(ylim = c(0.5, 1.0)) +
      labs(title = "Internal Test Set",
           x = "Time (months)", y = "AUC") +
      theme_pub()
    save_fig(p_int, "temporal_auc_internal", w = 9, h = 6)
  }

  # ---- 6b. Temporal (ML + SENECA) ----------------------------------------
  auc_temp <- auc_df %>% filter(cohort == "temporal")
  if (nrow(auc_temp) > 0) {
    p_temp <- ggplot(auc_temp, aes(x = timepoint, y = auc, color = model)) +
      geom_line(linewidth = 0.9) +
      geom_point(size = 2.5) +
      geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper, fill = model),
                  alpha = 0.15, color = NA) +
      scale_color_manual(values = c("ML" = col_ml, "SENECA" = col_seneca)) +
      scale_fill_manual(values  = c("ML" = col_ml, "SENECA" = col_seneca)) +
      scale_x_continuous(breaks = c(6, 12, 18, 24)) +
      coord_cartesian(ylim = c(0.5, 1.0)) +
      labs(title = "Temporal Validation",
           x = "Time (months)", y = "AUC") +
      theme_pub()
    save_fig(p_temp, "temporal_auc_temporal", w = 9, h = 6)
  }
}

# ==========================================================================
# 7. TEMPORAL BRIER — per-cohort plots with CI bands
# ==========================================================================
cat("=== Temporal Brier ===\n")

brier_file <- file.path(args$r_input_dir, "temporal_brier.csv")
if (file.exists(brier_file)) {
  brier_df <- read_csv(brier_file, show_col_types = FALSE)

  for (coh in c("internal", "temporal")) {
    brier_coh <- brier_df %>% filter(cohort == coh)
    if (nrow(brier_coh) == 0) next

    cohort_label <- ifelse(coh == "internal", "Internal Test Set", "Temporal Validation")

    has_ci <- all(c("ci_lower", "ci_upper") %in% names(brier_coh)) &&
              any(!is.na(brier_coh$ci_lower))

    p <- ggplot(brier_coh, aes(x = timepoint, y = brier, color = model)) +
      geom_line(linewidth = 0.9) +
      geom_point(size = 2.5) +
      {if (has_ci)
        geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper, fill = model),
                    alpha = 0.15, color = NA)
      } +
      scale_color_manual(values = c("ML" = col_ml)) +
      scale_fill_manual(values  = c("ML" = col_ml)) +
      scale_x_continuous(breaks = c(6, 12, 18, 24)) +
      coord_cartesian(ylim = c(0, 0.35)) +
      labs(title = sprintf("Brier Scores (IPCW) — %s", cohort_label),
           x = "Time (months)", y = "Brier Score") +
      theme_pub()

    save_fig(p, sprintf("temporal_brier_%s", coh), w = 9, h = 6)
  }
}

# ==========================================================================
# 8. RISK SCORE DISTRIBUTIONS (3 types)
# ==========================================================================
cat("=== Risk Score Distributions ===\n")

# 8a. Global prognostic score: internal vs temporal overlay
dens_int_file  <- file.path(args$r_input_dir, "risk_density_internal.csv")
dens_temp_file <- file.path(args$r_input_dir, "risk_density_temporal.csv")
if (file.exists(dens_int_file) && file.exists(dens_temp_file)) {
  dens_int  <- read_csv(dens_int_file,  show_col_types = FALSE) %>% mutate(cohort = "Internal")
  dens_temp <- read_csv(dens_temp_file, show_col_types = FALSE) %>% mutate(cohort = "Temporal")
  dens_all  <- bind_rows(dens_int, dens_temp)

  # Load 15-85 risk thresholds for vertical lines
  thresh_file <- file.path(args$tables_dir, "risk_thresholds_15-85.json")
  thresh_lines <- list()
  if (file.exists(thresh_file)) {
    thresh <- fromJSON(thresh_file)
    thresh_lines$low  <- thresh$low_threshold
    thresh_lines$high <- thresh$high_threshold
    cat(sprintf("  thresholds 15-85: low=%.4f, high=%.4f\n", thresh_lines$low, thresh_lines$high))
  } else {
    cat("  risk_thresholds_15-85.json not found, skipping threshold lines\n")
  }

  p <- ggplot(dens_all, aes(x = ml_risk_score, fill = cohort, color = cohort)) +
    geom_density(alpha = 0.3, linewidth = 0.6) +
    scale_fill_manual(values = c("Internal" = col_internal, "Temporal" = col_temporal)) +
    scale_color_manual(values = c("Internal" = col_internal, "Temporal" = col_temporal)) +
    labs(title = "Prognostic Score Distribution",
         x = "ML Risk Score", y = "Density") +
    theme_pub()

  if (length(thresh_lines) == 2) {
    p <- p +
      geom_vline(xintercept = thresh_lines$low,  linetype = "dashed", color = col_low,  linewidth = 0.7) +
      geom_vline(xintercept = thresh_lines$high, linetype = "dashed", color = col_high, linewidth = 0.7) +
      annotate("text", x = thresh_lines$low,  y = Inf, label = "Low / Medium", vjust = 1.5, hjust = 1.05, size = 3, fontface = "bold", color = col_low) +
      annotate("text", x = thresh_lines$high, y = Inf, label = "Medium / High", vjust = 1.5, hjust = -0.05, size = 3, fontface = "bold", color = col_high)
  }

  save_fig(p, "risk_score_distribution_overlay")
}

# 8b. Horizon-specific P(T <= t) distributions at 6 m and 18 m
pred_dist_file <- file.path(args$r_input_dir, "predicted_risk_distributions.csv")
if (file.exists(pred_dist_file)) {
  pred_dist <- read_csv(pred_dist_file, show_col_types = FALSE)

  for (tp in c(6, 18)) {
    pcol <- paste0("p_event_", tp, "m")
    if (!(pcol %in% names(pred_dist))) next

    df_tp <- pred_dist %>% filter(!is.na(.data[[pcol]]))

    p_overall <- ggplot(df_tp, aes(x = .data[[pcol]], fill = cohort, color = cohort)) +
      geom_density(alpha = 0.3, linewidth = 0.6) +
      scale_fill_manual(values = c("internal" = col_internal, "temporal" = col_temporal)) +
      scale_color_manual(values = c("internal" = col_internal, "temporal" = col_temporal)) +
      coord_cartesian(ylim = c(0, 5)) +
      labs(title = sprintf("P(T ≤ %d mo) Distribution", tp),
           x = sprintf("Predicted P(event ≤ %d months)", tp), y = "Density") +
      theme_pub()
    save_fig(p_overall, sprintf("predicted_risk_%dm", tp))
  }
}

# 8c. IPCW-stratified P(T <= t) by event status
ipcw_dist_file <- file.path(args$r_input_dir, "ipcw_stratified_distributions.csv")
if (file.exists(ipcw_dist_file)) {
  ipcw_dist <- read_csv(ipcw_dist_file, show_col_types = FALSE)

  for (tp in c(6, 18)) {
    pcol <- paste0("p_event_", tp, "m")
    if (!(pcol %in% names(ipcw_dist))) next

    df_tp <- ipcw_dist %>%
      filter(timepoint == tp, !is.na(.data[[pcol]])) %>%
      mutate(status_label = factor(event_status,
                                   levels = c("event", "no_event"),
                                   labels = c(sprintf("Event by %d mo", tp),
                                              sprintf("No event by %d mo", tp))))

    p <- ggplot(df_tp, aes(x = .data[[pcol]], fill = status_label, weight = ipcw_weight)) +
      geom_density(alpha = 0.4, linewidth = 0.6) +
      scale_fill_manual(values = c(col_high, col_low)) +
      scale_x_continuous(limits = c(0, 1)) +
      coord_cartesian(ylim = c(0, 5)) +
      labs(title = sprintf("IPCW-Weighted P(T ≤ %d mo) by Event Status — Temporal", tp),
           x = sprintf("Predicted P(event ≤ %d months)", tp), y = "Weighted Density") +
      theme_pub()
    save_fig(p, sprintf("ipcw_risk_stratified_%dm", tp))
  }
}

# ==========================================================================
# COPY PYTHON-GENERATED FIGURES (KM + SHAP) INTO OUTPUT
# ==========================================================================
cat("=== Copying Python-generated figures ===\n")

# Copy KM figures from compute_metrics
km_dir <- file.path(args$python_figures_dir, "km")
if (dir.exists(km_dir)) {
  km_pngs <- list.files(km_dir, pattern = "\\.png$", full.names = TRUE)
  km_pdfs <- list.files(km_dir, pattern = "\\.pdf$", full.names = TRUE)
  file.copy(km_pngs, png_dir, overwrite = TRUE)
  file.copy(km_pdfs, pdf_dir, overwrite = TRUE)
  cat(sprintf("  copied %d KM PNGs + %d PDFs from Python\n", length(km_pngs), length(km_pdfs)))
} else {
  cat("  KM figures directory not found, skipping\n")
}

# Copy SHAP figures (optional step)
if (!is.null(args$shap_figures_dir) && nchar(args$shap_figures_dir) > 0) {
  shap_png_src <- file.path(args$shap_figures_dir, "figures", "png")
  shap_pdf_src <- file.path(args$shap_figures_dir, "figures", "pdf")
  if (dir.exists(shap_png_src)) {
    shap_pngs <- list.files(shap_png_src, pattern = "\\.png$", full.names = TRUE)
    file.copy(shap_pngs, png_dir, overwrite = TRUE)
    cat(sprintf("  copied %d SHAP PNGs\n", length(shap_pngs)))
  }
  if (dir.exists(shap_pdf_src)) {
    shap_pdfs <- list.files(shap_pdf_src, pattern = "\\.pdf$", full.names = TRUE)
    file.copy(shap_pdfs, pdf_dir, overwrite = TRUE)
    cat(sprintf("  copied %d SHAP PDFs\n", length(shap_pdfs)))
  }
} else {
  cat("  SHAP figures not provided (optional step), skipping\n")
}

# ==========================================================================
# MLflow Logging
# ==========================================================================
cat("=== MLflow Logging ===\n")

tryCatch({
  suppressPackageStartupMessages(library(reticulate))
  use_python("/opt/pyenv/bin/python", required = TRUE)
  mlflow <- import("mlflow")
  cat("  MLflow client loaded\n")

  for (subdir in c("png", "pdf", "csv")) {
    sub_path <- file.path(args$output_dir, subdir)
    sub_files <- list.files(sub_path, full.names = TRUE)
    for (f in sub_files) {
      mlflow$log_artifact(f, artifact_path = paste0("figures/", subdir))
    }
    cat(sprintf("  %d artifacts logged under 'figures/%s/'\n", length(sub_files), subdir))
  }

}, error = function(e) {
  cat("  MLflow logging skipped:", conditionMessage(e), "\n")
})

cat("=== Publication Figures Complete ===\n")
