#!/usr/bin/env Rscript
# Figures of the temporal validation, from the outputs of compute_metrics.py.
#
#   Figure 2A-B  decision curves at 6 and 18 months (dca_supplementary_highlighted)
#   Figure 2C-D  calibration at 6 and 18 months (calibration_combined_temporal)
#   Fig. S3A-B   IPCW-weighted predicted risk by event status (ipcw_risk_stratified_*)
#   Fig. S3C-D   predicted risk, internal vs temporal cohort (predicted_risk_*)
#
# Also drawn, not in the paper: ROC and precision-recall curves, AUC and Brier
# score over time, risk-score densities, and the same figures for the Cox
# benchmark (*_cox). The Kaplan-Meier figures (Figure 3) and the SHAP figure
# (Figure 4) are made in Python and copied to the output folder. Panels were
# assembled into the published figures by hand.
#
# Output: png/ (300 dpi), pdf/, csv/ (numbers behind the figures, including
# the calibration summary reported in the Results).

suppressPackageStartupMessages({
  library(argparse)
  library(readr)
  library(jsonlite)
  library(dplyr)
  library(survival)
  library(dcurves)
  library(ggplot2)
  library(patchwork)
})

theme_pub <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(axis.title = element_text(face = "bold"), axis.text = element_text(color = "black"),
          plot.title = element_text(face = "bold", size = base_size + 2),
          plot.subtitle = element_text(color = "grey40"), legend.position = "bottom",
          legend.title = element_blank(), strip.text = element_text(face = "bold"),
          strip.background = element_blank())
}
title_theme <- theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))

col_ml <- "#0072B2"; col_seneca <- "#D55E00"; col_cox <- "#CC79A7"
col_low <- "#4DAF4A"; col_high <- "#E41A1C"; col_reference <- "grey50"
col_internal <- "#0072B2"; col_temporal <- "#D55E00"
model_colors <- c("ML" = col_ml, "Cox PH" = col_cox, "SENECA" = col_seneca)
relabel_models <- function(model) ifelse(model == "Cox", "Cox PH", model)
order_models <- function(model) factor(model, levels = intersect(names(model_colors), unique(model)))

parser <- ArgumentParser(description = "Figures of the temporal validation")
parser$add_argument("--metrics_dir", required = TRUE, help = "metrics/ of compute_metrics.py")
parser$add_argument("--r_input_dir", required = TRUE, help = "r_input/ of compute_metrics.py")
parser$add_argument("--tables_dir", required = TRUE, help = "tables/ of compute_metrics.py")
parser$add_argument("--python_figures_dir", required = TRUE, help = "Kaplan-Meier figures of compute_metrics.py")
parser$add_argument("--shap_figures_dir", default = NULL, help = "Output of shap_analysis.py")
parser$add_argument("--output_dir", required = TRUE)
args <- parser$parse_args()

png_dir <- file.path(args$output_dir, "png")
pdf_dir <- file.path(args$output_dir, "pdf")
csv_dir <- file.path(args$output_dir, "csv")
for (d in c(png_dir, pdf_dir, csv_dir)) dir.create(d, recursive = TRUE, showWarnings = FALSE)

save_fig <- function(plot_obj, name, w = 7, h = 6) {
  ggsave(file.path(png_dir, paste0(name, ".png")), plot_obj, width = w, height = h, dpi = 300)
  ggsave(file.path(pdf_dir, paste0(name, ".pdf")), plot_obj, width = w, height = h, device = cairo_pdf)
}
save_table <- function(tbl, name) write_csv(tbl, file.path(csv_dir, paste0(name, ".csv")))
r_input <- function(name) file.path(args$r_input_dir, name)

# ---- Calibration (Figure 2C-D) ------------------------------------------------

# Kaplan-Meier risk 1 - S(t) at the horizon; carried forward if follow-up ends earlier
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

# Deciles of predicted risk: mean predicted vs Kaplan-Meier observed risk
build_calibration_table <- function(df, pred_col, t_horizon, n_groups = 10) {
  df %>%
    mutate(cal_group = ntile(.data[[pred_col]], n_groups)) %>%
    group_by(cal_group) %>%
    summarise(n = n(), mean_predicted = mean(.data[[pred_col]], na.rm = TRUE),
              observed = km_risk_at_time(pick(everything()), t_horizon), .groups = "drop") %>%
    mutate(difference = observed - mean_predicted, timepoint_months = t_horizon)
}

# Patients with known status at the horizon: death before it (1) or follow-up beyond it (0)
make_horizon_labelable_data <- function(df, horizon, pred_col, eps = 1e-6) {
  df %>%
    transmute(tte = tte, event = event, pred = pmin(pmax(.data[[pred_col]], eps), 1 - eps),
              outcome = case_when(event == 1 & tte <= horizon ~ 1, tte > horizon ~ 0, TRUE ~ NA_real_)) %>%
    filter(!is.na(outcome)) %>%
    mutate(lp = qlogis(pred))
}

# O:E = Kaplan-Meier observed risk at the horizon (all patients) / mean predicted risk,
# 95% CI from the Greenwood SE on the log scale with the expected risk held fixed.
# Calibration slope = logistic regression of the status on the predicted log-odds in
# the patients with known status at the horizon, Wald 95% CI.
compute_calibration_metrics <- function(df, horizon, pred_col) {
  z <- qnorm(0.975)
  km <- summary(survfit(Surv(tte, event) ~ 1, data = df), times = horizon, extend = TRUE)
  observed <- 1 - km$surv
  oe <- observed / mean(df[[pred_col]], na.rm = TRUE)
  se_log <- km$std.err / observed
  dat <- make_horizon_labelable_data(df, horizon, pred_col)
  out <- data.frame(timepoint_months = horizon, n_total = nrow(df), oe_km = round(oe, 4),
                    oe_km_lower_95 = round(oe * exp(-z * se_log), 4),
                    oe_km_upper_95 = round(oe * exp(z * se_log), 4), n_labelable = nrow(dat),
                    slope = NA_real_, slope_lower_95 = NA_real_, slope_upper_95 = NA_real_)
  if (nrow(dat) >= 5) {
    fit <- glm(outcome ~ lp, family = binomial(), data = dat)
    est <- unname(coef(fit)["lp"])
    se <- unname(sqrt(diag(vcov(fit)))["lp"])
    out$slope <- round(est, 4)
    out$slope_lower_95 <- round(est - z * se, 4)
    out$slope_upper_95 <- round(est + z * se, 4)
  }
  out
}

build_calibration_plot <- function(cal_tbl, cal_metrics, title, col = col_ml) {
  p <- ggplot(cal_tbl, aes(x = mean_predicted, y = observed)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = col_reference) +
    geom_smooth(method = "loess", se = TRUE, color = col, fill = col, alpha = 0.15) +
    geom_point(shape = 21, size = 3, fill = col, color = "black", stroke = 0.3) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    scale_x_continuous(breaks = seq(0, 1, 0.1)) +
    scale_y_continuous(breaks = seq(0, 1, 0.1)) +
    labs(title = title, x = "Predicted probability", y = "Observed probability (KM)") +
    theme_pub()
  if (!is.na(cal_metrics$slope[1])) {
    lbl <- sprintf("O:E (KM) = %.2f [%.2f, %.2f], n = %d\nSlope = %.2f [%.2f, %.2f], n = %d",
                   cal_metrics$oe_km[1], cal_metrics$oe_km_lower_95[1], cal_metrics$oe_km_upper_95[1],
                   cal_metrics$n_total[1], cal_metrics$slope[1], cal_metrics$slope_lower_95[1],
                   cal_metrics$slope_upper_95[1], cal_metrics$n_labelable[1])
    p <- p + annotate("text", x = 0.02, y = 0.95, label = lbl, hjust = 0, vjust = 1, size = 3,
                      family = "mono", color = "black")
  }
  p
}

cal_df <- read_csv(r_input("calibration_input_temporal.csv"), show_col_types = FALSE)
for (spec in list(list(prefix = "", suffix = "", col = col_ml, title = ""),
                  list(prefix = "cox_", suffix = "_cox", col = col_cox, title = " (Cox PH benchmark)"))) {
  panels <- list()
  for (tp in c(6, 18)) {
    pred_col <- sprintf("%sp_event_%dm", spec$prefix, tp)
    if (!(pred_col %in% names(cal_df))) next
    deciles <- build_calibration_table(cal_df, pred_col, tp)
    metrics <- compute_calibration_metrics(cal_df, tp, pred_col)
    save_table(metrics, sprintf("calibration_metrics_%dm_temporal%s", tp, spec$suffix))
    save_table(deciles, sprintf("calibration_deciles_%dm_temporal%s", tp, spec$suffix))
    panels[[as.character(tp)]] <- build_calibration_plot(deciles, metrics, sprintf("%d Months", tp), spec$col)
  }
  if (length(panels) == 2) {
    save_fig((panels[["6"]] | panels[["18"]]) +
               plot_annotation(title = paste0("Calibration, temporal validation", spec$title), theme = title_theme),
             paste0("calibration_combined_temporal", spec$suffix), w = 14, h = 6)
  }
}

# ---- Decision curves (Figure 2A-B) --------------------------------------------

# Net benefit of the ensemble [and the Cox benchmark] vs treat all / treat none;
# the shaded ranges are the prespecified clinically relevant thresholds
highlight_ranges <- list("6" = c(0.60, 0.95), "18" = c(0.35, 0.80))
dca_colors <- scale_color_manual(values = c("Treat All" = "#F8766D", "Treat None" = "#00BA38",
                                            "ML Ensemble" = "#619CFF", "Cox PH" = col_cox))
dca_plots <- list()
dca_plots_highlighted <- list()
for (tp in c(6, 18)) {
  pred_col <- paste0("p_event_", tp, "m")
  if (!(pred_col %in% names(cal_df))) next
  vars <- pred_col
  labels <- setNames(list("ML Ensemble"), pred_col)
  cox_col <- paste0("cox_", pred_col)
  if (cox_col %in% names(cal_df) && !anyNA(cal_df[[cox_col]])) {
    vars <- c(pred_col, cox_col)
    labels[[cox_col]] <- "Cox PH"
  }
  dca_obj <- dca(as.formula(sprintf("Surv(tte, event) ~ %s", paste(vars, collapse = " + "))),
                 data = cal_df, time = tp, thresholds = seq(0.01, 0.99, 0.01), label = labels)
  save_table(as.data.frame(dca_obj$dca), sprintf("net_benefit_%dm_temporal", tp))

  base <- plot(dca_obj, smooth = TRUE) + coord_cartesian(ylim = c(-0.05, 0.50)) + dca_colors + theme_pub()
  dca_plots[[as.character(tp)]] <- base +
    labs(title = sprintf("%d Months", tp), x = "Threshold Probability", y = "Net Benefit") +
    scale_x_continuous(breaks = seq(0, 1, 0.2))
  hr <- highlight_ranges[[as.character(tp)]]
  dca_plots_highlighted[[as.character(tp)]] <- base +
    annotate("rect", xmin = hr[1], xmax = hr[2], ymin = -Inf, ymax = Inf, fill = "#87CEEB", alpha = 0.25) +
    labs(title = sprintf("%d Months", tp), x = "Threshold Probability", y = "Net Benefit",
         subtitle = sprintf("Highlighted range: %.0f%%–%.0f%%", hr[1] * 100, hr[2] * 100)) +
    scale_x_continuous(breaks = seq(0, 1, 0.1))
}
if (length(dca_plots) == 2) {
  save_fig((dca_plots[["6"]] | dca_plots[["18"]]) +
             plot_annotation(title = "Decision curve analysis, temporal validation", theme = title_theme),
           "dca_combined_temporal", w = 14, h = 6)
  save_fig((dca_plots_highlighted[["6"]] | dca_plots_highlighted[["18"]]) +
             plot_annotation(title = "Decision curve analysis, clinically relevant ranges", theme = title_theme),
           "dca_supplementary_highlighted", w = 14, h = 6)
}

# ---- ROC and precision-recall curves (not in the paper) -----------------------

label_colors <- function(df) {
  lab <- df %>% distinct(model, label)
  lab <- lab[order(match(lab$model, names(model_colors))), ]
  setNames(unname(model_colors[lab$model]), lab$label)
}

build_roc_panel <- function(roc_df, tp) {
  roc_df <- roc_df %>% mutate(model = relabel_models(model)) %>%
    group_by(model) %>% mutate(label = sprintf("%s (AUC = %.3f)", model, first(auc))) %>% ungroup()
  colors <- label_colors(roc_df)
  roc_df$label <- factor(roc_df$label, levels = names(colors))
  ggplot(roc_df, aes(x = fpr, y = tpr, color = label)) +
    geom_line(linewidth = 0.8) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
    scale_color_manual(values = colors) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = sprintf("%d Months", tp), x = "1 − Specificity", y = "Sensitivity") +
    theme_pub()
}

for (coh in list(c("internal", "Internal test set"), c("temporal_full", "Temporal validation"))) {
  panels <- lapply(c(6, 18), function(tp)
    build_roc_panel(read_csv(r_input(sprintf("roc_curves_%s_%dm.csv", coh[1], tp)), show_col_types = FALSE), tp))
  save_fig((panels[[1]] | panels[[2]]) +
             plot_annotation(title = sprintf("ROC curves (IPCW), %s", coh[2]), theme = title_theme),
           sprintf("roc_combined_%s", sub("_full", "", coh[1])), w = 14, h = 6)
}

# From 18 months the positive class is survival beyond the horizon
build_pr_panel <- function(pr_df, tp) {
  pr_df <- pr_df %>% mutate(model = relabel_models(model)) %>%
    group_by(model) %>% mutate(label = sprintf("%s (AP = %.3f)", model, first(average_precision))) %>% ungroup()
  colors <- label_colors(pr_df)
  pr_df$label <- factor(pr_df$label, levels = names(colors))
  inverted <- any(pr_df$inverted == TRUE, na.rm = TRUE)
  ggplot(pr_df, aes(x = recall, y = precision, color = label)) +
    geom_line(linewidth = 0.8) +
    scale_color_manual(values = colors) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = if (inverted) sprintf("P(survival > %d mo)", tp) else sprintf("P(event ≤ %d mo)", tp),
         x = if (inverted) "Recall (Specificity)" else "Recall (Sensitivity)",
         y = if (inverted) "Precision (NPV)" else "Precision (PPV)") +
    theme_pub()
}
pr_panels <- lapply(c(6, 18), function(tp)
  build_pr_panel(read_csv(r_input(sprintf("pr_curves_temporal_full_%dm.csv", tp)), show_col_types = FALSE), tp))
save_fig((pr_panels[[1]] | pr_panels[[2]]) +
           plot_annotation(title = "Precision-recall (IPCW), temporal validation", theme = title_theme),
         "pr_combined_temporal", w = 14, h = 6)

# ---- AUC and Brier score over time (not in the paper) -------------------------

over_time_plot <- function(df, y, ylim, title, ylab) {
  df$model <- order_models(df$model)
  p <- ggplot(df, aes(x = timepoint, y = .data[[y]], color = model)) +
    geom_line(linewidth = 0.9) + geom_point(size = 2.5)
  if (any(!is.na(df$ci_lower))) {
    p <- p + geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper, fill = model), alpha = 0.15, color = NA)
  }
  p + scale_color_manual(values = model_colors) + scale_fill_manual(values = model_colors) +
    scale_x_continuous(breaks = c(6, 12, 18, 24)) + coord_cartesian(ylim = ylim) +
    labs(title = title, x = "Time (months)", y = ylab) + theme_pub()
}
auc_df <- read_csv(r_input("temporal_auc.csv"), show_col_types = FALSE) %>% mutate(model = relabel_models(model))
brier_df <- read_csv(r_input("temporal_brier.csv"), show_col_types = FALSE) %>% mutate(model = relabel_models(model))
for (coh in list(c("internal", "Internal Test Set"), c("temporal", "Temporal Validation"))) {
  save_fig(over_time_plot(filter(auc_df, cohort == coh[1]), "auc", c(0.5, 1.0), coh[2], "AUC"),
           paste0("temporal_auc_", coh[1]), w = 9, h = 6)
  save_fig(over_time_plot(filter(brier_df, cohort == coh[1]), "brier", c(0, 0.35),
                          sprintf("Brier Scores (IPCW), %s", coh[2]), "Brier Score"),
           paste0("temporal_brier_", coh[1]), w = 9, h = 6)
}

# ---- Risk distributions (Fig. S3) ---------------------------------------------

dens_all <- bind_rows(
  read_csv(r_input("risk_density_internal.csv"), show_col_types = FALSE) %>% mutate(cohort = "Internal"),
  read_csv(r_input("risk_density_temporal.csv"), show_col_types = FALSE) %>% mutate(cohort = "Temporal"))
cohort_fill <- scale_fill_manual(values = c("Internal" = col_internal, "Temporal" = col_temporal))
cohort_color <- scale_color_manual(values = c("Internal" = col_internal, "Temporal" = col_temporal))
for (spec in list(list(score = "ml_risk_score", thr = "risk_thresholds_15-85.json", suffix = "",
                       title = "Prognostic Score Distribution", xlab = "ML Risk Score"),
                  list(score = "cox_risk_score", thr = "risk_thresholds_cox_15-85.json", suffix = "_cox",
                       title = "Prognostic Score Distribution — Cox PH benchmark",
                       xlab = "Cox PH Risk Score (linear predictor)"))) {
  if (!(spec$score %in% names(dens_all))) next
  thr <- fromJSON(file.path(args$tables_dir, spec$thr))
  p <- ggplot(dens_all, aes(x = .data[[spec$score]], fill = cohort, color = cohort)) +
    geom_density(alpha = 0.3, linewidth = 0.6) + cohort_fill + cohort_color +
    geom_vline(xintercept = thr$low_threshold, linetype = "dashed", color = col_low, linewidth = 0.7) +
    geom_vline(xintercept = thr$high_threshold, linetype = "dashed", color = col_high, linewidth = 0.7) +
    annotate("text", x = thr$low_threshold, y = Inf, label = "Low / Medium", vjust = 1.5, hjust = 1.05,
             size = 3, fontface = "bold", color = col_low) +
    annotate("text", x = thr$high_threshold, y = Inf, label = "Medium / High", vjust = 1.5, hjust = -0.05,
             size = 3, fontface = "bold", color = col_high) +
    labs(title = spec$title, x = spec$xlab, y = "Density") + theme_pub()
  save_fig(p, paste0("risk_score_distribution_overlay", spec$suffix))
}

# Fig. S3C-D: predicted P(event <= t), internal test set vs temporal cohort
pred_dist <- read_csv(r_input("predicted_risk_distributions.csv"), show_col_types = FALSE)
for (prefix in c("", "cox_")) {
  for (tp in c(6, 18)) {
    pcol <- sprintf("%sp_event_%dm", prefix, tp)
    if (!(pcol %in% names(pred_dist))) next
    p <- ggplot(filter(pred_dist, !is.na(.data[[pcol]])), aes(x = .data[[pcol]], fill = cohort, color = cohort)) +
      geom_density(alpha = 0.3, linewidth = 0.6) +
      scale_fill_manual(values = c("internal" = col_internal, "temporal" = col_temporal)) +
      scale_color_manual(values = c("internal" = col_internal, "temporal" = col_temporal)) +
      coord_cartesian(ylim = c(0, 5)) +
      labs(title = sprintf("P(T ≤ %d mo) Distribution%s", tp, if (prefix == "") "" else " — Cox PH benchmark"),
           x = sprintf("Predicted P(event ≤ %d months)", tp), y = "Density") +
      theme_pub()
    save_fig(p, sprintf("predicted_risk_%dm%s", tp, if (prefix == "") "" else "_cox"))
  }
}

# Fig. S3A-B: IPCW-weighted predicted P(event <= t) by event status at t, temporal cohort
for (spec in list(c("ipcw_stratified_distributions.csv", "", ""),
                  c("ipcw_stratified_distributions_cox.csv", "cox_", "_cox"))) {
  if (!file.exists(r_input(spec[1]))) next
  ipcw_dist <- read_csv(r_input(spec[1]), show_col_types = FALSE)
  for (tp in c(6, 18)) {
    pcol <- sprintf("%sp_event_%dm", spec[2], tp)
    if (!(pcol %in% names(ipcw_dist))) next
    df_tp <- ipcw_dist %>%
      filter(timepoint == tp, !is.na(.data[[pcol]])) %>%
      mutate(status_label = factor(event_status, levels = c("event", "no_event"),
                                   labels = c(sprintf("Event by %d mo", tp), sprintf("No event by %d mo", tp))))
    p <- ggplot(df_tp, aes(x = .data[[pcol]], fill = status_label, weight = ipcw_weight)) +
      geom_density(alpha = 0.4, linewidth = 0.6) +
      scale_fill_manual(values = c(col_high, col_low)) +
      scale_x_continuous(limits = c(0, 1)) +
      coord_cartesian(ylim = c(0, 5)) +
      labs(title = sprintf("IPCW-Weighted P(T ≤ %d mo) by Event Status — Temporal%s", tp,
                           if (spec[2] == "") "" else ", Cox PH benchmark"),
           x = sprintf("Predicted P(event ≤ %d months)", tp), y = "Weighted Density") +
      theme_pub()
    save_fig(p, sprintf("ipcw_risk_stratified_%dm%s", tp, spec[3]))
  }
}

# ---- Figures made in Python (Kaplan-Meier, SHAP) ------------------------------

copy_figures <- function(from, extension, to) {
  invisible(file.copy(list.files(from, pattern = paste0("\\.", extension, "$"), full.names = TRUE), to,
                      overwrite = TRUE))
}
km_dir <- file.path(args$python_figures_dir, "km")
copy_figures(km_dir, "png", png_dir)
copy_figures(km_dir, "pdf", pdf_dir)
if (!is.null(args$shap_figures_dir) && nchar(args$shap_figures_dir) > 0) {
  copy_figures(file.path(args$shap_figures_dir, "figures", "png"), "png", png_dir)
  copy_figures(file.path(args$shap_figures_dir, "figures", "pdf"), "pdf", pdf_dir)
}
cat("Figures written to", args$output_dir, "\n")
