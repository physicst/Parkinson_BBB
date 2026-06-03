# Step 4b  -  Longitudinal mixed-effects models: does Qalb modify UPDRS3 slope?
#
# Pre-specified in docs/2026-05-22-pd-bbb-csf-qalb-analysis-design.md (Step 4).
#
# Three pre-specified models, all exploratory / post-hoc:
#   1. PRIMARY             -  Time x Qalb_z, full Qalb cohort
#   2. POSITIVE CONTROL    -  Time x plasma-NfL (log, z), same model structure
#   3. STAGE-STRATIFIED    -  Time x Qalb_z, advanced subset (baseline UPDRS3 > median)
#
# Model mirrors the pre-registered primary endpoint (51_lmer_primary.R):
#   UPDRS3 ~ Time * <predictor> + Age + Sex + UPDRS3_Baseline + LEDD
#            + Neutrophil_pct + Monocyte_pct + (1 + Time | PATNO)
# REML, bobyqa, Satterthwaite df via lmerTest.
#
# Run: "C:/Program Files/R/R-4.3.2/bin/Rscript.exe" code/94_qalb_lmer.R

suppressPackageStartupMessages({
  library(arrow)
  library(lme4)
  library(lmerTest)
})

df <- as.data.frame(read_parquet("results/step6/qalb_modeling_cohort_long.parquet"))
df$PATNO <- as.factor(df$PATNO)
df$Sex <- as.factor(df$Sex)
cat(stepf("Loaded %d visits, %d subjects\n", nrow(df), nlevels(df$PATNO)))

fit_model <- function(data, predictor, label) {
  data <- droplevels(data)
  f <- as.formula(paste0(
    "UPDRS3_Total ~ Time * ", predictor,
    " + Age_at_Baseline + Sex + UPDRS3_Baseline + LEDD_total",
    " + Neutrophil_pct + Monocyte_pct + (1 + Time | PATNO)"))
  m <- lmer(f, data = data, REML = TRUE,
            control = lmerControl(optimizer = "bobyqa"))
  co <- summary(m)$coefficients
  ix <- paste0("Time:", predictor)
  est <- co[ix, "Estimate"]; se <- co[ix, "Std. Error"]
  sing <- isSingular(m)
  cat(stepf("\n[%s]\n  n=%d visits, %d subjects\n", label,
              nrow(data), nlevels(data$PATNO)))
  cat(stepf("  %s: estimate=%.3f, SE=%.3f, df=%.1f, t=%.2f, p=%.4f%s\n",
              ix, est, se, co[ix, "df"], co[ix, "t value"],
              co[ix, "Pr(>|t|)"], if (sing) "  [SINGULAR FIT]" else ""))
  data.frame(
    model = label, term = ix,
    estimate = est, se = se, df = co[ix, "df"],
    t = co[ix, "t value"], p = co[ix, "Pr(>|t|)"],
    ci_lo = est - 1.96 * se, ci_hi = est + 1.96 * se,
    n_obs = nrow(data), n_subj = nlevels(data$PATNO),
    singular = sing)
}

res <- rbind(
  fit_model(df, "qalb_z",
            "Primary: Qalb -> UPDRS3 slope (full cohort)"),
  fit_model(df[!is.na(df$nfl_log_z), ], "nfl_log_z",
            "Positive control: plasma NfL -> UPDRS3 slope"),
  fit_model(df[df$advanced, ], "qalb_z",
            "Stage-stratified: Qalb -> UPDRS3 slope (advanced subset)")
)

write.table(res, "results/step6/qalb_lmer_coefs.tsv",
            sep = "\t", row.names = FALSE, quote = FALSE)
cat("\n-> results/step6/qalb_lmer_coefs.tsv\n")
cat("\n=== Summary ===\n")
print(res[, c("model", "estimate", "se", "p", "n_subj", "singular")],
      row.names = FALSE)
