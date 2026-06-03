## Step 4 Task 2  -  Sensitivity lmer with v2 transcriptomic BBB score
## (instead of Olink PC1) on the pre-registered primary endpoint.
##
## Tests whether the null primary endpoint holds when we use the brain-derived
## v2 score directly as the BBB_score covariate. If the v2-score model is also
## null, the decoupling story is reinforced: neither modality predicts UPDRS3
## slope, but for different biological reasons.
##
## Same lmer formula as pre-reg §7, only BBB_score is replaced by v2_score
## (mean z(log-CPM) over 118 v2 module genes on PPMI baseline blood RNA-seq).

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(arrow)
  library(dplyr)
})

# Load modeling cohort (Olink-based) and v2-score-per-PATNO from Task 1
df <- as.data.frame(read_parquet("results/step3/modeling_cohort_long.parquet"))
v2 <- as.data.frame(read_parquet("results/step4/cross_modal_scores.parquet"))
v2$PATNO <- as.character(v2$PATNO)
df$PATNO <- as.character(df$PATNO)

cat(stepf("Original Olink modeling cohort: %d subjects, %d visits\n",
            length(unique(df$PATNO)), nrow(df)))
cat(stepf("Cross-modal cohort with v2 score: %d patients\n", nrow(v2)))

# Inner join: only patients with BOTH RNA-derived v2 score AND longitudinal
# UPDRS3 + covariates pass into this sensitivity model
drop_cols <- intersect(c("bbb_score", "bbb_score_pc1", "vascular_class"), colnames(df))
df_v2 <- df |>
  select(-all_of(drop_cols)) |>
  inner_join(v2[, c("PATNO", "v2_score")], by = "PATNO")

keep_pat <- names(which(table(df_v2$PATNO) >= 3))
df_v2 <- df_v2[df_v2$PATNO %in% keep_pat, ]
n_pat <- length(unique(df_v2$PATNO))
cat(stepf("\nSensitivity cohort (≥3 visits AND v2 score): %d subjects, %d visits\n",
            n_pat, nrow(df_v2)))

if (n_pat < 20) {
  stop("Sensitivity cohort too small (<20 subjects). Aborting.")
}

# Cohort breakdown
cat("\nCohort_Current breakdown (per subject):\n")
print(table(df_v2[!duplicated(df_v2$PATNO), "Cohort_Current"]))

# Prepare model variables
df_v2$Sex <- factor(df_v2$Sex, levels = c("Female", "Male"))
df_v2$v2_score_z <- as.numeric(scale(df_v2$v2_score))

# Pre-registered formula, with v2_score_z replacing BBB_score_z
cat("\nFitting sensitivity lmer (REML) with v2 transcriptomic score...\n")
m <- lmer(
  UPDRS3_Total ~ Time * v2_score_z
                 + Age_at_Baseline + Sex + UPDRS3_Baseline
                 + LEDD_total
                 + Neutrophil_pct + Monocyte_pct
                 + (1 + Time | PATNO),
  data = df_v2,
  REML = TRUE,
  control = lmerControl(optimizer = "bobyqa", check.conv.singular = "warning")
)

cat("\n=== Model summary (sensitivity, v2 score) ===\n")
print(summary(m))

# 95% CI on the v2 interaction term
cat("\n=== Time:v2_score_z 95% CI ===\n")
ci <- tryCatch(
  confint(m, parm = "Time:v2_score_z", method = "profile", quiet = TRUE),
  error = function(e) {
    cat("Profile CI failed; falling back to Wald.\n")
    confint(m, parm = "Time:v2_score_z", method = "Wald")
  }
)
print(ci)

cat("\n=== Variance components ===\n")
print(VarCorr(m))

# Pull and save primary coefficient
coefs <- summary(m)$coefficients
key <- "Time:v2_score_z"
if (key %in% rownames(coefs)) {
  est  <- coefs[key, "Estimate"]
  se   <- coefs[key, "Std. Error"]
  pval <- coefs[key, "Pr(>|t|)"]
  cat(stepf("\n=== SENSITIVITY (v2 brain transcriptomic score) ===\n"))
  cat(stepf("Time:v2_score_z  estimate = %+.4f, SE = %.4f, p = %.4f (two-sided)\n",
              est, se, pval))
  cat(stepf("Interpretation: per +1 SD of v2 brain-derived score, UPDRS3 slope is %+.3f points/year\n",
              est))
}

# Compare to Olink primary
olink_pri <- read.table("results/step3/lmer_primary_coefs.tsv",
                         sep = "\t", header = TRUE, check.names = FALSE)
cat("\n=== Side-by-side: Olink PC1 (primary) vs v2 transcriptomic (sensitivity) ===\n")
o <- olink_pri["Time:BBB_score_z", ]
cat(stepf("  Olink PC1 (n=61, primary):    estimate=%+.4f, SE=%.4f, p=%.4f\n",
            o$Estimate, o$`Std. Error`, o$`Pr(>|t|)`))
cat(stepf("  v2 brain (n=%d, sensitivity): estimate=%+.4f, SE=%.4f, p=%.4f\n",
            n_pat, est, se, pval))

# Save
saveRDS(m, "results/step4/lmer_sensitivity_v2_model.rds")
write.table(coefs,
            "results/step4/lmer_sensitivity_v2_coefs.tsv",
            sep = "\t", quote = FALSE)
cat("\n→ Wrote results/step4/lmer_sensitivity_v2_model.rds + _coefs.tsv\n")
