## Step 5 Task 51  -  Pre-registered primary endpoint mixed-effects model.
##
## Pre-reg §7 model:
##   lmer(UPDRS3_Total ~ Time * BBB_score
##                     + Age_at_Baseline + Sex + UPDRS3_Baseline + LEDD
##                     + Neutrophil_frac + Monocyte_frac
##                     + (1 + Time | PATNO),
##        data, REML=TRUE)
##
## Hypothesis test: two-sided test on Time:BBB_score coefficient
## (Satterthwaite d.f. via lmerTest), α=0.05.
##
## Pre-reg deviation 1 (§14): BBB_score is now PC1 of Olink BBB protein panel.
## Model formula otherwise unchanged.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(arrow)
  library(dplyr)
})

# Load modeling cohort
df <- as.data.frame(read_parquet("results/step3/modeling_cohort_long.parquet"))
cat(stepf("Modeling cohort: %d subjects, %d visits\n",
            length(unique(df$PATNO)), nrow(df)))

# Cohort summary
cat(stepf("  Cohort_Current x vascular_class:\n"))
print(table(df[!duplicated(df$PATNO), c("Cohort_Current", "vascular_class")]))

# Convert Sex to factor
df$Sex <- factor(df$Sex, levels = c("Female", "Male"))

# Scale BBB_score for interpretable interaction effect (1 SD = 1 unit)
df$BBB_score_z <- as.numeric(scale(df$bbb_score))

# Fit pre-registered model
cat("\nFitting pre-registered lmer (REML)...\n")
m <- lmer(
  UPDRS3_Total ~ Time * BBB_score_z
                 + Age_at_Baseline + Sex + UPDRS3_Baseline
                 + LEDD_total
                 + Neutrophil_pct + Monocyte_pct
                 + (1 + Time | PATNO),
  data = df,
  REML = TRUE,
  control = lmerControl(optimizer = "bobyqa", check.conv.singular = "warning")
)

cat("\n=== Model summary ===\n")
print(summary(m))

# Confidence interval on the primary coefficient
cat("\n=== Time:BBB_score_z 95% CI (profile likelihood) ===\n")
ci <- tryCatch(confint(m, parm = "Time:BBB_score_z", method = "profile", quiet = TRUE),
               error = function(e) {
                 cat("Profile CI failed; falling back to Wald.\n")
                 confint(m, parm = "Time:BBB_score_z", method = "Wald")
               })
print(ci)

# Variance components
cat("\n=== Variance components ===\n")
print(VarCorr(m))

# Marginal + conditional R^2 (omit if MuMIn unavailable)
if (requireNamespace("MuMIn", quietly = TRUE)) {
  r2 <- MuMIn::r.squaredGLMM(m)
  cat(stepf("\nMarginal R^2 = %.3f, Conditional R^2 = %.3f\n",
              r2[1, "R2m"], r2[1, "R2c"]))
}

# Pull primary coefficient explicitly
coefs <- summary(m)$coefficients
key <- "Time:BBB_score_z"
if (key %in% rownames(coefs)) {
  est  <- coefs[key, "Estimate"]
  se   <- coefs[key, "Std. Error"]
  pval <- coefs[key, "Pr(>|t|)"]
  cat(stepf("\n=== PRIMARY ENDPOINT ===\n"))
  cat(stepf("Time:BBB_score_z  estimate = %+.4f, SE = %.4f, p = %.4f (two-sided)\n",
              est, se, pval))
  cat(stepf("Interpretation: per +1 SD of BBB_score, UPDRS3 slope is %+.3f points/year\n",
              est))
}

# Save model object + coefficient table
saveRDS(m, "results/step3/lmer_primary_model.rds")
write.table(coefs,
            "results/step3/lmer_primary_coefs.tsv",
            sep = "\t", quote = FALSE)
cat("\n→ Wrote results/step3/lmer_primary_model.rds + lmer_primary_coefs.tsv\n")
