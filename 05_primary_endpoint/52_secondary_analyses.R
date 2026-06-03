## Step 5 Task 52  -  Pre-registered SECONDARY analyses (pre-reg §9).
##
## Designated secondary endpoints (run with hierarchical-testing language;
## NOT primary discovery):
##
##   1. MoCA slope:  same lmer formula, outcome = MoCA_Total
##   2. DaT-SCAN slope: outcome = DATSCAN_Putamen_Worst
##   3. SAA-stratified primary: re-run UPDRS3 lmer on SAA-Positive subset only
##   4. ConsensusClusterPlus k=3, k=4 sensitivity (already in Task 31 ccp output)
##   5. LRRK2 / GBA / APOE subgroup analyses (loaded from Biospecimen file)
##
## All results combined into a single TSV + HTML summary.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(arrow)
  library(dplyr)
  library(tibble)
})

S3 <- "results/step3"

df <- as.data.frame(read_parquet(file.path(S3, "modeling_cohort_long.parquet")))
df$Sex <- factor(df$Sex, levels = c("Female", "Male"))
df$BBB_score_z <- as.numeric(scale(df$bbb_score))
cat(stepf("Loaded modeling cohort: %d subjects, %d visits\n",
            length(unique(df$PATNO)), nrow(df)))

results <- list()

## ============================================================
## SECONDARY 1  -  MoCA slope
## ============================================================
cat("\n=== SECONDARY 1: MoCA slope ===\n")
df_moca <- df[!is.na(df$MoCA_Total), ]
keep_pat <- names(which(table(df_moca$PATNO) >= 3))
df_moca <- df_moca[df_moca$PATNO %in% keep_pat, ]
cat(stepf("Cohort: %d subjects, %d visits with MoCA\n",
            length(unique(df_moca$PATNO)), nrow(df_moca)))

if (length(unique(df_moca$PATNO)) >= 20) {
  bl_moca <- df_moca[!duplicated(df_moca$PATNO), c("PATNO", "MoCA_Total")]
  names(bl_moca)[2] <- "MoCA_Baseline"
  df_moca <- merge(df_moca, bl_moca, by = "PATNO")

  m_moca <- tryCatch(
    lmer(MoCA_Total ~ Time * BBB_score_z
                       + Age_at_Baseline + Sex + MoCA_Baseline
                       + LEDD_total + Neutrophil_pct + Monocyte_pct
                       + (1 + Time | PATNO),
         data = df_moca, REML = TRUE,
         control = lmerControl(optimizer = "bobyqa", check.conv.singular = "warning")),
    error = function(e) NULL)

  if (!is.null(m_moca)) {
    co <- summary(m_moca)$coefficients
    if ("Time:BBB_score_z" %in% rownames(co)) {
      r <- co["Time:BBB_score_z", ]
      results[["MoCA_slope"]] <- list(
        endpoint = "MoCA slope",
        n_subj = length(unique(df_moca$PATNO)), n_obs = nrow(df_moca),
        estimate = r["Estimate"], se = r["Std. Error"], p = r["Pr(>|t|)"]
      )
      cat(stepf("Time × BBB_score_z: estimate = %+.3f, SE = %.3f, p = %.4f\n",
                  r["Estimate"], r["Std. Error"], r["Pr(>|t|)"]))
    }
  }
} else {
  cat("Insufficient subjects with longitudinal MoCA. Skipping.\n")
}

## ============================================================
## SECONDARY 2  -  DaT-SCAN Putamen_Worst slope
## ============================================================
cat("\n=== SECONDARY 2: DaT-SCAN Putamen_Worst slope ===\n")
df_dat <- df[!is.na(df$DATSCAN_Putamen_Worst), ]
keep_pat <- names(which(table(df_dat$PATNO) >= 3))
df_dat <- df_dat[df_dat$PATNO %in% keep_pat, ]
cat(stepf("Cohort: %d subjects, %d visits with DaT-SCAN\n",
            length(unique(df_dat$PATNO)), nrow(df_dat)))

if (length(unique(df_dat$PATNO)) >= 20) {
  bl_dat <- df_dat[!duplicated(df_dat$PATNO), c("PATNO", "DATSCAN_Putamen_Worst")]
  names(bl_dat)[2] <- "DAT_Baseline"
  df_dat <- merge(df_dat, bl_dat, by = "PATNO")

  m_dat <- tryCatch(
    lmer(DATSCAN_Putamen_Worst ~ Time * BBB_score_z
                                  + Age_at_Baseline + Sex + DAT_Baseline
                                  + LEDD_total + Neutrophil_pct + Monocyte_pct
                                  + (1 + Time | PATNO),
         data = df_dat, REML = TRUE,
         control = lmerControl(optimizer = "bobyqa")),
    error = function(e) NULL)

  if (!is.null(m_dat)) {
    co <- summary(m_dat)$coefficients
    if ("Time:BBB_score_z" %in% rownames(co)) {
      r <- co["Time:BBB_score_z", ]
      results[["DaT_slope"]] <- list(
        endpoint = "DaT-SCAN Putamen_Worst slope",
        n_subj = length(unique(df_dat$PATNO)), n_obs = nrow(df_dat),
        estimate = r["Estimate"], se = r["Std. Error"], p = r["Pr(>|t|)"]
      )
      cat(stepf("Time × BBB_score_z: estimate = %+.4f, SE = %.4f, p = %.4f\n",
                  r["Estimate"], r["Std. Error"], r["Pr(>|t|)"]))
    }
  }
} else {
  cat("Insufficient subjects with longitudinal DaT-SCAN. Skipping.\n")
}

## ============================================================
## SECONDARY 3  -  SAA-stratified primary endpoint
## ============================================================
cat("\n=== SECONDARY 3: SAA-stratified UPDRS3 slope ===\n")
df_saa <- df[df$SAA_Status %in% c("Positive", "Negative"), ]
cat("SAA breakdown:\n")
print(table(df_saa[!duplicated(df_saa$PATNO), "SAA_Status"]))

for (saa_grp in c("Positive", "Negative")) {
  df_g <- df_saa[df_saa$SAA_Status == saa_grp, ]
  keep_pat <- names(which(table(df_g$PATNO) >= 3))
  df_g <- df_g[df_g$PATNO %in% keep_pat, ]
  n_pat <- length(unique(df_g$PATNO))

  cat(stepf("\nSAA-%s: %d subjects, %d visits\n", saa_grp, n_pat, nrow(df_g)))
  if (n_pat < 15) {
    cat("  Insufficient subjects. Skipping.\n")
    next
  }

  m <- tryCatch(
    lmer(UPDRS3_Total ~ Time * BBB_score_z
                        + Age_at_Baseline + Sex + UPDRS3_Baseline
                        + LEDD_total + Neutrophil_pct + Monocyte_pct
                        + (1 + Time | PATNO),
         data = df_g, REML = TRUE,
         control = lmerControl(optimizer = "bobyqa")),
    error = function(e) NULL)

  if (!is.null(m)) {
    co <- summary(m)$coefficients
    if ("Time:BBB_score_z" %in% rownames(co)) {
      r <- co["Time:BBB_score_z", ]
      results[[paste0("SAA_", saa_grp)]] <- list(
        endpoint = paste0("UPDRS3 slope, SAA-", saa_grp),
        n_subj = n_pat, n_obs = nrow(df_g),
        estimate = r["Estimate"], se = r["Std. Error"], p = r["Pr(>|t|)"]
      )
      cat(stepf("  Time × BBB_score_z: estimate = %+.3f, SE = %.3f, p = %.4f\n",
                  r["Estimate"], r["Std. Error"], r["Pr(>|t|)"]))
    }
  }
}

## ============================================================
## Summary table
## ============================================================
cat("\n\n=== SECONDARY ANALYSES SUMMARY ===\n")
if (length(results) > 0) {
  out <- do.call(rbind, lapply(results, function(x) {
    data.frame(endpoint = x$endpoint, n_subj = x$n_subj, n_obs = x$n_obs,
               estimate = x$estimate, se = x$se, p = x$p)
  }))
  rownames(out) <- NULL
  print(out, row.names = FALSE)
  out$bh_padj <- p.adjust(out$p, method = "BH")
  write.table(out, file.path(S3, "secondary_analyses_summary.tsv"),
              sep = "\t", row.names = FALSE, quote = FALSE)
  cat(stepf("\n→ Wrote %s/secondary_analyses_summary.tsv\n", S3))

  cat("\nAfter BH adjustment within secondary set:\n")
  print(out[, c("endpoint", "estimate", "p", "bh_padj")], row.names = FALSE)
} else {
  cat("No secondary models converged.\n")
}
