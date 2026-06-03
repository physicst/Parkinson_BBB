## Step 5 Task 53  -  Pre-registered subgroup analyses (pre-reg §9).
##
## Subgroups: SAA-Positive vs Negative; LRRK2 carriers; GBA carriers;
## APOE e4 carriers (cognitive analysis on MoCA).
##
## Genotype data from Current_Biospecimen_Analysis_Results: TESTNAME ApoE Genotype,
## GBA, LRRK2 Variant Identified.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(arrow)
  library(dplyr)
})

S3 <- "results/step3"

df <- as.data.frame(read_parquet(file.path(S3, "modeling_cohort_long.parquet")))
df$Sex <- factor(df$Sex, levels = c("Female", "Male"))
df$BBB_score_z <- as.numeric(scale(df$bbb_score))

# ---- Propagate SAA_Status per subject (one SAA test per patient typically) ----
saa_per_patient <- df |>
  filter(!is.na(SAA_Status), SAA_Status %in% c("Positive", "Negative")) |>
  distinct(PATNO, SAA_Status)
cat(stepf("SAA breakdown (per subject): %s\n",
            paste(table(saa_per_patient$SAA_Status), collapse = " / ")))

df_saa <- df |> select(-SAA_Status) |> left_join(saa_per_patient, by = "PATNO")

# ---- Load genotype data from Biospecimen ----
biospec_path <- "D:/Parkinson_file/Current_Biospecimen_Analysis_Results_02May2026.csv"
biospec <- read.csv(biospec_path, stringsAsFactors = FALSE)
biospec$PATNO <- as.character(biospec$PATNO)

# APOE
apoe <- biospec |>
  filter(toupper(TESTNAME) %in% c("APOE GENOTYPE")) |>
  distinct(PATNO, TESTVALUE) |>
  rename(APOE = TESTVALUE)
apoe$APOE_e4_carrier <- grepl("e4|E4", apoe$APOE)
cat(stepf("\nAPOE genotype available for %d patients; e4 carriers: %d\n",
            nrow(apoe), sum(apoe$APOE_e4_carrier)))

# GBA  -  check for any GBA-related test result indicating mutation
gba <- biospec |>
  filter(grepl("^GBA", TESTNAME) | grepl("^Full GBA", TESTNAME) | TESTNAME == "GBA") |>
  distinct(PATNO, TESTNAME, TESTVALUE)

gba_per_pat <- gba |>
  group_by(PATNO) |>
  summarise(any_gba_call = paste(unique(TESTVALUE), collapse = ";"),
            .groups = "drop")
gba_per_pat$GBA_carrier <- !grepl("^WT|wild|negative|none|no muta|^-$|^$",
                                   gba_per_pat$any_gba_call,
                                   ignore.case = TRUE) &
                          gba_per_pat$any_gba_call != "" &
                          !is.na(gba_per_pat$any_gba_call)
cat(stepf("GBA tested for %d patients; called as carrier: %d\n",
            nrow(gba_per_pat), sum(gba_per_pat$GBA_carrier)))

# LRRK2  -  same approach
lrrk2 <- biospec |>
  filter(grepl("LRRK2", TESTNAME) & !grepl("Fluorospot", TESTNAME)) |>
  distinct(PATNO, TESTNAME, TESTVALUE)
lrrk2_per_pat <- lrrk2 |>
  group_by(PATNO) |>
  summarise(any_lrrk2_call = paste(unique(TESTVALUE), collapse = ";"),
            .groups = "drop")
lrrk2_per_pat$LRRK2_carrier <- !grepl("^WT|wild|negative|none|^-$|^$",
                                       lrrk2_per_pat$any_lrrk2_call,
                                       ignore.case = TRUE) &
                              lrrk2_per_pat$any_lrrk2_call != "" &
                              !is.na(lrrk2_per_pat$any_lrrk2_call)
cat(stepf("LRRK2 tested for %d patients; called as carrier: %d\n",
            nrow(lrrk2_per_pat), sum(lrrk2_per_pat$LRRK2_carrier)))

# Merge all
df_anno <- df_saa |>
  left_join(apoe |> select(PATNO, APOE_e4_carrier), by = "PATNO") |>
  left_join(gba_per_pat |> select(PATNO, GBA_carrier), by = "PATNO") |>
  left_join(lrrk2_per_pat |> select(PATNO, LRRK2_carrier), by = "PATNO")

# Per-subject summary of subgroup membership
subj_anno <- df_anno |> distinct(PATNO, .keep_all = TRUE) |>
  select(PATNO, Cohort_Current, SAA_Status, APOE_e4_carrier, GBA_carrier, LRRK2_carrier)
cat("\nPer-subject subgroup distribution within modeling cohort:\n")
cat(stepf("  SAA Positive=%d, Negative=%d, Unknown=%d\n",
            sum(subj_anno$SAA_Status == "Positive", na.rm=TRUE),
            sum(subj_anno$SAA_Status == "Negative", na.rm=TRUE),
            sum(is.na(subj_anno$SAA_Status))))
cat(stepf("  APOE e4 carrier=%d, non=%d, NA=%d\n",
            sum(subj_anno$APOE_e4_carrier, na.rm=TRUE),
            sum(!subj_anno$APOE_e4_carrier, na.rm=TRUE),
            sum(is.na(subj_anno$APOE_e4_carrier))))
cat(stepf("  GBA carrier=%d, non=%d, NA=%d\n",
            sum(subj_anno$GBA_carrier, na.rm=TRUE),
            sum(!subj_anno$GBA_carrier, na.rm=TRUE),
            sum(is.na(subj_anno$GBA_carrier))))
cat(stepf("  LRRK2 carrier=%d, non=%d, NA=%d\n",
            sum(subj_anno$LRRK2_carrier, na.rm=TRUE),
            sum(!subj_anno$LRRK2_carrier, na.rm=TRUE),
            sum(is.na(subj_anno$LRRK2_carrier))))

# ---- Run lmer per subgroup ----
results <- list()

run_subgroup_lmer <- function(df_g, label, outcome = "UPDRS3_Total",
                              baseline_col = "UPDRS3_Baseline") {
  keep_pat <- names(which(table(df_g$PATNO) >= 3))
  df_g <- df_g[df_g$PATNO %in% keep_pat, ]
  n_pat <- length(unique(df_g$PATNO))
  if (n_pat < 15) {
    cat(stepf("  %s: only %d subjects, skipping\n", label, n_pat))
    return(NULL)
  }
  formula_str <- stepf(
    "%s ~ Time * BBB_score_z + Age_at_Baseline + Sex + %s + LEDD_total + Neutrophil_pct + Monocyte_pct + (1 + Time | PATNO)",
    outcome, baseline_col)
  m <- tryCatch(
    lmer(as.formula(formula_str), data = df_g, REML = TRUE,
         control = lmerControl(optimizer = "bobyqa", check.conv.singular = "ignore")),
    error = function(e) { cat(stepf("  %s: lmer error: %s\n", label, conditionMessage(e))); NULL })
  if (is.null(m)) return(NULL)
  co <- summary(m)$coefficients
  if (!"Time:BBB_score_z" %in% rownames(co)) return(NULL)
  r <- co["Time:BBB_score_z", ]
  cat(stepf("  %s: n=%d, estimate=%+.3f, p=%.4f\n", label, n_pat,
              r["Estimate"], r["Pr(>|t|)"]))
  list(label = label, n_subj = n_pat, n_obs = nrow(df_g),
       estimate = r["Estimate"], se = r["Std. Error"], p = r["Pr(>|t|)"])
}

cat("\n=== SAA-stratified UPDRS3 slope ===\n")
results[["SAA_Pos"]] <- run_subgroup_lmer(df_anno[df_anno$SAA_Status == "Positive" &
                                                     !is.na(df_anno$SAA_Status), ], "SAA-Pos")
results[["SAA_Neg"]] <- run_subgroup_lmer(df_anno[df_anno$SAA_Status == "Negative" &
                                                     !is.na(df_anno$SAA_Status), ], "SAA-Neg")

cat("\n=== APOE e4 stratified  -  UPDRS3 + MoCA slopes ===\n")
results[["APOE_e4_carrier_UPDRS3"]] <- run_subgroup_lmer(
  df_anno[df_anno$APOE_e4_carrier %in% TRUE, ], "APOE-e4 carrier (UPDRS3)")
results[["APOE_non_e4_UPDRS3"]] <- run_subgroup_lmer(
  df_anno[df_anno$APOE_e4_carrier %in% FALSE, ], "APOE-non-e4 (UPDRS3)")

# MoCA APOE-stratified
df_moca <- df_anno[!is.na(df_anno$MoCA_Total), ]
bl_moca <- df_moca[!duplicated(df_moca$PATNO), c("PATNO", "MoCA_Total")]
names(bl_moca)[2] <- "MoCA_Baseline"
df_moca <- merge(df_moca, bl_moca, by = "PATNO")
results[["APOE_e4_carrier_MoCA"]] <- run_subgroup_lmer(
  df_moca[df_moca$APOE_e4_carrier %in% TRUE, ], "APOE-e4 carrier (MoCA)",
  outcome = "MoCA_Total", baseline_col = "MoCA_Baseline")
results[["APOE_non_e4_MoCA"]] <- run_subgroup_lmer(
  df_moca[df_moca$APOE_e4_carrier %in% FALSE, ], "APOE-non-e4 (MoCA)",
  outcome = "MoCA_Total", baseline_col = "MoCA_Baseline")

cat("\n=== LRRK2 / GBA carrier subgroups ===\n")
results[["LRRK2_carrier"]] <- run_subgroup_lmer(
  df_anno[df_anno$LRRK2_carrier %in% TRUE, ], "LRRK2 carrier (UPDRS3)")
results[["GBA_carrier"]] <- run_subgroup_lmer(
  df_anno[df_anno$GBA_carrier %in% TRUE, ], "GBA carrier (UPDRS3)")
results[["non_carrier"]] <- run_subgroup_lmer(
  df_anno[(df_anno$LRRK2_carrier %in% FALSE) &
           (df_anno$GBA_carrier %in% FALSE), ], "non-LRRK2/GBA carrier (UPDRS3)")

# ---- Summary ----
results <- Filter(Negate(is.null), results)
cat("\n\n=== SUBGROUP ANALYSES SUMMARY ===\n")
if (length(results) > 0) {
  out <- do.call(rbind, lapply(results, function(x) {
    data.frame(label = x$label, n_subj = x$n_subj, n_obs = x$n_obs,
               estimate = x$estimate, se = x$se, p = x$p)
  }))
  rownames(out) <- NULL
  out$bh_padj <- p.adjust(out$p, method = "BH")
  print(out, row.names = FALSE)
  write.table(out, file.path(S3, "subgroup_analyses_summary.tsv"),
              sep = "\t", row.names = FALSE, quote = FALSE)
  cat(stepf("\n→ Wrote %s/subgroup_analyses_summary.tsv\n", S3))

  any_sig <- any(out$p < 0.05, na.rm = TRUE)
  if (any_sig) {
    cat("\n*** AT LEAST ONE SUBGROUP HAS p < 0.05 (uncorrected) ***\n")
    print(out[out$p < 0.05, ], row.names = FALSE)
  } else {
    cat("\nAll subgroups null at uncorrected p < 0.05.\n")
  }
} else {
  cat("No subgroup models converged with sufficient sample size.\n")
}
