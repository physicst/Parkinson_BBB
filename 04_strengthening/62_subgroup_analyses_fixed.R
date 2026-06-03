## Step 4 Task 3  -  FIXED LRRK2 / GBA / APOE carrier-call parsing.
##
## The original Step 5 Task 53 used loose grepl("LRRK2"|"GBA") that captured
## CSF protein assays, fluorospot panels, expression rep arrays, etc., and
## flagged ~all tested patients as "carriers". This rewrite restricts to the
## actual genotyping TESTNAMEs and uses explicit allele/variant rules.
##
## Carrier definitions:
##   - APOE e4: TESTNAME "APOE GENOTYPE", TESTVALUE contains "e4" or "E4"
##   - LRRK2:   any of the 4 SNP TESTNAMEs (G2019S, R1441H, Y1699C, I2020T)
##              with a heterozygous/homozygous variant call (i.e. NOT the WT
##              homozygous reference).
##   - GBA:     rs76763715_GBA_p.N370S = C/T (one carrier allele), OR
##              "Full GBA gene sequencing" with a pathogenic call
##              (anything NOT WT/WT and NOT "Non pathogenic copy number loss").
##
## Re-runs all pre-registered subgroup lmer models with corrected groups,
## writes results/step4/subgroup_analyses_fixed.tsv.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(arrow)
  library(dplyr)
})

S3 <- "results/step3"
S4 <- "results/step4"
dir.create(S4, showWarnings = FALSE, recursive = TRUE)

df <- as.data.frame(read_parquet(file.path(S3, "modeling_cohort_long.parquet")))
df$Sex <- factor(df$Sex, levels = c("Female", "Male"))
df$BBB_score_z <- as.numeric(scale(df$bbb_score))
df$PATNO <- as.character(df$PATNO)

# Propagate SAA per subject
saa_per_patient <- df |>
  filter(!is.na(SAA_Status), SAA_Status %in% c("Positive", "Negative")) |>
  distinct(PATNO, SAA_Status)
df_saa <- df |> select(-SAA_Status) |> left_join(saa_per_patient, by = "PATNO")

# ---- Load biospecimen ----
biospec <- read.csv("D:/Parkinson_file/Current_Biospecimen_Analysis_Results_02May2026.csv",
                    stringsAsFactors = FALSE)
biospec$PATNO <- as.character(biospec$PATNO)

# ---- APOE e4 ----
apoe <- biospec[toupper(biospec$TESTNAME) == "APOE GENOTYPE", c("PATNO", "TESTVALUE")]
apoe <- apoe[!duplicated(apoe$PATNO), ]
names(apoe)[2] <- "APOE_genotype"
apoe$APOE_e4_carrier <- grepl("e4|E4", apoe$APOE_genotype)
cat(stepf("APOE: %d patients tested; e4 carriers: %d (%.1f%%)\n",
            nrow(apoe), sum(apoe$APOE_e4_carrier),
            100 * mean(apoe$APOE_e4_carrier)))

# ---- LRRK2 SNP carriers ----
lrrk2_snps <- c("rs34637584_LRRK2_p.G2019S",
                "rs34995376_LRRK2_p.R1441H",
                "rs35801418_LRRK2_p.Y1699C",
                "rs35870237_LRRK2_p.I2020T")
# WT homozygotes per SNP (from inspection: G/G, G/G, A/A, T/T)
wt_call <- c("rs34637584_LRRK2_p.G2019S" = "G/G",
             "rs34995376_LRRK2_p.R1441H" = "G/G",
             "rs35801418_LRRK2_p.Y1699C" = "A/A",
             "rs35870237_LRRK2_p.I2020T" = "T/T")

lrrk2_long <- biospec[biospec$TESTNAME %in% lrrk2_snps,
                      c("PATNO", "TESTNAME", "TESTVALUE")]
lrrk2_long$is_variant <- mapply(function(tn, tv) {
  if (is.na(tv) || tv == "") return(NA)
  tv != wt_call[[tn]]
}, lrrk2_long$TESTNAME, lrrk2_long$TESTVALUE)

lrrk2_per_pat <- lrrk2_long |>
  group_by(PATNO) |>
  summarise(LRRK2_tested = TRUE,
            LRRK2_variants = paste(TESTNAME[is_variant %in% TRUE], collapse = ";"),
            LRRK2_carrier = any(is_variant %in% TRUE),
            .groups = "drop")
cat(stepf("LRRK2 SNP-tested: %d patients; carriers: %d\n",
            nrow(lrrk2_per_pat), sum(lrrk2_per_pat$LRRK2_carrier)))
if (sum(lrrk2_per_pat$LRRK2_carrier) > 0) {
  cat("  Variant breakdown among carriers:\n")
  print(table(lrrk2_per_pat$LRRK2_variants[lrrk2_per_pat$LRRK2_carrier]))
}

# ---- GBA carriers ----
# Strict definition: N370S het OR pathogenic call in full sequencing.
# Excluded: "Non pathogenic copy number loss" treated as non-carrier per name.
gba_n370s <- biospec[biospec$TESTNAME == "rs76763715_GBA_p.N370S",
                     c("PATNO", "TESTVALUE")]
gba_n370s$N370S_carrier <- gba_n370s$TESTVALUE != "T/T" &
                           !is.na(gba_n370s$TESTVALUE) & gba_n370s$TESTVALUE != ""

gba_seq <- biospec[biospec$TESTNAME == "Full GBA gene sequencing",
                   c("PATNO", "TESTVALUE")]
gba_seq$Seq_carrier <- !(gba_seq$TESTVALUE %in% c("WT/WT", "Non pathogenic copy number loss")) &
                       !is.na(gba_seq$TESTVALUE) & gba_seq$TESTVALUE != ""

gba_per_pat <- full_join(
  gba_n370s |> rename(N370S_TESTVALUE = TESTVALUE) |> select(PATNO, N370S_TESTVALUE, N370S_carrier),
  gba_seq   |> rename(Seq_TESTVALUE   = TESTVALUE) |> select(PATNO, Seq_TESTVALUE,   Seq_carrier),
  by = "PATNO"
)
gba_per_pat$GBA_tested <- !is.na(gba_per_pat$N370S_carrier) | !is.na(gba_per_pat$Seq_carrier)
gba_per_pat$GBA_carrier <- (gba_per_pat$N370S_carrier %in% TRUE) |
                           (gba_per_pat$Seq_carrier   %in% TRUE)
cat(stepf("GBA tested (N370S or full sequencing): %d patients; carriers: %d\n",
            sum(gba_per_pat$GBA_tested), sum(gba_per_pat$GBA_carrier)))

# Merge into modeling cohort
df_anno <- df_saa |>
  left_join(apoe |> select(PATNO, APOE_e4_carrier), by = "PATNO") |>
  left_join(lrrk2_per_pat |> select(PATNO, LRRK2_carrier), by = "PATNO") |>
  left_join(gba_per_pat   |> select(PATNO, GBA_carrier),   by = "PATNO")

subj_anno <- df_anno |> distinct(PATNO, .keep_all = TRUE) |>
  select(PATNO, Cohort_Current, SAA_Status, APOE_e4_carrier, LRRK2_carrier, GBA_carrier)

cat("\nWithin Olink modeling cohort (n=61):\n")
cat(stepf("  APOE e4 carrier=%d, non=%d, NA=%d\n",
            sum(subj_anno$APOE_e4_carrier %in% TRUE),
            sum(subj_anno$APOE_e4_carrier %in% FALSE),
            sum(is.na(subj_anno$APOE_e4_carrier))))
cat(stepf("  LRRK2 carrier=%d, non=%d, NA=%d\n",
            sum(subj_anno$LRRK2_carrier %in% TRUE),
            sum(subj_anno$LRRK2_carrier %in% FALSE),
            sum(is.na(subj_anno$LRRK2_carrier))))
cat(stepf("  GBA carrier=%d, non=%d, NA=%d\n",
            sum(subj_anno$GBA_carrier %in% TRUE),
            sum(subj_anno$GBA_carrier %in% FALSE),
            sum(is.na(subj_anno$GBA_carrier))))

# ---- lmer per subgroup ----
results <- list()

run_subgroup_lmer <- function(df_g, label, outcome = "UPDRS3_Total",
                              baseline_col = "UPDRS3_Baseline") {
  keep_pat <- names(which(table(df_g$PATNO) >= 3))
  df_g <- df_g[df_g$PATNO %in% keep_pat, ]
  n_pat <- length(unique(df_g$PATNO))
  if (n_pat < 15) {
    cat(stepf("  %s: n=%d (<15), skipping\n", label, n_pat))
    return(NULL)
  }
  formula_str <- stepf(
    "%s ~ Time * BBB_score_z + Age_at_Baseline + Sex + %s + LEDD_total + Neutrophil_pct + Monocyte_pct + (1 + Time | PATNO)",
    outcome, baseline_col)
  m <- tryCatch(
    lmer(as.formula(formula_str), data = df_g, REML = TRUE,
         control = lmerControl(optimizer = "bobyqa", check.conv.singular = "ignore")),
    error = function(e) { cat(stepf("  %s: error: %s\n", label, conditionMessage(e))); NULL })
  if (is.null(m)) return(NULL)
  co <- summary(m)$coefficients
  if (!"Time:BBB_score_z" %in% rownames(co)) return(NULL)
  r <- co["Time:BBB_score_z", ]
  cat(stepf("  %s: n=%d, est=%+.3f, p=%.4f\n", label, n_pat,
              r["Estimate"], r["Pr(>|t|)"]))
  list(label = label, n_subj = n_pat, n_obs = nrow(df_g),
       estimate = r["Estimate"], se = r["Std. Error"], p = r["Pr(>|t|)"])
}

cat("\n=== SAA-stratified UPDRS3 slope ===\n")
results[["SAA_Pos"]] <- run_subgroup_lmer(
  df_anno[df_anno$SAA_Status %in% "Positive", ], "SAA-Pos")
results[["SAA_Neg"]] <- run_subgroup_lmer(
  df_anno[df_anno$SAA_Status %in% "Negative", ], "SAA-Neg")

cat("\n=== APOE e4 stratified UPDRS3 + MoCA slopes ===\n")
results[["APOE_e4_carrier_UPDRS3"]] <- run_subgroup_lmer(
  df_anno[df_anno$APOE_e4_carrier %in% TRUE, ], "APOE-e4 carrier (UPDRS3)")
results[["APOE_non_e4_UPDRS3"]] <- run_subgroup_lmer(
  df_anno[df_anno$APOE_e4_carrier %in% FALSE, ], "APOE-non-e4 (UPDRS3)")

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

cat("\n=== LRRK2 / GBA carrier subgroups (corrected calls) ===\n")
results[["LRRK2_carrier"]] <- run_subgroup_lmer(
  df_anno[df_anno$LRRK2_carrier %in% TRUE, ], "LRRK2 carrier (UPDRS3)")
results[["GBA_carrier"]] <- run_subgroup_lmer(
  df_anno[df_anno$GBA_carrier %in% TRUE, ], "GBA carrier (UPDRS3)")
results[["non_carrier"]] <- run_subgroup_lmer(
  df_anno[(df_anno$LRRK2_carrier %in% FALSE) &
           (df_anno$GBA_carrier %in% FALSE), ],
  "non-LRRK2/GBA carrier (UPDRS3)")

# Summary
results <- Filter(Negate(is.null), results)
cat("\n\n=== FIXED SUBGROUP ANALYSES SUMMARY ===\n")
if (length(results) > 0) {
  out <- do.call(rbind, lapply(results, function(x) {
    data.frame(label = x$label, n_subj = x$n_subj, n_obs = x$n_obs,
               estimate = x$estimate, se = x$se, p = x$p)
  }))
  rownames(out) <- NULL
  out$bh_padj <- p.adjust(out$p, method = "BH")
  print(out, row.names = FALSE)
  write.table(out, file.path(S4, "subgroup_analyses_fixed.tsv"),
              sep = "\t", row.names = FALSE, quote = FALSE)
  cat(stepf("\n→ Wrote %s/subgroup_analyses_fixed.tsv\n", S4))

  # Save corrected per-subject annotation
  write.table(subj_anno, file.path(S4, "subject_genotype_annotation_fixed.tsv"),
              sep = "\t", row.names = FALSE, quote = FALSE, na = "")
  cat(stepf("→ Wrote %s/subject_genotype_annotation_fixed.tsv\n", S4))
} else {
  cat("No subgroup models converged.\n")
}
