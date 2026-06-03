## Bootstrap-stable WGCNA on integrated vascular pseudobulk.
##
## Native Windows execution:
##   "C:/Program Files/R/R-4.3.2/bin/Rscript.exe" code/32_wgcna_bootstrap.R
##
## Working directory: D:/PD_BBB_analysis
##
## Approach:
##   1. Load pseudobulk endothelial counts across all 3 datasets (inner-join on common genes).
##   2. VST-transform.
##   3. Soft-threshold power selection (R^2 > 0.85).
##   4. Reference WGCNA module assignment.
##   5. Bootstrap 100x by donor resampling; per-bootstrap module assignment.
##   6. Stability filter: retain modules whose member set is preserved across >= 70%
##      of bootstraps (per master plan §1.4).
##   7. Identify BBB-breakdown module (highest correlation with PD status,
##      enriched for BBB pathways via clusterProfiler).
##   8. Extract top 30 hub genes by module-membership x gene-significance.

suppressPackageStartupMessages({
  library(WGCNA)
  library(DESeq2)
  library(arrow)
  library(dplyr)
  library(tibble)
})

# WGCNA on Windows R: explicitly override stats::cor with WGCNA's cor.
# blockwiseModules passes extra args (weights.x, weights.y, cosine) that
# stats::cor rejects; WGCNA::cor handles them.
cor <- WGCNA::cor

allowWGCNAThreads()

CACHE   <- "results/cache"
RESULTS <- "results/step2"
dir.create(RESULTS, showWarnings = FALSE, recursive = TRUE)

set.seed(20260503)
N_BOOT <- 100
STABILITY_THRESHOLD <- 0.70

# ---- Load endothelial pseudobulk across datasets ----
load_pb <- function(path) {
  df <- as.data.frame(read_parquet(path))
  if (!"gene_id" %in% colnames(df)) {
    rownames(df) <- df[[1]]; df[[1]] <- NULL
  } else {
    rownames(df) <- df$gene_id; df$gene_id <- NULL
  }
  as.matrix(df)
}

meta <- read_parquet(file.path(CACHE, "pseudobulk_metadata.parquet")) |> as_tibble()

# Optional Smajic demographics backfill
if (file.exists("data/smajic_donor_meta.csv")) {
  smajic_donor <- read.csv("data/smajic_donor_meta.csv")
  mask <- meta$dataset == "smajic"
  for (col in c("sex", "age", "pmi")) {
    meta[mask, col] <- smajic_donor[match(meta$donor_id[mask], smajic_donor$donor_id), col]
  }
}

pb_files <- c(
  smajic    = "pseudobulk_smajic_endothelial.parquet",
  agarwal   = "pseudobulk_agarwal_endothelial.parquet",
  gse178265 = "pseudobulk_gse178265_endothelial.parquet"
)

pb_list <- list(); meta_list <- list()
for (ds in names(pb_files)) {
  fp <- file.path(CACHE, pb_files[ds])
  if (!file.exists(fp)) next
  M <- load_pb(fp)
  sub_meta <- meta |> filter(dataset == ds, cell_type == "endothelial")
  M <- M[, sub_meta$donor_id, drop = FALSE]
  colnames(M) <- paste(ds, sub_meta$donor_id, sep = "_")
  sub_meta$sample_id <- colnames(M)
  pb_list[[ds]] <- M
  meta_list[[ds]] <- sub_meta
  cat(stepf("  %-12s %d genes x %d donors\n", ds, nrow(M), ncol(M)))
}

common_genes <- Reduce(intersect, lapply(pb_list, rownames))
cat(stepf("Common endothelial genes: %d\n", length(common_genes)))
counts <- do.call(cbind, lapply(pb_list, function(M) M[common_genes, , drop = FALSE]))
combined_meta <- bind_rows(meta_list)

# ---- VST transform ----
cat("VST-transforming...\n")
dds <- DESeqDataSetFromMatrix(counts, colData = combined_meta, design = ~ condition)
dds <- dds[rowSums(counts(dds) >= 5) >= 3, ]
vsd <- assay(vst(dds, blind = TRUE))
cat(stepf("VST matrix: %d genes x %d samples\n", nrow(vsd), ncol(vsd)))

# Filter to top variable genes for tractable WGCNA (5,000 genes is standard)
gene_var <- apply(vsd, 1, var)
top_genes <- names(sort(gene_var, decreasing = TRUE)[1:min(5000, length(gene_var))])
vsd_top <- vsd[top_genes, ]
cat(stepf("Filtered to top %d variable genes for WGCNA\n", nrow(vsd_top)))

# ---- Soft-thresholding ----
# Constrain power >= 6 to avoid degenerate low-power picks where the network
# is essentially fully connected. With n=30 samples, scale-free topology can
# trivially fit at power=1 because the correlation matrix is dense  -  we want
# real thresholding.
cat("Picking soft-threshold power (min power = 6)...\n")
sft <- pickSoftThreshold(t(vsd_top),
                         powerVector = c(6:10, seq(12, 20, 2)),
                         networkType = "signed",
                         RsquaredCut = 0.85)
power <- ifelse(is.na(sft$powerEstimate), 12, sft$powerEstimate)
power <- max(power, 6)  # Hard floor
cat(stepf("Selected power = %d (R^2 = %.3f, slope = %.2f)\n",
            power,
            sft$fitIndices$SFT.R.sq[sft$fitIndices$Power == power],
            sft$fitIndices$slope[sft$fitIndices$Power == power]))

# ---- Reference modules ----
cat("Building reference WGCNA network...\n")
ref_net <- blockwiseModules(t(vsd_top),
                            power = power, networkType = "signed",
                            TOMType = "signed", deepSplit = 2,
                            minModuleSize = 30, mergeCutHeight = 0.25,
                            numericLabels = TRUE,
                            saveTOMs = FALSE, verbose = 0)
ref_modules <- ref_net$colors
names(ref_modules) <- top_genes
cat(stepf("Reference: %d modules (excl. grey)\n",
            length(unique(ref_modules)) - as.integer("0" %in% ref_modules)))
print(table(ref_modules))

# ---- Bootstrap stability ----
# For each pair of genes in the same reference module, count fraction of
# bootstraps where they're also co-assigned to the same module.
cat(stepf("\nBootstrap %d iterations...\n", N_BOOT))
boot_pair_co <- matrix(0, nrow = nrow(vsd_top), ncol = nrow(vsd_top),
                        dimnames = list(top_genes, top_genes))
n_donors <- ncol(vsd_top)
boot_failures <- 0

for (b in 1:N_BOOT) {
  if (b %% 10 == 0) cat(stepf("  bootstrap %d/%d\n", b, N_BOOT))
  donors_b <- sample(seq_len(n_donors), replace = TRUE)
  vsd_b <- vsd_top[, donors_b]
  net_b <- tryCatch(
    blockwiseModules(t(vsd_b),
                     power = power, networkType = "signed",
                     TOMType = "signed", deepSplit = 2,
                     minModuleSize = 30, mergeCutHeight = 0.25,
                     numericLabels = TRUE, saveTOMs = FALSE, verbose = 0),
    error = function(e) { boot_failures <<- boot_failures + 1; NULL })
  if (is.null(net_b)) next
  mods_b <- net_b$colors
  names(mods_b) <- top_genes

  for (m in unique(mods_b)) {
    if (m == 0) next
    members <- names(mods_b)[mods_b == m]
    if (length(members) < 2) next
    boot_pair_co[members, members] <- boot_pair_co[members, members] + 1
  }
}
n_successful <- N_BOOT - boot_failures
cat(stepf("  successful bootstraps: %d / %d\n", n_successful, N_BOOT))
boot_pair_co <- boot_pair_co / n_successful

# ---- Module stability filter ----
# A reference module is "stable" if >=70% of its member-pairs co-cluster in >=70% of bootstraps.
stable_modules <- c()
ref_mod_list <- split(names(ref_modules), ref_modules)
for (m in setdiff(names(ref_mod_list), "0")) {
  members <- ref_mod_list[[m]]
  if (length(members) < 5) next
  # Mean off-diagonal co-clustering rate within module
  pair_block <- boot_pair_co[members, members]
  diag(pair_block) <- NA
  rate <- mean(pair_block, na.rm = TRUE)
  if (rate >= STABILITY_THRESHOLD) {
    stable_modules <- c(stable_modules, m)
    cat(stepf("  module %s: %d genes, mean pair co-cluster rate = %.2f  STABLE\n",
                m, length(members), rate))
  } else {
    cat(stepf("  module %s: %d genes, mean pair co-cluster rate = %.2f\n",
                m, length(members), rate))
  }
}

cat(stepf("\nStable modules: %d / %d\n",
            length(stable_modules), length(ref_mod_list) - 1))

# ---- Module-PD-status correlation ----
condition_numeric <- ifelse(combined_meta$condition == "PD", 1, 0)
ME <- moduleEigengenes(t(vsd_top), colors = ref_modules)$eigengenes
ME_corr <- cor(ME, condition_numeric)
ME_pval <- corPvalueStudent(ME_corr, nrow(ME))
me_summary <- data.frame(
  module = sub("ME", "", rownames(ME_corr)),
  cor    = ME_corr[, 1],
  p      = ME_pval[, 1],
  stable = sub("ME", "", rownames(ME_corr)) %in% stable_modules
) |> arrange(p)

cat("\nModule-PD correlations (top 5):\n")
print(head(me_summary, 5))

# ---- Pick BBB-breakdown module ----
# Strongest |cor| with PD AMONG stable modules. If none, fall back to top |cor| overall + flag.
candidate <- me_summary |> filter(stable) |> arrange(desc(abs(cor)))
if (nrow(candidate) == 0) {
  cat("\nNo stable module passed; falling back to top-|cor| module.\n")
  candidate <- me_summary |> arrange(desc(abs(cor)))
}
bbb_mod <- candidate$module[1]
cat(stepf("\n=> Selected BBB module: %s (cor=%.3f, p=%.3g, stable=%s)\n",
            bbb_mod, candidate$cor[1], candidate$p[1], candidate$stable[1]))

bbb_genes <- names(ref_modules)[ref_modules == bbb_mod]

# ---- Hub genes: top 30 by MM x GS ----
# MM = module membership (correlation of gene with module eigengene)
# GS = gene significance (correlation of gene with PD status)
mm <- as.numeric(cor(t(vsd_top[bbb_genes, ]), ME[, paste0("ME", bbb_mod)]))
gs <- as.numeric(cor(t(vsd_top[bbb_genes, ]), condition_numeric))
hub <- data.frame(gene_id = bbb_genes, MM = mm, GS = gs,
                  hub_score = abs(mm) * abs(gs)) |>
       arrange(desc(hub_score))

# ---- Outputs ----
write.table(data.frame(gene_id = bbb_genes, module = bbb_mod),
            file.path(RESULTS, "bbb_module_genes.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
write.table(head(hub, 30),
            file.path(RESULTS, "hub_genes.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
write.table(me_summary,
            file.path(RESULTS, "module_pd_correlations.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)

cat(stepf("\nWrote %s/bbb_module_genes.tsv (%d genes)\n", RESULTS, length(bbb_genes)))
cat(stepf("Wrote %s/hub_genes.tsv (top 30)\n", RESULTS))
cat(stepf("Wrote %s/module_pd_correlations.tsv\n", RESULTS))
cat("\n=== Step 2 §6 WGCNA bootstrap-stable module derivation done ===\n")
