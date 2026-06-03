## Pseudobulk DESeq2 with full pre-registered design across all 3 datasets.
##
## Design: ~ condition + sex + age + pmi + dataset
##
## Native Windows execution (no Docker, no KISTI):
##   "C:/Program Files/R/R-4.3.2/bin/Rscript.exe" code/31_deseq2_integrated.R
##
## Working directory: D:/PD_BBB_analysis
##
## Inputs (parquets from Step 2 Task 26):
##   results/cache/pseudobulk_smajic_endothelial.parquet      (11 donors)
##   results/cache/pseudobulk_smajic_pericyte.parquet         (9 donors)
##   results/cache/pseudobulk_agarwal_endothelial.parquet     (1 donor)
##   results/cache/pseudobulk_agarwal_pericyte.parquet        (1 donor)
##   results/cache/pseudobulk_gse178265_endothelial.parquet   (18 donors)
##   results/cache/pseudobulk_metadata.parquet                (combined metadata)
##   data/smajic_donor_meta.csv                                (Smajic sex/age/pmi backfill)
##
## Outputs:
##   results/step2/integrated_de_endothelial.tsv
##   results/step2/integrated_de_pericyte.tsv  (if power permits)

suppressPackageStartupMessages({
  library(DESeq2)
  library(arrow)
  library(dplyr)
  library(tibble)
})

CACHE   <- "results/cache"
RESULTS <- "results/step2"
dir.create(RESULTS, showWarnings = FALSE, recursive = TRUE)

# ---- Load metadata + optional Smajic backfill ----
meta <- read_parquet(file.path(CACHE, "pseudobulk_metadata.parquet")) |> as_tibble()
cat(stepf("Metadata rows: %d\n", nrow(meta)))

smajic_meta_path <- "data/smajic_donor_meta.csv"
if (file.exists(smajic_meta_path)) {
  cat(stepf("Backfilling Smajic demographics from %s\n", smajic_meta_path))
  smajic_donor <- read.csv(smajic_meta_path, stringsAsFactors = FALSE)
  mask <- meta$dataset == "smajic"
  for (col in c("sex", "age", "pmi")) {
    meta[mask, col] <- smajic_donor[match(meta$donor_id[mask], smajic_donor$donor_id), col]
  }
} else {
  cat("WARNING: data/smajic_donor_meta.csv not found  -  Smajic sex/age/pmi will be NA.\n")
  cat("         Design will fall back to ~ condition + dataset for the rows missing covariates.\n")
}

# ---- Load pseudobulks: inner-join on common Ensembl gene IDs ----
load_pb <- function(path) {
  df <- as.data.frame(read_parquet(path))
  if (!"gene_id" %in% colnames(df)) {
    # arrow may have used the index column name; check first column
    rownames(df) <- df[[1]]
    df[[1]] <- NULL
  } else {
    rownames(df) <- df$gene_id
    df$gene_id <- NULL
  }
  as.matrix(df)
}

pb_paths <- list(
  c("smajic", "endothelial", "pseudobulk_smajic_endothelial.parquet"),
  c("smajic", "pericyte",    "pseudobulk_smajic_pericyte.parquet"),
  c("agarwal", "endothelial", "pseudobulk_agarwal_endothelial.parquet"),
  c("agarwal", "pericyte",    "pseudobulk_agarwal_pericyte.parquet"),
  c("gse178265", "endothelial", "pseudobulk_gse178265_endothelial.parquet")
)

run_de <- function(cell_type) {
  cat(stepf("\n=== %s pseudobulk DE ===\n", cell_type))

  pb_list <- list()
  meta_list <- list()
  for (entry in pb_paths) {
    ds <- entry[1]; ct <- entry[2]; fn <- entry[3]
    if (ct != cell_type) next
    if (!file.exists(file.path(CACHE, fn))) next
    M <- load_pb(file.path(CACHE, fn))
    sub_meta <- meta |> filter(dataset == ds, cell_type == !!cell_type)
    if (nrow(sub_meta) == 0 || ncol(M) == 0) next
    # Align matrix columns to metadata donor_ids
    M <- M[, sub_meta$donor_id, drop = FALSE]
    colnames(M) <- paste(ds, sub_meta$donor_id, sep = "_")
    sub_meta$sample_id <- colnames(M)
    pb_list[[ds]] <- M
    meta_list[[ds]] <- sub_meta
    cat(stepf("  %-12s %d genes x %d donors\n", ds, nrow(M), ncol(M)))
  }

  if (length(pb_list) < 1) {
    cat(stepf("  No data for %s; skipping.\n", cell_type))
    return(invisible(NULL))
  }

  # Inner-join on common genes
  common_genes <- Reduce(intersect, lapply(pb_list, rownames))
  cat(stepf("  Common genes across datasets: %d\n", length(common_genes)))
  counts <- do.call(cbind, lapply(pb_list, function(M) M[common_genes, , drop = FALSE]))
  combined_meta <- bind_rows(meta_list)

  # DROP rows with missing covariates
  keep <- complete.cases(combined_meta[, c("condition", "sex", "age", "pmi")])
  if (sum(!keep) > 0) {
    cat(stepf("  Dropping %d rows with missing covariates (likely Smajic without backfill).\n",
                sum(!keep)))
    counts <- counts[, keep, drop = FALSE]
    combined_meta <- combined_meta[keep, ]
  }

  # Need at least 2 levels of condition AND at least 4 samples per condition
  cond_n <- table(combined_meta$condition)
  cat(stepf("  Condition counts: %s\n", paste(names(cond_n), cond_n, sep="=", collapse=", ")))
  if (length(cond_n) < 2 || min(cond_n) < 3) {
    cat("  Insufficient condition coverage; skipping.\n")
    return(invisible(NULL))
  }

  # If only one dataset survives, drop dataset covariate
  has_dataset <- length(unique(combined_meta$dataset)) >= 2
  design_formula <- if (has_dataset) {
    ~ condition + sex + age + pmi + dataset
  } else {
    ~ condition + sex + age + pmi
  }
  cat(stepf("  Design: %s\n", deparse(design_formula)))

  combined_meta$condition <- factor(combined_meta$condition,
                                     levels = c("Control", setdiff(unique(combined_meta$condition), "Control")))
  combined_meta$sex     <- factor(combined_meta$sex)
  if (has_dataset) combined_meta$dataset <- factor(combined_meta$dataset)

  dds <- DESeqDataSetFromMatrix(
    countData = counts,
    colData   = combined_meta,
    design    = design_formula
  )
  dds <- dds[rowSums(counts(dds) >= 5) >= 3, ]   # mild gene filter
  dds <- DESeq(dds, parallel = FALSE)

  # Pull contrast
  res <- results(dds, contrast = c("condition", "PD", "Control"))

  # Save
  res_df <- as.data.frame(res) |>
    rownames_to_column("gene_id") |>
    arrange(padj)
  out_path <- file.path(RESULTS, stepf("integrated_de_%s.tsv", cell_type))
  write.table(res_df, out_path, sep = "\t", row.names = FALSE, quote = FALSE)
  cat(stepf("  Wrote %s (%d genes, %d significant at padj<0.05, %d at padj<0.1)\n",
              out_path, nrow(res_df),
              sum(res_df$padj < 0.05, na.rm=TRUE),
              sum(res_df$padj < 0.10, na.rm=TRUE)))

  # Print top hits
  cat("\n  Top 10 by padj:\n")
  print(head(res_df, 10), row.names = FALSE)
}

run_de("endothelial")
run_de("pericyte")

cat("\n=== Done ===\n")
