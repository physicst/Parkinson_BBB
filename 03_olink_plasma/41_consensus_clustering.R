## Step 3 Task 31  -  ConsensusClusterPlus k=2 (primary endotype) on Olink BBB panel.
##
## Inputs:
##   results/step3/olink_bbb_panel_baseline.parquet  (227 patients × 20 proteins)
##
## Outputs:
##   results/step3/consensus_k2_assignments.tsv      (PATNO -> cluster, vascular_class)
##   results/step3/figures/consensus_cdf.png
##   results/step3/figures/cluster_heatmap_k2.png
##
## Per pre-reg §14 deviation 1: ConsensusClusterPlus k=2, 1000 reps,
## hierarchical Ward.D2 + Pearson distance. Vascular-High = cluster with
## higher mean NPX across canonical endothelial-activation markers
## (VCAM1, ICAM1, SELE, VWF).

suppressPackageStartupMessages({
  library(ConsensusClusterPlus)
  library(arrow)
  library(dplyr)
  library(tibble)
  library(pheatmap)
})

set.seed(20260504)

# ---- Load panel matrix ----
panel <- as.data.frame(read_parquet("results/step3/olink_bbb_panel_baseline.parquet"))
rownames(panel) <- panel$PATNO
panel$PATNO <- NULL
cat(stepf("Loaded panel matrix: %d patients × %d proteins\n",
            nrow(panel), ncol(panel)))

# Drop any remaining NAs for clustering (impute with column mean to keep patients in)
n_na_before <- sum(is.na(panel))
if (n_na_before > 0) {
  for (j in seq_along(panel)) {
    miss <- is.na(panel[[j]])
    if (any(miss)) panel[[j]][miss] <- mean(panel[[j]], na.rm = TRUE)
  }
  cat(stepf("Imputed %d NA cells with column means.\n", n_na_before))
}

# Z-score per protein
z <- scale(panel)

# ConsensusClusterPlus expects features (proteins) as ROWS, samples as COLUMNS
ccp_input <- t(z)

cat("\nRunning ConsensusClusterPlus (1000 reps, Pearson + Ward.D2, k=2..6)...\n")
out_dir <- "results/step3/ccp"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ccp <- ConsensusClusterPlus(
  d = ccp_input,
  maxK = 6,
  reps = 1000,
  pItem = 0.8,
  pFeature = 1.0,
  clusterAlg = "hc",
  distance = "pearson",
  innerLinkage = "ward.D2",
  finalLinkage = "ward.D2",
  title = out_dir,
  plot = "png",
  seed = 20260504,
  verbose = FALSE,
)

# k=2 primary
k2 <- ccp[[2]]$consensusClass
assignments <- data.frame(
  PATNO = colnames(ccp_input),
  cluster = as.integer(k2)
)

# Identify Vascular-High by mean of canonical endothelial-activation markers
ENDO_ACTIV <- c("VCAM1", "ICAM1", "SELE", "VWF")
endo_in <- intersect(ENDO_ACTIV, colnames(panel))
cat(stepf("\nUsing endothelial-activation markers for cluster labeling: %s\n",
            paste(endo_in, collapse = ", ")))

mean_per_cluster <- assignments |>
  inner_join(
    panel |> rownames_to_column("PATNO"), by = "PATNO"
  ) |>
  group_by(cluster) |>
  summarise(across(all_of(endo_in), mean), .groups = "drop") |>
  mutate(mean_endo = rowMeans(across(all_of(endo_in))))

print(mean_per_cluster)

vascular_high_cluster <- mean_per_cluster$cluster[which.max(mean_per_cluster$mean_endo)]
assignments$vascular_class <- ifelse(
  assignments$cluster == vascular_high_cluster, "Vascular_High", "Vascular_Low"
)

cat(stepf("\nCluster sizes:\n"))
print(table(assignments$vascular_class))

# Save assignments
write.table(assignments,
            "results/step3/consensus_k2_assignments.tsv",
            sep = "\t", row.names = FALSE, quote = FALSE)
cat("\n→ Wrote results/step3/consensus_k2_assignments.tsv\n")

# Heatmap of z-scored panel ordered by cluster
fig_dir <- "results/step3/figures"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

annot <- data.frame(
  vascular_class = assignments$vascular_class,
  row.names = assignments$PATNO
)
ord <- order(assignments$vascular_class, assignments$PATNO)

png(file.path(fig_dir, "cluster_heatmap_k2.png"), width = 1100, height = 700, res = 110)
pheatmap(
  t(z[ord, ]),
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  annotation_col = annot,
  show_colnames = FALSE,
  scale = "none",
  main = "Olink BBB panel (z-scored NPX) by consensus k=2 endotype",
  fontsize_row = 9,
)
dev.off()
cat(stepf("→ Wrote %s/cluster_heatmap_k2.png\n", fig_dir))

# Cluster size summary
cat(stepf("\n=== Step 3 Task 31 done ===\n"))
cat(stepf("Vascular-High: %d patients\n", sum(assignments$vascular_class == "Vascular_High")))
cat(stepf("Vascular-Low:  %d patients\n", sum(assignments$vascular_class == "Vascular_Low")))
