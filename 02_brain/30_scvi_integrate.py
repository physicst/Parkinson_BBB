"""scVI multi-dataset integration of vascular cells from GSE178265 + Smajic + Agarwal.

Designed to run inside the pd-bbb:gpu Docker container with GPU passthrough.
  bash docker/run.sh python code/30_scvi_integrate.py

Inputs (mounted at /data/snrnaseq/cache/):
  - gse178265_vascular.h5ad     (14,903 endothelial)
  - smajic_vascular.h5ad        (2,952 endo+peri, 11 donors)
  - agarwal_vascular.h5ad       (265 endo+peri, 5 control donors)

Outputs (written to /workspace/results/step2/):
  - integrated_vascular.h5ad    (concatenated + scVI-integrated)
  - scvi_latent.parquet         (cell × 30-dim latent)
  - figures/umap_scvi_integrated.png
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scvi
import torch

# Paths inside the container
DATA = Path("/data/snrnaseq/cache")
RESULTS = Path("/workspace/results/step2")
RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / "figures").mkdir(exist_ok=True)

print(f"torch {torch.__version__} / cuda available: {torch.cuda.is_available()}")
print(f"scvi {scvi.__version__}")


def harmonize_obs(ad: anndata.AnnData, dataset_name: str) -> anndata.AnnData:
    """Ensure each AnnData has obs columns: dataset, donor_id, condition, vascular_class."""
    ad.obs["dataset"] = dataset_name

    # donor_id: harmonize across naming conventions
    if "donor_id" not in ad.obs.columns:
        for cand in ("patient", "Donor", "donor", "Donor_ID"):
            if cand in ad.obs.columns:
                ad.obs["donor_id"] = ad.obs[cand].astype(str)
                break
        else:
            raise KeyError(f"{dataset_name}: no donor column found in obs {list(ad.obs.columns)}")

    # condition
    if "condition" not in ad.obs.columns:
        for cand in ("Status", "disease", "diagnosis"):
            if cand in ad.obs.columns:
                ad.obs["condition"] = ad.obs[cand].astype(str)
                break
        else:
            ad.obs["condition"] = "Unknown"

    # vascular_class must already be set by Task 25
    if "vascular_class" not in ad.obs.columns:
        raise KeyError(f"{dataset_name}: vascular_class missing  -  re-run Task 25")

    # Type cleanup
    for col in ("dataset", "donor_id", "condition", "vascular_class"):
        ad.obs[col] = ad.obs[col].astype(str)

    return ad


def main() -> None:
    print("Loading vascular AnnDatas...")
    smajic = sc.read_h5ad(DATA / "smajic_vascular.h5ad")
    agarwal = sc.read_h5ad(DATA / "agarwal_vascular.h5ad")
    gse = sc.read_h5ad(DATA / "gse178265_vascular.h5ad")

    smajic = harmonize_obs(smajic, "smajic")
    agarwal = harmonize_obs(agarwal, "agarwal")
    gse = harmonize_obs(gse, "gse178265")

    print(f"  smajic   {smajic.shape}")
    print(f"  agarwal  {agarwal.shape}")
    print(f"  gse178265 {gse.shape}")

    # Concatenate (outer join on genes  -  different gene sets across datasets)
    print("Concatenating...")
    adata = anndata.concat(
        [smajic, agarwal, gse],
        join="outer",
        index_unique="-",
        keys=["smajic", "agarwal", "gse178265"],
        merge="same",
    )
    print(f"  combined {adata.shape}")
    print(f"  cells per dataset:\n{adata.obs['dataset'].value_counts()}")

    # Use raw counts. AnnData.X for these h5ads is raw int counts (Task 23/24/25 outputs).
    # scVI requires raw counts on .X (or specify layer).

    # scVI setup
    print("Setting up scVI...")
    scvi.model.SCVI.setup_anndata(
        adata,
        layer=None,
        batch_key="dataset",
        categorical_covariate_keys=["donor_id"],
    )

    # Train. n_latent=30 standard for cross-batch integration.
    model = scvi.model.SCVI(
        adata,
        n_layers=2,
        n_latent=30,
        gene_likelihood="nb",
    )
    print("Training scVI (max 400 epochs, early stopping)...")
    model.train(
        max_epochs=400,
        early_stopping=True,
        early_stopping_patience=20,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    # Latent representation
    latent = model.get_latent_representation()
    adata.obsm["X_scvi"] = latent

    # Diagnostic UMAP
    print("Computing UMAP for integration diagnostic...")
    sc.pp.neighbors(adata, use_rep="X_scvi", n_neighbors=30)
    sc.tl.umap(adata, min_dist=0.3)
    sc.settings.figdir = str(RESULTS / "figures")
    sc.pl.umap(
        adata,
        color=["dataset", "vascular_class", "condition", "donor_id"],
        wspace=0.4,
        save="_scvi_integrated.png",
        show=False,
    )

    # Save outputs
    print("Saving outputs...")
    adata.write_h5ad(RESULTS / "integrated_vascular.h5ad")

    pd.DataFrame(latent, index=adata.obs_names).to_parquet(
        RESULTS / "scvi_latent.parquet"
    )

    # Summary
    print(f"\n=== scVI integration done ===")
    print(f"Cells:   {adata.n_obs:,}")
    print(f"Genes:   {adata.n_vars:,}")
    print(f"Donors:  {adata.obs['donor_id'].nunique()}")
    print(f"Wrote {RESULTS/'integrated_vascular.h5ad'}")
    print(f"Wrote {RESULTS/'scvi_latent.parquet'}")
    print(f"Wrote {RESULTS/'figures/umap_scvi_integrated.png'}")


if __name__ == "__main__":
    main()
