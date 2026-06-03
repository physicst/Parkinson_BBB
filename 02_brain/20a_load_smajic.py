"""Load Smajic 2022 GSE157783 midbrain snRNA-seq into AnnData.

Sources (paths in code/config.yaml under `snrnaseq.smajic_*_tar`):
  - GSE157783_IPDCO_hg_midbrain_UMI.tar.gz    -> IPDCO_hg_midbrain_UMI.tsv
        Genes x cells dense TSV. Header line = 41,435 cell barcodes.
        Data rows = 26,737 genes (no row index column; gene order matches genes TSV).
        Uncompressed ~2.2 GB.
  - GSE157783_IPDCO_hg_midbrain_cell.tar.gz   -> IPDCO_hg_midbrain_cell.tsv
        Columns: barcode, cell_ontology, patient (e.g. PD3, C1).
  - GSE157783_IPDCO_hg_midbrain_genes.tar.gz  -> IPDCO_hg_midbrain_genes.tsv
        Columns: gene (Ensembl ID, no version), row (1-indexed), vst.* stats.

Strategy:
  1. Extract each tarball in-memory via `tarfile`; each contains exactly one TSV.
  2. Read UMI matrix in row-chunks (gene-chunks) to bound peak memory. Each chunk
     is converted to scipy.sparse.csr_matrix and stacked vstack -> final sparse
     genes x cells, then transposed to cells x genes for AnnData.
  3. Build obs from cell metadata: order obs to match matrix header column order.
  4. Build var from genes metadata: sorted by `row` to match matrix data-row order.
  5. Add `condition` column to obs: PD* -> IPD, C* -> Control.
  6. Strip Ensembl version suffix (.N) defensively even though Smajic IDs are
     already unversioned.

Memory budget:
  Dense full matrix would be 26737 x 41435 x 4 bytes ~ 4.4 GB; we never hold that.
  Each chunk of 2000 genes x 41435 cells x 4 bytes ~ 330 MB dense (transient),
  shrunk to <50 MB sparse before next chunk loads. Final sparse <500 MB in memory.

Output: D:/Parkinson_file/snrnaseq/cache/smajic_full.h5ad
"""
from __future__ import annotations
import io
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from lib.config import load_config


CHUNK_GENES = 2000  # rows per pandas chunk; tuned for ~330 MB transient peak
EXPECTED_DONORS = {"C1", "C2", "C3", "C4", "C5", "C6", "PD1", "PD2", "PD3", "PD4", "PD5"}


def extract_single_tsv(tar_path: Path) -> bytes:
    """Read the (single) .tsv member from a .tar.gz into bytes (in-memory)."""
    with tarfile.open(tar_path, mode="r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".tsv")]
        if len(members) != 1:
            raise RuntimeError(f"{tar_path}: expected exactly 1 .tsv, found {len(members)}: {[m.name for m in members]}")
        f = tf.extractfile(members[0])
        if f is None:
            raise RuntimeError(f"{tar_path}: could not open member {members[0].name}")
        return f.read()


def strip_ensembl_version(gene_id: str) -> str:
    """ENSG00000123456.7 -> ENSG00000123456. Idempotent if no suffix."""
    return gene_id.split(".")[0]


def load_umi_sparse(umi_bytes: bytes, n_genes_expected: int, n_cells_expected: int) -> tuple[sp.csr_matrix, list[str]]:
    """Stream the genes x cells UMI TSV into a sparse matrix.

    Returns (genes x cells csr_matrix, list of cell barcodes from header).
    """
    buf = io.BytesIO(umi_bytes)
    # Peek the header to capture cell barcodes (these become column names of the matrix).
    header_line = buf.readline().decode("utf-8").rstrip("\n").rstrip("\r")
    cell_barcodes = header_line.split("\t")
    if len(cell_barcodes) != n_cells_expected:
        raise RuntimeError(f"UMI header has {len(cell_barcodes)} cells, expected {n_cells_expected}")

    # Reset and let pandas read the full file in row-chunks. We re-include the header
    # so pandas builds proper column names (the cell barcodes).
    buf.seek(0)
    chunks: list[sp.csr_matrix] = []
    rows_read = 0
    reader = pd.read_csv(
        buf,
        sep="\t",
        header=0,
        dtype=np.int32,  # UMI counts; safe for int32
        chunksize=CHUNK_GENES,
        low_memory=False,
        engine="c",
    )
    for chunk in reader:
        if chunk.shape[1] != n_cells_expected:
            raise RuntimeError(f"UMI chunk has {chunk.shape[1]} cols, expected {n_cells_expected}")
        # chunk values: (rows_in_chunk, n_cells), int32. Convert to csr.
        chunks.append(sp.csr_matrix(chunk.values))
        rows_read += chunk.shape[0]
        print(f"  read {rows_read}/{n_genes_expected} gene rows")

    if rows_read != n_genes_expected:
        raise RuntimeError(f"UMI matrix has {rows_read} gene rows, expected {n_genes_expected}")

    mat = sp.vstack(chunks, format="csr")  # genes x cells
    return mat, cell_barcodes


def main() -> None:
    cfg = load_config()
    snr = cfg["snrnaseq"]
    umi_tar = Path(snr["smajic_umi_tar"])
    cell_tar = Path(snr["smajic_cell_tar"])
    genes_tar = Path(snr["smajic_genes_tar"])

    cache_dir = Path(snr["smajic_umi_tar"]).parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "smajic_full.h5ad"

    print(f"[1/5] Reading cell metadata: {cell_tar}")
    cell_bytes = extract_single_tsv(cell_tar)
    cells = pd.read_csv(io.BytesIO(cell_bytes), sep="\t", dtype=str)
    print(f"  cells: {cells.shape}, columns: {list(cells.columns)}")

    print(f"[2/5] Reading gene metadata: {genes_tar}")
    genes_bytes = extract_single_tsv(genes_tar)
    genes = pd.read_csv(io.BytesIO(genes_bytes), sep="\t")
    print(f"  genes: {genes.shape}, columns: {list(genes.columns)}")
    # Sort genes by `row` to lock the order matching the matrix data-row order.
    genes = genes.sort_values("row", kind="stable").reset_index(drop=True)
    genes["gene_id"] = genes["gene"].astype(str).map(strip_ensembl_version)

    print(f"[3/5] Streaming UMI matrix: {umi_tar} (~2.2 GB uncompressed)")
    umi_bytes = extract_single_tsv(umi_tar)
    mat_gc, header_barcodes = load_umi_sparse(
        umi_bytes,
        n_genes_expected=len(genes),
        n_cells_expected=len(cells),
    )
    print(f"  UMI matrix (genes x cells): {mat_gc.shape}, nnz={mat_gc.nnz}")
    # Free the raw bytes ASAP.
    del umi_bytes

    print("[4/5] Aligning obs to matrix header order")
    # Matrix columns are in the exact order of header_barcodes; reorder cells df to match.
    cells_indexed = cells.set_index("barcode")
    missing = set(header_barcodes) - set(cells_indexed.index)
    if missing:
        raise RuntimeError(f"{len(missing)} matrix barcodes missing from cell metadata; first: {list(missing)[:3]}")
    obs = cells_indexed.loc[header_barcodes].copy()
    obs.index.name = "barcode"
    # Derive condition from patient prefix: PD* -> IPD, C* -> Control.
    obs["condition"] = np.where(obs["patient"].str.startswith("PD"), "IPD", "Control")

    donors = set(obs["patient"].unique())
    if donors != EXPECTED_DONORS:
        raise RuntimeError(f"Donor mismatch. Got {sorted(donors)}, expected {sorted(EXPECTED_DONORS)}")
    if set(obs["condition"].unique()) != {"IPD", "Control"}:
        raise RuntimeError(f"Both conditions required; got {set(obs['condition'].unique())}")

    # Build var: indexed by stripped Ensembl ID, in matrix-row order.
    var = genes[["gene_id", "gene"]].rename(columns={"gene": "gene_id_raw"}).copy()
    var.index = var["gene_id"].astype(str)
    var.index.name = "gene_id"
    if var.index.duplicated().any():
        # Keep first occurrence; log how many dupes.
        n_dup = int(var.index.duplicated().sum())
        print(f"  WARNING: {n_dup} duplicate gene_ids after version strip; keeping first")
        keep_mask = ~var.index.duplicated(keep="first")
        var = var.loc[keep_mask]
        # Subset matrix rows to keep accordingly (rare; Smajic should be unique).
        mat_gc = mat_gc[np.where(keep_mask.values)[0], :]

    print("[5/5] Building AnnData (cells x genes)")
    # Transpose: AnnData expects obs=cells (rows), var=genes (columns).
    X = mat_gc.T.tocsr()
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["dataset"] = "Smajic_2022_GSE157783"
    adata.uns["tissue"] = "midbrain"

    # Asserts.
    assert adata.n_obs > 10000, f"n_obs too small: {adata.n_obs}"
    assert adata.n_vars > 15000, f"n_vars too small: {adata.n_vars}"

    print(f"\nAnnData: {adata}")
    print(f"\nDonor x condition table:")
    print(obs.groupby(["condition", "patient"], observed=True).size().unstack(fill_value=0))
    print(f"\nTop cell types:")
    print(obs["cell_ontology"].value_counts().head(10))

    print(f"\nWriting {out_path}")
    adata.write_h5ad(out_path, compression="gzip")
    size_mb = out_path.stat().st_size / 1024**2
    print(f"-> Wrote {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
