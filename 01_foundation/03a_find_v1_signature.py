"""Step 1 Task 17  -  locate or reconstruct the v1 1,015-gene BBB signature.

The v1 manuscript (D:/Parkinson_file/Manuscript.md) reports a 1,015-gene
BBB signature derived from PD midbrain endothelial pseudobulk DE
(GSE178265 endothelial cells: PD vs Control).

Step 1: Defensive scan  -  walk D:/Parkinson_file/ for any tabular file
        containing ~1,015 unique HGNC-ish gene symbols. If hit -> "found".
Step 2: Reconstruct from D:/Parkinson_file/Human Endothelial.h5ad:
        - 14,903 cells x 33,295 genes (Ensembl IDs in var.index, symbols in
          var["feature_name"]); 18 donors (6 PD, 4 LBD, 8 normal).
        - Pseudobulk: sum raw counts per donor_id (X is raw counts; max=3384).
        - DE: PD vs normal (drop LBD), Welch t-test on log2(CPM+1) per donor.
        - Filter p<0.05, sort by |log2FC| desc, take top 1015.

Output: results/v1_baseline/v1_signature_genes.tsv (gene, log2fc, source).

Run:
    conda run --no-capture-output -n pd-bbb python code/03a_find_v1_signature.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import stats

from lib.config import load_config

# Allow running from repo root or from code/
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Crude HGNC-like symbol pattern: starts with letter, alnum/-, 1-15 chars,
# excludes pure ENSG ids and obvious non-gene tokens.
GENE_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9\-\.]{0,14}$")
ENSG_RE = re.compile(r"^ENSG\d+")


def _looks_like_gene_symbols(values: pd.Series) -> int:
    """Return number of unique entries that look like HGNC gene symbols."""
    s = values.dropna().astype(str).str.strip().str.upper()
    s = s[s.map(lambda x: bool(GENE_SYMBOL_RE.match(x)) and not ENSG_RE.match(x))]
    return s.nunique()


def scan_for_existing_signature(data_root: Path, target: int = 1015,
                                tolerance: int = 50) -> Path | None:
    """Walk data_root for any tsv/csv/txt/xlsx with ~target unique symbols.

    Returns first matching file path, or None.
    """
    print(f"[scan] walking {data_root} for tabular files with ~{target} unique gene symbols")
    candidates = []
    for ext in ("*.tsv", "*.csv", "*.txt", "*.tsv.gz", "*.csv.gz", "*.txt.gz"):
        candidates.extend(data_root.rglob(ext))

    for fp in candidates:
        try:
            # Skip huge files
            if fp.stat().st_size > 50 * 1024 * 1024:
                continue
            sep = "\t" if fp.suffix in (".tsv", ".gz") and ".tsv" in fp.name else None
            df = pd.read_csv(fp, sep=sep, engine="python", nrows=2000,
                             on_bad_lines="skip", low_memory=False)
        except Exception:
            continue
        # Try each column
        for col in df.columns:
            try:
                n = _looks_like_gene_symbols(df[col])
            except Exception:
                continue
            if abs(n - target) <= tolerance and n >= 200:
                # Re-read full file to confirm
                try:
                    full = pd.read_csv(fp, sep=sep, engine="python",
                                       on_bad_lines="skip", low_memory=False)
                    n_full = _looks_like_gene_symbols(full[col])
                except Exception:
                    n_full = n
                if abs(n_full - target) <= tolerance:
                    print(f"[scan] HIT: {fp} col={col!r} n_symbols={n_full}")
                    return fp
    print("[scan] no existing tabular file with ~1015 unique gene symbols")
    return None


def pseudobulk_by_donor(ad) -> tuple[pd.DataFrame, pd.Series]:
    """Sum raw counts per donor_id. Returns (gene x donor DataFrame, donor->disease Series)."""
    print(f"[pseudobulk] cells={ad.n_obs} genes={ad.n_vars}")
    donors = ad.obs["donor_id"].astype(str).values
    unique_donors = sorted(set(donors))
    X = ad.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)

    # Build donor x gene sum matrix
    out = np.zeros((len(unique_donors), ad.n_vars), dtype=np.float64)
    for i, d in enumerate(unique_donors):
        mask = donors == d
        out[i] = np.asarray(X[mask].sum(axis=0)).ravel()

    # gene x donor DataFrame
    pb = pd.DataFrame(out.T, index=ad.var_names, columns=unique_donors)

    # donor -> disease label (each donor has only one disease)
    donor_disease = (
        ad.obs[["donor_id", "disease"]].drop_duplicates()
        .set_index("donor_id")["disease"]
    )
    donor_disease = donor_disease.loc[unique_donors]
    print(f"[pseudobulk] donors={len(unique_donors)}; disease map:")
    print(donor_disease.value_counts().to_string())
    return pb, donor_disease


def de_pd_vs_normal(pb: pd.DataFrame, donor_disease: pd.Series,
                    feature_name_map: pd.Series) -> pd.DataFrame:
    """Welch t-test on log2(CPM+1) per donor, PD vs normal. Returns a DataFrame
    sorted by |log2fc| descending, with columns gene, log2fc, pvalue, mean_pd, mean_ctrl."""
    pd_donors = donor_disease[donor_disease == "Parkinson disease"].index.tolist()
    ctrl_donors = donor_disease[donor_disease == "normal"].index.tolist()
    print(f"[de] PD donors={len(pd_donors)}  Ctrl donors={len(ctrl_donors)}")

    # log2(CPM + 1) per donor
    libsize = pb.sum(axis=0)
    cpm = pb.divide(libsize, axis=1) * 1e6
    logcpm = np.log2(cpm + 1.0)

    pd_mat = logcpm[pd_donors].values
    ct_mat = logcpm[ctrl_donors].values

    # Welch t-test row-wise
    tstat, pval = stats.ttest_ind(pd_mat, ct_mat, axis=1, equal_var=False,
                                  nan_policy="omit")
    log2fc = pd_mat.mean(axis=1) - ct_mat.mean(axis=1)

    # Map ensembl id -> symbol (fall back to ensembl id if no symbol)
    sym = feature_name_map.reindex(pb.index).astype(str)
    sym = sym.where(sym.str.len() > 0, pb.index.to_series())

    out = pd.DataFrame({
        "ensembl_id": pb.index,
        "gene": sym.values,
        "log2fc": log2fc,
        "pvalue": pval,
        "mean_pd": pd_mat.mean(axis=1),
        "mean_ctrl": ct_mat.mean(axis=1),
    })
    # Drop rows where t-test failed (NaN p-value: e.g., zero variance)
    out = out.dropna(subset=["pvalue", "log2fc"])
    return out


def select_top_signature(de: pd.DataFrame, target: int = 1015,
                         pthresh: float = 0.05) -> pd.DataFrame:
    """Filter p<pthresh, sort by |log2fc| desc, take top `target`."""
    sig = de[de["pvalue"] < pthresh].copy()
    sig["abs_log2fc"] = sig["log2fc"].abs()
    sig = sig.sort_values("abs_log2fc", ascending=False)
    # de-dup by gene symbol, keep most extreme
    sig = sig.drop_duplicates(subset=["gene"], keep="first")
    sig = sig.head(target).reset_index(drop=True)
    print(f"[select] sig genes after p<{pthresh} & |log2fc| sort: {len(sig)}")
    return sig


def main() -> int:
    cfg = load_config()
    data_root = Path(cfg["data_root"])
    out_dir = Path(cfg["results"]["v1_baseline_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v1_signature_genes.tsv"

    # Step 1: defensive scan
    found = scan_for_existing_signature(data_root, target=1015, tolerance=50)
    if found is not None:
        # Re-read and pull the symbol column, plus a log2fc column if present
        sep = "\t" if ".tsv" in found.name else ","
        df = pd.read_csv(found, sep=sep, engine="python",
                         on_bad_lines="skip", low_memory=False)
        # Pick the column with most gene-symbol-like entries
        best_col, best_n = None, 0
        for col in df.columns:
            try:
                n = _looks_like_gene_symbols(df[col])
            except Exception:
                continue
            if n > best_n:
                best_col, best_n = col, n
        genes = df[best_col].astype(str).str.strip()
        # Optional log2fc column
        l2_col = next((c for c in df.columns
                       if c.lower() in {"log2fc", "log2foldchange", "logfc"}), None)
        log2fc = df[l2_col].astype(float) if l2_col else np.nan
        out_df = pd.DataFrame({"gene": genes, "log2fc": log2fc, "source": "found"})
        out_df = out_df.drop_duplicates(subset=["gene"]).reset_index(drop=True)
        out_df.to_csv(out_path, sep="\t", index=False)
        print(f"[done:found] {len(out_df)} genes -> {out_path}")
        return 0

    # Step 2: reconstruct from h5ad
    h5ad_path = cfg["snrnaseq"]["gse178265_endothelial_h5ad"]
    print(f"[load] {h5ad_path}")
    ad = sc.read_h5ad(h5ad_path)
    print(f"[load] shape={ad.shape}  X.max={ad.X.max():.1f}  layers={list(ad.layers.keys())}")
    if ad.X.max() < 100:
        print("[load] WARNING: X looks log-normalized (max<100). Aborting  -  need raw counts.")
        return 2

    pb, donor_disease = pseudobulk_by_donor(ad)
    feature_name_map = ad.var["feature_name"].astype(str)
    de = de_pd_vs_normal(pb, donor_disease, feature_name_map)
    sig = select_top_signature(de, target=1015, pthresh=0.05)

    out_df = pd.DataFrame({
        "gene": sig["gene"].values,
        "log2fc": sig["log2fc"].values,
        "source": "reconstructed",
    })
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"[done:reconstructed] {len(out_df)} genes -> {out_path}")

    # Print top 5 by |log2fc| for sanity check
    print("[top5]")
    top5 = sig.head(5)[["gene", "log2fc", "pvalue"]]
    print(top5.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
