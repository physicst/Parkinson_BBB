"""Step 2 Task 25: Vascular subset extraction (endothelial + pericyte) per dataset.

Per Step 2 design (docs/2026-05-03-pd-bbb-step2-design.md §3.2), extract
endothelial and pericyte populations from each midbrain snRNA-seq dataset to
build the brain-side vascular signature in subsequent tasks.

Datasets handled here:
  1. Smajic 2022 (GSE157783) -- comes pre-annotated; trust `cell_ontology`.
       Output: D:/Parkinson_file/snrnaseq/cache/smajic_vascular.h5ad
  2. Agarwal 2020 (GSE140231 SN-only) -- no labels; marker-score with
       scanpy.tl.score_genes and apply assignment rules below.
       Output: D:/Parkinson_file/snrnaseq/cache/agarwal_vascular.h5ad
  3. GSE178265 endothelial subset -- already endothelial-only at
       D:/Parkinson_file/Human Endothelial.h5ad. Just relabel and copy.
       Output: D:/Parkinson_file/snrnaseq/cache/gse178265_vascular.h5ad

GSE178265 pericytes are deferred to KISTI per design (need full ~5 GB mtx).

Marker sets (HGNC symbols):
  ENDO = ["CLDN5", "FLT1", "PECAM1", "VWF", "SLC2A1", "MFSD2A"]
  PERI = ["PDGFRB", "RGS5", "ABCC9", "KCNJ8", "NOTCH3", "MCAM"]
  SMC  = ["ACTA2", "MYH11", "TAGLN"]   # excluded if score>0.5

Assignment rules (per cell):
  if  score(SMC)  > 0.5                       : smooth_muscle  (excluded)
  elif score(ENDO) > 0.3 and score(ENDO) > score(PERI) : endothelial
  elif score(PERI) > 0.3 and score(PERI) > score(ENDO) : pericyte
  else                                        : other          (excluded)

Output `vascular_class` values: {endothelial, pericyte}.

After per-dataset extraction, prints a per-(dataset, vascular_class) and per-
(dataset, donor, vascular_class) cell-count summary, and flags any donor x
celltype combo with < 30 cells (master-plan §1.2 QC threshold; will be dropped
at pseudobulk in Task 26).
"""
from __future__ import annotations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


CACHE_DIR = Path("D:/Parkinson_file/snrnaseq/cache")

ENDO_MARKERS = ["CLDN5", "FLT1", "PECAM1", "VWF", "SLC2A1", "MFSD2A"]
PERI_MARKERS = ["PDGFRB", "RGS5", "ABCC9", "KCNJ8", "NOTCH3", "MCAM"]
SMC_MARKERS = ["ACTA2", "MYH11", "TAGLN"]

# Default thresholds from design §3.2.
SMC_THRESH = 0.5
ENDO_THRESH = 0.3
PERI_THRESH = 0.3

# QC threshold from master-plan §1.2: any donor x celltype combo with fewer
# than this many cells is flagged (will be dropped at pseudobulk in Task 26).
MIN_CELLS_PER_DONOR_CELLTYPE = 30

# Smajic uses these exact strings in `cell_ontology`.
SMAJIC_ENDO_LABEL = "Endothelial cells"
SMAJIC_PERI_LABEL = "Pericytes"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _print_section(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def _ensgs_for_symbols(var: pd.DataFrame, symbol_col: str, symbols: list[str]) -> list[str]:
    """Return the Ensembl IDs (var.index values) corresponding to given symbols.

    Uses `var[symbol_col]` -> var.index mapping. Symbols not found are skipped
    (and reported). If a symbol maps to multiple Ensembl IDs we keep all of
    them (score_genes will average across them, which is fine).
    """
    sym_to_ensg: dict[str, list[str]] = {}
    for ensg, sym in zip(var.index.astype(str), var[symbol_col].astype(str)):
        sym_to_ensg.setdefault(sym, []).append(ensg)
    found: list[str] = []
    missing: list[str] = []
    for s in symbols:
        if s in sym_to_ensg:
            found.extend(sym_to_ensg[s])
        else:
            missing.append(s)
    if missing:
        print(f"    [warn] markers not found in var (skipped): {missing}")
    return found


def _score_marker_set(adata: ad.AnnData, gene_list: list[str], score_name: str) -> None:
    """Run scanpy.tl.score_genes; tolerate empty/short gene lists by writing 0.

    score_genes writes adata.obs[score_name] in place.
    """
    if len(gene_list) == 0:
        adata.obs[score_name] = 0.0
        print(f"    [warn] {score_name}: no markers available, score set to 0")
        return
    sc.tl.score_genes(adata, gene_list=gene_list, score_name=score_name, use_raw=False)


def _assign_vascular_class(
    score_endo: np.ndarray,
    score_peri: np.ndarray,
    score_smc: np.ndarray,
    smc_thresh: float = SMC_THRESH,
    endo_thresh: float = ENDO_THRESH,
    peri_thresh: float = PERI_THRESH,
) -> np.ndarray:
    """Apply assignment rules vectorized. Returns array of length n_cells."""
    out = np.full(score_endo.shape[0], "other", dtype=object)
    smc_mask = score_smc > smc_thresh
    out[smc_mask] = "smooth_muscle"
    # Among non-SMC cells, apply endo/peri rules.
    rest = ~smc_mask
    endo_mask = rest & (score_endo > endo_thresh) & (score_endo > score_peri)
    peri_mask = rest & (score_peri > peri_thresh) & (score_peri > score_endo) & (~endo_mask)
    out[endo_mask] = "endothelial"
    out[peri_mask] = "pericyte"
    return out


# ---------------------------------------------------------------------------
# Dataset 1: Smajic (pre-annotated, just subset)
# ---------------------------------------------------------------------------
def run_smajic() -> ad.AnnData:
    _print_section("[1/3] Smajic (pre-annotated)")
    in_path = CACHE_DIR / "smajic_full.h5ad"
    out_path = CACHE_DIR / "smajic_vascular.h5ad"
    print(f"  loading {in_path}")
    adata = ad.read_h5ad(in_path)
    print(f"  shape: {adata.shape}")

    if "cell_ontology" not in adata.obs.columns:
        raise RuntimeError("Smajic obs missing 'cell_ontology' column")

    counts = adata.obs["cell_ontology"].value_counts()
    print(f"  cell_ontology counts (top):\n{counts.head(15)}")

    keep_labels = {SMAJIC_ENDO_LABEL, SMAJIC_PERI_LABEL}
    present = set(adata.obs["cell_ontology"].unique())
    missing = keep_labels - present
    if missing:
        raise RuntimeError(
            f"Smajic cell_ontology missing expected labels: {missing}. "
            f"Present labels: {sorted(present)}"
        )

    mask = adata.obs["cell_ontology"].isin(keep_labels).values
    sub = adata[mask].copy()

    # Normalize labels to canonical {endothelial, pericyte}.
    label_map = {SMAJIC_ENDO_LABEL: "endothelial", SMAJIC_PERI_LABEL: "pericyte"}
    sub.obs["vascular_class"] = sub.obs["cell_ontology"].map(label_map).astype(str)
    # Donor column harmonization (called `donor_id` downstream).
    if "donor_id" not in sub.obs.columns and "patient" in sub.obs.columns:
        sub.obs["donor_id"] = sub.obs["patient"].astype(str)
    sub.uns["dataset"] = "Smajic_2022_GSE157783"
    sub.uns["vascular_extraction_method"] = "pre_annotated_cell_ontology"

    print(f"  vascular subset: {sub.shape}")
    print(f"  by class: {sub.obs['vascular_class'].value_counts().to_dict()}")
    print(f"  writing {out_path}")
    sub.write_h5ad(out_path, compression="gzip")
    size_mb = out_path.stat().st_size / 1024**2
    print(f"  -> wrote ({size_mb:.1f} MB)")
    return sub


# ---------------------------------------------------------------------------
# Dataset 2: Agarwal (marker-score on Ensembl-indexed adata)
# ---------------------------------------------------------------------------
def run_agarwal() -> ad.AnnData:
    _print_section("[2/3] Agarwal (marker scoring)")
    in_path = CACHE_DIR / "agarwal_sn.h5ad"
    out_path = CACHE_DIR / "agarwal_vascular.h5ad"
    print(f"  loading {in_path}")
    adata = ad.read_h5ad(in_path)
    print(f"  shape: {adata.shape}")
    print(f"  var index sample: {list(adata.var.index[:3])}")
    print(f"  var columns: {list(adata.var.columns)}")

    if "gene_symbols" not in adata.var.columns:
        raise RuntimeError("Agarwal var missing 'gene_symbols' column")

    # Map symbol -> Ensembl IDs in this var.
    print("  mapping marker symbols -> Ensembl IDs in var")
    endo_ensg = _ensgs_for_symbols(adata.var, "gene_symbols", ENDO_MARKERS)
    peri_ensg = _ensgs_for_symbols(adata.var, "gene_symbols", PERI_MARKERS)
    smc_ensg = _ensgs_for_symbols(adata.var, "gene_symbols", SMC_MARKERS)
    print(f"    ENDO: {len(endo_ensg)} ensg from {len(ENDO_MARKERS)} symbols")
    print(f"    PERI: {len(peri_ensg)} ensg from {len(PERI_MARKERS)} symbols")
    print(f"    SMC : {len(smc_ensg)} ensg from {len(SMC_MARKERS)} symbols")

    # score_genes needs log-normalized data. Agarwal h5ad holds raw counts.
    # Normalize+log on a working copy of X (don't persist); score on .X.
    print("  normalizing + log1p (in-memory copy for scoring)")
    work = adata.copy()
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)

    print("  scoring ENDO/PERI/SMC")
    _score_marker_set(work, endo_ensg, "score_endo")
    _score_marker_set(work, peri_ensg, "score_peri")
    _score_marker_set(work, smc_ensg, "score_smc")

    score_endo = work.obs["score_endo"].to_numpy()
    score_peri = work.obs["score_peri"].to_numpy()
    score_smc = work.obs["score_smc"].to_numpy()

    smc_thresh, endo_thresh, peri_thresh = SMC_THRESH, ENDO_THRESH, PERI_THRESH

    # Apply default thresholds first.
    classes = _assign_vascular_class(score_endo, score_peri, score_smc,
                                     smc_thresh, endo_thresh, peri_thresh)
    n_endo = int((classes == "endothelial").sum())
    n_peri = int((classes == "pericyte").sum())
    n_smc = int((classes == "smooth_muscle").sum())
    n_other = int((classes == "other").sum())
    n_vasc = n_endo + n_peri
    print(f"  default thresholds (smc>{smc_thresh}, endo>{endo_thresh}, peri>{peri_thresh}):")
    print(f"    endothelial={n_endo}  pericyte={n_peri}  smc={n_smc}  other={n_other}")
    print(f"    total vascular = {n_vasc}")

    # Sanity sweep: if vascular count looks off (per spec: <50 or >1500), try
    # a relaxed pass and report. Default pass is still what we save.
    overrides = []
    if n_vasc < 50 or n_vasc > 1500:
        relaxed = _assign_vascular_class(score_endo, score_peri, score_smc,
                                         smc_thresh=0.5, endo_thresh=0.15, peri_thresh=0.15)
        n_endo_r = int((relaxed == "endothelial").sum())
        n_peri_r = int((relaxed == "pericyte").sum())
        n_vasc_r = n_endo_r + n_peri_r
        msg = (f"DEFAULTS_PRODUCED_IMPLAUSIBLE_COUNT vasc={n_vasc} "
               f"(50<=expected<=1500). Relaxed (0.15) -> endo={n_endo_r} "
               f"peri={n_peri_r} total={n_vasc_r}")
        print(f"  [concern] {msg}")
        overrides.append(msg)
        # Heuristic: if defaults gave <50, switch to relaxed for the saved
        # file; otherwise keep defaults (large counts are noisier but >1500
        # would mean too many false-positives, not too few cells).
        if n_vasc < 50 and n_vasc_r >= 50:
            print("  applying relaxed thresholds (0.15) for saved output")
            classes = relaxed
            endo_thresh = 0.15
            peri_thresh = 0.15
            n_endo, n_peri = n_endo_r, n_peri_r
            n_vasc = n_vasc_r

    # Persist scores + class onto the original (raw-count) adata.
    adata.obs["score_endo"] = score_endo
    adata.obs["score_peri"] = score_peri
    adata.obs["score_smc"] = score_smc
    adata.obs["vascular_class_full"] = classes  # includes smooth_muscle/other

    keep = np.isin(classes, ["endothelial", "pericyte"])
    sub = adata[keep].copy()
    sub.obs["vascular_class"] = classes[keep]

    sub.uns["dataset"] = "Agarwal_2020_GSE140231_SN"
    sub.uns["vascular_extraction_method"] = "marker_score_genes"
    sub.uns["thresholds"] = {
        "smc": float(smc_thresh),
        "endo": float(endo_thresh),
        "peri": float(peri_thresh),
    }
    if overrides:
        sub.uns["threshold_overrides"] = overrides

    print(f"  vascular subset: {sub.shape}")
    print(f"  by class: {sub.obs['vascular_class'].value_counts().to_dict()}")
    print(f"  writing {out_path}")
    sub.write_h5ad(out_path, compression="gzip")
    size_mb = out_path.stat().st_size / 1024**2
    print(f"  -> wrote ({size_mb:.1f} MB)")
    return sub


# ---------------------------------------------------------------------------
# Dataset 3: GSE178265 endothelial subset (already extracted)
# ---------------------------------------------------------------------------
def run_gse178265() -> ad.AnnData:
    _print_section("[3/3] GSE178265 (already endothelial-only)")
    in_path = Path("D:/Parkinson_file/Human Endothelial.h5ad")
    out_path = CACHE_DIR / "gse178265_vascular.h5ad"
    print(f"  loading {in_path}")
    adata = ad.read_h5ad(in_path)
    print(f"  shape: {adata.shape}")

    # Sanity: confirm shape close to expected from Step 1 task 17 logs.
    if adata.n_obs < 10000 or adata.n_obs > 20000:
        print(f"  [warn] unexpected n_obs {adata.n_obs} (expected ~14,903)")

    adata.obs["vascular_class"] = "endothelial"
    adata.uns["dataset"] = "GSE178265_endothelial_only"
    adata.uns["vascular_extraction_method"] = "preselected_endothelial_subset"
    adata.uns["pericytes_deferred_to_KISTI"] = True

    print(f"  by class: {adata.obs['vascular_class'].value_counts().to_dict()}")
    print(f"  writing {out_path}")
    adata.write_h5ad(out_path, compression="gzip")
    size_mb = out_path.stat().st_size / 1024**2
    print(f"  -> wrote ({size_mb:.1f} MB)")
    return adata


# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------
def _resolve_donor_col(obs: pd.DataFrame, dataset_name: str) -> str:
    """Pick the donor column for each dataset."""
    if dataset_name == "Smajic":
        for c in ("donor_id", "patient"):
            if c in obs.columns:
                return c
    if dataset_name == "Agarwal":
        return "donor_id"
    if dataset_name == "GSE178265":
        return "donor_id"
    raise RuntimeError(f"unknown dataset {dataset_name}")


def print_summary(smajic: ad.AnnData, agarwal: ad.AnnData, gse: ad.AnnData) -> None:
    _print_section("Combined per-(dataset, vascular_class) summary")

    rows = []
    for name, a in (("Smajic", smajic), ("Agarwal", agarwal), ("GSE178265", gse)):
        vc = a.obs["vascular_class"].value_counts().to_dict()
        rows.append({"dataset": name,
                     "endothelial": int(vc.get("endothelial", 0)),
                     "pericyte": int(vc.get("pericyte", 0)),
                     "total": int(a.n_obs)})
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    _print_section("Per-(dataset, donor, vascular_class) breakdown")
    flagged = []
    for name, a in (("Smajic", smajic), ("Agarwal", agarwal), ("GSE178265", gse)):
        donor_col = _resolve_donor_col(a.obs, name)
        grp = (
            a.obs.groupby([donor_col, "vascular_class"], observed=True)
            .size()
            .reset_index(name="n_cells")
        )
        grp.insert(0, "dataset", name)
        print(f"\n  --- {name} (donor col: {donor_col}) ---")
        print(grp.to_string(index=False))
        for _, r in grp.iterrows():
            if int(r["n_cells"]) < MIN_CELLS_PER_DONOR_CELLTYPE:
                flagged.append((name, str(r[donor_col]), str(r["vascular_class"]), int(r["n_cells"])))

    _print_section(f"Donor x celltype combos below {MIN_CELLS_PER_DONOR_CELLTYPE}-cell threshold")
    if flagged:
        df = pd.DataFrame(flagged, columns=["dataset", "donor", "vascular_class", "n_cells"])
        print(df.to_string(index=False))
        print(f"\n  -> {len(flagged)} combos flagged. These will be dropped at pseudobulk (Task 26).")
    else:
        print("  (none)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    smajic = run_smajic()
    agarwal = run_agarwal()
    gse = run_gse178265()
    print_summary(smajic, agarwal, gse)
    _print_section("Done")
    print("Outputs:")
    for p in (
        CACHE_DIR / "smajic_vascular.h5ad",
        CACHE_DIR / "agarwal_vascular.h5ad",
        CACHE_DIR / "gse178265_vascular.h5ad",
    ):
        if p.exists():
            print(f"  {p}  ({p.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
