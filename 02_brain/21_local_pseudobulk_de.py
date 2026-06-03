"""Step 2 Task 27: local sensitivity DESeq2 on Smajic + GSE178265 overlap.

Per Step 2 design (`docs/2026-05-03-pd-bbb-step2-design.md` §3.4) and plan
Task 27, this script runs a local pseudobulk DESeq2 (via `pydeseq2`) on the
Smajic vascular subsets, then compares the Smajic endothelial top-DE list to
the Step 1 GSE178265 endothelial DE list (`results/v1_baseline/
v1_signature_genes.tsv`) as a cross-dataset biological-consistency check.

Agarwal is excluded from this step: it is control-only (5 controls, no PD
donors  -  discovered during Task 24), so PD-vs-Control DE is not possible
locally. Agarwal contributes only to the integrated KISTI analysis (§3.5).

Design caveat (Smajic-only, intentional)
----------------------------------------
Per design §3.4 the canonical local design is `~ condition + sex + age`. The
pseudobulk_metadata table built in Task 26 has `sex=Unknown` and `age=NaN`
for all Smajic donors (the source h5ad obs lacks those covariates; flagged
during Task 26). We therefore drop sex/age/PMI for the LOCAL Smajic DE and
fit `~ condition` only. Covariates will be re-added on KISTI via the
paper-supplementary backfill (master plan §3.4 / Step 2 design §3.5).
This is documented as DONE_WITH_CONCERNS-eligible in the plan.

Inputs
------
- results/cache/pseudobulk_smajic_endothelial.parquet  (26,737 genes × 11 donors)
- results/cache/pseudobulk_smajic_pericyte.parquet     (26,737 genes ×  9 donors)
- results/cache/pseudobulk_metadata.parquet            (per-donor metadata)
- results/cache/ensg_to_symbol.tsv                     (Step 1 mygene map)
- results/v1_baseline/v1_signature_genes.tsv           (Step 1 GSE178265 DE)

Outputs (results/step2/)
--------------------------
- local_de_smajic_endothelial.tsv
- local_de_smajic_pericyte.tsv
- local_de_smajic_vs_gse178265_overlap.tsv
- (printed) hypergeometric fold-enrichment + p-value, sanity-floor check

Pitfalls handled
----------------
- pydeseq2 expects `samples × genes` orientation; parquet is `genes × samples`
  -> we transpose before passing.
- pydeseq2 emits NaN padj for low-count genes -> we drop those before ranking.
- Smajic n is small (11 donors endo, 9 pericyte). We use padj < 0.1 (per plan
  §Task 27 Step 1) and fall back to top 100 by |log2FC| if too few pass.
- Smajic's var has no gene-symbol column (only Ensembl). For the symbol-level
  overlap with GSE178265 (HGNC symbols) we build an Ensembl->symbol map from
  the GSE178265 vascular h5ad's `var` (`feature_name` column; 33,295 genes;
  same source the v1 signature symbols were built from in Step 1). This is
  full-Ensembl-coverage and internally consistent with the GSE178265 list.
  The Step 1 mygene map at results/cache/ensg_to_symbol.tsv only covers
  the v1 signature symbols (839/1015) and is unsuitable for mapping Smajic
  Ensembl IDs in the other direction.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference
from scipy.stats import hypergeom


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
ROOT = Path("D:/PD_BBB_analysis")
CACHE = ROOT / "results" / "cache"
STEP2 = ROOT / "results" / "step2"
V1_BASE = ROOT / "results" / "v1_baseline"

# GSE178265 vascular h5ad  -  used solely for its `var` (Ensembl + symbol).
GSE178265_H5AD = Path(
    "D:/Parkinson_file/snrnaseq/cache/gse178265_vascular.h5ad"
)

TOP_N = 100
PADJ_THRESH = 0.10
SANITY_FLOOR_FRAC = 0.15  # plan §Task 27 Step 4
# Background gene universe for hypergeometric: total expressed genes, master
# plan rough figure ~25,000 (used as the universe size).
HYPERGEOM_BACKGROUND = 25_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_section(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def _load_pseudobulk(cell_type: str) -> pd.DataFrame:
    p = CACHE / f"pseudobulk_smajic_{cell_type}.parquet"
    df = pd.read_parquet(p)
    # Ensure index name + integer dtype
    df.index = df.index.astype(str)
    df.index.name = "gene_id"
    return df


def _load_meta(cell_type: str) -> pd.DataFrame:
    meta = pd.read_parquet(CACHE / "pseudobulk_metadata.parquet")
    sub = meta[(meta.dataset == "smajic") & (meta.cell_type == cell_type)].copy()
    sub["donor_id"] = sub["donor_id"].astype(str)
    sub = sub.set_index("donor_id")
    return sub


def _load_ensg_to_symbol() -> dict[str, str]:
    """Build Ensembl -> HGNC symbol map from the GSE178265 vascular h5ad var.

    The GSE178265 var carries `feature_name` populated from the same gene
    metadata used to build v1_signature_genes.tsv in Step 1, so this map
    is internally consistent with the GSE178265 DE list. Coverage spans the
    full ~33k-gene reference (vs. the Step 1 mygene map which only covers
    the 839 v1 signature symbols and is therefore unusable for mapping
    arbitrary Smajic Ensembl IDs).

    Where `feature_name` equals the Ensembl ID itself (GSE178265 falls back
    to the ID for unmapped lncRNAs/pseudogenes) we drop the entry.
    """
    if not GSE178265_H5AD.exists():
        raise FileNotFoundError(
            f"Missing {GSE178265_H5AD}; expected from Task 25."
        )
    a = ad.read_h5ad(GSE178265_H5AD, backed="r")
    var = a.var.copy()
    ensg = var.index.astype(str)
    sym = var["feature_name"].astype(str).values
    out: dict[str, str] = {}
    for e, s in zip(ensg, sym):
        if not s or s == e or s == "nan":
            continue
        out[e] = s
    return out


def _run_deseq2(
    counts_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    cell_type: str,
) -> pd.DataFrame:
    """Run pydeseq2 on a Smajic pseudobulk; return ranked DE table.

    Parameters
    ----------
    counts_df : DataFrame
        genes × samples (the parquet orientation).
    meta_df : DataFrame
        Indexed by donor_id, with `condition` column ∈ {PD, Control}.

    Returns
    -------
    DataFrame with columns gene_id, log2fc, p, padj, baseMean.
    """
    # Align meta to counts columns
    samples = list(counts_df.columns)
    meta_aligned = meta_df.loc[samples].copy()
    if meta_aligned["condition"].isna().any():
        raise ValueError("Some Smajic donors have NaN condition.")
    n_per = meta_aligned["condition"].value_counts().to_dict()
    print(f"  [{cell_type}] donors: {len(samples)}, condition counts: {n_per}")

    if n_per.get("Control", 0) < 3 or n_per.get("PD", 0) < 3:
        raise RuntimeError(
            f"[{cell_type}] insufficient donors per condition for DE: {n_per}"
        )

    # Ensure Control is the reference level by ordering categories.
    meta_aligned["condition"] = pd.Categorical(
        meta_aligned["condition"],
        categories=["Control", "PD"],
        ordered=False,
    )

    # pydeseq2 expects samples × genes
    counts_T = counts_df.T.astype(np.int32)

    inference = DefaultInference(n_cpus=4)
    dds = DeseqDataSet(
        counts=counts_T,
        metadata=meta_aligned,
        design_factors=["condition"],
        refit_cooks=True,
        inference=inference,
        quiet=True,
    )
    dds.deseq2()

    stats_res = DeseqStats(
        dds,
        contrast=["condition", "PD", "Control"],
        inference=inference,
        quiet=True,
    )
    stats_res.summary()
    res = stats_res.results_df.copy()
    # Standardize column names to spec: gene_id, log2fc, p, padj, baseMean
    res = res.rename(columns={
        "log2FoldChange": "log2fc",
        "pvalue": "p",
        "padj": "padj",
        "baseMean": "baseMean",
    })
    res = res.reset_index().rename(columns={"index": "gene_id"})
    res = res[["gene_id", "log2fc", "p", "padj", "baseMean"]]
    # Rank by |log2fc| descending; NaN log2fc to bottom
    res["abs_log2fc"] = res["log2fc"].abs()
    res = res.sort_values(
        ["abs_log2fc", "padj"],
        ascending=[False, True],
        na_position="last",
    ).drop(columns=["abs_log2fc"])
    res = res.reset_index(drop=True)
    return res


def _smajic_top_with_symbols(
    de_df: pd.DataFrame,
    ensg_to_sym: dict[str, str],
    top_n: int,
    padj_thresh: float,
) -> pd.DataFrame:
    """Filter Smajic endothelial DE -> top N by |log2fc| with padj<thresh.

    Drops NaN padj, applies padj cutoff, then takes top_n by |log2fc|.
    Falls back to top_n by |log2fc| from the padj-non-NaN set if too few pass
    (per plan §Task 27 Step 1 / pitfalls note).
    """
    cleaned = de_df.dropna(subset=["padj", "log2fc"]).copy()
    cleaned["abs_log2fc"] = cleaned["log2fc"].abs()
    passing = cleaned[cleaned["padj"] < padj_thresh].copy()
    if len(passing) >= top_n:
        chosen = passing.nlargest(top_n, "abs_log2fc")
        used_filter = f"padj<{padj_thresh}"
    else:
        # Fallback: top_n by |log2fc| from cleaned (any padj). Surface a note.
        print(f"    NOTE: only {len(passing)} genes pass padj<{padj_thresh}; "
              f"falling back to top {top_n} by |log2fc| (any padj).")
        chosen = cleaned.nlargest(top_n, "abs_log2fc")
        used_filter = f"top {top_n} by |log2fc| (fallback)"
    chosen = chosen.copy()
    chosen["gene_symbol"] = chosen["gene_id"].map(ensg_to_sym)
    chosen["filter_used"] = used_filter
    chosen = chosen.reset_index(drop=True)
    chosen["smajic_rank"] = np.arange(1, len(chosen) + 1)
    return chosen


def _compute_overlap(
    smajic_top: pd.DataFrame,
    gse_top: pd.DataFrame,
) -> pd.DataFrame:
    """Build overlap table at the symbol level."""
    smajic_syms = set(s for s in smajic_top["gene_symbol"].dropna().tolist())
    gse_syms = set(gse_top["gene"].dropna().astype(str).tolist())
    intersect = smajic_syms & gse_syms

    rows = []
    sm_rank = dict(zip(smajic_top["gene_symbol"], smajic_top["smajic_rank"]))
    gse_top_idx = gse_top.reset_index(drop=True).copy()
    gse_top_idx["gse178265_rank"] = np.arange(1, len(gse_top_idx) + 1)
    gse_rank = dict(zip(gse_top_idx["gene"].astype(str),
                        gse_top_idx["gse178265_rank"]))

    # Union of unique symbols across both top lists, plus marker
    union = sorted(smajic_syms | gse_syms)
    for sym in union:
        rows.append({
            "gene_symbol": sym,
            "smajic_rank": sm_rank.get(sym, np.nan),
            "gse178265_rank": gse_rank.get(sym, np.nan),
            "in_overlap": sym in intersect,
        })
    out = pd.DataFrame(rows)
    # Order: overlap first by combined-rank, then non-overlap
    out["combined_rank"] = out[["smajic_rank", "gse178265_rank"]].mean(axis=1)
    out = out.sort_values(
        ["in_overlap", "combined_rank"],
        ascending=[False, True],
        na_position="last",
    ).drop(columns=["combined_rank"]).reset_index(drop=True)
    return out


def _hypergeom_test(
    overlap_n: int,
    smajic_n: int,
    gse_n: int,
    background: int,
) -> tuple[float, float]:
    """Hypergeometric test for top-list overlap.

    P(X >= overlap_n) where:
      population size      M = background
      successes in pop     n = gse_n
      sample size          N = smajic_n
    Fold-enrichment = observed / expected.

    Returns (fold_enrichment, p_value).
    """
    expected = (smajic_n * gse_n) / background
    fold = overlap_n / expected if expected > 0 else float("nan")
    # sf(k-1) gives P(X >= k)
    p = float(hypergeom.sf(overlap_n - 1, background, gse_n, smajic_n))
    return fold, p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    STEP2.mkdir(parents=True, exist_ok=True)

    # ---- Load inputs ------------------------------------------------------
    _print_section("Load inputs")
    ensg_to_sym = _load_ensg_to_symbol()
    print(f"  ensg_to_symbol map: {len(ensg_to_sym)} entries")

    v1 = pd.read_csv(V1_BASE / "v1_signature_genes.tsv", sep="\t")
    print(f"  v1_signature_genes.tsv: {len(v1)} rows, columns={list(v1.columns)}")

    de_outputs: dict[str, Path] = {}

    # ---- Smajic endothelial ----------------------------------------------
    _print_section("Smajic endothelial DE (pydeseq2)")
    endo_counts = _load_pseudobulk("endothelial")
    endo_meta = _load_meta("endothelial")
    print(f"  pseudobulk: {endo_counts.shape[0]} genes × {endo_counts.shape[1]} donors")
    endo_de = _run_deseq2(endo_counts, endo_meta, "endothelial")
    endo_out = STEP2 / "local_de_smajic_endothelial.tsv"
    endo_de.to_csv(endo_out, sep="\t", index=False)
    de_outputs["endothelial"] = endo_out
    print(f"  -> wrote {endo_out.name}: {len(endo_de)} rows")
    print(f"  NaN padj: {int(endo_de['padj'].isna().sum())}")
    print(f"  padj<0.10: {int((endo_de['padj'] < 0.10).sum())}; "
          f"padj<0.05: {int((endo_de['padj'] < 0.05).sum())}")

    # ---- Smajic pericyte (if runnable) -----------------------------------
    _print_section("Smajic pericyte DE (pydeseq2)")
    peri_counts = _load_pseudobulk("pericyte")
    peri_meta = _load_meta("pericyte")
    print(f"  pseudobulk: {peri_counts.shape[0]} genes × {peri_counts.shape[1]} donors")
    peri_n_per = peri_meta["condition"].value_counts().to_dict()
    peri_de = None
    peri_concern = None
    if peri_n_per.get("Control", 0) >= 3 and peri_n_per.get("PD", 0) >= 3:
        try:
            peri_de = _run_deseq2(peri_counts, peri_meta, "pericyte")
            peri_out = STEP2 / "local_de_smajic_pericyte.tsv"
            peri_de.to_csv(peri_out, sep="\t", index=False)
            de_outputs["pericyte"] = peri_out
            print(f"  -> wrote {peri_out.name}: {len(peri_de)} rows")
            print(f"  NaN padj: {int(peri_de['padj'].isna().sum())}")
            print(f"  padj<0.10: {int((peri_de['padj'] < 0.10).sum())}; "
                  f"padj<0.05: {int((peri_de['padj'] < 0.05).sum())}")
        except Exception as exc:
            peri_concern = f"pericyte DE failed: {exc!r}"
            print(f"  CONCERN: {peri_concern}")
    else:
        peri_concern = (f"pericyte: insufficient donors per condition "
                        f"({peri_n_per}); skipped.")
        print(f"  CONCERN: {peri_concern}")

    # ---- Top genes summary (endothelial) ---------------------------------
    _print_section("Smajic endothelial: top 5 up + top 5 down (by |log2fc|, padj<0.10)")
    endo_padj_passing = endo_de.dropna(subset=["padj", "log2fc"])
    endo_padj_passing = endo_padj_passing[endo_padj_passing["padj"] < PADJ_THRESH]
    endo_padj_passing = endo_padj_passing.copy()
    endo_padj_passing["gene_symbol"] = endo_padj_passing["gene_id"].map(ensg_to_sym)
    top_up = endo_padj_passing[endo_padj_passing["log2fc"] > 0].nlargest(5, "log2fc")
    top_dn = endo_padj_passing[endo_padj_passing["log2fc"] < 0].nsmallest(5, "log2fc")
    print("  TOP UP:")
    print(top_up[["gene_id", "gene_symbol", "log2fc", "padj"]].to_string(index=False))
    print("  TOP DOWN:")
    print(top_dn[["gene_id", "gene_symbol", "log2fc", "padj"]].to_string(index=False))

    if peri_de is not None:
        _print_section("Smajic pericyte: top 5 up + top 5 down (by |log2fc|, padj<0.10)")
        pp = peri_de.dropna(subset=["padj", "log2fc"])
        pp = pp[pp["padj"] < PADJ_THRESH].copy()
        pp["gene_symbol"] = pp["gene_id"].map(ensg_to_sym)
        if len(pp) == 0:
            print(f"  (no genes pass padj<{PADJ_THRESH})")
        else:
            ptop_up = pp[pp["log2fc"] > 0].nlargest(5, "log2fc")
            ptop_dn = pp[pp["log2fc"] < 0].nsmallest(5, "log2fc")
            print("  TOP UP:")
            print(ptop_up[["gene_id", "gene_symbol", "log2fc", "padj"]].to_string(index=False))
            print("  TOP DOWN:")
            print(ptop_dn[["gene_id", "gene_symbol", "log2fc", "padj"]].to_string(index=False))

    # ---- Overlap with GSE178265 v1 signature ------------------------------
    _print_section(
        f"Overlap: Smajic endothelial top {TOP_N} vs GSE178265 (v1) top {TOP_N}"
    )
    smajic_top = _smajic_top_with_symbols(endo_de, ensg_to_sym, TOP_N, PADJ_THRESH)
    smajic_top_n = len(smajic_top)
    smajic_mapped_n = int(smajic_top["gene_symbol"].notna().sum())
    print(f"  Smajic top: {smajic_top_n} rows; "
          f"with symbol: {smajic_mapped_n} ({smajic_mapped_n / max(smajic_top_n,1):.1%})")

    # GSE178265 top 100 by |log2fc|. v1_signature_genes.tsv is already sorted
    # by |log2fc| descending (per Step 1 build), but re-sort defensively.
    v1_sorted = v1.assign(abs_log2fc=v1["log2fc"].abs()) \
                  .sort_values("abs_log2fc", ascending=False) \
                  .head(TOP_N).drop(columns=["abs_log2fc"])
    gse_top = v1_sorted.copy()
    print(f"  GSE178265 top: {len(gse_top)} symbols")

    overlap_df = _compute_overlap(smajic_top, gse_top)
    overlap_path = STEP2 / "local_de_smajic_vs_gse178265_overlap.tsv"
    overlap_df.to_csv(overlap_path, sep="\t", index=False)
    print(f"  -> wrote {overlap_path.name}: {len(overlap_df)} rows")

    n_overlap = int(overlap_df["in_overlap"].sum())
    overlap_frac = n_overlap / TOP_N
    print(f"  overlap count: {n_overlap} / {TOP_N} ({overlap_frac:.1%})")

    if smajic_mapped_n < TOP_N:
        # We can only "find" overlaps among Smajic genes that have a symbol.
        # Use the mapped-only Smajic n as the effective sample size for
        # hypergeometric  -  otherwise we systematically under-estimate
        # enrichment because unmapped Smajic genes have zero chance of overlap.
        eff_smajic_n = smajic_mapped_n
        print(f"  NOTE: using mapped-Smajic n={eff_smajic_n} as hypergeom "
              "sample size (unmapped genes can't overlap by construction).")
    else:
        eff_smajic_n = smajic_top_n

    fold, pval = _hypergeom_test(
        overlap_n=n_overlap,
        smajic_n=eff_smajic_n,
        gse_n=len(gse_top),
        background=HYPERGEOM_BACKGROUND,
    )
    expected = (eff_smajic_n * len(gse_top)) / HYPERGEOM_BACKGROUND
    print(f"  expected overlap (random, background={HYPERGEOM_BACKGROUND}): "
          f"{expected:.2f}")
    print(f"  fold-enrichment: {fold:.2f}x")
    print(f"  hypergeometric p-value (P[X >= {n_overlap}]): {pval:.3e}")

    floor_met = overlap_frac >= SANITY_FLOOR_FRAC
    print(f"  sanity floor (>={SANITY_FLOOR_FRAC:.0%}): "
          f"{'MET' if floor_met else 'NOT MET'} "
          f"(observed {overlap_frac:.1%})")

    # ---- Final summary ---------------------------------------------------
    _print_section("Summary")
    for ct, p in de_outputs.items():
        print(f"  smajic/{ct} DE -> {p}")
    print(f"  overlap table -> {overlap_path}")
    print(f"  overlap: {n_overlap}/{TOP_N} ({overlap_frac:.1%}); "
          f"fold={fold:.2f}, p={pval:.3e}; "
          f"floor {'MET' if floor_met else 'NOT MET'}")
    if peri_concern:
        print(f"  pericyte note: {peri_concern}")
    print("Done.")


if __name__ == "__main__":
    main()
