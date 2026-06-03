"""Step A  -  Build the CSF Olink BBB PC1 score from Project 277 (Olink HT).

Pre-specified in docs/2026-05-24-pd-bbb-csf-olink-analysis-design.md (Step A).

Mirrors the plasma Olink BBB pipeline (Step 3 Methods 4.5) with the same
20 pre-registered BBB-panel proteins. All 20 have been verified present in
the Olink Explore HT panel.

Inputs:
  D:/Parkinson_file/PPMI_Project_277_CSF_Screened_NPX_20260130.parquet

Outputs:
  results/step7/csf_olink_bbb_score.parquet  (one row per BL patient)
  results/step7/csf_olink_per_protein_loadings.tsv
  results/step7/csf_olink_qc.txt
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

F = "D:/Parkinson_file/PPMI_Project_277_CSF_Screened_NPX_20260130.parquet"
OUT = Path("results/step7")
OUT.mkdir(parents=True, exist_ok=True)

PANEL = [
    "PECAM1", "CDH5", "CD34", "ICAM1", "ICAM2", "ICAM3", "VCAM1", "MCAM",
    "TIE1", "SELE", "SELP", "ANGPT1", "ANGPTL4", "MMP9", "NOTCH3", "PDGFRA",
    "PDGFRB", "VWF", "NEFL", "HSPA1A",
]

log: list[str] = []


def emit(msg: str = "") -> None:
    print(msg)
    log.append(msg)


def main() -> None:
    emit("=== Step A - Build CSF Olink BBB PC1 score (Project 277) ===\n")

    df = pd.read_parquet(F)
    emit(f"Loaded {len(df):,} rows from Project 277")

    # 1. Filter to BL + SampleQC PASS + real patients
    df = df[(df["EVENT_ID"] == "BL")
            & (df["SampleQC"] == "PASS")
            & (df["PATNO"].notna())].copy()
    df["PATNO"] = df["PATNO"].astype(int)
    emit(f"After BL + PASS + non-null PATNO filter: {len(df):,} rows, "
         f"{df['PATNO'].nunique()} patients")

    # 2. Subset to the 20 BBB-panel proteins and drop assay-level QC failures
    df = df[df["Assay"].isin(PANEL)].copy()
    if "AssayQC" in df.columns:
        before = len(df)
        df = df[df["AssayQC"].isin(["PASS", None]) | df["AssayQC"].isna()]
        emit(f"  assay-QC pass filter: {len(df):,}/{before:,} rows kept")
    emit(f"After panel subset: {len(df):,} rows, "
         f"{df['PATNO'].nunique()} patients, "
         f"{df['Assay'].nunique()} proteins (expected 20)")

    # 3. Pivot to wide: PATNO x Assay, mean NPX (collapses any duplicate
    #    plate/well measurements per patient)
    wide = df.pivot_table(index="PATNO", columns="Assay",
                          values="NPX", aggfunc="mean")
    # ensure column order matches PANEL
    wide = wide.reindex(columns=PANEL)
    emit(f"Wide matrix: {wide.shape[0]} patients x {wide.shape[1]} proteins")
    emit(f"  per-protein missingness (count of NaN):")
    miss = wide.isna().sum()
    for p, n in miss.items():
        if n:
            emit(f"    {p}: {n}")
    if miss.sum() == 0:
        emit("    (no missing values)")

    # 4. Drop patients with any missing panel value (mirror plasma pipeline)
    before = len(wide)
    wide = wide.dropna()
    emit(f"After complete-panel filter: {len(wide)} / {before} patients")

    # 5. Z-score each protein across the BL cohort
    Xs = StandardScaler().fit_transform(wide.values)

    # 6. PC1
    pca = PCA(n_components=1).fit(Xs)
    pc1 = pca.transform(Xs)[:, 0]
    var_pct = pca.explained_variance_ratio_[0] * 100
    emit(f"\nPC1 explains {var_pct:.1f}% of panel variance")

    # 7. Loadings
    loadings = pd.DataFrame({
        "protein": wide.columns,
        "pc1_loading": pca.components_[0],
    }).sort_values("pc1_loading", ascending=False)
    n_pos = (loadings["pc1_loading"] > 0).sum()
    n_neg = (loadings["pc1_loading"] < 0).sum()
    emit(f"Per-protein PC1 loadings: {n_pos} positive, {n_neg} negative")
    emit("  top 5 loadings (positive):")
    for _, r in loadings.head(5).iterrows():
        emit(f"    {r['protein']:8s}  {r['pc1_loading']:+.3f}")
    emit("  bottom 5 loadings:")
    for _, r in loadings.tail(5).iterrows():
        emit(f"    {r['protein']:8s}  {r['pc1_loading']:+.3f}")
    sign_consistent = (n_pos == 20) or (n_neg == 20)
    emit(f"\nSign-consistency check (all-same-sign expected for a single "
         f"shared activation axis): {'YES' if sign_consistent else 'NO'}")
    # Conventional sign: make PC1 positive in mean (flip if needed) -- matches
    # the plasma pipeline convention so higher PC1 = greater activation
    if pc1.mean() < 0 and not sign_consistent:
        pc1 = -pc1
        loadings["pc1_loading"] = -loadings["pc1_loading"]
        emit("  (PC1 sign flipped to keep convention consistent with plasma)")

    # 8. Save
    out = pd.DataFrame({"PATNO": wide.index.astype(int),
                        "csf_olink_pc1": pc1})
    out.to_parquet(OUT / "csf_olink_bbb_score.parquet", index=False)
    loadings.to_csv(OUT / "csf_olink_per_protein_loadings.tsv",
                    sep="\t", index=False)
    emit(f"\n-> {OUT}/csf_olink_bbb_score.parquet  ({len(out)} patients)")
    emit(f"-> {OUT}/csf_olink_per_protein_loadings.tsv")

    (OUT / "csf_olink_qc.txt").write_text("\n".join(log) + "\n",
                                          encoding="utf-8")
    emit(f"-> {OUT}/csf_olink_qc.txt")
    emit("\n=== Step A done ===")


if __name__ == "__main__":
    main()
