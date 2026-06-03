"""Step 3 Task 30  -  Olink BBB protein panel matrix at baseline.

Input: 3 Olink CSVs (Cardio + INF + NEURO panels), long-format NPX values.
Locked panel: 20 proteins (per pre-reg §14 deviation 1, hash accc4e9):
  Endothelial activation (10): VCAM1 ICAM1 ICAM2 ICAM3 SELE SELP VWF CDH5 PECAM1 CD34
  Pericyte / mural (4):        PDGFRB PDGFRA NOTCH3 MCAM
  Vascular damage (4):         MMP9 ANGPTL4 ANGPT1 TIE1
  Neuronal damage (1):         NEFL
  Vascular stress (1):         HSPA1A

Filter to EVENT_ID = "BL" (baseline). One row per PATNO; columns = panel proteins.
QC: drop a protein from the panel if NPX > LOD in <50% of baseline samples.

Output: results/step3/olink_bbb_panel_baseline.parquet
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import load_config
from lib.ppmi_ids import normalize_patno

PANEL = [
    # Endothelial activation
    "VCAM1", "ICAM1", "ICAM2", "ICAM3", "SELE", "SELP", "VWF",
    "CDH5", "PECAM1", "CD34",
    # Pericyte / mural cell
    "PDGFRB", "PDGFRA", "NOTCH3", "MCAM",
    # Vascular damage / matrix
    "MMP9", "ANGPTL4", "ANGPT1", "TIE1",
    # Neuronal damage
    "NEFL",
    # Vascular stress (cross-modal anchor)
    "HSPA1A",
]


def load_olink(path: Path, panel_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["panel_source"] = panel_name
    return df


def main() -> None:
    cfg = load_config()

    print("Loading 3 Olink panels...")
    cardio = load_olink(Path(cfg["proteomics"]["olink_cardio"]), "Cardio")
    inf    = load_olink(Path(cfg["proteomics"]["olink_inf"]),    "INF")
    neuro  = load_olink(Path(cfg["proteomics"]["olink_neuro"]),  "NEURO")
    olink = pd.concat([cardio, inf, neuro], ignore_index=True)
    print(f"  total rows: {len(olink):,}")

    # Subset to panel proteins
    in_panel = olink["ASSAY"].isin(PANEL)
    panel_olink = olink[in_panel].copy()
    found_proteins = sorted(panel_olink["ASSAY"].unique())
    print(f"\nPanel proteins found in Olink data: {len(found_proteins)} / {len(PANEL)}")
    missing = sorted(set(PANEL) - set(found_proteins))
    if missing:
        print(f"  MISSING from Olink: {missing}")
    print(f"  Found: {found_proteins}")

    # Pre-reg lock said HSPA1A is on the platform  -  verify earlier check held.
    # If a panel protein appears in multiple Olink panels (rare overlap), keep one with
    # better LOD coverage at baseline.

    # QC: NPX > LOD coverage at baseline
    panel_olink["PATNO"] = normalize_patno(panel_olink["PATNO"])
    bl = panel_olink[panel_olink["EVENT_ID"] == "BL"].copy()
    print(f"\nBaseline rows in panel: {len(bl):,}")
    print(f"  unique PATNO at BL:   {bl['PATNO'].nunique():,}")

    # NPX values: many panels mark below-LOD with QC_WARNING and a small NPX value.
    # PPMI Olink convention: NPX is reported regardless of LOD; LOD column gives the floor.
    # Treat "below LOD" if NPX <= LOD.
    bl["above_lod"] = bl["NPX"] > bl["LOD"]
    if "QC_WARNING" in bl.columns:
        bl["above_lod"] &= (bl["QC_WARNING"].astype(str).str.upper() != "WARN")

    cov = bl.groupby("ASSAY")["above_lod"].mean().rename("frac_above_LOD").sort_values(ascending=False)
    print("\nFraction of baseline samples with NPX > LOD per panel protein:")
    print(cov.to_string())

    keep = cov[cov >= 0.50].index.tolist()
    drop = cov[cov < 0.50].index.tolist()
    print(f"\n  Retained (≥50% above LOD): {len(keep)}")
    if drop:
        print(f"  Dropped (<50% above LOD):  {drop}")

    # Pivot to wide. If a protein appears in multiple panel sources for a single (PATNO, EVENT_ID),
    # take the one with the higher NPX (consistent with "max signal" convention; documented choice).
    bl_keep = bl[bl["ASSAY"].isin(keep)].copy()
    bl_keep = (
        bl_keep.sort_values(["PATNO", "ASSAY", "NPX"])
              .drop_duplicates(subset=["PATNO", "ASSAY"], keep="last")
    )
    wide = bl_keep.pivot(index="PATNO", columns="ASSAY", values="NPX")
    print(f"\nWide matrix: {wide.shape[0]} patients x {wide.shape[1]} proteins")

    # Drop patients with too many missing proteins (at most 2 missing)
    n_missing_per_patient = wide.isna().sum(axis=1)
    keep_pat = n_missing_per_patient <= 2
    n_dropped_pat = (~keep_pat).sum()
    if n_dropped_pat > 0:
        print(f"  Dropping {n_dropped_pat} patients with >2 missing panel proteins")
    wide = wide.loc[keep_pat]
    print(f"  Final: {wide.shape[0]} patients x {wide.shape[1]} proteins")

    # Save
    out_dir = Path("results/step3")
    out_dir.mkdir(parents=True, exist_ok=True)
    wide_out = wide.reset_index()
    wide_out.to_parquet(out_dir / "olink_bbb_panel_baseline.parquet")
    print(f"\n→ Wrote {out_dir/'olink_bbb_panel_baseline.parquet'}")

    # Sanity print per protein
    print("\nNPX summary stats per panel protein:")
    print(wide.describe().T[["mean", "std", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
