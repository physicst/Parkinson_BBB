"""Build the canonical patient-visit table by merging all clinical sources.

Output: results/cache/canonical_clinical.parquet
  One row per (PATNO, EVENT_ID).

Subject-level fields (broadcast across all visits per PATNO):
  Sex, BIRTHDT, Age_at_Baseline, Race_*, Hispanic, Cohort_at_Enrollment, Cohort_Current.
Visit-level fields (per-visit):
  INFODT, Visit_Months_From_BL, UPDRS3_Total, UPDRS3_State, PDTRTMNT,
  MoCA_Total, DaT-SCAN columns, SAA_Status, SAA_Type, RNA_*.

Anchor: a patient-visit row exists if ANY of the visit-level sources has a row
for (PATNO, EVENT_ID).
"""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

# Local imports  -  code/ is on sys.path so per-source modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from importlib import import_module
demographics = import_module("01a_demographics")
cohort = import_module("01b_cohort")
updrs3 = import_module("01c_updrs3")
moca = import_module("01d_moca")
datscan = import_module("01e_datscan")
saa = import_module("01f_saa")
rna_link = import_module("01g_rna_link")

from lib.config import load_config


def build() -> pd.DataFrame:
    print("Loading sources...")
    demo = demographics.load_demographics()
    print(f"  demographics: {len(demo)}")
    coh = cohort.load_cohort()
    print(f"  cohort:       {len(coh)}")
    u3 = updrs3.load_updrs3()
    print(f"  UPDRS3:       {len(u3)}")
    mc = moca.load_moca()
    print(f"  MoCA:         {len(mc)}")
    ds = datscan.load_datscan()
    print(f"  DaT-SCAN:     {len(ds)}")
    sa = saa.load_saa()
    print(f"  SAA:          {len(sa)}")
    rl = rna_link.load_rna_link()
    print(f"  RNA link:     {len(rl)}")

    # Subject-level frame
    subj = demo.merge(coh, on="PATNO", how="outer")
    n_subj_pre = subj["PATNO"].nunique()
    print(f"\nSubject-level frame: {len(subj)} rows, {n_subj_pre} unique PATNO")

    # Compute baseline date per patient = earliest INFODT across UPDRS3 + MoCA visits
    visit_dates = pd.concat([
        u3[["PATNO", "EVENT_ID", "INFODT"]],
        mc[["PATNO", "EVENT_ID", "INFODT"]],
    ], ignore_index=True).dropna(subset=["INFODT"])
    baseline = (
        visit_dates.sort_values("INFODT")
                   .drop_duplicates(subset="PATNO", keep="first")
                   .rename(columns={"INFODT": "Baseline_INFODT"})
                   [["PATNO", "Baseline_INFODT"]]
    )
    subj = subj.merge(baseline, on="PATNO", how="left")
    subj["Age_at_Baseline"] = (
        (subj["Baseline_INFODT"] - subj["BIRTHDT"]).dt.days / 365.25
    ).round(2)

    # Visit-level frame: outer-join all visit sources on (PATNO, EVENT_ID)
    visit = u3.merge(mc[["PATNO", "EVENT_ID", "MoCA_Total"]],
                     on=["PATNO", "EVENT_ID"], how="outer")
    visit = visit.merge(
        ds.drop(columns="DATSCAN_DATE", errors="ignore"),
        on=["PATNO", "EVENT_ID"], how="outer", suffixes=("", "_ds"))
    visit = visit.merge(sa, on=["PATNO", "EVENT_ID"], how="outer")
    visit = visit.merge(rl, on=["PATNO", "EVENT_ID"], how="outer")

    # Visit_Months_From_BL
    visit = visit.merge(baseline, on="PATNO", how="left")
    visit["Visit_Months_From_BL"] = (
        (visit["INFODT"] - visit["Baseline_INFODT"]).dt.days / 30.4375
    ).round(2)

    # Final merge: subject-level fields broadcast onto every visit row
    final = visit.merge(subj.drop(columns="Baseline_INFODT", errors="ignore"),
                        on="PATNO", how="left")

    assert (final.groupby(["PATNO", "EVENT_ID"]).size() == 1).all(), \
        "Duplicate (PATNO, EVENT_ID) in final canonical frame"

    return final


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["results"]["cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "canonical_clinical.parquet"

    df = build()

    df.to_parquet(out_path, index=False)
    print(f"\n→ Wrote {out_path}")
    print(f"  rows:    {len(df):,}")
    print(f"  cols:    {len(df.columns)}")
    print(f"  patients: {df['PATNO'].nunique():,}")
    print(f"\nCohort_Current x has_RNA:")
    df["_has_RNA"] = df["RNA_HudAlphaID"].notna()
    print(df.groupby("Cohort_Current")["_has_RNA"].sum())


if __name__ == "__main__":
    main()
