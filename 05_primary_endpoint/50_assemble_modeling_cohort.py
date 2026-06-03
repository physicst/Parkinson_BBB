"""Step 5 Task 50  -  Assemble the modeling cohort for the pre-registered
primary endpoint.

Pre-registered model:
  lmer(UPDRS3_Total ~ Time * BBB_score
                    + Age_at_Baseline + Sex + UPDRS3_Baseline + LEDD
                    + Neutrophil_frac + Monocyte_frac
                    + (1 + Time | PATNO),
       data, REML=TRUE)

Cohort criteria (from pre-reg §2):
  - Cohort_Current ∈ {PD, Prodromal}
  - Olink BBB score at baseline (Step 3 deviation: PC1 of 20-protein panel)
  - ≥3 UPDRS3 visits in first 36 months post-baseline

Inputs:
  results/step3/bbb_protein_score.parquet           (227 patients with BBB score)
  results/cache/canonical_clinical.parquet            (longitudinal UPDRS3 + demographics)
  D:/Parkinson_file/LEDD_Concomitant_Medication_Log_02May2026.csv
  D:/Parkinson_file/Blood_Chemistry___Hematology_02May2026.csv

Output:
  results/step3/modeling_cohort_long.parquet     (one row per visit)
  results/step3/modeling_cohort_subjects.parquet (one row per subject)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.ppmi_ids import normalize_patno, parse_ppmi_date

S3 = Path("results/step3")
CACHE = Path("results/cache")


def load_ledd_per_visit(canonical_visits: pd.DataFrame) -> pd.DataFrame:
    """For each (PATNO, INFODT), sum LEDD across all medications active on that date."""
    log = pd.read_csv("D:/Parkinson_file/LEDD_Concomitant_Medication_Log_02May2026.csv",
                     low_memory=False)
    log["PATNO"] = normalize_patno(log["PATNO"])
    log["LEDD"] = pd.to_numeric(log["LEDD"], errors="coerce").fillna(0)
    log["STARTDT"] = parse_ppmi_date(log["STARTDT"])
    log["STOPDT"]  = parse_ppmi_date(log["STOPDT"])
    # Treat null STOPDT as still-active (today)
    log["STOPDT"]  = log["STOPDT"].fillna(pd.Timestamp.today())

    # For each visit row, sum LEDD of medications where STARTDT <= INFODT <= STOPDT
    visits = canonical_visits[["PATNO", "EVENT_ID", "INFODT"]].dropna(subset=["INFODT"]).copy()
    visits["LEDD_total"] = 0.0

    # This is O(visits × meds) but small enough; cap with merge_asof if too slow
    for patno, sub in log.groupby("PATNO"):
        mask = visits["PATNO"] == patno
        if not mask.any():
            continue
        for _, m in sub.iterrows():
            if pd.isna(m["STARTDT"]):
                continue
            within = mask & visits["INFODT"].between(m["STARTDT"], m["STOPDT"])
            visits.loc[within, "LEDD_total"] += m["LEDD"]

    return visits[["PATNO", "EVENT_ID", "LEDD_total"]]


def load_cbc_per_visit(canonical_visits: pd.DataFrame) -> pd.DataFrame:
    """Get neutrophil_pct and monocyte_pct per (PATNO, EVENT_ID) from CBC file.

    Match by (PATNO, EVENT_ID) directly since CBC has those columns.
    """
    cbc = pd.read_csv("D:/Parkinson_file/Blood_Chemistry___Hematology_02May2026.csv",
                      low_memory=False)
    cbc["PATNO"] = normalize_patno(cbc["PATNO"])
    cbc["EVENT_ID"] = cbc["EVENT_ID"].astype(str)
    cbc["LSIRES"] = pd.to_numeric(cbc["LSIRES"], errors="coerce")

    # Filter to neutrophil + monocyte percentages (LTSTNAME = "Neutrophils (%)" / "Monocytes (%)")
    neut = cbc[cbc["LTSTNAME"] == "Neutrophils (%)"][["PATNO", "EVENT_ID", "LSIRES"]]
    neut = neut.rename(columns={"LSIRES": "Neutrophil_pct"}).dropna()
    mono = cbc[cbc["LTSTNAME"] == "Monocytes (%)"][["PATNO", "EVENT_ID", "LSIRES"]]
    mono = mono.rename(columns={"LSIRES": "Monocyte_pct"}).dropna()

    # Aggregate (some have multiple measurements per visit)
    neut = neut.groupby(["PATNO", "EVENT_ID"], as_index=False)["Neutrophil_pct"].mean()
    mono = mono.groupby(["PATNO", "EVENT_ID"], as_index=False)["Monocyte_pct"].mean()

    return neut.merge(mono, on=["PATNO", "EVENT_ID"], how="outer")


def main() -> None:
    # --- Load inputs ---
    bbb = pd.read_parquet(S3 / "bbb_protein_score.parquet")
    bbb["PATNO"] = bbb["PATNO"].astype(str)
    print(f"BBB score: {len(bbb)} patients")

    canon = pd.read_parquet(CACHE / "canonical_clinical.parquet")
    canon["PATNO"] = canon["PATNO"].astype(str)

    # Restrict canonical to PD + Prodromal
    canon = canon[canon["Cohort_Current"].isin(["PD", "Prodromal"])].copy()

    # ≥3 UPDRS3 visits in 0-36 months
    df36 = canon[canon["Visit_Months_From_BL"].between(0, 36)
                 & canon["UPDRS3_Total"].notna()]
    n_visits = df36.groupby("PATNO").size()
    keep_pat = n_visits[n_visits >= 3].index.tolist()
    df36 = df36[df36["PATNO"].isin(keep_pat)].copy()
    print(f"PD/Prodromal × ≥3 UPDRS3 visits in 0-36mo: {df36['PATNO'].nunique()} subjects, "
          f"{len(df36)} visits")

    # Subset further to subjects with BBB score
    has_bbb = set(bbb["PATNO"])
    df_long = df36[df36["PATNO"].isin(has_bbb)].copy()
    n_subj = df_long["PATNO"].nunique()
    print(f"With Olink BBB score:                       {n_subj} subjects, "
          f"{len(df_long)} visits")

    # Add UPDRS3_Baseline (per subject  -  the earliest UPDRS3 in the window)
    bl_updrs = (df_long.sort_values(["PATNO", "Visit_Months_From_BL"])
                       .drop_duplicates("PATNO", keep="first")
                       [["PATNO", "UPDRS3_Total"]]
                       .rename(columns={"UPDRS3_Total": "UPDRS3_Baseline"}))
    df_long = df_long.merge(bl_updrs, on="PATNO", how="left")

    # LEDD per visit
    print("\nComputing LEDD per visit...")
    ledd = load_ledd_per_visit(df_long)
    df_long = df_long.merge(ledd, on=["PATNO", "EVENT_ID"], how="left")
    df_long["LEDD_total"] = df_long["LEDD_total"].fillna(0)

    # CBC per visit
    print("Computing CBC fractions per visit...")
    cbc = load_cbc_per_visit(df_long)
    df_long = df_long.merge(cbc, on=["PATNO", "EVENT_ID"], how="left")

    # BBB score (per subject  -  baseline value, broadcast across visits)
    df_long = df_long.merge(bbb[["PATNO", "bbb_score", "vascular_class"]],
                            on="PATNO", how="left")

    # Time in years (for slope interpretability)
    df_long["Time"] = df_long["Visit_Months_From_BL"] / 12.0

    # --- Final cohort filter: complete covariates required by pre-reg ---
    required = ["UPDRS3_Total", "Time", "bbb_score",
                "Age_at_Baseline", "Sex", "UPDRS3_Baseline",
                "LEDD_total", "Neutrophil_pct", "Monocyte_pct"]
    n_before = len(df_long)
    df_complete = df_long.dropna(subset=required).copy()
    n_after = len(df_complete)
    print(f"\nAfter dropping rows with missing covariates: {n_after}/{n_before} visits, "
          f"{df_complete['PATNO'].nunique()} subjects")

    # Per-subject sample size constraint: need at least 3 visits AFTER covariate filter
    n_visits_post = df_complete.groupby("PATNO").size()
    keep_post = n_visits_post[n_visits_post >= 3].index.tolist()
    df_complete = df_complete[df_complete["PATNO"].isin(keep_post)].copy()
    print(f"Final modeling cohort: {df_complete['PATNO'].nunique()} subjects, "
          f"{len(df_complete)} visits")

    # Cohort breakdown
    subj = df_complete.drop_duplicates("PATNO")[
        ["PATNO", "Cohort_Current", "Sex", "Age_at_Baseline",
         "vascular_class", "bbb_score", "UPDRS3_Baseline"]
    ]
    print("\nCohort_Current x vascular_class:")
    print(pd.crosstab(subj["Cohort_Current"], subj["vascular_class"]))
    print(f"\nMean BBB score: {subj['bbb_score'].mean():.3f} (sd {subj['bbb_score'].std():.3f})")
    print(f"Mean baseline UPDRS3: {subj['UPDRS3_Baseline'].mean():.1f}")

    # MoCA + DaT-SCAN + SAA columns are already present in df_complete via the
    # canon dataframe (it was the source for df36 → df_long → df_complete);
    # they may be NaN for many visits, which secondary models filter on themselves.

    # Save
    out_long = df_complete[["PATNO", "EVENT_ID", "Time", "Visit_Months_From_BL",
                             "UPDRS3_Total", "UPDRS3_Baseline",
                             "MoCA_Total", "DATSCAN_Putamen_Worst", "SAA_Status",
                             "bbb_score", "vascular_class",
                             "Age_at_Baseline", "Sex", "Cohort_Current",
                             "LEDD_total", "Neutrophil_pct", "Monocyte_pct"]]
    out_long.to_parquet(S3 / "modeling_cohort_long.parquet")
    subj.to_parquet(S3 / "modeling_cohort_subjects.parquet")
    print(f"\n→ Wrote {S3/'modeling_cohort_long.parquet'} ({len(out_long)} rows)")
    print(f"→ Wrote {S3/'modeling_cohort_subjects.parquet'} ({len(subj)} subjects)")


if __name__ == "__main__":
    main()
