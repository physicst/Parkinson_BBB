"""Step 4a  -  Assemble the longitudinal Qalb modelling cohort.

Pre-specified in docs/2026-05-22-pd-bbb-csf-qalb-analysis-design.md (Step 4).

Builds its OWN cohort (not the n=61 Olink modelling cohort): PD patients
with a baseline Qalb, >=3 MDS-UPDRS III visits within 36 months, and complete
pre-registered covariates. Also attaches baseline plasma NfL as the
positive-control predictor.

Covariate loaders (LEDD, CBC) are adapted verbatim from
50_assemble_modeling_cohort.py so the covariate definitions match the
pre-registered primary analysis exactly.

Inputs:
  results/step6/qalb_baseline.parquet
  results/cache/canonical_clinical.parquet
  D:/Parkinson_file/LEDD_Concomitant_Medication_Log_02May2026.csv
  D:/Parkinson_file/Blood_Chemistry___Hematology_02May2026.csv
  D:/Parkinson_file/Current_Biospecimen_Analysis_Results_02May2026.csv  (plasma NfL)

Outputs:
  results/step6/qalb_modeling_cohort_long.parquet
  results/step6/qalb_modeling_cohort_subjects.parquet
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.ppmi_ids import normalize_patno, parse_ppmi_date

OUT = Path("results/step6")
CACHE = Path("results/cache")
BIO = "D:/Parkinson_file/Current_Biospecimen_Analysis_Results_02May2026.csv"


# --- covariate loaders (adapted from 50_assemble_modeling_cohort.py) ---
def load_ledd_per_visit(visits: pd.DataFrame) -> pd.DataFrame:
    log = pd.read_csv("D:/Parkinson_file/LEDD_Concomitant_Medication_Log_02May2026.csv",
                      low_memory=False)
    log["PATNO"] = normalize_patno(log["PATNO"])
    log["LEDD"] = pd.to_numeric(log["LEDD"], errors="coerce").fillna(0)
    log["STARTDT"] = parse_ppmi_date(log["STARTDT"])
    log["STOPDT"] = parse_ppmi_date(log["STOPDT"]).fillna(pd.Timestamp.today())
    v = visits[["PATNO", "EVENT_ID", "INFODT"]].dropna(subset=["INFODT"]).copy()
    v["LEDD_total"] = 0.0
    for patno, sub in log.groupby("PATNO"):
        mask = v["PATNO"] == patno
        if not mask.any():
            continue
        for _, m in sub.iterrows():
            if pd.isna(m["STARTDT"]):
                continue
            within = mask & v["INFODT"].between(m["STARTDT"], m["STOPDT"])
            v.loc[within, "LEDD_total"] += m["LEDD"]
    return v[["PATNO", "EVENT_ID", "LEDD_total"]]


def load_cbc_per_visit() -> pd.DataFrame:
    cbc = pd.read_csv("D:/Parkinson_file/Blood_Chemistry___Hematology_02May2026.csv",
                      low_memory=False)
    cbc["PATNO"] = normalize_patno(cbc["PATNO"])
    cbc["EVENT_ID"] = cbc["EVENT_ID"].astype(str)
    cbc["LSIRES"] = pd.to_numeric(cbc["LSIRES"], errors="coerce")
    neut = (cbc[cbc["LTSTNAME"] == "Neutrophils (%)"][["PATNO", "EVENT_ID", "LSIRES"]]
            .rename(columns={"LSIRES": "Neutrophil_pct"}).dropna()
            .groupby(["PATNO", "EVENT_ID"], as_index=False)["Neutrophil_pct"].mean())
    mono = (cbc[cbc["LTSTNAME"] == "Monocytes (%)"][["PATNO", "EVENT_ID", "LSIRES"]]
            .rename(columns={"LSIRES": "Monocyte_pct"}).dropna()
            .groupby(["PATNO", "EVENT_ID"], as_index=False)["Monocyte_pct"].mean())
    return neut.merge(mono, on=["PATNO", "EVENT_ID"], how="outer")


def load_plasma_nfl() -> pd.Series:
    """Baseline plasma NfL (Project 283, 'Average (pg/ml)'), per patient."""
    bio = pd.read_csv(BIO, low_memory=False)
    nfl = bio[(bio["TESTNAME"] == "NFL")
              & (bio["TYPE"].astype(str).str.upper() == "PLASMA")
              & (bio["CLINICAL_EVENT"] == "BL")
              & (bio["UNITS"] == "Average (pg/ml)")].copy()
    nfl["PATNO"] = normalize_patno(nfl["PATNO"])
    nfl["val"] = pd.to_numeric(nfl["TESTVALUE"], errors="coerce")
    s = nfl.dropna(subset=["val"]).groupby("PATNO")["val"].mean()
    return s.rename("plasma_nfl")


def main():
    # ---- Qalb PD patients (non-extreme) ----
    qalb = pd.read_parquet(OUT / "qalb_baseline.parquet")
    qalb["PATNO"] = normalize_patno(qalb["PATNO"])
    qalb = qalb[(qalb["qalb_qc_flag"] == "ok") & (qalb["COHORT"] == "PD")]
    qalb_pd = set(qalb["PATNO"])
    print(f"PD patients with clean baseline Qalb: {len(qalb_pd)}")

    # ---- Canonical visits, restricted to those patients ----
    canon = pd.read_parquet(CACHE / "canonical_clinical.parquet")
    canon["PATNO"] = normalize_patno(canon["PATNO"])
    canon = canon[canon["PATNO"].isin(qalb_pd)].copy()

    df = canon[canon["Visit_Months_From_BL"].between(0, 36)
               & canon["UPDRS3_Total"].notna()].copy()
    nvis = df.groupby("PATNO").size()
    keep = nvis[nvis >= 3].index
    df = df[df["PATNO"].isin(keep)].copy()
    print(f"PD x Qalb x >=3 UPDRS3 visits in 0-36mo: "
          f"{df['PATNO'].nunique()} subjects, {len(df)} visits")

    # ---- UPDRS3_Baseline (earliest in window) ----
    bl = (df.sort_values(["PATNO", "Visit_Months_From_BL"])
            .drop_duplicates("PATNO", keep="first")[["PATNO", "UPDRS3_Total"]]
            .rename(columns={"UPDRS3_Total": "UPDRS3_Baseline"}))
    df = df.merge(bl, on="PATNO", how="left")

    # ---- Covariates ----
    print("Computing LEDD per visit...")
    df = df.merge(load_ledd_per_visit(df), on=["PATNO", "EVENT_ID"], how="left")
    df["LEDD_total"] = df["LEDD_total"].fillna(0)
    print("Computing CBC fractions...")
    df = df.merge(load_cbc_per_visit(), on=["PATNO", "EVENT_ID"], how="left")

    # ---- Per-subject predictors: Qalb + plasma NfL ----
    df = df.merge(qalb[["PATNO", "qalb"]], on="PATNO", how="left")
    nfl = load_plasma_nfl()
    df = df.merge(nfl, left_on="PATNO", right_index=True, how="left")
    print(f"Subjects with baseline plasma NfL: "
          f"{df.dropna(subset=['plasma_nfl'])['PATNO'].nunique()}")

    df["Time"] = df["Visit_Months_From_BL"] / 12.0

    # ---- Complete-covariate filter (pre-registered covariate set) ----
    required = ["UPDRS3_Total", "Time", "qalb", "Age_at_Baseline", "Sex",
                "UPDRS3_Baseline", "LEDD_total", "Neutrophil_pct",
                "Monocyte_pct"]
    df = df.dropna(subset=required).copy()
    nvis2 = df.groupby("PATNO").size()
    df = df[df["PATNO"].isin(nvis2[nvis2 >= 3].index)].copy()
    print(f"Final Qalb modelling cohort: {df['PATNO'].nunique()} subjects, "
          f"{len(df)} visits")

    # ---- Standardised predictors ----
    subj = df.drop_duplicates("PATNO").set_index("PATNO")
    df["qalb_z"] = ((df["qalb"] - subj["qalb"].mean()) / subj["qalb"].std()).values
    # NfL is right-skewed -> log then z (mean/SD over the per-subject values)
    log_nfl = np.log(subj["plasma_nfl"].dropna())
    lm, ls = log_nfl.mean(), log_nfl.std()
    df["nfl_log_z"] = (np.log(df["plasma_nfl"]) - lm) / ls

    n_nfl = df.dropna(subset=["nfl_log_z"])["PATNO"].nunique()
    print(f"  with NfL positive-control predictor: {n_nfl} subjects")

    # advanced-subset flag (baseline UPDRS3 above cohort median)
    med = subj["UPDRS3_Baseline"].median()
    df["advanced"] = df["UPDRS3_Baseline"] > med
    print(f"  baseline-UPDRS3 median = {med:.1f}; "
          f"advanced subset = {df[df['advanced']]['PATNO'].nunique()} subjects")

    keep_cols = ["PATNO", "EVENT_ID", "Time", "Visit_Months_From_BL",
                 "UPDRS3_Total", "UPDRS3_Baseline", "qalb", "qalb_z",
                 "plasma_nfl", "nfl_log_z", "advanced",
                 "Age_at_Baseline", "Sex", "LEDD_total",
                 "Neutrophil_pct", "Monocyte_pct"]
    df[keep_cols].to_parquet(OUT / "qalb_modeling_cohort_long.parquet",
                             index=False)
    subj_out = df.drop_duplicates("PATNO")[
        ["PATNO", "Age_at_Baseline", "Sex", "UPDRS3_Baseline", "qalb",
         "qalb_z", "plasma_nfl", "advanced"]]
    subj_out.to_parquet(OUT / "qalb_modeling_cohort_subjects.parquet",
                        index=False)
    print(f"\n-> {OUT}/qalb_modeling_cohort_long.parquet ({len(df)} rows)")
    print(f"-> {OUT}/qalb_modeling_cohort_subjects.parquet "
          f"({df['PATNO'].nunique()} subjects)")


if __name__ == "__main__":
    main()
