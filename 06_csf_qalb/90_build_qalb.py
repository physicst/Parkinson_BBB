"""Step 1  -  Build the CSF/serum albumin ratio (Qalb) and run QC.

Pre-specified in docs/2026-05-22-pd-bbb-csf-qalb-analysis-design.md (Step 1).

Qalb = (CSF albumin / plasma albumin) x 1000.
Both albumin assays are PPMI Project 181, reported in ng/ml (same units),
so the ratio is dimensionless; x1000 places it on the conventional Qalb
scale (normal range ~5-8).

QC gate (pre-specified): Qalb must correlate positively with age. Albumin
permeability rises with age in every published cohort; if that correlation
is absent the ratio is mis-constructed and downstream steps must not run.

Inputs:
  D:/Parkinson_file/Current_Biospecimen_Analysis_Results_02May2026.csv
  results/cache/canonical_clinical.parquet   (Age_at_Baseline)

Outputs:
  results/step6/qalb_baseline.parquet
  results/step6/qalb_qc.txt
  results/step6/qalb_qc.png
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BIO = "D:/Parkinson_file/Current_Biospecimen_Analysis_Results_02May2026.csv"
CANON = "results/cache/canonical_clinical.parquet"
OUT = Path("results/step6")
OUT.mkdir(parents=True, exist_ok=True)

OKABE = {"blue": "#0072B2", "orange": "#E69F00", "red": "#D55E00",
         "grey": "#7A7A7A"}


def main():
    log = []

    def emit(msg=""):
        print(msg)
        log.append(msg)

    emit("=== Step 1  -  Build Qalb (CSF/serum albumin ratio) ===\n")

    # ---- Load albumin assays at baseline ----
    bio = pd.read_csv(BIO, low_memory=False)
    bio["TESTVALUE"] = pd.to_numeric(bio["TESTVALUE"], errors="coerce")

    def albumin(test):
        s = bio[(bio["TESTNAME"] == test) & (bio["CLINICAL_EVENT"] == "BL")]
        # one value per patient; if duplicated, take the mean
        return s.groupby("PATNO")["TESTVALUE"].mean()

    csf = albumin("CSF Albumin").rename("csf_albumin_ngml")
    plasma = albumin("Plasma Albumin").rename("plasma_albumin_ngml")
    df = pd.concat([csf, plasma], axis=1).dropna()
    df.index = df.index.astype(int)
    emit(f"Patients with both CSF + plasma albumin at BL: {len(df)}")

    # Carry cohort + sex from the biospecimen file itself (complete coverage,
    # unlike the canonical clinical table's Cohort_Current).
    meta = (bio[(bio["TESTNAME"] == "CSF Albumin") &
                (bio["CLINICAL_EVENT"] == "BL")]
            .drop_duplicates("PATNO").set_index("PATNO")[["COHORT", "SEX"]])
    meta.index = meta.index.astype(int)
    df = df.join(meta)
    emit("Cohort coverage (from biospecimen file):")
    for c, n in df["COHORT"].value_counts(dropna=False).items():
        emit(f"  {c}: {n}")

    # ---- Compute Qalb ----
    df["qalb"] = df["csf_albumin_ngml"] / df["plasma_albumin_ngml"] * 1000.0

    q = df["qalb"]
    emit(f"\nQalb distribution (x10^-3):")
    emit(f"  n        = {len(q)}")
    emit(f"  min      = {q.min():.2f}")
    emit(f"  Q1       = {q.quantile(0.25):.2f}")
    emit(f"  median   = {q.median():.2f}")
    emit(f"  Q3       = {q.quantile(0.75):.2f}")
    emit(f"  max      = {q.max():.2f}")
    emit(f"  mean+/-SD= {q.mean():.2f} +/- {q.std():.2f}")

    # ---- QC flags (report, do not drop) ----
    # Conventional interpretation: Qalb < 1 implausible; Qalb > 30 suggests a
    # traumatic (blood-contaminated) tap. Flag both, keep all rows.
    df["qalb_qc_flag"] = np.where(
        (df["qalb"] < 1) | (df["qalb"] > 30), "extreme", "ok")
    n_flag = (df["qalb_qc_flag"] == "extreme").sum()
    emit(f"\nExtreme-Qalb flags (Qalb<1 or >30, kept but marked): {n_flag}")
    # plasma albumin physiological plausibility (ng/ml -> g/L = /1e6)
    pl_gL = df["plasma_albumin_ngml"] / 1e6
    emit(f"Plasma albumin g/L: median {pl_gL.median():.1f} "
         f"(range {pl_gL.min():.1f}-{pl_gL.max():.1f})")
    csf_mgL = df["csf_albumin_ngml"] / 1e3
    emit(f"CSF albumin mg/L:   median {csf_mgL.median():.1f} "
         f"(range {csf_mgL.min():.1f}-{csf_mgL.max():.1f})")

    # ---- Age join + QC gate ----
    canon = pd.read_parquet(CANON)
    canon["PATNO"] = canon["PATNO"].astype(int)
    age = (canon[["PATNO", "Age_at_Baseline"]]
           .dropna().drop_duplicates("PATNO").set_index("PATNO")["Age_at_Baseline"])
    df = df.join(age)
    paired = df.dropna(subset=["Age_at_Baseline"])
    rho, p = spearmanr(paired["qalb"], paired["Age_at_Baseline"])
    emit(f"\n--- QC GATE: Qalb vs age ---")
    emit(f"  n with age   = {len(paired)}")
    emit(f"  Spearman rho = {rho:+.3f}")
    emit(f"  p-value      = {p:.2e}")
    # Use the non-extreme subset for the gate verdict
    ok = paired[paired["qalb_qc_flag"] == "ok"]
    rho_ok, p_ok = spearmanr(ok["qalb"], ok["Age_at_Baseline"])
    emit(f"  (excluding extreme flags: rho = {rho_ok:+.3f}, p = {p_ok:.2e}, "
         f"n = {len(ok)})")
    gate = "PASS" if (rho > 0 and p < 0.05) else "FAIL"
    emit(f"  QC GATE: {gate}  "
         f"(expected: positive rho, p<0.05  -  albumin permeability rises with age)")

    # ---- Save ----
    df = df.reset_index().rename(columns={"index": "PATNO"})
    df.to_parquet(OUT / "qalb_baseline.parquet", index=False)
    emit(f"\n-> {OUT}/qalb_baseline.parquet  ({len(df)} patients)")

    (OUT / "qalb_qc.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    emit(f"-> {OUT}/qalb_qc.txt")

    # ---- QC figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    ax0 = axes[0]
    ax0.hist(q, bins=40, color=OKABE["blue"], alpha=0.8, edgecolor="white")
    ax0.axvline(q.median(), color=OKABE["red"], lw=1.5, ls="--",
                label=f"median = {q.median():.1f}")
    ax0.axvspan(5, 8, color=OKABE["grey"], alpha=0.15,
                label="textbook normal 5-8")
    ax0.set_xlabel("Qalb (CSF/plasma albumin x 10$^{-3}$)")
    ax0.set_ylabel("Patients")
    ax0.set_title(f"Qalb distribution (n = {len(q)})")
    ax0.legend(fontsize=8, frameon=False)
    ax0.spines[["top", "right"]].set_visible(False)

    ax1 = axes[1]
    ax1.scatter(paired["Age_at_Baseline"], paired["qalb"], s=14, alpha=0.5,
                color=OKABE["blue"], edgecolor="white", linewidths=0.3)
    z = np.polyfit(paired["Age_at_Baseline"], paired["qalb"], 1)
    xx = np.linspace(paired["Age_at_Baseline"].min(),
                     paired["Age_at_Baseline"].max(), 50)
    ax1.plot(xx, np.polyval(z, xx), color=OKABE["red"], lw=1.8)
    ax1.set_xlabel("Age at baseline (years)")
    ax1.set_ylabel("Qalb (x 10$^{-3}$)")
    ax1.set_title(f"QC gate: Qalb vs age\nSpearman rho = {rho:+.3f}, "
                  f"p = {p:.1e}  [{gate}]")
    ax1.set_ylim(0, min(q.max(), 30) * 1.05)
    ax1.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "qalb_qc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    emit(f"-> {OUT}/qalb_qc.png")

    emit(f"\n=== Step 1 done  -  QC gate {gate} ===")
    if gate != "PASS":
        emit("QC GATE FAILED  -  do not proceed to Steps 2-4 until resolved.")


if __name__ == "__main__":
    main()
