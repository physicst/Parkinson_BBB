"""Step 2  -  Cross-sectional characterization of Qalb.

Pre-specified in docs/2026-05-22-pd-bbb-csf-qalb-analysis-design.md (Step 2).

Tests, on the 494 baseline-Qalb patients:
  - Qalb vs Hoehn & Yahr stage          (Spearman + Kruskal-Wallis)   [PD+Prodromal]
  - Qalb vs baseline MDS-UPDRS III      (Spearman)                    [PD+Prodromal]
  - Qalb vs age                         (Spearman)                    [all]
  - Qalb in PD/Prodromal vs HC          (Mann-Whitney)                [all]

The literature predicts Qalb rises with disease stage; in an early-stage
cohort like PPMI the gradient may be weak or absent. Either result is
informative and pre-specified as such.

NOTE: disease duration is NOT analysed here  -  it requires the PD
diagnosis-date file, which is not on disk. H&Y and baseline UPDRS3 serve
as the available stage axes. Flagged for the user.

Inputs:
  results/step6/qalb_baseline.parquet
  results/cache/canonical_clinical.parquet
  D:/Parkinson_file/MDS-UPDRS_Part_III_20Apr2026.csv

Outputs:
  results/step6/qalb_crosssectional.tsv
  results/step6/qalb_crosssectional.png
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal, mannwhitneyu

OUT = Path("results/step6")
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
         "red": "#D55E00", "grey": "#7A7A7A"}
SENTINEL = 101  # PPMI "untestable" code for NP3TOT / NHY


def main():
    rows = []  # result table

    qalb = pd.read_parquet(OUT / "qalb_baseline.parquet")
    qalb["PATNO"] = qalb["PATNO"].astype(int)
    # Pre-specified: primary analysis on non-extreme Qalb (kept-but-flagged)
    q = qalb[qalb["qalb_qc_flag"] == "ok"].copy()
    print(f"Qalb patients: {len(qalb)} total, {len(q)} after dropping "
          f"{len(qalb) - len(q)} extreme-flagged")

    # ---- Cohort: carried in the Step 1 parquet from the biospecimen file ----
    canon = pd.read_parquet("results/cache/canonical_clinical.parquet")
    canon["PATNO"] = canon["PATNO"].astype(int)
    q["cohort"] = q["COHORT"].astype(str).str.strip()

    # ---- Baseline UPDRS3 (worst non-sentinel value at BL) ----
    bl = canon[canon["EVENT_ID"] == "BL"].copy()
    bl["UPDRS3_Total"] = pd.to_numeric(bl["UPDRS3_Total"], errors="coerce")
    bl = bl[bl["UPDRS3_Total"] != SENTINEL]
    updrs_bl = bl.groupby("PATNO")["UPDRS3_Total"].max()
    q["updrs3_baseline"] = q["PATNO"].map(updrs_bl)

    # ---- Hoehn & Yahr at BL ----
    mds = pd.read_csv("D:/Parkinson_file/MDS-UPDRS_Part_III_20Apr2026.csv",
                      low_memory=False)
    mds_bl = mds[mds["EVENT_ID"] == "BL"].copy()
    mds_bl["NHY"] = pd.to_numeric(mds_bl["NHY"], errors="coerce")
    mds_bl = mds_bl[(mds_bl["NHY"].notna()) & (mds_bl["NHY"] != SENTINEL)]
    nhy = mds_bl.groupby("PATNO")["NHY"].max()
    q["hy_stage"] = q["PATNO"].map(nhy)

    print(f"  with cohort: {q['cohort'].notna().sum()}, "
          f"with baseline UPDRS3: {q['updrs3_baseline'].notna().sum()}, "
          f"with H&Y: {q['hy_stage'].notna().sum()}")
    print("\nCohort breakdown of Qalb patients:")
    print(q["cohort"].value_counts(dropna=False).to_string())

    # The CSF albumin sub-study (Project 181) enrolled only PD and Control
    # (no Prodromal). Severity-gradient tests therefore run within PD.
    spectrum = q[q["cohort"] == "PD"].copy()

    # ---- Test 1: Qalb vs age (all) ----
    a = q.dropna(subset=["Age_at_Baseline"])
    rho, p = spearmanr(a["qalb"], a["Age_at_Baseline"])
    rows.append(["Qalb vs age", "all cohorts", "Spearman", len(a),
                 f"rho={rho:+.3f}", p])

    # ---- Test 2: Qalb vs baseline UPDRS3 (PD) ----
    u = spectrum.dropna(subset=["updrs3_baseline"])
    rho, p = spearmanr(u["qalb"], u["updrs3_baseline"])
    rows.append(["Qalb vs baseline MDS-UPDRS III", "PD", "Spearman",
                 len(u), f"rho={rho:+.3f}", p])

    # ---- Test 3: Qalb vs H&Y stage (PD) ----
    h = spectrum.dropna(subset=["hy_stage"])
    rho, p = spearmanr(h["qalb"], h["hy_stage"])
    rows.append(["Qalb vs H&Y stage (ordinal)", "PD", "Spearman",
                 len(h), f"rho={rho:+.3f}", p])
    groups = [g["qalb"].values for _, g in h.groupby("hy_stage") if len(g) >= 5]
    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        rows.append(["Qalb across H&Y stages", "PD (stages n>=5)",
                     "Kruskal-Wallis", len(h), f"H={stat:.2f}", p])

    # ---- Test 4: Qalb PD vs Control ----
    case = q[q["cohort"] == "PD"]["qalb"]
    ctrl = q[q["cohort"] == "Control"]["qalb"]
    if len(ctrl) >= 5:
        stat, p = mannwhitneyu(case, ctrl, alternative="two-sided")
        rows.append(["Qalb PD vs Control", f"PD n={len(case)}, "
                     f"Control n={len(ctrl)}", "Mann-Whitney",
                     len(case) + len(ctrl),
                     f"median {case.median():.2f} vs {ctrl.median():.2f}", p])

    # ---- Results table ----
    res = pd.DataFrame(rows, columns=["test", "subset", "method", "n",
                                      "effect", "p_value"])
    res.to_csv(OUT / "qalb_crosssectional.tsv", sep="\t", index=False)
    print("\n=== Cross-sectional results ===")
    print(res.to_string(index=False))

    # ---- Figure ----
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    # (a) Qalb by H&Y stage
    ax = axes[0]
    hh = spectrum.dropna(subset=["hy_stage"])
    stages = sorted(hh["hy_stage"].unique())
    data = [hh[hh["hy_stage"] == s]["qalb"].values for s in stages]
    bp = ax.boxplot(data, positions=range(len(stages)), widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.2))
    for patch in bp["boxes"]:
        patch.set_facecolor(OKABE["blue"]); patch.set_alpha(0.55)
    rng = np.random.default_rng(1)
    for i, s in enumerate(stages):
        v = hh[hh["hy_stage"] == s]["qalb"].values
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.13, 0.13, len(v)), v,
                   s=12, alpha=0.5, color=OKABE["blue"], edgecolor="white",
                   linewidths=0.3)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([f"H&Y {int(s)}\n(n={len(d)})"
                        for s, d in zip(stages, data)])
    ax.set_ylabel("Qalb (x 10$^{-3}$)")
    rho_h, p_h = spearmanr(hh["qalb"], hh["hy_stage"])
    ax.set_title(f"Qalb by Hoehn & Yahr stage\nSpearman rho = {rho_h:+.3f}, "
                 f"p = {p_h:.2f}")
    ax.set_ylim(0, min(spectrum["qalb"].max(), 30) * 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.14, 1.05, "a", transform=ax.transAxes, fontsize=13,
            fontweight="bold")

    # (b) Qalb vs baseline UPDRS3
    ax = axes[1]
    uu = spectrum.dropna(subset=["updrs3_baseline"])
    ax.scatter(uu["updrs3_baseline"], uu["qalb"], s=16, alpha=0.5,
               color=OKABE["orange"], edgecolor="white", linewidths=0.3)
    z = np.polyfit(uu["updrs3_baseline"], uu["qalb"], 1)
    xx = np.linspace(uu["updrs3_baseline"].min(), uu["updrs3_baseline"].max(), 50)
    ax.plot(xx, np.polyval(z, xx), color=OKABE["red"], lw=1.8)
    rho_u, p_u = spearmanr(uu["qalb"], uu["updrs3_baseline"])
    ax.set_xlabel("Baseline MDS-UPDRS Part III")
    ax.set_ylabel("Qalb (x 10$^{-3}$)")
    ax.set_title(f"Qalb vs baseline motor severity\nSpearman rho = {rho_u:+.3f}, "
                 f"p = {p_u:.2f}  (n={len(uu)})")
    ax.set_ylim(0, min(spectrum["qalb"].max(), 30) * 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.14, 1.05, "b", transform=ax.transAxes, fontsize=13,
            fontweight="bold")

    # (c) Qalb by cohort
    ax = axes[2]
    order = [c for c in ["Control", "PD"] if c in q["cohort"].unique()]
    cdata = [q[q["cohort"] == c]["qalb"].values for c in order]
    bp = ax.boxplot(cdata, positions=range(len(order)), widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.2))
    cmap = {"Control": OKABE["green"], "PD": OKABE["red"]}
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(cmap.get(c, OKABE["grey"])); patch.set_alpha(0.55)
    rng = np.random.default_rng(2)
    for i, c in enumerate(order):
        v = q[q["cohort"] == c]["qalb"].values
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.13, 0.13, len(v)), v,
                   s=12, alpha=0.5, color=cmap.get(c, OKABE["grey"]),
                   edgecolor="white", linewidths=0.3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{c}\n(n={len(d)})" for c, d in zip(order, cdata)])
    ax.set_ylabel("Qalb (x 10$^{-3}$)")
    ax.set_title("Qalb by diagnostic group")
    ax.set_ylim(0, min(q["qalb"].max(), 30) * 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.14, 1.05, "c", transform=ax.transAxes, fontsize=13,
            fontweight="bold")

    fig.suptitle("Step 2 - Cross-sectional characterization of Qalb "
                 "(n=484, extreme-flagged excluded)", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "qalb_crosssectional.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {OUT}/qalb_crosssectional.tsv")
    print(f"-> {OUT}/qalb_crosssectional.png")
    print("\nNOTE: disease duration not analysed (needs PD diagnosis-date "
          "file, not on disk).")


if __name__ == "__main__":
    main()
