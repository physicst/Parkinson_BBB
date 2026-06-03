"""Step 3  -  Three-way decoupling: Qalb vs brain score vs Olink PC1.

Pre-specified in docs/2026-05-22-pd-bbb-csf-qalb-analysis-design.md (Step 3).

The published manuscript reports a 2-way decoupling (brain-derived
transcriptomic score vs Olink plasma PC1). This step adds Qalb  -  the
gold-standard BBB-permeability marker  -  as a third axis and computes all
three pairwise Spearman correlations on the patients who have all three.

Pre-specified decision rule (design §5): if |rho| < 0.2 for Qalb vs both
surrogates, the decoupling extends from 2 to 3 modalities.

Inputs:
  results/step6/qalb_baseline.parquet
  results/step4/cross_modal_scores.parquet   (v2_score, olink_pc1)

Outputs:
  results/step6/qalb_decoupling.tsv
  results/step6/qalb_decoupling.png
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = Path("results/step6")
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
         "red": "#D55E00", "purple": "#CC79A7", "grey": "#7A7A7A"}


def main():
    qalb = pd.read_parquet(OUT / "qalb_baseline.parquet")
    qalb["PATNO"] = qalb["PATNO"].astype(int)
    qalb = qalb[qalb["qalb_qc_flag"] == "ok"]  # pre-specified: drop extremes

    xmod = pd.read_parquet("results/step4/cross_modal_scores.parquet")
    xmod["PATNO"] = xmod["PATNO"].astype(int)

    # brain-vs-Olink on the full published cross-modal cohort, for continuity
    rho_bo_full, p_bo_full = spearmanr(xmod["v2_score"], xmod["olink_pc1"])
    print(f"Published 2-way (brain vs Olink), full n={len(xmod)}: "
          f"rho={rho_bo_full:+.3f}, p={p_bo_full:.3f}")

    # merge to the all-three-modalities set
    m = xmod.merge(qalb[["PATNO", "qalb"]], on="PATNO", how="inner")
    n = len(m)
    print(f"Patients with brain score + Olink PC1 + Qalb: n={n}")

    pairs = [
        ("Brain-derived score", "Olink BBB PC1", "v2_score", "olink_pc1"),
        ("Qalb (CSF permeability)", "Brain-derived score", "qalb", "v2_score"),
        ("Qalb (CSF permeability)", "Olink BBB PC1", "qalb", "olink_pc1"),
    ]
    rows = []
    for la, lb, ca, cb in pairs:
        rho, p = spearmanr(m[ca], m[cb])
        rows.append([f"{la} vs {lb}", ca, cb, n, rho, p])
        print(f"  {la:24s} vs {lb:20s}: rho={rho:+.3f}, p={p:.3f}")

    res = pd.DataFrame(rows, columns=["pair", "x", "y", "n", "spearman_rho",
                                      "p_value"])
    res.to_csv(OUT / "qalb_decoupling.tsv", sep="\t", index=False)

    # pre-specified decision rule
    q_brain = res.loc[1, "spearman_rho"]
    q_olink = res.loc[2, "spearman_rho"]
    decoupled = abs(q_brain) < 0.2 and abs(q_olink) < 0.2
    verdict = ("3-WAY DECOUPLED (|rho| < 0.2 for Qalb vs both surrogates)"
               if decoupled else
               "NOT fully decoupled (|rho| >= 0.2 for at least one pair)")
    print(f"\nDecision rule: {verdict}")

    # ---- Figure: 3 pairwise scatters ----
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    specs = [
        (axes[0], "v2_score", "olink_pc1", "Brain-derived BBB score",
         "Olink BBB PC1", OKABE["blue"], "a", res.loc[0]),
        (axes[1], "v2_score", "qalb", "Brain-derived BBB score",
         "Qalb (x 10$^{-3}$)", OKABE["purple"], "b", res.loc[1]),
        (axes[2], "olink_pc1", "qalb", "Olink BBB PC1",
         "Qalb (x 10$^{-3}$)", OKABE["orange"], "c", res.loc[2]),
    ]
    for ax, cx, cy, lx, ly, col, lab, r in specs:
        ax.scatter(m[cx], m[cy], s=18, alpha=0.55, color=col,
                   edgecolor="white", linewidths=0.3)
        z = np.polyfit(m[cx], m[cy], 1)
        xx = np.linspace(m[cx].min(), m[cx].max(), 50)
        ax.plot(xx, np.polyval(z, xx), color="black", lw=1.3, alpha=0.6)
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        sig = "" if r["p_value"] >= 0.05 else "  *"
        ax.set_title(f"Spearman rho = {r['spearman_rho']:+.3f}, "
                     f"p = {r['p_value']:.2f}{sig}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.16, 1.06, lab, transform=ax.transAxes, fontsize=13,
                fontweight="bold")

    band = "#e6ffec" if decoupled else "#fde2e2"
    edge = OKABE["green"] if decoupled else OKABE["red"]
    fig.suptitle(f"Step 3 - Three-way decoupling of PD vascular modalities "
                 f"(n = {n})", fontsize=12, fontweight="bold", y=1.04)
    fig.text(0.5, -0.06, verdict, ha="center", fontsize=11, fontweight="bold",
             color=edge,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=band,
                       edgecolor="none"))
    fig.tight_layout()
    fig.savefig(OUT / "qalb_decoupling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {OUT}/qalb_decoupling.tsv")
    print(f"-> {OUT}/qalb_decoupling.png")


if __name__ == "__main__":
    main()
