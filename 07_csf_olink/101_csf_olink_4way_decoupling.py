"""Step B - Four-way decoupling test (brain + plasma Olink + CSF Olink + Qalb).

Pre-specified in docs/2026-05-24-pd-bbb-csf-olink-analysis-design.md (Step B).
Decision rule: all 6 pairwise |rho| < 0.20 -> 4-WAY DECOUPLED.

Inputs:
  results/step4/cross_modal_scores.parquet    (brain v2_score + plasma olink_pc1)
  results/step6/qalb_baseline.parquet         (Qalb + qc flag)
  results/step7/csf_olink_bbb_score.parquet   (CSF Olink PC1, from Step A)

Outputs:
  results/step7/decoupling_4way.tsv
  results/step7/csf_olink_4way_summary.txt
"""
from __future__ import annotations
from pathlib import Path
from itertools import combinations

import pandas as pd
from scipy.stats import spearmanr

OUT = Path("results/step7")

log: list[str] = []


def emit(msg: str = "") -> None:
    print(msg)
    log.append(msg)


def main() -> None:
    emit("=== Step B - Four-way decoupling test at baseline ===\n")

    # Load all four per-patient scores at BL
    xmod = pd.read_parquet("results/step4/cross_modal_scores.parquet")
    xmod["PATNO"] = xmod["PATNO"].astype(int)
    qalb = pd.read_parquet("results/step6/qalb_baseline.parquet")
    qalb["PATNO"] = qalb["PATNO"].astype(int)
    qalb = qalb[qalb["qalb_qc_flag"] == "ok"]
    csf = pd.read_parquet("results/step7/csf_olink_bbb_score.parquet")
    csf["PATNO"] = csf["PATNO"].astype(int)

    # Inner join on PATNO
    m = (xmod[["PATNO", "v2_score", "olink_pc1"]]
         .merge(qalb[["PATNO", "qalb"]], on="PATNO", how="inner")
         .merge(csf[["PATNO", "csf_olink_pc1"]], on="PATNO", how="inner"))
    n = len(m)
    emit(f"Patients with ALL FOUR modalities at BL: n = {n}")
    emit(f"  brain transcriptomic, plasma Olink PC1, CSF Qalb, CSF Olink PC1")

    # Pretty labels
    cols = {
        "v2_score": "Brain transcriptomic",
        "olink_pc1": "Plasma Olink PC1",
        "csf_olink_pc1": "CSF Olink PC1",
        "qalb": "CSF Qalb",
    }
    order = ["v2_score", "olink_pc1", "csf_olink_pc1", "qalb"]

    # Six pairwise Spearman correlations
    emit("\n--- Pairwise Spearman correlations ---")
    rows = []
    for a, b in combinations(order, 2):
        rho, p = spearmanr(m[a], m[b])
        rows.append({"a": cols[a], "b": cols[b], "n": n,
                     "spearman_rho": rho, "p_value": p})
        emit(f"  {cols[a]:22s} vs {cols[b]:22s}  rho = {rho:+.3f}  "
             f"p = {p:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "decoupling_4way.tsv", sep="\t", index=False)

    # 4x4 matrix view
    emit("\n--- 4 x 4 Spearman correlation matrix ---")
    mat = pd.DataFrame(1.0, index=order, columns=order)
    for a, b in combinations(order, 2):
        r = res.loc[(res["a"] == cols[a]) & (res["b"] == cols[b]),
                    "spearman_rho"].iloc[0]
        mat.loc[a, b] = r
        mat.loc[b, a] = r
    pretty = mat.rename(index=cols, columns=cols).round(3)
    emit(pretty.to_string())

    # Pre-specified decision rule
    max_abs = res["spearman_rho"].abs().max()
    decoupled = max_abs < 0.20
    plasma_csf_olink_rho = res.loc[
        (res["a"] == "Plasma Olink PC1") &
        (res["b"] == "CSF Olink PC1"), "spearman_rho"].iloc[0]
    emit(f"\n--- Pre-specified decision rule ---")
    emit(f"Maximum |rho| across the 6 pairwise tests: {max_abs:.3f}")
    emit(f"Threshold for 4-way decoupling: |rho| < 0.20")
    emit(f"\nKEY NEW PAIR  -  plasma Olink vs CSF Olink (same 20 proteins, "
         f"two compartments):  rho = {plasma_csf_olink_rho:+.3f}")
    if decoupled:
        emit("\n>>> VERDICT: 4-WAY DECOUPLED")
        emit("    All six pairwise correlations are below the |rho|=0.20 "
             "decoupling threshold.")
        emit("    The decoupling extends from 3-way to 4-way; the same "
             "20 BBB proteins, measured in plasma versus CSF, are also "
             "statistically independent  -  removing the panel-composition "
             "reviewer attack on the central claim.")
    elif abs(plasma_csf_olink_rho) >= 0.30:
        emit("\n>>> VERDICT: COMPARTMENTAL COUPLING (plasma <-> CSF same "
             "proteins)")
        emit(f"    Plasma vs CSF Olink rho = {plasma_csf_olink_rho:+.3f} "
             "(>= 0.30), indicating the same proteins do partially track "
             "across compartments. Reframe per design Section 5.")
    else:
        emit("\n>>> VERDICT: PARTIAL 4-WAY DECOUPLING")
        emit(f"    Maximum |rho| = {max_abs:.3f}, above the 0.20 "
             "decoupling threshold but below the 0.30 coupling threshold. "
             "Reported transparently; specific pair flagged in Discussion.")

    (OUT / "csf_olink_4way_summary.txt").write_text("\n".join(log) + "\n",
                                                     encoding="utf-8")
    emit(f"\n-> {OUT}/decoupling_4way.tsv")
    emit(f"-> {OUT}/csf_olink_4way_summary.txt")
    emit("\n=== Step B done ===")


if __name__ == "__main__":
    main()
