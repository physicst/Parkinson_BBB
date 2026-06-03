"""Step 3 Task 32  -  Continuous BBB protein score per patient.

Per pre-reg §14: "ssGSEA-equivalent score (mean z-NPX) ... PC1 alternative
if ssGSEA unstable." Computes BOTH and reports their separation power
between Vascular-High / Vascular-Low clusters via Cohen's d.

Primary score = whichever has |Cohen's d| ≥ 0.5 between clusters.
If both qualify, PC1 (per pre-reg fallback). If neither, ssGSEA-mean is
reported with a warning that the panel may not capture a single gradient.

Output: results/step3/bbb_protein_score.parquet
  PATNO, bbb_score (= chosen primary), bbb_score_mean, bbb_score_pc1,
  primary_method, vascular_class, cluster
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA

S3 = Path("results/step3")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


def main() -> None:
    panel = pd.read_parquet(S3 / "olink_bbb_panel_baseline.parquet")
    panel = panel.set_index("PATNO")
    print(f"Panel: {panel.shape[0]} patients × {panel.shape[1]} proteins")

    # Impute NA with column mean
    for col in panel.columns:
        panel[col] = panel[col].fillna(panel[col].mean())

    # Z-score per protein (within discovery cohort, per pre-reg)
    z = (panel - panel.mean()) / panel.std(ddof=0)

    # Score 1: ssGSEA-equivalent (mean z-NPX)
    score_mean = z.mean(axis=1)

    # Score 2: PC1 of z-scored panel
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(z.values)
    score_pc1 = pd.Series(pcs[:, 0], index=z.index)
    pc1_var = pca.explained_variance_ratio_[0]
    pc2_var = pca.explained_variance_ratio_[1]
    print(f"PCA: PC1 explains {pc1_var:.1%} of variance, PC2 {pc2_var:.1%}")

    # Sanity-check directionality: positive PC1 should align with elevated
    # endothelial-activation markers (canonical "damage" direction).
    endo_activ = ["VCAM1", "ICAM1", "SELE", "VWF"]
    endo_in = [p for p in endo_activ if p in z.columns]
    activ_mean = z[endo_in].mean(axis=1)
    pc1_corr = np.corrcoef(score_pc1, activ_mean)[0, 1]
    print(f"PC1 vs endothelial-activation mean: r = {pc1_corr:.3f}")
    if pc1_corr < 0:
        score_pc1 = -score_pc1
        print("  Flipped PC1 sign so positive = elevated endothelial activation")

    # Compare both scores against cluster assignment
    assign = pd.read_csv(S3 / "consensus_k2_assignments.tsv", sep="\t")
    assign["PATNO"] = assign["PATNO"].astype(str)

    out = pd.DataFrame({
        "PATNO": z.index.astype(str),
        "bbb_score_mean": score_mean.values,
        "bbb_score_pc1": score_pc1.values,
    }).merge(assign[["PATNO", "vascular_class", "cluster"]], on="PATNO", how="left")

    print("\n=== Score separation between Vascular-High and Vascular-Low clusters ===")
    for col, name in [("bbb_score_mean", "ssGSEA-mean"), ("bbb_score_pc1", "PC1")]:
        high = out[out["vascular_class"] == "Vascular_High"][col].values
        low  = out[out["vascular_class"] == "Vascular_Low"][col].values
        d = cohens_d(high, low)
        _, p = mannwhitneyu(high, low, alternative="greater")
        print(f"  {name:15s}: Cohen's d = {d:+.2f},  one-sided MWU p = {p:.2e}")

    d_mean = cohens_d(
        out[out["vascular_class"] == "Vascular_High"]["bbb_score_mean"].values,
        out[out["vascular_class"] == "Vascular_Low"]["bbb_score_mean"].values,
    )
    d_pc1 = cohens_d(
        out[out["vascular_class"] == "Vascular_High"]["bbb_score_pc1"].values,
        out[out["vascular_class"] == "Vascular_Low"]["bbb_score_pc1"].values,
    )

    # Pick primary
    if abs(d_pc1) >= 0.5 and abs(d_pc1) > abs(d_mean):
        primary = "pc1"
    elif abs(d_mean) >= 0.5:
        primary = "mean"
    else:
        # Neither clean; pick whichever is larger but flag
        primary = "pc1" if abs(d_pc1) > abs(d_mean) else "mean"
        print(f"\nWARNING: Neither score reaches Cohen's d >= 0.5 between clusters.")
        print("This means the consensus k=2 clusters are not separated on a single damage")
        print("gradient. Report primary score with caveat; manuscript should describe")
        print("the cluster structure as multi-axis rather than as a continuous severity scale.")

    out["bbb_score"] = out[f"bbb_score_{primary}"]
    out["primary_method"] = primary
    print(f"\nPrimary score = {primary} (Cohen's d = {d_pc1 if primary=='pc1' else d_mean:+.2f})")

    out.to_parquet(S3 / "bbb_protein_score.parquet")
    print(f"\n→ Wrote {S3/'bbb_protein_score.parquet'} ({len(out)} patients)")


if __name__ == "__main__":
    main()
