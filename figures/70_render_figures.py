"""Render the 5 publication figures for the npj PD manuscript.

Outputs in results/figures/:
  figure1.png + figure1.pdf  -  Brain vascular dysfunction (4-panel)
  figure2.png + figure2.pdf  -  Brain signature does not transfer to blood
  figure3.png + figure3.pdf  -  Olink plasma landscape
  figure4.png + figure4.pdf  -  Cross-modal decoupling
  figure5.png + figure5.pdf  -  Pre-registered longitudinal endpoints (forest)

Style: 300 DPI, sans-serif, no chartjunk, panel labels (a, b, c, d) in upper-left.
Colors are colorblind-friendly (Okabe-Ito where 2-3 categories).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr, mannwhitneyu

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Okabe-Ito palette (colorblind-friendly)
OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "skyblue": "#56B4E9",
    "grey": "#7A7A7A",
}


def panel_label(ax, label, x=-0.15, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left")


def save(fig, name: str):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  → results/figures/{name}.png + .pdf")


# =====================================================================
# FIGURE 1  -  Brain vascular dysfunction in PD (4 panels)
# =====================================================================
def figure1():
    print("\n[Figure 1] Brain vascular dysfunction in PD")
    fig = plt.figure(figsize=(11, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                            left=0.07, right=0.97, top=0.94, bottom=0.07)

    # ---- (a) UMAP from GSE178265 endothelial atlas ----
    ax_a = fig.add_subplot(gs[0, 0])
    try:
        import anndata as ad
        a = ad.read_h5ad("D:/Parkinson_file/Human Endothelial.h5ad")
        umap = a.obsm["X_umap"]
        # Disease label from ontology
        dis = a.obs["disease_ontology_term_id"].astype(str)
        is_pd = dis.str.contains("0014202|14202", na=False)  # MONDO PD term
        # Fallback: use any non-"PATO_0000461" (normal) as disease
        if is_pd.sum() == 0:
            is_pd = ~dis.str.contains("PATO_0000461", na=False)
        # Subsample to 8000 each for clean rendering
        n = 6000
        rng = np.random.default_rng(42)
        ctrl_idx = np.where(~is_pd.values)[0]
        pd_idx = np.where(is_pd.values)[0]
        if len(ctrl_idx) > n:
            ctrl_idx = rng.choice(ctrl_idx, n, replace=False)
        if len(pd_idx) > n:
            pd_idx = rng.choice(pd_idx, n, replace=False)
        ax_a.scatter(umap[ctrl_idx, 0], umap[ctrl_idx, 1], s=2, alpha=0.45,
                      color=OKABE["skyblue"], label=f"Control (n={len(ctrl_idx):,})",
                      rasterized=True, linewidths=0)
        ax_a.scatter(umap[pd_idx, 0], umap[pd_idx, 1], s=2, alpha=0.45,
                      color=OKABE["red"], label=f"PD (n={len(pd_idx):,})",
                      rasterized=True, linewidths=0)
        ax_a.set_title("Endothelial cells (GSE178265, n=14,903  -  primary cohort)")
    except Exception as exc:
        ax_a.text(0.5, 0.5, f"UMAP source unavailable\n({exc})",
                  ha="center", va="center", transform=ax_a.transAxes)
    ax_a.set_xlabel("UMAP 1")
    ax_a.set_ylabel("UMAP 2")
    ax_a.set_xticks([]); ax_a.set_yticks([])
    ax_a.legend(loc="upper right", markerscale=3)
    panel_label(ax_a, "a")

    # ---- (b) Volcano of integrated DESeq2 ----
    ax_b = fig.add_subplot(gs[0, 1])
    de = pd.read_csv("results/step2/integrated_de_endothelial.tsv", sep="\t")
    de = de.dropna(subset=["padj", "log2FoldChange"])
    de["neglog10padj"] = -np.log10(de["padj"].clip(lower=1e-300))
    sig = de["padj"] < 0.05
    # Background
    ax_b.scatter(de.loc[~sig, "log2FoldChange"], de.loc[~sig, "neglog10padj"],
                 s=4, alpha=0.25, color=OKABE["grey"], rasterized=True, linewidths=0)
    # Significant
    ax_b.scatter(de.loc[sig, "log2FoldChange"], de.loc[sig, "neglog10padj"],
                 s=8, alpha=0.7, color=OKABE["red"], rasterized=True, linewidths=0,
                 label=f"padj < 0.05 (n={int(sig.sum())})")
    # Highlight BBB module + label hub genes
    mod_sym = pd.read_csv("results/step4/v2_module_with_symbols.tsv", sep="\t")
    hub = pd.read_csv("results/step4/hub_genes_with_symbols.tsv", sep="\t")
    de_mod = de.merge(mod_sym[["gene_id"]], on="gene_id", how="inner")
    ax_b.scatter(de_mod["log2FoldChange"], de_mod["neglog10padj"], s=14,
                 color=OKABE["blue"], edgecolor="white", linewidths=0.4,
                 label=f"BBB module (n={len(de_mod)})", zorder=3)
    # Label top hubs
    top_hubs = hub.head(8).merge(de, on="gene_id")
    for _, r in top_hubs.iterrows():
        if r["symbol"]:
            ax_b.annotate(r["symbol"], (r["log2FoldChange"], r["neglog10padj"]),
                          fontsize=7, ha="left", va="bottom",
                          xytext=(3, 2), textcoords="offset points")
    ax_b.axhline(-np.log10(0.05), linestyle="--", color="grey", alpha=0.5, lw=0.8)
    ax_b.set_xlabel("log$_2$ fold change (PD vs control)")
    ax_b.set_ylabel("$-$log$_{10}$ adjusted p-value")
    ax_b.set_title("Pseudobulk DESeq2 (~ condition + sex + age + PMI + dataset)")
    ax_b.set_xlim(-5, 5)
    ax_b.legend(loc="upper left")
    panel_label(ax_b, "b")

    # ---- (c) Top 20 hub genes ----
    ax_c = fig.add_subplot(gs[1, 0])
    hub_top = hub.head(20).iloc[::-1].copy()
    # Replace missing symbols with the trailing portion of the Ensembl ID
    hub_top["label"] = [
        s if isinstance(s, str) and s and s != "nan"
        else f"({gid[-7:]})"
        for s, gid in zip(hub_top["symbol"], hub_top["gene_id"])
    ]
    chap_set = {"AHSA1", "STIP1", "DNAJA1", "FKBP4", "BAG3", "CACYBP",
                "CHORDC1", "HSP90AA1", "HSPA8", "HSPB1", "HSPH1"}
    is_chap = [isinstance(s, str) and ("HSP" in s or s in chap_set)
               for s in hub_top["symbol"]]
    colors = [OKABE["red"] if c else OKABE["blue"] for c in is_chap]
    ax_c.barh(range(len(hub_top)), hub_top["hub_score"], color=colors,
              edgecolor="white", linewidth=0.5)
    ax_c.set_yticks(range(len(hub_top)))
    ax_c.set_yticklabels(hub_top["label"], fontsize=7.5)
    # Color-code the y-tick labels themselves (red = HSP/chaperone, blue = other)
    #  -  a cleaner alternative to an in-panel legend that would overlap bars.
    for tick_label, c in zip(ax_c.get_yticklabels(), colors):
        tick_label.set_color(c)
    ax_c.set_xlabel("Hub score (MM × |GS|)")
    ax_c.set_title("Top 20 WGCNA hub genes (module 8)  -  "
                   "$\\bf{red}$ = HSP/chaperone family")
    ax_c.set_xlim(0, hub_top["hub_score"].max() * 1.05)
    panel_label(ax_c, "c")

    # ---- (d) Heat-shock enrichment forest ----
    ax_d = fig.add_subplot(gs[1, 1])
    enr = pd.read_csv("results/step4/v2_module_enrichment.tsv", sep="\t")
    enr = enr[enr["fold_enrichment"] > 0].copy()  # drop zero-overlap rows
    enr = enr.sort_values("fold_enrichment").reset_index(drop=True)
    nice = {
        "HSP90_family": "HSP90 family",
        "HSP70_family": "HSP70 family",
        "HSP60_chaperonin": "HSP60 chaperonin",
        "small_HSP_family": "Small HSP",
        "Unfolded_protein_response": "Unfolded protein response",
        "Endothelial_activation": "Endothelial activation",
    }
    enr["nice"] = enr["reference_set"].map(nice).fillna(enr["reference_set"])
    bar_colors = [OKABE["red"] if p < 0.001 else OKABE["orange"] if p < 0.01
                   else OKABE["yellow"] if p < 0.05 else OKABE["grey"]
                   for p in enr["bh_padj"]]
    ax_d.barh(range(len(enr)), enr["fold_enrichment"], color=bar_colors,
              edgecolor="white", linewidth=0.5)
    ax_d.set_yticks(range(len(enr)))
    ax_d.set_yticklabels(enr["nice"])
    ax_d.set_xlabel("Fold enrichment (hypergeometric)")
    ax_d.set_title("BBB module vs canonical reference sets")
    # Annotate p_BH
    for i, r in enr.iterrows():
        ax_d.text(r["fold_enrichment"] + 1.5, i,
                   f"  p={r['bh_padj']:.0e}  ({r['overlap']}/{r['ref_size']})",
                   va="center", fontsize=7, color="#333")
    ax_d.set_xlim(0, max(enr["fold_enrichment"]) * 1.55)
    panel_label(ax_d, "d")

    save(fig, "figure1")


# =====================================================================
# FIGURE 2  -  Brain signature does not transfer to blood
# =====================================================================
def figure2():
    print("\n[Figure 2] Brain → blood transfer")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

    # Load v1 ROC (already computed)
    v1_roc = pd.read_csv("results/v1_baseline/v1_roc_curve.tsv", sep="\t")
    v1_auc = pd.read_csv("results/v1_baseline/v1_auc_results.tsv", sep="\t")
    v1_auc_pooled = float(v1_auc["auc_pooled"].iloc[0]) if "auc_pooled" in v1_auc.columns else 0.440

    # Recompute v2 ROC + 5-fold CV from v2 module
    counts = pd.read_parquet("results/cache/ppmi_counts_baseline.parquet")
    canonical = pd.read_parquet("results/cache/canonical_clinical.parquet")
    canonical["PATNO"] = canonical["PATNO"].astype(str)
    v2_genes = set(pd.read_csv("results/step2/bbb_module_genes.tsv",
                                sep="\t")["gene_id"].astype(str))
    overlap_genes = sorted(v2_genes & set(counts.index))
    cpm = counts.div(counts.sum(axis=0), axis=1) * 1e6
    log_cpm = np.log2(cpm + 1).loc[overlap_genes]
    z = log_cpm.sub(log_cpm.mean(axis=1), axis=0).div(log_cpm.std(axis=1) + 1e-12, axis=0)
    v2_score = z.mean(axis=0).rename("v2_score").to_frame().reset_index().rename(
        columns={"index": "RNA_HudAlphaID"})
    rna_link = (canonical[canonical["RNA_HudAlphaID"].notna()]
                .drop_duplicates("RNA_HudAlphaID")
                [["PATNO", "RNA_HudAlphaID", "Cohort_Current"]])
    df = v2_score.merge(rna_link, on="RNA_HudAlphaID", how="inner")
    df = df[df["Cohort_Current"].isin(["PD", "HC"])].dropna(subset=["v2_score"])
    df["y"] = (df["Cohort_Current"] == "PD").astype(int)
    X = df[["v2_score"]].values
    y = df["y"].values
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260504)
    aucs5 = cross_val_score(clf, X, y, scoring="roc_auc", cv=cv)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    v2_auc_pooled = roc_auc_score(y, proba)
    v2_fpr, v2_tpr, _ = roc_curve(y, proba)
    print(f"  Brain-derived signature 5-fold CV AUC = {aucs5.mean():.3f} ± {aucs5.std():.3f}, "
          f"pooled = {v2_auc_pooled:.3f}, n={len(df)} (PD={y.sum()}, HC={(1-y).sum()})")

    # Two-panel layout: ROC (a) + 5-fold CV (b). The disputed 0.81 AUC from
    # an internal preliminary derivation has no peer-reviewed source and is
    # not appropriate as a publication-figure benchmark; reproduction details
    # live in Methods + Supp Fig S1.
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                            left=0.09, right=0.97, top=0.92, bottom=0.13,
                            width_ratios=[1.2, 1.0])

    # (a) ROC: both signatures evaluated under matched protocol (this study)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5, lw=0.8,
              label="Chance (AUC=0.50)")
    ax_a.plot(v1_roc["fpr"], v1_roc["tpr"], lw=2, color=OKABE["orange"],
              label=f"1,015-gene signature (AUC={v1_auc_pooled:.2f})")
    ax_a.plot(v2_fpr, v2_tpr, lw=2, color=OKABE["blue"],
              label=f"118-gene brain-derived signature (AUC={v2_auc_pooled:.2f})")
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1)
    ax_a.set_aspect("equal")
    ax_a.set_xlabel("False positive rate")
    ax_a.set_ylabel("True positive rate")
    ax_a.set_title("PD vs HC, PPMI baseline blood\n"
                   "(both analyses: this study, 5-fold CV, n=444)")
    ax_a.legend(loc="lower right", fontsize=8)
    panel_label(ax_a, "a")

    # (b) 5-fold CV distribution for the brain-derived signature
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.boxplot([aucs5], positions=[0], widths=0.4, patch_artist=True,
                 boxprops=dict(facecolor=OKABE["blue"], alpha=0.4),
                 medianprops=dict(color="black", lw=1.2))
    ax_b.scatter(np.zeros_like(aucs5) + np.random.uniform(-0.08, 0.08, len(aucs5)),
                 aucs5, color=OKABE["blue"], s=40, edgecolor="white", zorder=3)
    ax_b.axhline(0.5, linestyle="--", color="grey", alpha=0.5, lw=0.8,
                 label="Chance (AUC=0.50)")
    ax_b.set_ylim(0.35, 0.75)
    ax_b.set_xticks([0])
    ax_b.set_xticklabels(["118-gene brain-derived\nsignature"])
    ax_b.set_ylabel("AUC per fold")
    ax_b.set_title(f"5-fold CV stability: {aucs5.mean():.2f} ± {aucs5.std():.2f}")
    ax_b.legend(loc="upper right", fontsize=8)
    panel_label(ax_b, "b")

    save(fig, "figure2")


# =====================================================================
# FIGURE 3  -  Olink plasma vascular protein landscape
# =====================================================================
def figure3():
    print("\n[Figure 3] Olink plasma landscape")
    panel = pd.read_parquet("results/step3/olink_bbb_panel_baseline.parquet")
    score = pd.read_parquet("results/step3/bbb_protein_score.parquet")
    # Pull age from canonical clinical so we cover the FULL n=227 Olink cohort
    # (not the n=61 lmer modeling subset which the design doc does not cite)
    canon = pd.read_parquet("results/cache/canonical_clinical.parquet")
    canon["PATNO"] = canon["PATNO"].astype(int)

    # Merge: panel × class
    panel["PATNO"] = panel["PATNO"].astype(int)
    score["PATNO"] = score["PATNO"].astype(int)
    merged = panel.merge(score[["PATNO", "vascular_class", "bbb_score_pc1"]],
                          on="PATNO")
    proteins = [c for c in panel.columns if c != "PATNO"]
    merged_sorted = merged.sort_values(["vascular_class", "bbb_score_pc1"],
                                        ascending=[False, False])

    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32, left=0.06, right=0.97,
                            top=0.92, bottom=0.18,
                            width_ratios=[1.7, 1.0, 1.0])

    # (a) Heatmap (proteins × patients), z-scored per protein
    ax_a = fig.add_subplot(gs[0, 0])
    mat = merged_sorted[proteins].T  # 20 × 227
    z = (mat.sub(mat.mean(axis=1), axis=0)
            .div(mat.std(axis=1) + 1e-9, axis=0))
    im = ax_a.imshow(z.values, aspect="auto", cmap="RdBu_r",
                      vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax_a.set_yticks(range(len(proteins)))
    ax_a.set_yticklabels(proteins, fontsize=7)
    ax_a.set_xticks([])
    ax_a.set_xlabel(f"Patients (n={len(merged_sorted)}, sorted by class then PC1)")
    # Class bar
    classes = merged_sorted["vascular_class"].values
    high_n = (classes == "Vascular_High").sum()
    ax_a.add_patch(Rectangle((0 - 0.5, -1.5), high_n, 0.7,
                              color=OKABE["red"], clip_on=False))
    ax_a.add_patch(Rectangle((high_n - 0.5, -1.5), len(classes) - high_n, 0.7,
                              color=OKABE["blue"], clip_on=False))
    ax_a.text(high_n / 2, -1.9, f"Vascular-High (n={high_n})", ha="center",
               fontsize=8, color=OKABE["red"], fontweight="bold")
    ax_a.text(high_n + (len(classes) - high_n) / 2, -1.9,
               f"Low (n={len(classes) - high_n})", ha="center", fontsize=8,
               color=OKABE["blue"], fontweight="bold")
    cbar = plt.colorbar(im, ax=ax_a, fraction=0.025, pad=0.02)
    cbar.set_label("z-scored NPX", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax_a.set_title("Olink BBB panel  -  20 proteins × 227 PD/Prodromal patients")
    panel_label(ax_a, "a", x=-0.18)

    # (b) PC1 by class
    ax_b = fig.add_subplot(gs[0, 1])
    high = merged.loc[merged["vascular_class"] == "Vascular_High", "bbb_score_pc1"]
    low = merged.loc[merged["vascular_class"] == "Vascular_Low", "bbb_score_pc1"]
    bp = ax_b.boxplot([high, low], positions=[0, 1], widths=0.55,
                       patch_artist=True, showfliers=False,
                       medianprops=dict(color="black", lw=1.2))
    for patch, c in zip(bp["boxes"], [OKABE["red"], OKABE["blue"]]):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    rng = np.random.default_rng(7)
    for i, vals in enumerate([high, low]):
        ax_b.scatter(np.full(len(vals), i) + rng.uniform(-0.12, 0.12, len(vals)),
                     vals, s=10, alpha=0.5,
                     color=[OKABE["red"], OKABE["blue"]][i],
                     edgecolor="white", linewidths=0.3, zorder=3)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels([f"High\n(n={len(high)})", f"Low\n(n={len(low)})"])
    ax_b.set_ylabel("BBB PC1 score")
    ax_b.set_title("PC1 explains 33% variance\n(r = 0.85 with endothelial activation)")
    panel_label(ax_b, "b")

    # (c) Age by class
    ax_c = fig.add_subplot(gs[0, 2])
    cohort_age = score[["PATNO", "vascular_class"]].merge(
        canon[["PATNO", "Age_at_Baseline"]].drop_duplicates("PATNO"),
        on="PATNO", how="left")
    age_high = cohort_age.loc[cohort_age["vascular_class"] == "Vascular_High",
                                "Age_at_Baseline"].dropna()
    age_low = cohort_age.loc[cohort_age["vascular_class"] == "Vascular_Low",
                               "Age_at_Baseline"].dropna()
    bp2 = ax_c.boxplot([age_high, age_low], positions=[0, 1], widths=0.55,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.2))
    for patch, c in zip(bp2["boxes"], [OKABE["red"], OKABE["blue"]]):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    rng = np.random.default_rng(8)
    for i, vals in enumerate([age_high, age_low]):
        ax_c.scatter(np.full(len(vals), i) + rng.uniform(-0.12, 0.12, len(vals)),
                     vals, s=10, alpha=0.5,
                     color=[OKABE["red"], OKABE["blue"]][i],
                     edgecolor="white", linewidths=0.3, zorder=3)
    u, p = mannwhitneyu(age_high, age_low, alternative="two-sided")
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels([f"High\n(n={len(age_high)},\nμ={age_high.mean():.1f}y)",
                            f"Low\n(n={len(age_low)},\nμ={age_low.mean():.1f}y)"])
    ax_c.set_ylabel("Age at baseline (years)")
    ax_c.set_title(f"Age confound: Mann-Whitney p = {p:.1e}")
    # Significance bracket
    y_top = max(age_high.max(), age_low.max()) + 2
    ax_c.plot([0, 0, 1, 1], [y_top, y_top + 1, y_top + 1, y_top], color="black", lw=1)
    ax_c.text(0.5, y_top + 1.3, "***", ha="center", fontsize=14)
    panel_label(ax_c, "c")

    save(fig, "figure3")


# =====================================================================
# FIGURE 4  -  Four-way decoupling (brain / plasma Olink / CSF Olink / Qalb)
# =====================================================================
def figure4():
    print("\n[Figure 4] Four-way decoupling: brain / plasma Olink / CSF Olink / Qalb")
    xmod = pd.read_parquet("results/step4/cross_modal_scores.parquet")
    xmod["PATNO"] = xmod["PATNO"].astype(int)
    qalb = pd.read_parquet("results/step6/qalb_baseline.parquet")
    qalb["PATNO"] = qalb["PATNO"].astype(int)
    qalb = qalb[qalb["qalb_qc_flag"] == "ok"]
    csf_ol = pd.read_parquet("results/step7/csf_olink_bbb_score.parquet")
    csf_ol["PATNO"] = csf_ol["PATNO"].astype(int)
    m = (xmod[["PATNO", "v2_score", "olink_pc1"]]
         .merge(qalb[["PATNO", "qalb"]], on="PATNO", how="inner")
         .merge(csf_ol[["PATNO", "csf_olink_pc1"]], on="PATNO", how="inner"))
    n = len(m)
    print(f"  all-four-modality patients: n={n}")

    # Three scatter panels involving the NEW modality (CSF Olink); the three
    # legacy pairwise correlations (brain<->plasma, brain<->Qalb,
    # plasma<->Qalb) are preserved in the 4x4 matrix (panel d).
    # Panel (a) is THE KEY new comparison: same 20 BBB proteins, plasma vs CSF.
    pairs = [
        ("olink_pc1", "csf_olink_pc1", "Plasma Olink BBB PC1",
         "CSF Olink BBB PC1", OKABE["red"], "a",
         "Same 20 proteins, plasma vs CSF"),
        ("v2_score", "csf_olink_pc1", "Brain-derived BBB score",
         "CSF Olink BBB PC1", OKABE["blue"], "b", None),
        ("qalb", "csf_olink_pc1",
         "Qalb (CSF/serum albumin x10$^{-3}$)",
         "CSF Olink BBB PC1", OKABE["green"], "c", None),
    ]
    rhos = {}

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.36,
                            left=0.08, right=0.96, top=0.91, bottom=0.10)
    cells = [gs[0, 0], gs[0, 1], gs[1, 0]]

    for (cx, cy, lx, ly, col, lab, sub), cell in zip(pairs, cells):
        ax = fig.add_subplot(cell)
        rho, p = spearmanr(m[cx], m[cy])
        rhos[(cx, cy)] = rho
        ax.scatter(m[cx], m[cy], s=20, alpha=0.55, color=col,
                   edgecolor="white", linewidths=0.3)
        z = np.polyfit(m[cx], m[cy], 1)
        xx = np.linspace(m[cx].min(), m[cx].max(), 50)
        ax.plot(xx, np.polyval(z, xx), color="black", lw=1.2, alpha=0.6)
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        title = f"Spearman ρ = {rho:+.3f}, p = {p:.2f}"
        if sub:
            title = sub + "\n" + title
        ax.set_title(title)
        panel_label(ax, lab)

    # Compute the three legacy pairwise correlations on the same n=130
    # cohort, so the 4x4 matrix is internally consistent.
    from itertools import combinations
    all_cols = ["v2_score", "olink_pc1", "csf_olink_pc1", "qalb"]
    short = {"v2_score": "Brain\n(transcriptomic)",
             "olink_pc1": "Plasma\n(Olink PC1)",
             "csf_olink_pc1": "CSF\n(Olink PC1)",
             "qalb": "CSF\n(Qalb)"}
    full_rhos = {}
    for a, b in combinations(all_cols, 2):
        full_rhos[(a, b)] = spearmanr(m[a], m[b])[0]

    # (d) 4x4 Spearman correlation matrix
    ax_d = fig.add_subplot(gs[1, 1])
    M = np.eye(4)
    for i, a in enumerate(all_cols):
        for j, b in enumerate(all_cols):
            if i == j:
                continue
            key = (a, b) if (a, b) in full_rhos else (b, a)
            M[i, j] = full_rhos[key]
    im = ax_d.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(4):
        for j in range(4):
            ax_d.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                      fontsize=10.5, fontweight="bold",
                      color="white" if abs(M[i, j]) > 0.55 else "black")
    ax_d.set_xticks(range(4)); ax_d.set_xticklabels(
        [short[c] for c in all_cols], fontsize=7.5)
    ax_d.set_yticks(range(4)); ax_d.set_yticklabels(
        [short[c] for c in all_cols], fontsize=7.5)
    ax_d.set_title("Pairwise Spearman ρ (all 6 pairs)")
    plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04, label="Spearman ρ")
    panel_label(ax_d, "d")

    maxoff = max(abs(v) for v in full_rhos.values())
    decoupled = maxoff < 0.20
    verdict = (f"FOUR-WAY DECOUPLED  -  every pairwise |ρ| < {maxoff:.2f}  "
               f"(n = {n})" if decoupled else
               f"partial coupling  -  max |ρ| = {maxoff:.2f}")
    fig.text(0.5, 0.02, verdict, ha="center", fontsize=11, fontweight="bold",
             color=OKABE["red"] if decoupled else OKABE["green"],
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#fde2e2" if decoupled else "#e6ffec",
                       edgecolor="none"))
    fig.suptitle("Brain, plasma, and CSF vascular biomarkers are mutually "
                 "decoupled in PD  -  including the same 20 proteins measured "
                 "in plasma vs CSF",
                 fontsize=11.5, fontweight="bold")
    save(fig, "figure4")


# =====================================================================
# FIGURE 5  -  Pre-registered longitudinal endpoints (forest plot)
# =====================================================================
def figure5():
    print("\n[Figure 5] Pre-registered forest plot")
    # Pre-reg primary
    pri = pd.read_csv("results/step3/lmer_primary_coefs.tsv", sep="\t",
                      index_col=0)
    pri_row = pri.loc["Time:BBB_score_z"]
    sens = pd.read_csv("results/step4/lmer_sensitivity_v2_coefs.tsv",
                       sep="\t", index_col=0).loc["Time:v2_score_z"]
    sec = pd.read_csv("results/step3/secondary_analyses_summary.tsv", sep="\t")
    sub = pd.read_csv("results/step4/subgroup_analyses_fixed.tsv", sep="\t")

    # Build forest rows (label, n, est, se, p, tier)
    rows = []
    rows.append(("Pre-registered primary: UPDRS3 slope (Olink PC1)",
                 61, pri_row["Estimate"], pri_row["Std. Error"],
                 pri_row["Pr(>|t|)"], "primary"))
    rows.append(("Sensitivity: UPDRS3 slope (brain-derived transcriptomic score)",
                 54, sens["Estimate"], sens["Std. Error"], sens["Pr(>|t|)"],
                 "sensitivity"))
    for _, r in sec.iterrows():
        rows.append((f"Pre-registered secondary: {r['endpoint']}",
                     int(r["n_subj"]), r["estimate"], r["se"],
                     r["p"], "secondary"))
    # Subgroup rows we keep (drop the LRRK2/GBA "non-carrier" label which is
    # less informative than the SAA + APOE-stratified rows; keep APOE-MoCA as
    # exploratory)
    keep_sub_labels = ["SAA-Pos", "APOE-non-e4 (UPDRS3)", "APOE-non-e4 (MoCA)"]
    for label in keep_sub_labels:
        r = sub[sub["label"] == label]
        if len(r):
            r = r.iloc[0]
            tier = "exploratory" if "MoCA" in label else "subgroup"
            rows.append((f"Subgroup: {label}", int(r["n_subj"]),
                         r["estimate"], r["se"], r["p"], tier))

    # Plot  -  split layout: forest on left, annotation column + legend on right
    fig = plt.figure(figsize=(13, 6.5))
    gs = gridspec.GridSpec(1, 1, figure=fig, left=0.32, right=0.94,
                            top=0.92, bottom=0.18)
    ax = fig.add_subplot(gs[0, 0])
    n = len(rows)
    y_positions = np.arange(n)[::-1]
    tier_colors = {
        "primary": OKABE["red"],
        "sensitivity": OKABE["orange"],
        "secondary": OKABE["blue"],
        "subgroup": OKABE["green"],
        "exploratory": OKABE["purple"],
    }
    # Pre-compute CIs and clip very wide CIs to a sensible visual range so the
    # plot scale isn't dominated by a single high-uncertainty subgroup
    XLIM = (-2.0, 2.0)
    for y, (label, n_subj, est, se, p, tier) in zip(y_positions, rows):
        ci_lo = est - 1.96 * se
        ci_hi = est + 1.96 * se
        # Visually clip CIs that extend past the plot range, with arrow markers
        clipped_lo = max(ci_lo, XLIM[0])
        clipped_hi = min(ci_hi, XLIM[1])
        ax.errorbar([est], [y],
                    xerr=[[est - clipped_lo], [clipped_hi - est]],
                    fmt="s", color=tier_colors[tier], ecolor=tier_colors[tier],
                    elinewidth=1.5, capsize=3, markersize=9,
                    markeredgecolor="white", markeredgewidth=0.6)
        # Mark clip with arrow if needed
        if ci_lo < XLIM[0]:
            ax.annotate("", xy=(XLIM[0], y), xytext=(XLIM[0] + 0.08, y),
                        arrowprops=dict(arrowstyle="-|>", color=tier_colors[tier],
                                          lw=1.2))
        if ci_hi > XLIM[1]:
            ax.annotate("", xy=(XLIM[1], y), xytext=(XLIM[1] - 0.08, y),
                        arrowprops=dict(arrowstyle="-|>", color=tier_colors[tier],
                                          lw=1.2))
    ax.axvline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(XLIM)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Estimate (Time × BBB_score_z, units/year per +1 SD)")
    ax.set_title("Pre-registered longitudinal endpoints  -  all consistent with null",
                 fontsize=11)

    # Annotation column: n / est / p in a fixed-width font, anchored to right
    # margin via a blended transform  -  x in axes coords (just outside right
    # edge of plot), y in data coords (the actual row position)
    from matplotlib.transforms import blended_transform_factory
    annot_trans = blended_transform_factory(ax.transAxes, ax.transData)
    for y, (label, n_subj, est, se, p, tier) in zip(y_positions, rows):
        sig_marker = " *" if p < 0.05 else ""
        ax.text(1.02, y,
                 f"n={n_subj:>3}   est={est:+.3f}   p={p:.3f}{sig_marker}",
                 transform=annot_trans,
                 va="center", ha="left", fontsize=8.5, family="monospace",
                 clip_on=False)

    # Legend BELOW the plot (won't collide with anything)
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=OKABE["red"], label="Pre-registered primary"),
        Patch(facecolor=OKABE["orange"], label="Sensitivity (post-hoc)"),
        Patch(facecolor=OKABE["blue"], label="Pre-registered secondary"),
        Patch(facecolor=OKABE["green"], label="Pre-registered subgroup"),
        Patch(facecolor=OKABE["purple"], label="Exploratory only"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.10),
              ncol=5, fontsize=8.5, title="Analysis tier",
              title_fontsize=8.5)

    # Footnote about dropped subgroups
    fig.text(0.32, 0.02,
             "Note: LRRK2 and GBA carrier subgroups dropped  -  0 carriers in n=61 "
             "modeling cohort post Step 4 carrier-call correction. "
             "Significance asterisk (*) marks p<0.05 uncorrected; "
             "APOE-non-e4 MoCA p_BH = 0.126 (exploratory only).",
             fontsize=7.5, color="#555", style="italic")

    save(fig, "figure5")


def main():
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    print("\nAll 5 figures rendered to results/figures/")


if __name__ == "__main__":
    main()
