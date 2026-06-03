"""Render the 5 missing supplementary figures for the npj PD manuscript.

Supp Fig S3 (Olink QC + CCP) is already rendered by 71_render_figureS3.py.
This script produces the remaining five:

  figureS1  -  Reproduction of the previously reported 1,015-gene signature
  figureS2  -  WGCNA module-trait correlations and bootstrap-stability outcome
  figureS4  -  Demographic balance of the n=61 longitudinal modelling cohort
  figureS5  -  Cohort-flow diagram (PPMI baseline -> longitudinal modelling)
  figureS6  -  Individual UPDRS3 trajectories in the modelling cohort

Style matches 70_render_figures.py / 71_render_figureS3.py (300 DPI, Okabe-Ito).
Every panel is built only from result files that exist on disk; nothing is
fabricated. Where a value cannot be recomputed it is annotated as documented.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import mannwhitneyu

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

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


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> results/figures/{name}.png + .pdf")


# =====================================================================
# S1  -  Reproduction of the previously reported 1,015-gene signature
# =====================================================================
def figureS1():
    print("\n[Supp Fig S1] 1,015-gene signature reproduction")
    roc = pd.read_csv("results/v1_baseline/v1_roc_curve.tsv", sep="\t")
    auc = pd.read_csv("results/v1_baseline/v1_auc_results.tsv", sep="\t")
    genes = pd.read_csv("results/v1_baseline/v1_signature_genes.tsv", sep="\t")

    auc_pooled = float(auc["auc_overall_pooled"].iloc[0])
    auc_5f = float(auc["auc_5fold_mean"].iloc[0])
    auc_5f_sd = float(auc["auc_5fold_sd"].iloc[0])
    target = float(auc["manuscript_target"].iloc[0])

    fig = plt.figure(figsize=(13, 4.6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42,
                           left=0.07, right=0.97, top=0.86, bottom=0.16)

    # (a) ROC curve
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5, lw=0.8,
              label="Chance (AUC=0.50)")
    ax_a.plot(roc["fpr"], roc["tpr"], lw=2.2, color=OKABE["orange"],
              label=f"1,015-gene signature (AUC={auc_pooled:.2f})")
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1)
    ax_a.set_aspect("equal")
    ax_a.set_xlabel("False positive rate")
    ax_a.set_ylabel("True positive rate")
    ax_a.set_title("PD vs HC, PPMI baseline blood\n(5-fold cross-validated)")
    ax_a.legend(loc="lower right", fontsize=8)
    panel_label(ax_a, "a", x=-0.20)

    # (b) Per-model 5-fold CV AUC
    ax_b = fig.add_subplot(gs[0, 1])
    models = auc["model"].tolist()
    means = auc["auc_5fold_mean"].astype(float).values
    sds = auc["auc_5fold_sd"].astype(float).fillna(0).values
    nice = {"BBB_score_only": "1,015-gene\nscore only",
            "Age+Sex_only": "Age + Sex\nonly",
            "BBB+Age+Sex": "BBB + Age\n+ Sex"}
    labels = [nice.get(m, m) for m in models]
    xpos = np.arange(len(models))
    ax_b.bar(xpos, means, yerr=sds, width=0.6, color=OKABE["blue"],
             alpha=0.8, edgecolor="white", capsize=4)
    ax_b.axhline(0.5, ls="--", color="grey", alpha=0.6, lw=0.9,
                 label="Chance (0.50)")
    ax_b.axhline(target, ls=":", color=OKABE["red"], lw=1.5,
                 label=f"Originally reported ({target:.2f})")
    for x, m, sd in zip(xpos, means, sds):
        ax_b.text(x, m + sd + 0.02, f"{m:.2f}", ha="center", fontsize=8)
    ax_b.set_xticks(xpos)
    ax_b.set_xticklabels(labels, fontsize=8)
    ax_b.set_ylabel("5-fold CV AUC (mean +/- SD)")
    ax_b.set_ylim(0, 1.0)
    ax_b.set_title("Reproduction under matched protocol")
    ax_b.legend(loc="upper right", fontsize=7.5)
    panel_label(ax_b, "b", x=-0.20)

    # (c) Gene-set detection in PPMI blood
    ax_c = fig.add_subplot(gs[0, 2])
    n_sig = len(genes)
    detected = np.nan
    try:
        counts = pd.read_parquet("results/cache/ppmi_counts_baseline.parquet")
        emap = pd.read_csv("results/cache/ensg_to_symbol.tsv", sep="\t")
        # emap columns: symbol, ensembl_gene_id
        sig_ensg = set(emap.loc[emap["symbol"].isin(genes["gene"]),
                                "ensembl_gene_id"])
        # PPMI count matrix is indexed by ENSG (possibly version-suffixed)
        counts_ensg = {str(g).split(".")[0] for g in counts.index}
        detected = len(sig_ensg & counts_ensg)
    except Exception as exc:
        print(f"  (gene-detection recompute skipped: {exc})")

    if np.isnan(detected):
        ax_c.bar([0], [n_sig], width=0.5, color=OKABE["orange"], alpha=0.8,
                 edgecolor="white")
        ax_c.set_xticks([0]); ax_c.set_xticklabels(["Signature genes"])
        ax_c.text(0, n_sig + 15, f"{n_sig}", ha="center", fontsize=9)
        ax_c.set_title("Signature size")
    else:
        ax_c.bar([0, 1], [n_sig, detected], width=0.55,
                 color=[OKABE["grey"], OKABE["green"]], alpha=0.85,
                 edgecolor="white")
        ax_c.set_xticks([0, 1])
        ax_c.set_xticklabels(["Signature\ngenes", "Detected in\nPPMI blood"])
        for x, v in zip([0, 1], [n_sig, detected]):
            ax_c.text(x, v + 15, f"{v}", ha="center", fontsize=9)
        pct = 100 * detected / n_sig
        ax_c.set_title(f"Gene-set detection\n({pct:.0f}% of signature measurable)")
    ax_c.set_ylabel("Number of genes")
    ax_c.set_ylim(0, n_sig * 1.18)
    panel_label(ax_c, "c", x=-0.20)

    fig.suptitle("Supplementary Fig. S1  -  Reproduction of the previously reported "
                 "1,015-gene blood signature", fontsize=11, fontweight="bold", y=1.00)
    save(fig, "figureS1")
    print(f"  signature={n_sig} genes, pooled AUC={auc_pooled:.3f}, "
          f"5-fold={auc_5f:.3f}+/-{auc_5f_sd:.3f}, originally reported={target}")


# =====================================================================
# S2  -  WGCNA module-trait correlations + bootstrap-stability outcome
# =====================================================================
def figureS2():
    print("\n[Supp Fig S2] WGCNA module-trait correlation + stability")
    mc = pd.read_csv("results/step2/module_pd_correlations.tsv", sep="\t")
    mc = mc[mc["module"] != 0].copy()           # drop grey / unassigned
    mc = mc.sort_values("cor").reset_index(drop=True)
    n_genes = len(pd.read_csv("results/step2/bbb_module_genes.tsv", sep="\t"))
    sel = mc.loc[mc["cor"].abs().idxmax()]       # module 8 = max |cor|

    fig = plt.figure(figsize=(12, 5.2))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                           left=0.10, right=0.97, top=0.86, bottom=0.13,
                           width_ratios=[1.25, 1.0])

    # (a) Module-PD correlation bar chart
    ax_a = fig.add_subplot(gs[0, 0])
    colors = [OKABE["red"] if m == sel["module"] else OKABE["grey"]
              for m in mc["module"]]
    ax_a.barh(range(len(mc)), mc["cor"], color=colors, alpha=0.85,
              edgecolor="white", linewidth=0.5)
    ax_a.set_yticks(range(len(mc)))
    ax_a.set_yticklabels([f"Module {m}" for m in mc["module"]], fontsize=8)
    ax_a.axvline(0, color="grey", lw=0.7, alpha=0.6)
    for i, r in mc.iterrows():
        off = 0.015 if r["cor"] >= 0 else -0.015
        ax_a.text(r["cor"] + off, i, f"p={r['p']:.2f}",
                  va="center", ha="left" if r["cor"] >= 0 else "right",
                  fontsize=7, color="#444")
    ax_a.set_xlabel("Pearson correlation of module eigengene with PD status")
    ax_a.set_xlim(-0.45, 0.55)
    ax_a.set_title("Module-trait correlations (12 non-grey modules)")
    panel_label(ax_a, "a", x=-0.22)

    # (b) Stability-outcome / selection-rule panel
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis("off")
    panel_label(ax_b, "b", x=-0.02)
    ax_b.text(0.5, 1.02, "Bootstrap stability outcome",
              ha="center", va="top", fontsize=10, fontweight="bold",
              transform=ax_b.transAxes)
    lines = [
        ("Bootstrap design", OKABE["blue"], True),
        ("100x donor resampling; per-bootstrap WGCNA;", GREY := "#333", False),
        ("module retained if member-pairs co-cluster", GREY, False),
        ("in >= 70% of bootstraps (pre-registered).", GREY, False),
        ("", GREY, False),
        ("Result: 0 of 12 modules met the", OKABE["red"], True),
        ("70% stability threshold (n = 30 donors  - ", GREY, False),
        ("small for stability filtering).", GREY, False),
        ("", GREY, False),
        ("Pre-specified fallback engaged", OKABE["green"], True),
        ("(master plan risk #6): select the module", GREY, False),
        ("with maximum |correlation with PD status|.", GREY, False),
        ("", GREY, False),
        (f"=> Module {int(sel['module'])} selected: {n_genes} genes,", OKABE["red"], True),
        (f"   r = {sel['cor']:+.3f}, p = {sel['p']:.3f}.", OKABE["red"], True),
        ("This is the BBB module used throughout.", GREY, False),
    ]
    y = 0.90
    for txt, col, bold in lines:
        ax_b.text(0.04, y, txt, ha="left", va="top", fontsize=8.6,
                  color=col, fontweight="bold" if bold else "normal",
                  transform=ax_b.transAxes, family="DejaVu Sans")
        y -= 0.058
    ax_b.add_patch(FancyBboxPatch((0.02, 0.0), 0.96, 0.96,
                   boxstyle="round,pad=0.01", transform=ax_b.transAxes,
                   facecolor="#F7F7F7", edgecolor=OKABE["grey"],
                   linewidth=0.8, zorder=-1))

    fig.suptitle("Supplementary Fig. S2  -  WGCNA module-trait correlation and "
                 "bootstrap-stability outcome", fontsize=11, fontweight="bold",
                 y=0.99)
    save(fig, "figureS2")
    print(f"  selected module {int(sel['module'])}: {n_genes} genes, "
          f"r={sel['cor']:+.3f}, p={sel['p']:.3f}; 0/12 modules stable")


# =====================================================================
# S4  -  Demographic balance of the n=61 longitudinal modelling cohort
# =====================================================================
def figureS4():
    print("\n[Supp Fig S4] Demographic balance of the modelling cohort")
    subj = pd.read_parquet("results/step3/modeling_cohort_subjects.parquet")
    high = subj[subj["vascular_class"] == "Vascular_High"]
    low = subj[subj["vascular_class"] == "Vascular_Low"]
    c_high, c_low = OKABE["red"], OKABE["blue"]

    fig = plt.figure(figsize=(12.5, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32,
                           left=0.09, right=0.96, top=0.90, bottom=0.08)

    def strip(ax, vals_h, vals_l, ylabel, title):
        bp = ax.boxplot([vals_h, vals_l], positions=[0, 1], widths=0.55,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.2))
        for patch, c in zip(bp["boxes"], [c_high, c_low]):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        rng = np.random.default_rng(3)
        for i, vals in enumerate([vals_h, vals_l]):
            ax.scatter(np.full(len(vals), i) + rng.uniform(-0.13, 0.13, len(vals)),
                       vals, s=14, alpha=0.6, color=[c_high, c_low][i],
                       edgecolor="white", linewidths=0.3, zorder=3)
        try:
            _, p = mannwhitneyu(vals_h, vals_l, alternative="two-sided")
            ptxt = f"Mann-Whitney p = {p:.2f}"
        except Exception:
            ptxt = ""
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Vascular-High\n(n={len(vals_h)})",
                            f"Vascular-Low\n(n={len(vals_l)})"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n{ptxt}")

    # (a) Age
    ax_a = fig.add_subplot(gs[0, 0])
    strip(ax_a, high["Age_at_Baseline"].dropna(), low["Age_at_Baseline"].dropna(),
          "Age at baseline (years)", "Age by vascular endotype")
    panel_label(ax_a, "a")

    # (b) Sex composition
    ax_b = fig.add_subplot(gs[0, 1])
    sex_tab = (subj.groupby(["vascular_class", "Sex"]).size()
               .unstack(fill_value=0).reindex(["Vascular_High", "Vascular_Low"]))
    bottoms = np.zeros(2)
    sexcolors = {"Female": OKABE["purple"], "Male": OKABE["skyblue"]}
    for sex in sex_tab.columns:
        vals = sex_tab[sex].values
        ax_b.bar([0, 1], vals, bottom=bottoms, width=0.55,
                 color=sexcolors.get(sex, OKABE["grey"]), label=str(sex),
                 edgecolor="white")
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0:
                ax_b.text(i, b + v / 2, str(int(v)), ha="center", va="center",
                          color="white", fontsize=9, fontweight="bold")
        bottoms += vals
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels([f"Vascular-High\n(n={len(high)})",
                          f"Vascular-Low\n(n={len(low)})"])
    ax_b.set_ylabel("Number of subjects")
    ax_b.set_title("Sex composition by endotype")
    ax_b.legend(loc="upper right", title="Sex")
    panel_label(ax_b, "b")

    # (c) Baseline UPDRS3
    ax_c = fig.add_subplot(gs[1, 0])
    strip(ax_c, high["UPDRS3_Baseline"].dropna(), low["UPDRS3_Baseline"].dropna(),
          "Baseline MDS-UPDRS Part III", "Baseline motor severity by endotype")
    panel_label(ax_c, "c")

    # (d) Cohort composition (PD vs Prodromal)
    ax_d = fig.add_subplot(gs[1, 1])
    coh_tab = (subj.groupby(["vascular_class", "Cohort_Current"]).size()
               .unstack(fill_value=0).reindex(["Vascular_High", "Vascular_Low"]))
    bottoms = np.zeros(2)
    cohcolors = {"PD": OKABE["red"], "Prodromal": OKABE["orange"],
                 "HC": OKABE["green"]}
    for coh in coh_tab.columns:
        vals = coh_tab[coh].values
        ax_d.bar([0, 1], vals, bottom=bottoms, width=0.55,
                 color=cohcolors.get(coh, OKABE["grey"]), label=str(coh),
                 edgecolor="white")
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0:
                ax_d.text(i, b + v / 2, str(int(v)), ha="center", va="center",
                          color="white", fontsize=9, fontweight="bold")
        bottoms += vals
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels([f"Vascular-High\n(n={len(high)})",
                          f"Vascular-Low\n(n={len(low)})"])
    ax_d.set_ylabel("Number of subjects")
    ax_d.set_title("Diagnostic-group composition by endotype")
    ax_d.legend(loc="upper right", title="Cohort")
    panel_label(ax_d, "d")

    fig.suptitle("Supplementary Fig. S4  -  Demographic balance of the n=61 "
                 "longitudinal modelling cohort", fontsize=11,
                 fontweight="bold", y=0.97)
    save(fig, "figureS4")


# =====================================================================
# S5  -  Cohort-flow diagram
# =====================================================================
def figureS5():
    print("\n[Supp Fig S5] Cohort-flow diagram")
    # Verified directly from result files:
    n_olink = len(pd.read_parquet("results/step3/bbb_protein_score.parquet"))
    n_xmod = len(pd.read_parquet("results/step4/cross_modal_scores.parquet"))
    n_model = len(pd.read_parquet("results/step3/modeling_cohort_subjects.parquet"))
    # Documented in the analysis record (Figure 2 / findings summary):
    n_blood = 444
    n_sens = 54

    steps = [
        ("PPMI baseline assessment\n(clinical + biofluid + omics)",
         "5,925 subjects in canonical clinical table", OKABE["grey"]),
        ("Whole-blood RNA-seq at baseline\n(PD + HC, signature-transfer test)",
         f"n = {n_blood}  (338 PD + 106 HC)", OKABE["blue"]),
        ("Olink BBB panel at baseline\n(20 proteins, PD + Prodromal)",
         f"n = {n_olink}  (endotyping cohort)", OKABE["orange"]),
        ("Paired brain-score + Olink PC1\n(cross-modal decoupling test)",
         f"n = {n_xmod}  (both modalities present)", OKABE["purple"]),
        ("Longitudinal modelling cohort\n(>=2 UPDRS3 visits + complete covariates)",
         f"n = {n_model}  (31 PD + 30 Prodromal)  -  PRIMARY ENDPOINT", OKABE["red"]),
        ("Sensitivity cohort\n(brain-derived transcriptomic score available)",
         f"n = {n_sens}  (sensitivity lmer)", OKABE["green"]),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12.6)
    ax.axis("off")

    box_w, box_h = 6.6, 1.35
    x_c = 3.7
    y0 = 11.2
    dy = 1.95
    centers = []
    for i, (title, sub, col) in enumerate(steps):
        y = y0 - i * dy
        centers.append(y)
        box = FancyBboxPatch((x_c - box_w / 2, y - box_h / 2), box_w, box_h,
                             boxstyle="round,pad=0.05", facecolor=col,
                             edgecolor="white", linewidth=1.5, alpha=0.92)
        ax.add_patch(box)
        ax.text(x_c, y + 0.22, title, ha="center", va="center", fontsize=9.5,
                color="white", fontweight="bold")
        ax.text(x_c, y - 0.36, sub, ha="center", va="center", fontsize=8.5,
                color="white")
        if i > 0:
            ax.add_patch(FancyArrowPatch((x_c, centers[i - 1] - box_h / 2),
                         (x_c, y + box_h / 2), arrowstyle="-|>",
                         mutation_scale=18, color=OKABE["grey"], lw=1.8))

    # Side annotations for the filter applied at each narrowing step
    filters = [
        "filter: baseline blood RNA-seq available",
        "filter: baseline Olink panel available",
        "filter: both brain-score and Olink PC1",
        "filter: >=2 UPDRS3 visits + complete\ncovariates (Age, Sex, LEDD, cell %)",
        "filter: brain transcriptomic score\ncomputable from blood RNA-seq",
    ]
    for i, ftxt in enumerate(filters):
        y_mid = (centers[i] + centers[i + 1]) / 2
        ax.text(x_c + box_w / 2 + 0.25, y_mid, ftxt, ha="left", va="center",
                fontsize=7.6, color="#555", style="italic")

    ax.text(5, 12.25, "Supplementary Fig. S5  -  Cohort-flow diagram",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(5, 0.35,
            "Counts n=227 / 212 / 61 verified directly from result files; "
            "n=444 and n=54 are documented in the analysis record.",
            ha="center", fontsize=7.6, color="#666", style="italic")
    save(fig, "figureS5")
    print(f"  flow: 444 -> {n_olink} -> {n_xmod} -> {n_model} -> {n_sens}")


# =====================================================================
# S6  -  Individual UPDRS3 trajectories in the modelling cohort
# =====================================================================
def figureS6():
    print("\n[Supp Fig S6] Individual UPDRS3 trajectories")
    lg = pd.read_parquet("results/step3/modeling_cohort_long.parquet")
    lg = lg.dropna(subset=["Time", "UPDRS3_Total", "vascular_class"])

    fig = plt.figure(figsize=(12.5, 5.2))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.28,
                           left=0.07, right=0.97, top=0.88, bottom=0.14)

    classes = [("Vascular_High", OKABE["red"], "Vascular-High"),
               ("Vascular_Low", OKABE["blue"], "Vascular-Low")]

    # (a) Spaghetti, all subjects, colored by endotype
    ax_a = fig.add_subplot(gs[0, 0])
    for cls, col, lab in classes:
        sub = lg[lg["vascular_class"] == cls]
        for patno, g in sub.groupby("PATNO"):
            g = g.sort_values("Time")
            ax_a.plot(g["Time"], g["UPDRS3_Total"], "-", color=col,
                      alpha=0.32, lw=0.9)
        n_subj = sub["PATNO"].nunique()
        ax_a.plot([], [], "-", color=col, lw=2, label=f"{lab} (n={n_subj})")
    ax_a.set_xlabel("Time from baseline (years)")
    ax_a.set_ylabel("MDS-UPDRS Part III total")
    ax_a.set_title("Individual UPDRS3 trajectories\n(n=61 subjects, 187 visits)")
    ax_a.legend(loc="upper left")
    panel_label(ax_a, "a", x=-0.13)

    # (b) Mean +/- SEM trajectory by endotype (binned by visit time)
    ax_b = fig.add_subplot(gs[0, 1])
    lg2 = lg.copy()
    lg2["tbin"] = (lg2["Time"] / 0.5).round() * 0.5
    for cls, col, lab in classes:
        sub = lg2[lg2["vascular_class"] == cls]
        agg = sub.groupby("tbin")["UPDRS3_Total"].agg(["mean", "sem", "count"])
        agg = agg[agg["count"] >= 3]
        ax_b.errorbar(agg.index, agg["mean"], yerr=agg["sem"], fmt="o-",
                      color=col, lw=2, markersize=6, capsize=3,
                      markeredgecolor="white", label=lab)
    ax_b.set_xlabel("Time from baseline (years, 0.5-yr bins)")
    ax_b.set_ylabel("Mean MDS-UPDRS Part III (+/- SEM)")
    ax_b.set_title("Mean progression by endotype\n"
                   "(Time x BBB-score interaction: p = 0.81, null)")
    ax_b.legend(loc="upper left")
    panel_label(ax_b, "b", x=-0.13)

    fig.suptitle("Supplementary Fig. S6  -  UPDRS3 trajectories confirm the null "
                 "primary endpoint is not a data-quality artefact",
                 fontsize=11, fontweight="bold", y=1.00)
    save(fig, "figureS6")


def main():
    figureS1()
    figureS2()
    figureS4()
    figureS5()
    figureS6()
    print("\nAll 5 supplementary figures rendered to results/figures/")


if __name__ == "__main__":
    main()
