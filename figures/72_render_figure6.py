"""Figure 6  -  Mechanistic depth + therapeutic context.

4-panel figure for the npj PD manuscript:
  (a) GO Biological Process enrichment of the 118-gene BBB module
  (b) Reactome pathway enrichment
  (c) Multi-database source breakdown (heat-shock dominance robust across
      knowledge sources)
  (d) Druggable target landscape  -  HSP90/HSP70/chaperone genes in the
      module that are targets of clinical-stage drugs

Inputs:
  results/step4/gprofiler_full_results.tsv   (gprofiler API output)
  results/step4/v2_module_with_symbols.tsv   (118-gene module + symbols)

Outputs:
  results/figures/figure6.png
  results/figures/figure6.pdf

Druggable target annotations are hand-curated from established pharmacology
(HSP90 inhibitors are textbook clinical-stage; mappings are conservative
and reference well-known compounds).
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

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Style  -  matches the rest of Figures 1-5
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

# Druggable target annotations  -  hand-curated from established pharmacology.
# Each entry: (gene_symbol, drug_class, representative compounds).
# Conservative  -  only includes well-published clinical or preclinical agents.
DRUG_TARGETS = [
    ("HSP90AA1", "HSP90 inhibitor", "17-AAG (tanespimycin), ganetespib, luminespib, retaspimycin"),
    ("HSP90AB1", "HSP90 inhibitor", "17-AAG, ganetespib, NVP-AUY922"),
    ("HSPA8",    "HSP70 inhibitor", "VER-155008, JG-98, MKT-077"),
    ("HSPA1A",   "HSP70 inhibitor", "KNK437, VER-155008"),
    ("HSPH1",    "HSPH1 (HSP110) modulator", "preclinical small-molecule binders"),
    ("HSPB1",    "small-HSP / αB-crystallin axis", "preclinical neuroprotection candidates"),
    ("FKBP4",    "Immunophilin / FKBP", "tacrolimus (FK506), FK1706 (preclinical CNS analog)"),
    ("FKBP5",    "Immunophilin / FKBP", "SAFit1, SAFit2 (preclinical CNS-penetrant)"),
    ("BAG3",     "Co-chaperone modulator", "preclinical"),
    ("DNAJA1",   "DNAJ co-chaperone", "preclinical"),
    ("STIP1",    "HSP90 co-chaperone (HOP)", "preclinical disruptors"),
    ("CDC37",    "HSP90 co-chaperone", "celastrol, withaferin A (natural products)"),
]


def panel_label(ax, label, x=-0.15, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left")


def _truncate(s, max_len=55):
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def lollipop(ax, df, name_col="name", value_col="neglog10p", n=12,
             color=OKABE["blue"], label_max_len=55):
    """Top-N lollipop chart; expects df sorted descending by value_col."""
    sub = df.head(n).iloc[::-1].copy()  # bottom = highest score
    y = np.arange(len(sub))
    ax.hlines(y, 0, sub[value_col], color=color, linewidth=1.6, alpha=0.8)
    ax.scatter(sub[value_col], y, color=color, s=55, edgecolor="white",
               linewidths=0.5, zorder=3)
    labels = [_truncate(s, label_max_len) for s in sub[name_col]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, sub[value_col].max() * 1.10)


def main():
    # Load gprofiler results
    enr = pd.read_csv("results/step4/gprofiler_full_results.tsv", sep="\t")
    enr["neglog10p"] = -np.log10(enr["p_value"].clip(lower=1e-300))
    print(f"Loaded {len(enr)} significant terms; sources: "
          f"{dict(enr['source'].value_counts())}")

    # Map source codes to readable names
    SOURCE_NAMES = {
        "GO:BP": "GO Biological Process",
        "GO:MF": "GO Molecular Function",
        "GO:CC": "GO Cellular Component",
        "REAC": "Reactome",
        "KEGG": "KEGG",
        "WP": "WikiPathways",
    }

    # Load module + symbols for druggable target intersection
    mod = pd.read_csv("results/step4/v2_module_with_symbols.tsv", sep="\t")
    module_symbols = set(s for s in mod["symbol"] if isinstance(s, str) and s)

    # ---- Figure ----
    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.65, wspace=0.55,
                            left=0.21, right=0.97, top=0.94, bottom=0.07)

    # ============================================================
    # Panel (a)  -  GO Biological Process, top 12
    # ============================================================
    ax_a = fig.add_subplot(gs[0, 0])
    go_bp = enr[enr["source"] == "GO:BP"].sort_values("neglog10p",
                                                        ascending=False).copy()
    lollipop(ax_a, go_bp, color=OKABE["blue"], n=12)
    ax_a.set_xlabel("$-$log$_{10}$ adjusted p-value")
    ax_a.set_title(f"GO Biological Process\n(top 12 of {len(go_bp)} significant terms)")
    panel_label(ax_a, "a")

    # ============================================================
    # Panel (b)  -  Reactome, top 12
    # ============================================================
    ax_b = fig.add_subplot(gs[0, 1])
    reac = enr[enr["source"] == "REAC"].sort_values("neglog10p",
                                                      ascending=False).copy()
    lollipop(ax_b, reac, color=OKABE["red"], n=12)
    ax_b.set_xlabel("$-$log$_{10}$ adjusted p-value")
    ax_b.set_title(f"Reactome pathways\n(top 12 of {len(reac)} significant terms)")
    panel_label(ax_b, "b")

    # ============================================================
    # Panel (c)  -  Multi-database source breakdown
    # Top 3 terms per source, color-coded; demonstrates that the heat-shock /
    # proteostasis signal is robust across multiple curated databases.
    # ============================================================
    ax_c = fig.add_subplot(gs[1, 0])
    rows = []
    src_colors = {
        "GO:BP": OKABE["blue"],
        "GO:MF": OKABE["green"],
        "GO:CC": OKABE["yellow"],
        "REAC": OKABE["red"],
        "KEGG": OKABE["purple"],
        "WP": OKABE["skyblue"],
    }
    src_order = ["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG", "WP"]
    for src in src_order:
        s = enr[enr["source"] == src].sort_values("neglog10p",
                                                    ascending=False).head(3)
        for _, r in s.iterrows():
            rows.append({"source": src, "name": r["name"],
                          "neglog10p": r["neglog10p"]})
    sub_c = pd.DataFrame(rows)
    if len(sub_c) > 0:
        sub_c = sub_c.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(sub_c))
        for i, r in sub_c.iterrows():
            color = src_colors[r["source"]]
            ax_c.hlines(i, 0, r["neglog10p"], color=color, linewidth=1.6,
                        alpha=0.8)
            ax_c.scatter([r["neglog10p"]], [i], color=color, s=55,
                          edgecolor="white", linewidths=0.5, zorder=3)
        ax_c.set_yticks(y)
        ax_c.set_yticklabels(sub_c["name"], fontsize=7)
        ax_c.set_xlabel("$-$log$_{10}$ adjusted p-value")
        ax_c.set_xlim(0, sub_c["neglog10p"].max() * 1.10)
    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=src_colors[s],
                      label=SOURCE_NAMES.get(s, s)) for s in src_order
                if s in enr["source"].values]
    ax_c.legend(handles=handles, loc="lower right", fontsize=7,
                title="Source database", title_fontsize=7)
    ax_c.set_title("Top 3 enriched terms per source database\n"
                    " -  heat-shock / proteostasis signal is multi-database-robust")
    panel_label(ax_c, "c")

    # ============================================================
    # Panel (d)  -  Druggable target landscape (horizontal bar chart with
    # representative drugs annotated to the right of each bar)
    # ============================================================
    ax_d = fig.add_subplot(gs[1, 1])
    # Filter DRUG_TARGETS to genes actually in the module + load hub_score
    hub = pd.read_csv("results/step4/hub_genes_with_symbols.tsv", sep="\t")
    hub_score_map = dict(zip(hub["symbol"].astype(str), hub["hub_score"]))
    drug_rows = []
    for g, cls, drugs in DRUG_TARGETS:
        if g in module_symbols:
            drug_rows.append({"gene": g, "class": cls, "drugs": drugs,
                              "hub_score": hub_score_map.get(g, 0.0)})
    print(f"Druggable targets in module: {len(drug_rows)}/{len(DRUG_TARGETS)}")
    drug_df = pd.DataFrame(drug_rows).sort_values("hub_score",
                                                    ascending=True)

    class_colors = {
        "HSP90 inhibitor": OKABE["red"],
        "HSP70 inhibitor": OKABE["orange"],
        "HSPH1 (HSP110) modulator": OKABE["yellow"],
        "small-HSP / αB-crystallin axis": OKABE["green"],
        "Immunophilin / FKBP": OKABE["purple"],
        "Co-chaperone modulator": OKABE["blue"],
        "DNAJ co-chaperone": OKABE["skyblue"],
        "HSP90 co-chaperone (HOP)": OKABE["red"],
        "HSP90 co-chaperone": OKABE["red"],
    }
    y = np.arange(len(drug_df))
    bar_colors = [class_colors.get(c, OKABE["grey"]) for c in drug_df["class"]]
    ax_d.barh(y, drug_df["hub_score"], color=bar_colors, alpha=0.75,
              edgecolor="white", linewidth=0.5)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(drug_df["gene"], fontsize=8.5, fontweight="bold")
    # Annotate each bar with class + representative drugs to the right
    xmax = drug_df["hub_score"].max() if len(drug_df) else 1.0
    for yi, (_, r) in zip(y, drug_df.iterrows()):
        ax_d.text(xmax * 1.05, yi,
                   f"{r['class']}\n  {_truncate(r['drugs'], 70)}",
                   va="center", ha="left", fontsize=6.5)
    ax_d.set_xlim(0, xmax * 2.5)
    ax_d.set_xlabel("Hub score (MM × |GS|)")
    ax_d.set_title(f"Druggable targets in the BBB module "
                    f"({len(drug_df)} genes)")
    panel_label(ax_d, "d")

    fig.savefig(OUT / "figure6.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figure6.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ {OUT}/figure6.png + .pdf")


if __name__ == "__main__":
    main()
