"""Supplementary Figure S7  -  CSF Qalb analysis (post-hoc, exploratory).

Composes the three Qalb panels (QC, cross-sectional, longitudinal) produced
by the Step 1-4 scripts into a single stacked supplementary figure.

Inputs:
  results/step6/qalb_qc.png
  results/step6/qalb_crosssectional.png
  results/step6/qalb_longitudinal.png

Output:
  results/figures/figureS7.png + .pdf
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

S6 = Path("results/step6")
OUT = Path("results/figures")

panels = [
    (S6 / "qalb_qc.png", "a", "Step 1  -  Qalb construction and QC "
     "(n=494; QC gate: Qalb rises with age)"),
    (S6 / "qalb_crosssectional.png", "b", "Step 2  -  Cross-sectional "
     "characterization (no significant disease/severity gradient)"),
    (S6 / "qalb_longitudinal.png", "c", "Step 4  -  Longitudinal models "
     "(Qalb does not modify UPDRS3 slope; positive control also null)"),
]

# height of each row proportional to its image aspect ratio
imgs = [mpimg.imread(str(p)) for p, _, _ in panels]
aspects = [im.shape[0] / im.shape[1] for im in imgs]
fig_w = 11.0
heights = [fig_w * a for a in aspects]
fig = plt.figure(figsize=(fig_w, sum(heights) + 1.0))
gs = fig.add_gridspec(3, 1, height_ratios=heights, hspace=0.12)

for i, (img, (_, lab, cap)) in enumerate(zip(imgs, panels)):
    ax = fig.add_subplot(gs[i])
    ax.imshow(img, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-0.01, 1.02, lab, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom", ha="right")
    ax.set_title(cap, fontsize=9, loc="left", pad=4)

fig.suptitle("Supplementary Fig. S7  -  Cerebrospinal-fluid barrier "
             "permeability (Qalb): post-hoc exploratory analysis",
             fontsize=11, fontweight="bold", y=0.997)
fig.savefig(OUT / "figureS7.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "figureS7.pdf", bbox_inches="tight")
plt.close(fig)
print(f"-> {OUT}/figureS7.png + .pdf")
