"""Step 4c  -  Forest plot of the longitudinal Qalb models.

Reads results/step6/qalb_lmer_coefs.tsv (from 94_qalb_lmer.R).
Writes results/step6/qalb_longitudinal.png
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/step6")
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "red": "#D55E00",
         "grey": "#7A7A7A"}

res = pd.read_csv(OUT / "qalb_lmer_coefs.tsv", sep="\t")

labels = [
    "Primary: Qalb -> UPDRS3 slope\n(full cohort)",
    "Positive control: plasma NfL\n-> UPDRS3 slope",
    "Stage-stratified: Qalb\n(advanced subset)",
]
colors = [OKABE["red"], OKABE["blue"], OKABE["orange"]]

fig, ax = plt.subplots(figsize=(9.5, 4.2))
y = list(range(len(res)))[::-1]
for yi, (_, r), col in zip(y, res.iterrows(), colors):
    ax.errorbar(r["estimate"], yi,
                xerr=[[r["estimate"] - r["ci_lo"]], [r["ci_hi"] - r["estimate"]]],
                fmt="s", color=col, ecolor=col, elinewidth=1.8, capsize=4,
                markersize=10, markeredgecolor="white", markeredgewidth=0.7)
    ax.text(1.02, yi, f"est={r['estimate']:+.3f}   p={r['p']:.3f}   "
            f"n={r['n_subj']}",
            transform=ax.get_yaxis_transform(), va="center", ha="left",
            fontsize=9, family="monospace")

ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.6)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_ylim(-0.6, len(res) - 0.4)
ax.set_xlim(-1.6, 1.6)
ax.set_xlabel("Time x predictor interaction (UPDRS3 points/year per +1 SD)")
ax.set_title("Step 4 - Longitudinal models: Qalb does not modify UPDRS3 slope\n"
             "Positive control (NfL) also null -> longitudinal test is "
             "inconclusive (underpowered), not a clean null",
             fontsize=10.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "qalb_longitudinal.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"-> {OUT}/qalb_longitudinal.png")
