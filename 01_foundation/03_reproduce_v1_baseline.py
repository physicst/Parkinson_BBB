"""Step 1 v1-baseline final report.

Inputs (from Tasks 17-20):
  - results/v1_baseline/v1_signature_genes.tsv
  - results/v1_baseline/v1_auc_results.tsv
  - results/v1_baseline/v1_roc_curve.tsv

Output:
  - results/v1_baseline/<today>-v1-reproduction.html (standalone, embedded ROC PNG as data URI)
"""
from __future__ import annotations
import base64
import datetime
import io
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from lib.config import load_config

TARGET_AUC = 0.811
CONCORDANCE_TOL = 0.05


def render_roc_png(roc: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(roc["fpr"], roc["tpr"], linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("v1 BBB-score-only ROC (5-fold CV pooled)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    cfg = load_config()
    v1_dir = Path(cfg["results"]["v1_baseline_dir"])
    today = datetime.date.today().isoformat()

    sig = pd.read_csv(v1_dir / "v1_signature_genes.tsv", sep="\t")
    auc = pd.read_csv(v1_dir / "v1_auc_results.tsv", sep="\t")
    roc = pd.read_csv(v1_dir / "v1_roc_curve.tsv", sep="\t")

    bbb_auc = float(auc.loc[auc["model"] == "BBB_score_only", "auc_overall_pooled"].iloc[0])
    delta = bbb_auc - TARGET_AUC
    reproduced = abs(delta) <= CONCORDANCE_TOL

    if reproduced:
        decision_html = (
            f'<p class="decision ok">REPRODUCED: BBB-score-only AUC = {bbb_auc:.3f} '
            f'(target {TARGET_AUC}, delta = {delta:+.3f})</p>'
        )
    else:
        decision_html = (
            f'<p class="decision warn">NOT REPRODUCED: BBB-score-only AUC = {bbb_auc:.3f} '
            f'(target {TARGET_AUC}, delta = {delta:+.3f})</p>'
            '<p>Possible explanations:</p>'
            '<ul>'
            '<li>v1 used a different gene list than what we found/reconstructed (we used GSE178265 endothelial pseudobulk Welch t-test PD vs normal; v1 may have pooled LBD or used different DE method).</li>'
            '<li>v1 used a non-CV evaluation (train-on-all, test-on-all) which inflates AUC vs. proper 5-fold CV.</li>'
            '<li>v1 included additional covariates or feature engineering.</li>'
            '<li>v1 used a different sample subset (e.g., Phase 1 only, different baseline cohort definition).</li>'
            '<li>v1 used HGNC symbols directly while PPMI matrix uses ENSG IDs; symbol-to-ENSG mapping coverage may have differed.</li>'
            '</ul>'
            '<p>Document explanation hypothesis in master plan section 7 decision log. '
            'For Step 2 onward, the new (re-derived, multi-dataset, deconvolution-adjusted) signature replaces v1 anyway, so this gap is informative but not blocking.</p>'
        )

    roc_uri = render_roc_png(roc)
    auc_table = auc.to_html(index=False, float_format="%.3f")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>PD-BBB v1 Reproduction - {today}</title>
<style>
body{{font-family:system-ui;max-width:900px;margin:2em auto;padding:0 1em;color:#222;}}
table{{border-collapse:collapse;margin:1em 0;}}
th,td{{border:1px solid #ccc;padding:6px 12px;text-align:right;}}
th{{background:#f5f5f5;text-align:left;}}
.decision{{padding:1em;border-left:4px solid;font-weight:bold;}}
.decision.ok{{background:#e6ffec;border-color:#22863a;}}
.decision.warn{{background:#fff7e6;border-color:#d4a72c;}}
img{{max-width:100%;}}
</style></head><body>
<h1>PD-BBB Step 1 - v1 Baseline Reproduction</h1>
<p><b>Date:</b> {today}<br>
<b>Goal:</b> Reproduce the v1 result from <code>D:\\Parkinson_file\\Manuscript.md</code>
(1,015-gene signature; logistic regression PD vs HC AUC = 0.811; Age+Sex baseline AUC = 0.510).
Acceptable concordance: AUC within +/-{CONCORDANCE_TOL}.</p>

<h2>Signature provenance</h2>
<ul>
<li>Size: {len(sig)} genes</li>
<li>Source: {sig['source'].iloc[0]}</li>
</ul>

<h2>Reproduction results</h2>
{auc_table}

<h2>ROC curve (BBB-score-only, 5-fold CV pooled)</h2>
<img src="{roc_uri}" alt="ROC curve">

<h2>Decision</h2>
{decision_html}

</body></html>"""

    out_path = v1_dir / f"{today}-v1-reproduction.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"-> Wrote {out_path}")
    print(f"  Reproduction status: {'REPRODUCED' if reproduced else 'NOT REPRODUCED'} "
          f"(AUC {bbb_auc:.3f} vs target {TARGET_AUC}, delta {delta:+.3f})")


if __name__ == "__main__":
    main()
