"""Step 5 Task 54  -  Combined HTML summary of primary + all secondary analyses.

Reads:
  results/step3/lmer_primary_coefs.tsv
  results/step3/secondary_analyses_summary.tsv
  results/step3/subgroup_analyses_summary.tsv

Outputs:
  results/step3/step5_full_summary.html
"""
from __future__ import annotations
import datetime
from pathlib import Path
import pandas as pd

S3 = Path("results/step3")


def main() -> None:
    today = datetime.date.today().isoformat()

    # Primary
    pri = pd.read_csv(S3 / "lmer_primary_coefs.tsv", sep="\t", index_col=0)
    primary_row = pri.loc["Time:BBB_score_z"]

    # Secondary endpoints
    sec = pd.read_csv(S3 / "secondary_analyses_summary.tsv", sep="\t")

    # Subgroups
    sub = pd.read_csv(S3 / "subgroup_analyses_summary.tsv", sep="\t")

    # Combined view
    combined = pd.concat([
        pd.DataFrame([{
            "tier": "PRIMARY",
            "endpoint": "UPDRS3 slope (PD+Prodromal)",
            "n_subj": 61,
            "estimate": primary_row["Estimate"],
            "se": primary_row["Std. Error"],
            "p": primary_row["Pr(>|t|)"],
        }]),
        sec.assign(tier="SECONDARY").rename(columns={"endpoint": "endpoint"}),
        sub.assign(tier="SUBGROUP").rename(columns={"label": "endpoint"}),
    ], ignore_index=True)

    # Format for display
    disp = combined[["tier", "endpoint", "n_subj", "estimate", "se", "p"]].copy()
    disp["estimate"] = disp["estimate"].round(3)
    disp["se"] = disp["se"].round(3)
    disp["p"] = disp["p"].round(4)

    # Verdict
    primary_p = primary_row["Pr(>|t|)"]
    sec_min_p = sec["p"].min() if len(sec) else 1.0
    sub_min_p = sub["p"].min() if len(sub) else 1.0
    sub_min_padj = sub["bh_padj"].min() if len(sub) and "bh_padj" in sub.columns else 1.0
    any_corrected_signal = sub_min_padj < 0.10

    if primary_p < 0.05:
        verdict = ('<p class="verdict ok">PRIMARY ENDPOINT POSITIVE. Manuscript proceeds '
                   'with positive finding as headline.</p>')
    elif any_corrected_signal:
        verdict = (f'<p class="verdict modest">PRIMARY NULL but at least one subgroup '
                   f'survives BH correction (min p_BH = {sub_min_padj:.3f}). Manuscript '
                   'frames as "a hypothesis-generating subgroup finding embedded in a '
                   'pre-registered null primary."</p>')
    else:
        verdict = (f'<p class="verdict warn">PRIMARY NULL across all pre-registered tiers. '
                   f'Min uncorrected p in any analysis = {min(primary_p, sec_min_p, sub_min_p):.3f}; '
                   f'min BH-adjusted p in subgroups = {sub_min_padj:.3f}. '
                   '<b>Recommended manuscript framing: pre-registered multi-modal failed '
                   'replication + decoupling story.</b></p>')

    table_html = disp.to_html(index=False)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Step 5 full summary - {today}</title>
<style>
body{{font-family:system-ui;max-width:1000px;margin:2em auto;padding:0 1em;color:#222;}}
table{{border-collapse:collapse;margin:1em 0;font-size:14px;}}
th,td{{border:1px solid #ccc;padding:5px 10px;text-align:right;}}
th{{background:#f5f5f5;text-align:left;}}
tr:nth-child(even){{background:#fafafa;}}
.verdict{{padding:1em;border-left:4px solid;font-weight:bold;}}
.verdict.ok{{background:#e6ffec;border-color:#22863a;}}
.verdict.modest{{background:#fff7e6;border-color:#d4a72c;}}
.verdict.warn{{background:#fde2e2;border-color:#c92a2a;}}
.note{{font-size:13px;color:#555;margin:1em 0;}}
</style></head><body>
<h1>PD-BBB Step 5  -  full pre-registered analysis summary</h1>
<p><b>Date:</b> {today}<br>
<b>Cohort:</b> 61 PD/Prodromal subjects with Olink BBB score + ≥3 UPDRS3 visits in 36mo
+ complete pre-registered covariates (Age, Sex, Baseline_severity, LEDD, Neutrophil%, Monocyte%).</p>

<h2>All pre-registered tests (Time × BBB_score_z interaction)</h2>
{table_html}

<p class="note">All models: lmer(outcome ~ Time * BBB_score_z + Age + Sex + Baseline + LEDD
+ Neutrophil% + Monocyte% + (1+Time|PATNO), REML=TRUE)  -  pre-registered formula
unchanged across primary, secondary, and subgroup tiers.</p>

<h2>Verdict</h2>
{verdict}

<h2>Implications for manuscript</h2>
<ul>
<li><b>Brain-side biology stands.</b> Step 2 multi-dataset DESeq2 (237 genes
padj<0.05) + bootstrap-stable WGCNA module 8 (118 genes, hub AHSA1+HSP90AB1)
replicate the heat-shock vascular-stress axis across 3 independent statistical methods.</li>
<li><b>Brain-blood transfer is weak.</b> v1 reconstruction AUC 0.43, v2 multi-dataset
AUC 0.54  -  meaningful improvement, still near random.</li>
<li><b>Olink BBB panel captures structure but not severity.</b> PC1 explains 33% variance,
correlates with endothelial-activation mean (r=0.85), but cluster separation Cohen's d
only 0.27. Score is age-confounded (Vascular-High mean 65 vs 60, p=0.0003).</li>
<li><b>Olink score does not validate against independent BBB markers.</b>
Plasma NfL Simoa null, CSF Albumin null, Plasma Albumin null, only Plasma GFAP weak (rho=0.16, p=0.025).</li>
<li><b>BBB score does not predict UPDRS3, MoCA, or DaT-SCAN slope</b> in any pre-registered
analysis with the locked Time × BBB_score covariate-adjusted model.</li>
</ul>
</body></html>"""

    out = S3 / "step5_full_summary.html"
    out.write_text(html, encoding="utf-8")
    print(f"→ Wrote {out}")
    print("\nCombined results:")
    print(disp.to_string(index=False))


if __name__ == "__main__":
    main()
