"""Step 4 Day 5  -  Integration summary across the 5 strengthening tasks.

Ties every Step 4 result to the corresponding manuscript Figure / Table
claim and produces a single source-of-truth HTML status page for the
manuscript-writing step to refer back to.

Outputs:
  results/step4/step4_integration_summary.html
"""
from __future__ import annotations
import datetime
from pathlib import Path
import pandas as pd

S4 = Path("results/step4")
today = datetime.date.today().isoformat()

# ---- Pull headline numbers from the per-task outputs ----
xmod = pd.read_csv(S4 / "cross_modal_decoupling_summary.tsv", sep="\t").iloc[0]
sens = pd.read_csv(S4 / "lmer_sensitivity_v2_coefs.tsv", sep="\t", index_col=0)
sens_row = sens.loc["Time:v2_score_z"]
sub = pd.read_csv(S4 / "subgroup_analyses_fixed.tsv", sep="\t")
enr = pd.read_csv(S4 / "v2_module_enrichment.tsv", sep="\t")

# Olink primary (Step 5)
pri = pd.read_csv("results/step3/lmer_primary_coefs.tsv", sep="\t", index_col=0)
pri_row = pri.loc["Time:BBB_score_z"]


def fmt_p(p):
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


# Build per-claim status table
claims = [
    {
        "manuscript_target": "Figure 1 (headline biology)",
        "claim": "Brain heat-shock vascular-stress axis is real and cross-method replicated",
        "evidence_files": "results/step2/* (DESeq2 237 genes, WGCNA module 8, hub list)",
        "supporting_step4": "Task 4  -  HSP90 family 90× enriched (p_BH=1.8e-10)",
        "status": "STRONG",
    },
    {
        "manuscript_target": "Figure 2 (transfer)",
        "claim": "Brain signature does not transfer to peripheral blood",
        "evidence_files": "results/step2/v1_v2_comparison/* (AUC 0.43 / 0.54)",
        "supporting_step4": "Task 2  -  v2 score in lmer also null (p=0.68)",
        "status": "ROBUST",
    },
    {
        "manuscript_target": "Figure 3 (plasma landscape)",
        "claim": "Plasma BBB protein PC1 captures structure but is age-confounded",
        "evidence_files": "results/step3/bbb_protein_score.parquet (PC1 33% var; "
                          "age d=0.27)",
        "supporting_step4": " - ",
        "status": "STRONG",
    },
    {
        "manuscript_target": "Figure 4 (decoupling)",
        "claim": "Brain v2 transcriptomic and plasma Olink signatures are decoupled",
        "evidence_files": "results/step4/cross_modal_decoupling.html",
        "supporting_step4": (
            f"Task 1  -  Spearman ρ={xmod['spearman_rho']:+.3f} "
            f"(p={fmt_p(xmod['spearman_p'])}); "
            f"Cohen's κ={xmod['cohens_kappa']:+.3f} (n={int(xmod['n_patients'])})"
        ),
        "status": "EMPIRICALLY SUPPORTED",
    },
    {
        "manuscript_target": "Figure 5 (forest plot)",
        "claim": "Pre-registered primary + 2 secondaries + 4 subgroups all null "
                 "(except APOE-non-e4 MoCA hint)",
        "evidence_files": "results/step3/lmer_primary_coefs.tsv + "
                          "secondary_analyses_summary.tsv + "
                          "results/step4/subgroup_analyses_fixed.tsv",
        "supporting_step4": (
            f"Task 3  -  corrected genotype calls; "
            f"LRRK2 carriers in cohort = 0 (drop from forest); "
            f"GBA carriers = 0 (drop from forest); "
            f"APOE-non-e4 MoCA: est={sub.loc[sub['label']=='APOE-non-e4 (MoCA)','estimate'].iloc[0]:+.3f}, "
            f"p_BH={sub.loc[sub['label']=='APOE-non-e4 (MoCA)','bh_padj'].iloc[0]:.3f}"
        ),
        "status": "ROBUST",
    },
    {
        "manuscript_target": "Table 2 (lit comparison)",
        "claim": "Our null effect size is informative, not under-powered",
        "evidence_files": "results/step4/lit_comparison_table2.tsv",
        "supporting_step4": (
            "Task 5  -  9-study scaffold; 10 [VERIFY] cells need confirmation "
            "from source papers during manuscript writing"
        ),
        "status": "SCAFFOLD READY",
    },
]
status_df = pd.DataFrame(claims)

# Render
status_html = status_df.to_html(index=False, escape=False)

# Headline numbers panel
headline = f"""
<table>
<tr><th>Pre-reg primary endpoint (Olink PC1)</th>
    <td>Time:BBB_score_z = {pri_row['Estimate']:+.3f} ± {pri_row['Std. Error']:.3f},
        p = {fmt_p(pri_row['Pr(>|t|)'])} (n=61)</td></tr>
<tr><th>Sensitivity (v2 brain transcriptomic)</th>
    <td>Time:v2_score_z = {sens_row['Estimate']:+.3f} ± {sens_row['Std. Error']:.3f},
        p = {fmt_p(sens_row['Pr(>|t|)'])} (n=54)</td></tr>
<tr><th>Cross-modal Spearman ρ</th>
    <td>{xmod['spearman_rho']:+.3f}, p = {fmt_p(xmod['spearman_p'])}, n = {int(xmod['n_patients'])}</td></tr>
<tr><th>Cross-modal Cohen's κ</th>
    <td>{xmod['cohens_kappa']:+.3f} (κ &lt; 0.20 → decoupled)</td></tr>
<tr><th>v2 module HSP90 enrichment</th>
    <td>6/12 overlap, {enr.loc[enr['reference_set']=='HSP90_family','fold_enrichment'].iloc[0]:.1f}×,
        p_BH = {enr.loc[enr['reference_set']=='HSP90_family','bh_padj'].iloc[0]:.1e}</td></tr>
<tr><th>v2 module HSP70 enrichment</th>
    <td>6/19 overlap, {enr.loc[enr['reference_set']=='HSP70_family','fold_enrichment'].iloc[0]:.1f}×,
        p_BH = {enr.loc[enr['reference_set']=='HSP70_family','bh_padj'].iloc[0]:.1e}</td></tr>
</table>
"""

html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Step 4 integration summary  -  {today}</title>
<style>
body{{font-family:system-ui;max-width:1300px;margin:2em auto;padding:0 1em;color:#222;}}
table{{border-collapse:collapse;margin:1em 0;font-size:13px;}}
th,td{{border:1px solid #ccc;padding:6px 12px;vertical-align:top;text-align:left;}}
th{{background:#f5f5f5;}}
tr:nth-child(even){{background:#fafafa;}}
.gate{{background:#e6ffec;padding:1em;border-left:4px solid #22863a;margin:1em 0;}}
.note{{background:#fff7e6;padding:1em;border-left:4px solid #d4a72c;margin:1em 0;}}
</style></head><body>

<h1>Step 4 Integration Summary  -  {today}</h1>
<p>Single source-of-truth status of every claim in the npj PD manuscript
(<i>2026-05-04-pd-bbb-manuscript-design.md</i>) after the 5 strengthening tasks.</p>

<h2>Headline numbers</h2>
{headline}

<h2>Per-claim manuscript status</h2>
{status_html}

<div class="gate"><b>STEP 4 GATE PASSED.</b> All 5 strengthening tasks
completed and committed. The manuscript is ready to enter writing phase
(Week 2: figures finalized + Methods drafted, per design §5 timeline).
Two follow-ups deferred to writing step: (i) Kamath 2022 / Wang 2024
Suppl-Table overlap (Task 4 stub); (ii) Table 2 [VERIFY] cells must be
confirmed against source papers (Task 5).</div>

<div class="note"><b>Important Methods caveats surfaced by Step 4:</b>
<ul>
<li><b>Genotype-call fix (Task 3):</b> within the n=61 modeling cohort,
LRRK2 carriers = 0, GBA carriers = 0, APOE e4 carriers = 6 (too small
for stratified UPDRS3); 31/61 patients ungenotyped. The Figure 5 forest
plot must <i>drop</i> the LRRK2 and GBA boxes (or replace with "n.a.  - 
no carriers in modeling cohort"). The Step 5 Task 53
<i>subgroup_analyses_summary.tsv</i> is now superseded by
<i>results/step4/subgroup_analyses_fixed.tsv</i>.</li>
<li><b>Sensitivity lmer singular fit (Task 2):</b> with n=54 the random-
intercept variance collapsed to zero. Point estimate and CI remain
interpretable for the sensitivity claim, but Methods must explicitly
flag the boundary fit.</li>
<li><b>APOE-non-e4 MoCA hint (Task 3, robust to fix):</b> est=-0.469,
p=0.032 uncorrected, p_BH=0.126. To be flagged in Figure 5 caption as
exploratory, hypothesis-generating only  -  not a positive finding. The
hint is robust to the carrier-call correction because APOE was already
parsed correctly in Step 5.</li>
</ul></div>

<h2>Files of record</h2>
<ul>
<li><code>results/step4/cross_modal_decoupling.html / .tsv / .parquet</code>  -  Figure 4</li>
<li><code>results/step4/lmer_sensitivity_v2_coefs.tsv / _model.rds</code>  -  Figure 5 sensitivity row</li>
<li><code>results/step4/subgroup_analyses_fixed.tsv + subject_genotype_annotation_fixed.tsv</code>  -  Figure 5 forest (corrected)</li>
<li><code>results/step4/v2_module_enrichment.tsv + cross_cohort_context.html + hub_genes_with_symbols.tsv</code>  -  Discussion + Figure 1 caption</li>
<li><code>results/step4/lit_comparison_table2.tsv / .html</code>  -  Table 2 scaffold</li>
</ul>

</body></html>"""

out = S4 / "step4_integration_summary.html"
out.write_text(html, encoding="utf-8")
print(f"→ Wrote {out}")
print("\nHeadline:")
print(f"  Olink primary:        {pri_row['Estimate']:+.3f} ± {pri_row['Std. Error']:.3f}, p={pri_row['Pr(>|t|)']:.4f}")
print(f"  v2 sensitivity:       {sens_row['Estimate']:+.3f} ± {sens_row['Std. Error']:.3f}, p={sens_row['Pr(>|t|)']:.4f}")
print(f"  Spearman cross-modal: {xmod['spearman_rho']:+.3f}, p={xmod['spearman_p']:.2e}")
print(f"  Cohen's κ:            {xmod['cohens_kappa']:+.3f}")
print(f"  HSP90 enrichment:     6/12, p_BH={enr.loc[enr['reference_set']=='HSP90_family','bh_padj'].iloc[0]:.1e}")
print(f"  HSP70 enrichment:     6/19, p_BH={enr.loc[enr['reference_set']=='HSP70_family','bh_padj'].iloc[0]:.1e}")
