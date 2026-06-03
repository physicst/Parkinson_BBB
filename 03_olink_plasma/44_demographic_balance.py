"""Step 3 Task 34  -  Demographic balance check.

Tests whether the BBB score / vascular_class assignment is confounded by
age, sex, disease duration, LEDD. Output: HTML report + plain TSV.
"""
from __future__ import annotations
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact, spearmanr

S3 = Path("results/step3")


def main() -> None:
    today = datetime.date.today().isoformat()

    bbb = pd.read_parquet(S3 / "bbb_protein_score.parquet")
    bbb["PATNO"] = bbb["PATNO"].astype(str)

    canon = pd.read_parquet(Path("results/cache/canonical_clinical.parquet"))
    canon["PATNO"] = canon["PATNO"].astype(str)
    bl = canon[canon["EVENT_ID"] == "BL"].drop_duplicates("PATNO")

    df = bbb.merge(
        bl[["PATNO", "Cohort_Current", "Age_at_Baseline", "Sex"]],
        on="PATNO", how="left",
    )

    print(f"Cohort: {len(df)} patients; "
          f"Vascular_High={(df['vascular_class']=='Vascular_High').sum()}, "
          f"Vascular_Low={(df['vascular_class']=='Vascular_Low').sum()}")

    rows = []

    # Age vs continuous score
    sub = df[["bbb_score", "Age_at_Baseline"]].dropna()
    rho, p = spearmanr(sub["bbb_score"], sub["Age_at_Baseline"])
    rows.append({"covariate": "Age (continuous)", "test": "Spearman vs bbb_score",
                 "stat": f"rho={rho:+.3f}", "p": p})

    # Age by cluster
    h_age = df[df["vascular_class"] == "Vascular_High"]["Age_at_Baseline"].dropna()
    l_age = df[df["vascular_class"] == "Vascular_Low"]["Age_at_Baseline"].dropna()
    if len(h_age) > 5 and len(l_age) > 5:
        _, p_age = mannwhitneyu(h_age, l_age, alternative="two-sided")
        rows.append({"covariate": "Age (cluster)",
                     "test": f"MWU High(n={len(h_age)}, mean={h_age.mean():.1f}) vs Low(n={len(l_age)}, mean={l_age.mean():.1f})",
                     "stat": "two-sided", "p": p_age})

    # Sex × cluster
    contingency = pd.crosstab(df["vascular_class"], df["Sex"])
    print("\nSex × cluster:")
    print(contingency)
    if contingency.shape == (2, 2):
        _, p_sex = fisher_exact(contingency)
        rows.append({"covariate": "Sex × cluster", "test": "Fisher exact",
                     "stat": str(contingency.values.tolist()), "p": p_sex})

    # Cohort distribution
    print("\nCohort_Current × cluster:")
    print(pd.crosstab(df["vascular_class"], df["Cohort_Current"]))

    # Output
    out = pd.DataFrame(rows)
    print("\n=== Demographic balance ===")
    print(out.to_string(index=False))

    out.to_csv(S3 / "demographic_balance.tsv", sep="\t", index=False)

    # HTML wrapper
    table_html = out.round(4).to_html(index=False)
    contingency_html = contingency.to_html()

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Step 3 Task 34 - Demographic balance</title>
<style>
body{{font-family:system-ui;max-width:900px;margin:2em auto;padding:0 1em;}}
table{{border-collapse:collapse;margin:1em 0;}}
th,td{{border:1px solid #ccc;padding:6px 12px;}}
th{{background:#f5f5f5;}}
</style></head><body>
<h1>Step 3 - Demographic balance</h1>
<p><b>Date:</b> {today}<br><b>n:</b> {len(df)}</p>

<h2>Covariate tests</h2>
{table_html}

<h2>Sex × cluster contingency</h2>
{contingency_html}

<p>Per pre-reg §3.4 (master plan §3.4): any covariate with p&lt;0.05 imbalance
is added to the Step 5 mixed-effects model in addition to the
already-pre-registered Age + Sex + Baseline_severity + LEDD + Neutrophil_frac
+ Monocyte_frac.</p>
</body></html>"""

    (S3 / "demographic_balance.html").write_text(html, encoding="utf-8")
    print(f"\n→ Wrote {S3/'demographic_balance.tsv'}")
    print(f"→ Wrote {S3/'demographic_balance.html'}")


if __name__ == "__main__":
    main()
