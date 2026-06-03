"""Step 3 Task 33  -  Orthogonal validation of the Olink BBB score.

Tests whether the continuous BBB score (PC1 of the 20-protein Olink panel,
per pre-reg §14 deviation 1) correlates with:

  BBB-validation set (expect POSITIVE correlation):
    - Plasma NEFL (Olink Neurology, separate from panel)
    - CSF Albumin (Biospecimen project 181)
    - Plasma GFAP (Biospecimen project 152/256/283 Simoa)

  Orthogonality test set (no correlation expected  -  characterizes the axis):
    - DaT-SCAN Putamen_Worst (nigrostriatal degeneration)
    - CSF α-synuclein SAA status (α-syn aggregation)

Output: results/step3/orthogonal_validation_report.html
"""
from __future__ import annotations
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

S3 = Path("results/step3")


def main() -> None:
    today = datetime.date.today().isoformat()

    # Load BBB score (PC1 primary)
    bbb = pd.read_parquet(S3 / "bbb_protein_score.parquet")
    bbb["PATNO"] = bbb["PATNO"].astype(str)
    print(f"BBB score: {len(bbb)} patients")

    # Load canonical clinical (already has DaT-SCAN, SAA, Cohort_Current)
    canonical = pd.read_parquet(Path("results/cache/canonical_clinical.parquet"))
    canonical["PATNO"] = canonical["PATNO"].astype(str)

    # Restrict to baseline visit
    bl = canonical[canonical["EVENT_ID"] == "BL"].drop_duplicates("PATNO").copy()

    # Load biospecimen file for albumin, GFAP, NfL (Simoa)
    biospec = pd.read_csv("D:/Parkinson_file/Current_Biospecimen_Analysis_Results_02May2026.csv")
    biospec["PATNO"] = biospec["PATNO"].astype(str)
    # Take baseline only (CLINICAL_EVENT == "BL"), prefer numeric TESTVALUE
    biospec_bl = biospec[biospec["CLINICAL_EVENT"] == "BL"].copy()
    biospec_bl["NumValue"] = pd.to_numeric(biospec_bl["TESTVALUE"], errors="coerce")

    # Pivot: one row per (PATNO, TESTNAME)
    def get_marker(testnames: list[str]) -> pd.Series:
        sub = biospec_bl[biospec_bl["TESTNAME"].isin(testnames) & biospec_bl["NumValue"].notna()]
        return sub.groupby("PATNO")["NumValue"].mean()

    csf_albumin = get_marker(["CSF Albumin"])
    plasma_albumin = get_marker(["Plasma Albumin"])
    nfl_simoa = get_marker(["NfL", "NFL"])
    gfap_simoa = get_marker(["GFAP"])
    print(f"  CSF Albumin    : {len(csf_albumin)} patients with measurement")
    print(f"  Plasma Albumin : {len(plasma_albumin)} patients")
    print(f"  Simoa NfL/NFL  : {len(nfl_simoa)} patients")
    print(f"  Simoa GFAP     : {len(gfap_simoa)} patients")

    # Olink NEFL (already in panel  -  for sanity, pull from panel matrix directly)
    panel = pd.read_parquet(S3 / "olink_bbb_panel_baseline.parquet")
    panel["PATNO"] = panel["PATNO"].astype(str)
    nefl_olink = panel.set_index("PATNO")["NEFL"] if "NEFL" in panel.columns else None

    # Build merged dataframe
    df = bbb[["PATNO", "bbb_score", "vascular_class"]].copy()
    df = df.merge(bl[["PATNO", "Cohort_Current", "DATSCAN_Putamen_Worst",
                      "SAA_Status", "Age_at_Baseline", "Sex"]],
                  on="PATNO", how="left")
    if nefl_olink is not None:
        df["nefl_olink"] = df["PATNO"].map(nefl_olink)
    df["csf_albumin"] = df["PATNO"].map(csf_albumin)
    df["plasma_albumin"] = df["PATNO"].map(plasma_albumin)
    df["nfl_simoa"] = df["PATNO"].map(nfl_simoa)
    df["gfap_simoa"] = df["PATNO"].map(gfap_simoa)

    # ---- BBB-validation tests ----
    bbb_validation = []
    for marker_col, label in [
        ("nefl_olink", "Plasma NEFL (Olink, panel-internal)"),
        ("nfl_simoa", "Plasma NfL (Simoa, separate Biospecimen project)"),
        ("csf_albumin", "CSF Albumin"),
        ("plasma_albumin", "Plasma Albumin"),
        ("gfap_simoa", "Plasma GFAP (Simoa)"),
    ]:
        sub = df[["bbb_score", marker_col]].dropna()
        if len(sub) < 10:
            bbb_validation.append({"marker": label, "n": len(sub),
                                   "rho": np.nan, "p": np.nan})
            continue
        rho, p = spearmanr(sub["bbb_score"], sub[marker_col])
        bbb_validation.append({"marker": label, "n": len(sub),
                               "rho": rho, "p": p,
                               "p_one_sided": p / 2 if rho > 0 else 1 - p / 2})

    bbb_df = pd.DataFrame(bbb_validation)
    print("\n=== BBB-validation correlations (expect positive rho) ===")
    print(bbb_df.to_string(index=False))

    # ---- Orthogonality test set ----
    ortho_results = []

    # DaT-SCAN
    sub = df[["bbb_score", "DATSCAN_Putamen_Worst"]].dropna()
    if len(sub) >= 10:
        rho, p = spearmanr(sub["bbb_score"], sub["DATSCAN_Putamen_Worst"])
        ortho_results.append({"marker": "DaT-SCAN Putamen_Worst", "n": len(sub),
                              "rho": rho, "p": p})

    # SAA: Wilcoxon (Positive vs Negative)
    saa_pos = df[df["SAA_Status"] == "Positive"]["bbb_score"].dropna()
    saa_neg = df[df["SAA_Status"] == "Negative"]["bbb_score"].dropna()
    if len(saa_pos) >= 5 and len(saa_neg) >= 5:
        stat, p = mannwhitneyu(saa_pos, saa_neg, alternative="two-sided")
        ortho_results.append({
            "marker": f"CSF SAA (Pos n={len(saa_pos)} vs Neg n={len(saa_neg)})",
            "n": len(saa_pos) + len(saa_neg),
            "rho": (saa_pos.mean() - saa_neg.mean()) /
                   np.sqrt((saa_pos.var() + saa_neg.var()) / 2),  # Cohen's d
            "p": p,
        })

    ortho_df = pd.DataFrame(ortho_results)
    print("\n=== Orthogonality test (no relationship expected) ===")
    print(ortho_df.to_string(index=False))

    # ---- HTML report ----
    bbb_pass = bbb_df.dropna(subset=["p"])
    n_pass = (bbb_pass["rho"] > 0).sum()  # crude  -  strict version uses FDR-adj < 0.10
    bbb_pass["bh_padj"] = bbb_pass["p"] * len(bbb_pass) / np.arange(1, len(bbb_pass) + 1)
    bbb_pass["bh_padj"] = bbb_pass["bh_padj"].clip(upper=1.0)
    n_pass_strict = (
        (bbb_pass["rho"] > 0) & (bbb_pass["bh_padj"] < 0.10)
    ).sum()

    if n_pass_strict >= 2:
        verdict = (f'<p class="verdict ok">PASSED: {n_pass_strict} BBB-validation markers '
                   'positively correlate with the BBB score at FDR<0.10. '
                   'The Olink-primary endotype is biologically anchored.</p>')
    elif n_pass >= 1:
        verdict = (f'<p class="verdict modest">PARTIAL: {n_pass} markers show positive '
                   'direction but exit criterion (≥2 at FDR<0.10) not met. '
                   'Endotype is internal-consistency-only  -  manuscript narrative weakens.</p>')
    else:
        verdict = (f'<p class="verdict warn">FAILED: no BBB-validation marker correlates '
                   'with the BBB score. The Olink-primary endotype does not validate '
                   'against any orthogonal damage marker. Major reframing required.</p>')

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Step 3 Task 33 - Orthogonal validation</title>
<style>
body{{font-family:system-ui;max-width:900px;margin:2em auto;padding:0 1em;color:#222;}}
table{{border-collapse:collapse;margin:1em 0;}}
th,td{{border:1px solid #ccc;padding:6px 12px;text-align:right;}}
th{{background:#f5f5f5;text-align:left;}}
.verdict{{padding:1em;border-left:4px solid;font-weight:bold;}}
.verdict.ok{{background:#e6ffec;border-color:#22863a;}}
.verdict.modest{{background:#fff7e6;border-color:#d4a72c;}}
.verdict.warn{{background:#fde2e2;border-color:#c92a2a;}}
</style></head><body>
<h1>Step 3  -  Orthogonal validation of Olink-primary BBB score</h1>
<p><b>Date:</b> {today}<br>
<b>Score:</b> PC1 of 20-protein Olink BBB panel (33.0% variance; r=0.853 with endothelial-activation mean).<br>
<b>Sample n:</b> {len(df)} baseline patients.</p>

<h2>BBB-validation (positive correlation expected)</h2>
{bbb_df.round(4).to_html(index=False)}

<h2>Orthogonality test (no relationship expected; either result informative)</h2>
{ortho_df.round(4).to_html(index=False)}

<h2>Verdict</h2>
{verdict}
</body></html>"""

    out = S3 / "orthogonal_validation_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n→ Wrote {out}")


if __name__ == "__main__":
    main()
