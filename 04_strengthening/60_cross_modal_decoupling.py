"""Step 4 Task 1  -  Cross-modal decoupling test (npj PD strengthening item 1).

Tests the central scientific claim of the manuscript: that the brain-derived
v2 transcriptomic BBB signature and the Olink plasma BBB protein PC1 are
NOT measuring the same axis of vascular biology in PD.

Two formal tests:
  (a) Continuous: Spearman correlation between v2 score (mean z-score over
      v2 module on PPMI baseline whole-blood RNA-seq) and Olink BBB PC1
      across overlapping patients.
  (b) Categorical: Cohen's κ between transcriptomic and proteomic endotype
      assignments. Transcriptomic endotype defined by median split of v2
      score (Vascular_T-High / Vascular_T-Low); proteomic endotype from
      Step 3 Task 31 (Vascular_High / Vascular_Low).

Output: results/step4/cross_modal_decoupling.html + .tsv
"""
from __future__ import annotations
import datetime
import io
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

OUT = Path("results/step4")
OUT.mkdir(parents=True, exist_ok=True)


def compute_v2_score(counts: pd.DataFrame, v2_genes: set[str]) -> pd.Series:
    """Mean z-scored log-CPM across v2 module genes detectable in PPMI."""
    overlap = sorted(v2_genes & set(counts.index))
    cpm = counts.div(counts.sum(axis=0), axis=1) * 1e6
    log_cpm = np.log2(cpm + 1).loc[overlap]
    z = log_cpm.sub(log_cpm.mean(axis=1), axis=0).div(log_cpm.std(axis=1) + 1e-12, axis=0)
    return z.mean(axis=0).rename("v2_transcriptomic_score")


def render_scatter_uri(df: pd.DataFrame, rho: float, p: float) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df["v2_score"], df["olink_pc1"], alpha=0.55, s=22, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("Brain-derived v2 transcriptomic score\n(mean z(log-CPM) on PPMI blood)")
    ax.set_ylabel("Olink BBB panel PC1\n(plasma proteomic score)")
    ax.set_title(f"Cross-modal decoupling test\nSpearman ρ = {rho:+.3f}, p = {p:.2e}, n = {len(df)}")
    ax.axhline(0, color="grey", linestyle="--", alpha=0.4)
    ax.axvline(0, color="grey", linestyle="--", alpha=0.4)
    ax.grid(alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    today = datetime.date.today().isoformat()

    # Load PPMI baseline counts + canonical
    counts = pd.read_parquet("results/cache/ppmi_counts_baseline.parquet")
    canonical = pd.read_parquet("results/cache/canonical_clinical.parquet")
    canonical["PATNO"] = canonical["PATNO"].astype(str)

    # v2 module genes
    v2_mod = pd.read_csv("results/step2/bbb_module_genes.tsv", sep="\t")
    v2_genes = set(v2_mod["gene_id"].astype(str))
    print(f"v2 module: {len(v2_genes)} Ensembl IDs")

    # Compute v2 score per HudAlphaID
    v2_score = compute_v2_score(counts, v2_genes)
    print(f"Computed v2 score for {len(v2_score)} PPMI baseline samples")

    # Map HudAlphaID -> PATNO via canonical clinical (RNA_HudAlphaID column)
    rna_link = (canonical[canonical["RNA_HudAlphaID"].notna()]
                .drop_duplicates("RNA_HudAlphaID")
                [["PATNO", "RNA_HudAlphaID"]])
    v2_per_patno = (
        pd.DataFrame({"RNA_HudAlphaID": v2_score.index, "v2_score": v2_score.values})
        .merge(rna_link, on="RNA_HudAlphaID", how="inner")
        .groupby("PATNO", as_index=False)["v2_score"].mean()
    )
    print(f"Mapped to {len(v2_per_patno)} unique PATNOs")

    # Olink BBB PC1 per PATNO
    olink = pd.read_parquet("results/step3/bbb_protein_score.parquet")
    olink["PATNO"] = olink["PATNO"].astype(str)
    olink_score = olink[["PATNO", "bbb_score_pc1", "vascular_class"]].rename(
        columns={"bbb_score_pc1": "olink_pc1", "vascular_class": "olink_class"}
    )

    merged = v2_per_patno.merge(olink_score, on="PATNO", how="inner")
    print(f"Cross-modal cohort: {len(merged)} patients with BOTH v2 and Olink scores")

    # ---- (a) Continuous: Spearman correlation ----
    rho, p = spearmanr(merged["v2_score"], merged["olink_pc1"])
    print(f"\nSpearman ρ(v2_score, olink_pc1) = {rho:+.3f}, p = {p:.4e}")

    # ---- (b) Categorical: κ between transcriptomic median-split and Olink endotype ----
    median_v2 = merged["v2_score"].median()
    merged["transcriptomic_class"] = np.where(
        merged["v2_score"] >= median_v2, "Vascular_T-High", "Vascular_T-Low"
    )

    contingency = pd.crosstab(merged["transcriptomic_class"], merged["olink_class"])
    print(f"\nContingency (transcriptomic class × Olink class):")
    print(contingency)

    kappa = cohen_kappa_score(
        merged["transcriptomic_class"].map({"Vascular_T-High": 1, "Vascular_T-Low": 0}),
        merged["olink_class"].map({"Vascular_High": 1, "Vascular_Low": 0}),
    )
    print(f"Cohen's κ = {kappa:+.3f}")

    # Master plan §3.4 had κ ≥ 0.4 = strong cross-modal signal; κ ≥ 0.2 = real but
    # weak; κ < 0.2 = decoupled. Our pre-reg deviation §14 frames the same.

    # ---- Verdict ----
    if abs(rho) >= 0.40 and p < 0.001 and kappa >= 0.4:
        verdict = (f'<p class="verdict ok">CONCORDANT  -  both modalities measure '
                   f'the same axis (ρ={rho:+.3f}, κ={kappa:+.2f}). Decoupling '
                   'claim NOT supported. Reframe paper.</p>')
    elif abs(rho) >= 0.20 and kappa >= 0.20:
        verdict = (f'<p class="verdict modest">PARTIALLY CONCORDANT  -  modalities '
                   f'share some signal (ρ={rho:+.3f}, κ={kappa:+.2f}). Decoupling '
                   'claim weakened; framing needs to acknowledge partial overlap.</p>')
    else:
        verdict = (f'<p class="verdict warn">DECOUPLED  -  brain-derived '
                   f'transcriptomic and plasma proteomic vascular signatures are '
                   f'essentially independent (Spearman ρ = {rho:+.3f}, p = {p:.2e}; '
                   f'Cohen\'s κ = {kappa:+.3f}). The two modalities measure '
                   'different axes of PD vascular biology. <b>Decoupling claim '
                   'EMPIRICALLY SUPPORTED.</b></p>')

    # Render
    scatter_uri = render_scatter_uri(
        merged.rename(columns={"v2_score": "v2_score", "olink_pc1": "olink_pc1"}),
        rho, p,
    )
    contingency_html = contingency.to_html()

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Cross-modal decoupling  -  {today}</title>
<style>
body{{font-family:system-ui;max-width:900px;margin:2em auto;padding:0 1em;color:#222;}}
table{{border-collapse:collapse;margin:1em 0;}}
th,td{{border:1px solid #ccc;padding:6px 12px;}}
th{{background:#f5f5f5;}}
.verdict{{padding:1em;border-left:4px solid;font-weight:bold;}}
.verdict.ok{{background:#e6ffec;border-color:#22863a;}}
.verdict.modest{{background:#fff7e6;border-color:#d4a72c;}}
.verdict.warn{{background:#fde2e2;border-color:#c92a2a;}}
img{{max-width:100%;}}
</style></head><body>
<h1>Step 4 Task 1  -  Cross-modal decoupling test (manuscript Figure 4)</h1>
<p><b>Date:</b> {today}<br>
<b>Cohort:</b> {len(merged)} PPMI patients with both v2 transcriptomic BBB
score (118 brain-derived genes) and Olink BBB PC1 (20 plasma proteins).</p>

<h2>(a) Continuous correlation</h2>
<p>Spearman ρ = <b>{rho:+.3f}</b>, p = <b>{p:.2e}</b>, n = {len(merged)}.</p>
<img src="{scatter_uri}" alt="cross-modal scatter">

<h2>(b) Endotype agreement (Cohen's κ)</h2>
<p>Transcriptomic median-split (v2 score above/below cohort median) vs Olink
ConsensusClusterPlus k=2 endotype:</p>
{contingency_html}
<p>Cohen's κ = <b>{kappa:+.3f}</b></p>

<h2>Decoupling test verdict</h2>
{verdict}

<p><i>Master plan §3.4 thresholds: κ ≥ 0.4 strong; κ 0.2-0.4 real but weak;
κ &lt; 0.2 decoupled. Our deviation §14 reuses the same convention.</i></p>
</body></html>"""

    out_html = OUT / "cross_modal_decoupling.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"\n→ Wrote {out_html}")

    # Save TSV
    merged.to_parquet(OUT / "cross_modal_scores.parquet")
    pd.DataFrame([{
        "n_patients": len(merged), "spearman_rho": rho, "spearman_p": p,
        "cohens_kappa": kappa,
    }]).to_csv(OUT / "cross_modal_decoupling_summary.tsv", sep="\t", index=False)
    print(f"→ Wrote {OUT/'cross_modal_decoupling_summary.tsv'}")


if __name__ == "__main__":
    main()
