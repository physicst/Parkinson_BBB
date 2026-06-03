"""Task 29  -  v1 vs v2 BBB signature comparison on PPMI baseline.

Compares two signatures by projecting both onto PPMI baseline whole-blood
RNA-seq and running 5-fold CV logistic regression PD vs HC:

  v1 = the reconstructed 1,015-gene signature from Step 1 (Welch t-test on
       GSE178265 endothelial pseudobulk; AUC 0.43 in Step 1 Task 20)
  v2 = the 118-gene Step 2 BBB module (multi-dataset integrated WGCNA,
       module-PD correlation 0.387, p=0.0347)

Outputs an HTML report at results/step2/<today>-v1-vs-v2-comparison.html.
"""
from __future__ import annotations
import base64
import datetime
import io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import hypergeom
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import load_config

CACHE = Path(load_config()["results"]["cache_dir"])
V1_DIR = Path(load_config()["results"]["v1_baseline_dir"])
S2_DIR = CACHE.parent / "step2"


def load_v1_genes(ensg_map: pd.DataFrame) -> set[str]:
    """v1 signature genes were stored as HGNC symbols. Map to Ensembl."""
    sig = pd.read_csv(V1_DIR / "v1_signature_genes.tsv", sep="\t")
    sym_to_ensg = dict(zip(ensg_map["symbol"], ensg_map["ensembl_gene_id"]))
    ensgs = {sym_to_ensg[s] for s in sig["gene"].astype(str) if s in sym_to_ensg}
    print(f"v1 signature: {len(sig)} symbols -> {len(ensgs)} mapped ENSGs")
    return ensgs


def load_v2_genes() -> set[str]:
    """v2 module genes already in Ensembl."""
    mod = pd.read_csv(S2_DIR / "bbb_module_genes.tsv", sep="\t")
    ensgs = set(mod["gene_id"].astype(str))
    print(f"v2 module: {len(ensgs)} ENSGs")
    return ensgs


def score_per_sample(counts: pd.DataFrame, gene_set: set[str]) -> pd.Series:
    """Mean z-scored log-CPM across signature genes detectable in PPMI."""
    overlap = sorted(gene_set & set(counts.index))
    print(f"  detectable in PPMI: {len(overlap)} / {len(gene_set)}")
    cpm = counts.div(counts.sum(axis=0), axis=1) * 1e6
    log_cpm = np.log2(cpm + 1).loc[overlap]
    z = log_cpm.sub(log_cpm.mean(axis=1), axis=0).div(log_cpm.std(axis=1) + 1e-12, axis=0)
    return z.mean(axis=0)


def auc_via_cv(score: pd.Series, canonical: pd.DataFrame, name: str) -> dict:
    """5-fold CV logistic regression PD vs HC."""
    df = pd.DataFrame({"score": score, "RNA_HudAlphaID": score.index}).merge(
        canonical[["RNA_HudAlphaID", "Cohort_Current", "Age_at_Baseline", "Sex"]]
        .drop_duplicates("RNA_HudAlphaID"),
        on="RNA_HudAlphaID", how="left",
    )
    df = df[df["Cohort_Current"].isin(["PD", "HC"])].dropna(subset=["score"])
    df["y"] = (df["Cohort_Current"] == "PD").astype(int)

    X = df[["score"]].values
    y = df["y"].values
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260504)
    aucs = cross_val_score(clf, X, y, scoring="roc_auc", cv=cv)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc_pooled = roc_auc_score(y, proba)
    fpr, tpr, _ = roc_curve(y, proba)

    print(f"  {name}: n={len(df)} (PD={(y==1).sum()}, HC={(y==0).sum()}), "
          f"5-fold CV AUC = {aucs.mean():.3f} +/- {aucs.std():.3f}, pooled = {auc_pooled:.3f}")

    return {
        "name": name,
        "n_samples": len(df),
        "n_pd": int((y == 1).sum()),
        "n_hc": int((y == 0).sum()),
        "auc_5fold_mean": aucs.mean(),
        "auc_5fold_sd": aucs.std(),
        "auc_pooled": auc_pooled,
        "fpr": fpr,
        "tpr": tpr,
    }


def render_roc(v1: dict, v2: dict) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(v1["fpr"], v1["tpr"], linewidth=2, label=f"v1 (AUC={v1['auc_pooled']:.3f})")
    ax.plot(v2["fpr"], v2["tpr"], linewidth=2, label=f"v2 (AUC={v2['auc_pooled']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("PPMI baseline PD-vs-HC discrimination")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    today = datetime.date.today().isoformat()

    # Load PPMI baseline counts + canonical clinical
    print("Loading PPMI baseline matrix...")
    counts = pd.read_parquet(CACHE / "ppmi_counts_baseline.parquet")
    canonical = pd.read_parquet(CACHE / "canonical_clinical.parquet")
    print(f"  matrix: {counts.shape[0]} genes x {counts.shape[1]} samples")

    # v1 signature uses HGNC symbols; need ENSG mapping
    ensg_map = pd.read_csv(CACHE / "ensg_to_symbol.tsv", sep="\t")
    v1_genes = load_v1_genes(ensg_map)
    v2_genes = load_v2_genes()

    # Gene overlap
    overlap = v1_genes & v2_genes
    background_n = len(set(counts.index))
    rv = hypergeom(M=background_n, n=len(v1_genes), N=len(v2_genes))
    pval = rv.sf(len(overlap) - 1)
    expected = len(v1_genes) * len(v2_genes) / background_n
    fold = len(overlap) / expected if expected > 0 else float("nan")
    print(f"\nv1 ∩ v2 gene overlap: {len(overlap)} (expected by chance ≈ {expected:.2f}, "
          f"fold={fold:.1f}x, hypergeom p={pval:.2e})")

    # Score samples on each
    print("\nScoring v1 signature on PPMI baseline...")
    v1_score = score_per_sample(counts, v1_genes)
    print("Scoring v2 module on PPMI baseline...")
    v2_score = score_per_sample(counts, v2_genes)

    # Logistic regression
    print("\nLogistic regression:")
    v1_res = auc_via_cv(v1_score, canonical, "v1 signature (1,015 genes)")
    v2_res = auc_via_cv(v2_score, canonical, "v2 BBB module (118 genes)")

    # Render
    roc_uri = render_roc(v1_res, v2_res)
    summary_table = pd.DataFrame([
        {"signature": v1_res["name"], "n_genes_input": len(v1_genes),
         "n_samples": v1_res["n_samples"], "PD": v1_res["n_pd"], "HC": v1_res["n_hc"],
         "AUC_pooled": f"{v1_res['auc_pooled']:.3f}",
         "AUC_5foldCV": f"{v1_res['auc_5fold_mean']:.3f} ± {v1_res['auc_5fold_sd']:.3f}"},
        {"signature": v2_res["name"], "n_genes_input": len(v2_genes),
         "n_samples": v2_res["n_samples"], "PD": v2_res["n_pd"], "HC": v2_res["n_hc"],
         "AUC_pooled": f"{v2_res['auc_pooled']:.3f}",
         "AUC_5foldCV": f"{v2_res['auc_5fold_mean']:.3f} ± {v2_res['auc_5fold_sd']:.3f}"},
    ]).to_html(index=False)

    delta = v2_res["auc_pooled"] - v1_res["auc_pooled"]
    if v2_res["auc_pooled"] >= 0.65 and delta > 0.10:
        verdict_html = (
            f'<p class="verdict ok">v2 outperforms v1 in PPMI baseline (Δ AUC = {delta:+.3f}). '
            'The pre-registered, multi-dataset, deconvolution-aware approach produces '
            'a meaningfully better blood-detectable signature than the v1 reconstruction. '
            'The new signature replaces v1 as the basis for Step 3 endotyping.</p>'
        )
    elif v2_res["auc_pooled"] > 0.55 and delta > 0.05:
        verdict_html = (
            f'<p class="verdict modest">v2 modestly outperforms v1 (Δ AUC = {delta:+.3f}). '
            'Improvement is real but the absolute discrimination is modest. The Step 3 '
            'endotyping cohort split should still proceed using the v2 module.</p>'
        )
    else:
        verdict_html = (
            f'<p class="verdict warn">v2 does NOT meaningfully outperform v1 (Δ AUC = {delta:+.3f}). '
            'Both signatures perform near random in PPMI baseline blood. This is consistent '
            'with the Step 1 finding that the v1 1,015-gene signature did not reproduce '
            '(claimed 0.811 vs measured 0.43). The brain-derived BBB signature, even after '
            'rigorous multi-dataset re-derivation, does NOT translate cleanly to PPMI blood. '
            'Master plan §3.4 risk #1 is now realized: <b>downstream Step 3 endotyping should '
            'pivot to plasma proteomics (Olink) as the primary modality</b>; transcriptomic '
            'endotyping moves to exploratory.</p>'
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>v1 vs v2 BBB signature - {today}</title>
<style>
body{{font-family:system-ui;max-width:900px;margin:2em auto;padding:0 1em;color:#222;}}
table{{border-collapse:collapse;margin:1em 0;}}
th,td{{border:1px solid #ccc;padding:6px 12px;text-align:right;}}
th{{background:#f5f5f5;text-align:left;}}
.verdict{{padding:1em;border-left:4px solid;font-weight:bold;}}
.verdict.ok{{background:#e6ffec;border-color:#22863a;}}
.verdict.modest{{background:#fff7e6;border-color:#d4a72c;}}
.verdict.warn{{background:#fde2e2;border-color:#c92a2a;}}
img{{max-width:100%;}}
</style></head><body>
<h1>PD-BBB Step 2  -  v1 vs v2 BBB signature comparison</h1>
<p><b>Date:</b> {today}<br>
<b>Test:</b> 5-fold CV logistic regression PD vs HC on PPMI baseline whole-blood RNA-seq.</p>

<h2>Signatures compared</h2>
<ul>
<li><b>v1</b>: 1,015 HGNC symbols from Step 1 best-effort reconstruction (Welch t-test PD-vs-normal on GSE178265 endothelial h5ad pseudobulk; ~839 mapped to Ensembl).</li>
<li><b>v2</b>: 118 Ensembl IDs from Step 2 multi-dataset integrated WGCNA module 8 (correlation with PD status: 0.387, p=0.0347; bootstrap-stable filter not satisfied  -  DEG-ranking fallback engaged per master plan §5 risk #6).</li>
</ul>

<h2>Gene overlap</h2>
<p>v1 ∩ v2 = <b>{len(overlap)}</b> genes (expected by chance ≈ {expected:.2f}; fold-enrichment = {fold:.1f}×; hypergeometric p = {pval:.2e}).</p>

<h2>PPMI baseline AUC results</h2>
{summary_table}

<h2>ROC curves</h2>
<img src="{roc_uri}" alt="v1 vs v2 ROC">

<h2>Verdict</h2>
{verdict_html}
</body></html>"""

    out_path = S2_DIR / f"{today}-v1-vs-v2-comparison.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n→ Wrote {out_path}")


if __name__ == "__main__":
    main()
