"""Compute v1 BBB score per PPMI baseline sample.

Score = mean of z-scored log-CPM expression across the 1,015-gene signature.
Genes not present in PPMI are dropped; coverage is reported.

Handles symbol->ENSG mapping if PPMI matrix uses Ensembl IDs.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from lib.config import load_config


def main() -> None:
    cfg = load_config()
    cache = Path(cfg["results"]["cache_dir"])
    v1_dir = Path(cfg["results"]["v1_baseline_dir"])

    sig = pd.read_csv(v1_dir / "v1_signature_genes.tsv", sep="\t")
    counts = pd.read_parquet(cache / "ppmi_counts_baseline.parquet")
    print(f"Signature: {len(sig)} genes ({sig['source'].iloc[0]})")
    print(f"PPMI matrix: {counts.shape[0]} genes x {counts.shape[1]} samples")

    sig_set = set(sig["gene"].dropna().astype(str))

    if str(counts.index[0]).startswith("ENSG"):
        try:
            anno = pd.read_csv(cache / "ensg_to_symbol.tsv", sep="\t")
        except FileNotFoundError:
            raise SystemExit(
                "Need ENSG<->symbol map. Create it once with:\n"
                "  conda run -n pd-bbb python code/03b_make_ensg_map.py"
            )
        sym_to_ensg = dict(zip(anno["symbol"], anno["ensembl_gene_id"]))
        sig_ensgs = [sym_to_ensg[g] for g in sig_set if g in sym_to_ensg]
        print(f"Symbol->ENSG mapping: {len(sig_ensgs)}/{len(sig_set)} resolved")
    else:
        sig_ensgs = [g for g in sig_set if g in counts.index]

    overlap = sorted(set(sig_ensgs) & set(counts.index))
    print(f"Signature genes detectable in PPMI: {len(overlap)} / {len(sig_set)}")

    # log-CPM
    cpm = counts.div(counts.sum(axis=0), axis=1) * 1e6
    log_cpm = np.log2(cpm + 1).loc[overlap]

    # Z-score per gene across samples
    z = log_cpm.sub(log_cpm.mean(axis=1), axis=0).div(log_cpm.std(axis=1) + 1e-12, axis=0)
    score = z.mean(axis=0).rename("BBB_score_v1")

    out = pd.DataFrame(score).reset_index().rename(columns={"index": "RNA_HudAlphaID"})
    out_path = v1_dir / "ppmi_baseline_v1_scores.parquet"
    out.to_parquet(out_path)
    print(f"-> Wrote {out_path} ({len(out)} samples scored)")


if __name__ == "__main__":
    main()
