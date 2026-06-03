"""Build a symbol -> ENSG mapping for the v1 signature genes.

PPMI featureCounts output uses Ensembl gene IDs (e.g., ENSG00000223972) as the
gene index, but our v1 signature is HGNC symbols. Map symbols to ENSGs using
mygene (which queries NCBI/Ensembl).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import mygene

from lib.config import load_config


def main() -> None:
    cfg = load_config()
    sig = pd.read_csv(
        Path(cfg["results"]["v1_baseline_dir"]) / "v1_signature_genes.tsv",
        sep="\t",
    )
    symbols = sig["gene"].dropna().astype(str).unique().tolist()
    print(f"Querying {len(symbols)} symbols against mygene...")

    mg = mygene.MyGeneInfo()
    res = mg.querymany(symbols, scopes="symbol", fields="ensembl.gene",
                       species="human", returnall=False)

    rows = []
    for r in res:
        if "ensembl" not in r:
            continue
        ens = r["ensembl"]
        if isinstance(ens, list):
            # Multi-mapping  -  take the first (canonical)
            for e in ens:
                gid = e.get("gene")
                if gid:
                    rows.append((r["query"], gid))
                    break
        elif isinstance(ens, dict):
            gid = ens.get("gene")
            if gid:
                rows.append((r["query"], gid))

    out = pd.DataFrame(rows, columns=["symbol", "ensembl_gene_id"]).drop_duplicates(subset="symbol")
    out_path = Path(cfg["results"]["cache_dir"]) / "ensg_to_symbol.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    print(f"-> Wrote {len(out)} mappings to {out_path}")
    print(f"Coverage: {len(out)}/{len(symbols)} symbols ({100*len(out)/len(symbols):.1f}%)")


if __name__ == "__main__":
    main()
