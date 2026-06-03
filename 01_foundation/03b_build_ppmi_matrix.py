"""Build PPMI baseline-visit gene x sample count matrix from featureCounts.

Strategy:
  1. Restrict to baseline visits (EVENT_ID == 'BL') with QCflag pass.
  2. Map each HudAlphaID -> featureCounts file path.
  3. Read each featureCounts file (skip comment line, take Geneid + last column).
  4. Strip Ensembl version suffix '.NN' from gene IDs.
  5. Output: parquet with index=gene_id, columns=HudAlphaID.

Memory budget: ~60K genes x ~1500 baseline samples x 4 bytes = ~360 MB int32.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from lib.config import load_config


def read_featurecounts(path: Path) -> pd.Series:
    """Return a Series indexed by stripped gene_id with int counts."""
    df = pd.read_csv(path, sep="\t", comment="#")
    s = df.iloc[:, [0, -1]].rename(columns={df.columns[0]: "gene", df.columns[-1]: "count"})
    s["gene"] = s["gene"].astype(str).str.split(".").str[0]
    return s.set_index("gene")["count"].astype("int32")


def main() -> None:
    cfg = load_config()
    canonical = pd.read_parquet(Path(cfg["results"]["cache_dir"]) / "canonical_clinical.parquet")

    base = canonical[
        (canonical["EVENT_ID"] == "BL")
        & (canonical["RNA_QCflag"] == "pass")
        & canonical["RNA_HudAlphaID"].notna()
    ].drop_duplicates(subset="RNA_HudAlphaID")
    print(f"Baseline pass-QC RNA samples: {len(base)}")

    counts_dir = Path(cfg["rna_seq"]["counts_dir"])
    available = {p.name: p for p in counts_dir.glob("*.featureCounts.GencodeV29.txt")}
    print(f"Available featureCounts files: {len(available)}")

    matched_paths = []
    matched_ids = []
    missing = []
    for hud in base["RNA_HudAlphaID"]:
        candidates = [name for name in available if f".{hud}." in name]
        if candidates:
            matched_paths.append(available[candidates[0]])
            matched_ids.append(hud)
        else:
            missing.append(hud)
    print(f"Matched: {len(matched_paths)}; missing: {len(missing)}")
    if missing[:3]:
        print(f"  example missing: {missing[:3]}")

    print("Reading featureCounts files (this takes ~5-15 min)...")
    series_list = []
    for hud, path in tqdm(list(zip(matched_ids, matched_paths))):
        s = read_featurecounts(path).rename(hud)
        series_list.append(s)

    mat = pd.concat(series_list, axis=1).fillna(0).astype("int32")
    print(f"\nMatrix: {mat.shape[0]} genes x {mat.shape[1]} samples")

    out_path = Path(cfg["results"]["cache_dir"]) / "ppmi_counts_baseline.parquet"
    mat.to_parquet(out_path)
    print(f"-> Wrote {out_path}")


if __name__ == "__main__":
    main()
