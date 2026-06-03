"""Load Agarwal 2020 GSE140231 snRNA-seq into a single AnnData (SN-only).

Source: D:/Parkinson_file/snrnaseq/GSE140231_RAW.tar  (config.snrnaseq.agarwal_raw_tar)

Structure ("tar of tar"):
  GSE140231_RAW.tar
    +-- GSM4157067_Sample_5_C3.tar.gz
    +-- GSM4157068_Sample_6_N3.tar.gz
    +-- ... 12 nested tarballs total
    Each nested tarball contains a legacy (Cell Ranger v2-style) 10X bundle:
        <Sample_dir>/matrix.mtx
        <Sample_dir>/barcodes.tsv
        <Sample_dir>/genes.tsv     (Ensembl ID  TAB  symbol; no feature-type col)

Sample-name decoding (verified against per-GSM GEO metadata 2026-05-03):
  Sample_<n>_<TISSUE><DONOR>[B]
    TISSUE: 'C' = Cortex, 'N' = substantia Nigra (NOT condition!)
    DONOR : single digit 1-5 (matches donor across cortex/SN pair)
    B     : batch-2 sample (some donors were run on Day 1 AND Day 2)
  All 12 samples are CONTROL donors  -  GSE140231 is the Agarwal SN reference
  atlas, not a PD-vs-Control study. So `condition` is uniformly 'Control'.

Filter: drop the 6 cortex samples; keep the 6 substantia-nigra samples.

Output: D:/Parkinson_file/snrnaseq/cache/agarwal_sn.h5ad

Notes:
- read_10x_mtx with var_names='gene_ids' uses Ensembl IDs as the var index and
  preserves the symbol as `var['gene_symbols']`.
- Ensembl version suffix is stripped defensively (Agarwal IDs from this 10X
  bundle look unversioned, but normalize anyway to match Smajic conventions).
- Barcodes collide across samples; anndata.concat(index_unique='-') makes them
  unique by suffixing the per-sample key.
"""
from __future__ import annotations
import re
import tarfile
import tempfile
from pathlib import Path

import anndata as ad
import scanpy as sc

from lib.config import load_config


# Per-GSM metadata harvested from GEO sample pages (2026-05-03).
# All donors are controls. Donor IDs are derived from the trailing digit in the
# sample-name code (e.g. C3 / N3 -> donor 3). Age/sex/PMI taken from GEO.
SAMPLE_META: dict[str, dict[str, object]] = {
    "GSM4157067": {"sample_name": "Sample_5_C3",   "donor_id": "D3", "region": "cortex", "sex": "M", "age": 59, "pmi": 28, "batch": "Day1"},
    "GSM4157068": {"sample_name": "Sample_6_N3",   "donor_id": "D3", "region": "SN",     "sex": "M", "age": 59, "pmi": 28, "batch": "Day1"},
    "GSM4157069": {"sample_name": "Sample_7_N4",   "donor_id": "D4", "region": "SN",     "sex": "M", "age": 70, "pmi": 42, "batch": "Day1"},
    "GSM4157070": {"sample_name": "Sample_8_N5",   "donor_id": "D5", "region": "SN",     "sex": "F", "age": 56, "pmi": 35, "batch": "Day1"},
    "GSM4157071": {"sample_name": "Sample_9_C1B",  "donor_id": "D1", "region": "cortex", "sex": "M", "age": 70, "pmi": 43, "batch": "Day2"},
    "GSM4157072": {"sample_name": "Sample_10_N1B", "donor_id": "D1", "region": "SN",     "sex": "M", "age": 70, "pmi": 43, "batch": "Day2"},
    "GSM4157073": {"sample_name": "Sample_11_C2B", "donor_id": "D2", "region": "cortex", "sex": "M", "age": 55, "pmi": 18, "batch": "Day2"},
    "GSM4157074": {"sample_name": "Sample_12_N2B", "donor_id": "D2", "region": "SN",     "sex": "M", "age": 55, "pmi": 18, "batch": "Day2"},
    "GSM4157075": {"sample_name": "Sample_13_C4B", "donor_id": "D4", "region": "cortex", "sex": "M", "age": 70, "pmi": 42, "batch": "Day2"},
    "GSM4157076": {"sample_name": "Sample_14_N4B", "donor_id": "D4", "region": "SN",     "sex": "M", "age": 70, "pmi": 42, "batch": "Day2"},
    "GSM4157077": {"sample_name": "Sample_15_C5B", "donor_id": "D5", "region": "cortex", "sex": "F", "age": 56, "pmi": 35, "batch": "Day2"},
    "GSM4157078": {"sample_name": "Sample_16_N5B", "donor_id": "D5", "region": "SN",     "sex": "F", "age": 56, "pmi": 35, "batch": "Day2"},
}

# Disease state for ALL samples in GSE140231  -  this is a control-only SN atlas.
DEFAULT_CONDITION = "Control"

GSM_TARNAME_RE = re.compile(r"^(GSM\d+)_Sample_\d+_[A-Z0-9]+\.tar\.gz$")


def strip_ensembl_version(gene_id: str) -> str:
    """ENSG00000123456.7 -> ENSG00000123456. Idempotent if no suffix."""
    return gene_id.split(".")[0]


def parse_inner_member_name(name: str) -> str | None:
    """Return GSM id from outer-tar member name, or None if it doesn't match."""
    base = Path(name).name
    m = GSM_TARNAME_RE.match(base)
    return m.group(1) if m else None


def load_sample(inner_tar_path: Path, gsm_id: str, work_dir: Path) -> ad.AnnData:
    """Extract one nested 10X bundle and return its AnnData (cells x genes).

    Annotates obs with sample/donor/region/condition + Ensembl-stripped var index.
    """
    sample_dir = work_dir / gsm_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(inner_tar_path, mode="r:gz") as tf:
        tf.extractall(sample_dir)

    # The bundle is at sample_dir/<Sample_x_XX>/{matrix.mtx,barcodes.tsv,genes.tsv}.
    bundle_dirs = [p for p in sample_dir.iterdir() if p.is_dir()]
    if len(bundle_dirs) != 1:
        raise RuntimeError(f"{gsm_id}: expected 1 bundle dir, found {bundle_dirs}")
    bundle = bundle_dirs[0]

    # Confirm the legacy-format files are present (uncompressed in this dataset).
    for needed in ("matrix.mtx", "barcodes.tsv", "genes.tsv"):
        if not (bundle / needed).exists():
            raise RuntimeError(f"{gsm_id}: bundle missing {needed} in {bundle}")

    # var_names='gene_ids' -> Ensembl IDs as var index, symbol preserved as 'gene_symbols'.
    adata = sc.read_10x_mtx(bundle, var_names="gene_ids", make_unique=True)

    # Strip Ensembl version (idempotent if absent).
    new_index = adata.var.index.map(strip_ensembl_version)
    adata.var.index = new_index
    adata.var.index.name = "gene_id"
    # If stripping creates duplicates (rare), drop subsequent occurrences.
    if adata.var.index.duplicated().any():
        keep = ~adata.var.index.duplicated(keep="first")
        adata = adata[:, keep].copy()

    meta = SAMPLE_META[gsm_id]
    adata.obs["gsm_id"] = gsm_id
    adata.obs["sample_name"] = meta["sample_name"]
    adata.obs["donor_id"] = meta["donor_id"]
    adata.obs["region"] = meta["region"]
    adata.obs["condition"] = DEFAULT_CONDITION
    adata.obs["sex"] = meta["sex"]
    adata.obs["age"] = meta["age"]
    adata.obs["pmi"] = meta["pmi"]
    adata.obs["batch"] = meta["batch"]
    return adata


def main() -> None:
    cfg = load_config()
    snr = cfg["snrnaseq"]
    outer_tar = Path(snr["agarwal_raw_tar"])

    cache_dir = outer_tar.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "agarwal_sn.h5ad"

    print(f"[1/4] Inspecting outer tar: {outer_tar}")
    with tarfile.open(outer_tar, mode="r") as outer:
        members = [m for m in outer.getmembers() if m.isfile() and m.name.endswith(".tar.gz")]
    print(f"  found {len(members)} nested tarballs")
    if len(members) != 12:
        raise RuntimeError(f"expected 12 nested tarballs, got {len(members)}")

    # Map: GSM id -> outer-tar member name.
    gsm_to_member: dict[str, str] = {}
    for m in members:
        gsm = parse_inner_member_name(m.name)
        if gsm is None:
            raise RuntimeError(f"unparseable inner-tar name: {m.name}")
        gsm_to_member[gsm] = m.name
    unknown = set(gsm_to_member) - set(SAMPLE_META)
    if unknown:
        raise RuntimeError(f"GSM IDs not in SAMPLE_META: {sorted(unknown)}")

    # SN-only filter: keep the 6 nigra samples, drop the 6 cortex.
    sn_gsm_ids = sorted(g for g, meta in SAMPLE_META.items() if meta["region"] == "SN")
    cortex_gsm_ids = sorted(g for g, meta in SAMPLE_META.items() if meta["region"] == "cortex")
    print(f"  SN samples to keep    ({len(sn_gsm_ids)}): {[SAMPLE_META[g]['sample_name'] for g in sn_gsm_ids]}")
    print(f"  cortex samples to drop ({len(cortex_gsm_ids)}): {[SAMPLE_META[g]['sample_name'] for g in cortex_gsm_ids]}")

    # Extract + load each SN sample. Use one TemporaryDirectory for the whole run
    # so cleanup happens automatically; per-sample subdirs keep things tidy.
    print(f"[2/4] Loading {len(sn_gsm_ids)} SN 10X bundles")
    adatas: list[ad.AnnData] = []
    sample_keys: list[str] = []
    with tempfile.TemporaryDirectory(prefix="agarwal_") as tmpdir:
        tmp = Path(tmpdir)
        # Stage: extract just the SN inner tarballs from the outer tar.
        with tarfile.open(outer_tar, mode="r") as outer:
            for gsm in sn_gsm_ids:
                member_name = gsm_to_member[gsm]
                outer.extract(member_name, path=tmp)
                inner_path = tmp / member_name
                if not inner_path.exists():
                    # In case the outer tar wraps members in a directory.
                    inner_path = next(tmp.rglob(Path(member_name).name))
                print(f"  {gsm} ({SAMPLE_META[gsm]['sample_name']}): extracting + loading")
                adata = load_sample(inner_path, gsm, tmp / "bundles")
                print(f"    -> {adata.n_obs} cells x {adata.n_vars} genes")
                adatas.append(adata)
                sample_keys.append(gsm)

        print(f"[3/4] Concatenating {len(adatas)} SN AnnDatas (join='outer')")
        combined = ad.concat(
            adatas,
            join="outer",
            label="gsm_id_concat",
            keys=sample_keys,
            index_unique="-",
            merge="same",
        )
        # Drop the duplicate concat label column (we already have gsm_id in obs).
        if "gsm_id_concat" in combined.obs.columns:
            del combined.obs["gsm_id_concat"]
        combined.uns["dataset"] = "Agarwal_2020_GSE140231"
        combined.uns["tissue"] = "substantia_nigra"
        combined.uns["sn_only_filtered"] = True

    print(f"[4/4] AnnData built: {combined}")

    # --- Validation asserts (relaxed for control-only nature of this dataset) ---
    n_sn_samples = combined.obs["gsm_id"].nunique()
    n_donors = combined.obs["donor_id"].nunique()
    conditions = set(combined.obs["condition"].unique())
    assert n_sn_samples >= 4, f"expected >=4 SN samples, got {n_sn_samples}"
    assert n_donors >= 3, f"expected >=3 donors, got {n_donors}"
    # Note: GSE140231 is a control-only SN atlas. The "both conditions present"
    # assertion from the task spec was written assuming PD/Ctrl mixing  -  that
    # assumption is incorrect for this dataset (verified against per-GSM GEO
    # metadata). We assert that exactly one condition (Control) is present.
    assert conditions == {"Control"}, f"expected only Control, got {conditions}"
    assert (combined.obs["region"] == "SN").all(), "non-SN cells leaked through filter"

    print("\nSummary:")
    print(f"  shape: {combined.shape}")
    print(f"  SN samples retained: {n_sn_samples}")
    print(f"  donors: {n_donors} ({sorted(combined.obs['donor_id'].unique())})")
    print(f"  conditions: {conditions}")
    print("\nCells per sample:")
    print(combined.obs.groupby(["donor_id", "sample_name", "batch"], observed=True).size())

    print(f"\nWriting {out_path}")
    combined.write_h5ad(out_path, compression="gzip")
    size_mb = out_path.stat().st_size / 1024**2
    print(f"-> Wrote {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
