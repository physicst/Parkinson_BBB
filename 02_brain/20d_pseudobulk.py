"""Step 2 Task 26: Donor-level pseudobulk per (dataset, cell_type).

Per Step 2 design (docs/2026-05-03-pd-bbb-step2-design.md §3.3) and master
plan §1.2, build donor-level pseudobulk matrices for each (dataset, cell_type)
combo from the vascular subsets produced in Task 25.

Inputs
------
- D:/Parkinson_file/snrnaseq/cache/smajic_vascular.h5ad     (2,952 cells)
- D:/Parkinson_file/snrnaseq/cache/agarwal_vascular.h5ad    (265 cells)
- D:/Parkinson_file/snrnaseq/cache/gse178265_vascular.h5ad  (14,903 endo only)

Outputs (D:/PD_BBB_analysis/results/cache/)
-------------------------------------------
- pseudobulk_smajic_endothelial.parquet
- pseudobulk_smajic_pericyte.parquet
- pseudobulk_agarwal_endothelial.parquet
- pseudobulk_agarwal_pericyte.parquet
- pseudobulk_gse178265_endothelial.parquet
- pseudobulk_metadata.parquet

Each pseudobulk parquet has:
  index   = gene_id (Ensembl, version-stripped  -  already the var index)
  columns = donor_id (string)
  values  = int32 sum of raw counts across cells in (donor, cell_type) group

The metadata parquet has columns:
  dataset, donor_id, cell_type, condition, sex, age, pmi, n_cells_aggregated
keyed by (dataset, donor_id, cell_type).

QC rule (master plan §1.2): drop donor x cell_type combos with < 30 cells.

Pitfalls handled
----------------
- Per-dataset gene index is preserved as-is; no cross-dataset alignment yet
  (KISTI integration step does that).
- Sparse -> dense per-group: np.asarray(X[group_mask].sum(axis=0)).ravel()
  (avoids materializing the full matrix as dense).
- Float32 raw counts (Agarwal, GSE178265) are integer-valued; we cast the
  per-donor sum to int32 after aggregation.
- Different obs schemas: defensive .get with "Unknown"/NaN fallbacks.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
CACHE_IN = Path("D:/Parkinson_file/snrnaseq/cache")
CACHE_OUT = Path("D:/PD_BBB_analysis/results/cache")

MIN_CELLS_PER_DONOR_CELLTYPE = 30  # master-plan §1.2

# Map source disease/condition strings to canonical labels we keep.
# We preserve whatever is there (no aggressive renaming) but normalize a few
# common synonyms so the metadata table is consistent.
SMAJIC_CONDITION_MAP = {"IPD": "PD", "Control": "Control"}
GSE_DISEASE_MAP = {
    "normal": "Control",
    "Parkinson disease": "PD",
    "Lewy body dementia": "LBD",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_section(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def _normalize_sex(val) -> str:
    """Map various sex encodings to {'M', 'F', 'Unknown'}."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "Unknown"
    s = str(val).strip().lower()
    if s in ("m", "male"):
        return "M"
    if s in ("f", "female"):
        return "F"
    return "Unknown"


def _parse_dev_stage_age(val) -> float:
    """Parse age in years from strings like '79-year-old stage'. NaN if unknown."""
    if val is None:
        return float("nan")
    s = str(val).strip()
    if not s:
        return float("nan")
    # Take leading integer
    head = s.split("-", 1)[0]
    try:
        return float(head)
    except ValueError:
        return float("nan")


def _to_float(val) -> float:
    if val is None:
        return float("nan")
    try:
        f = float(val)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _aggregate_donor_celltype(
    adata: ad.AnnData,
    donor_col: str,
    celltype_col: str,
    celltype_value: str,
) -> tuple[pd.DataFrame, dict[str, int], list[tuple[str, int]]]:
    """Sum raw counts across cells per donor for a single cell_type.

    Returns
    -------
    pseudobulk : DataFrame
        Index = gene_id (var.index); columns = donor_id (str); int32 counts.
        Only donors with >= MIN_CELLS_PER_DONOR_CELLTYPE cells are kept.
    kept : dict[str, int]
        donor_id -> n_cells_aggregated for kept donors.
    dropped : list[tuple[str, int]]
        (donor_id, n_cells) for donors below threshold.
    """
    mask_ct = (adata.obs[celltype_col].astype(str) == celltype_value).values
    if mask_ct.sum() == 0:
        return (pd.DataFrame(index=adata.var.index.astype(str)),
                {}, [])

    sub = adata[mask_ct]
    donors = sub.obs[donor_col].astype(str).values
    X = sub.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    else:
        X = X.tocsr()

    gene_index = adata.var.index.astype(str)

    kept: dict[str, int] = {}
    dropped: list[tuple[str, int]] = []
    cols: dict[str, np.ndarray] = {}

    for donor in pd.unique(donors):
        rows = np.where(donors == donor)[0]
        n = int(rows.size)
        if n < MIN_CELLS_PER_DONOR_CELLTYPE:
            dropped.append((str(donor), n))
            continue
        # Sparse-friendly column sum: select rows then sum axis=0.
        col_sum = np.asarray(X[rows, :].sum(axis=0)).ravel()
        # Values are integer-valued raw counts; cast safely.
        col_int = np.rint(col_sum).astype(np.int32)
        cols[str(donor)] = col_int
        kept[str(donor)] = n

    if cols:
        # Stable column order: sort donor_id alphabetically for reproducibility.
        ordered = sorted(cols.keys())
        mat = np.column_stack([cols[d] for d in ordered])
        df = pd.DataFrame(mat, index=gene_index, columns=ordered, dtype=np.int32)
    else:
        df = pd.DataFrame(index=gene_index)
    df.index.name = "gene_id"
    return df, kept, dropped


# ---------------------------------------------------------------------------
# Per-dataset metadata extraction
# ---------------------------------------------------------------------------
def _meta_smajic(adata: ad.AnnData, donor: str) -> dict:
    """Extract per-donor metadata for Smajic. condition only; sex/age/pmi N/A."""
    obs = adata.obs[adata.obs["donor_id"].astype(str) == donor]
    if len(obs) == 0:
        return {"condition": "Unknown", "sex": "Unknown",
                "age": float("nan"), "pmi": float("nan")}
    cond_raw = str(obs["condition"].iloc[0]) if "condition" in obs.columns else "Unknown"
    cond = SMAJIC_CONDITION_MAP.get(cond_raw, cond_raw)
    return {
        "condition": cond,
        "sex": _normalize_sex(obs["sex"].iloc[0]) if "sex" in obs.columns else "Unknown",
        "age": _to_float(obs["age"].iloc[0]) if "age" in obs.columns else float("nan"),
        "pmi": _to_float(obs["pmi"].iloc[0]) if "pmi" in obs.columns else float("nan"),
    }


def _meta_agarwal(adata: ad.AnnData, donor: str) -> dict:
    obs = adata.obs[adata.obs["donor_id"].astype(str) == donor]
    if len(obs) == 0:
        return {"condition": "Control", "sex": "Unknown",
                "age": float("nan"), "pmi": float("nan")}
    return {
        "condition": str(obs["condition"].iloc[0]) if "condition" in obs.columns else "Control",
        "sex": _normalize_sex(obs["sex"].iloc[0]) if "sex" in obs.columns else "Unknown",
        "age": _to_float(obs["age"].iloc[0]) if "age" in obs.columns else float("nan"),
        "pmi": _to_float(obs["pmi"].iloc[0]) if "pmi" in obs.columns else float("nan"),
    }


def _meta_gse178265(adata: ad.AnnData, donor: str) -> dict:
    obs = adata.obs[adata.obs["donor_id"].astype(str) == donor]
    if len(obs) == 0:
        return {"condition": "Unknown", "sex": "Unknown",
                "age": float("nan"), "pmi": float("nan")}
    disease_raw = str(obs["disease"].iloc[0]) if "disease" in obs.columns else "Unknown"
    cond = GSE_DISEASE_MAP.get(disease_raw, disease_raw)
    age = _parse_dev_stage_age(obs["development_stage"].iloc[0]) \
        if "development_stage" in obs.columns else float("nan")
    pmi = _to_float(obs["author_Donor_PMI"].iloc[0]) \
        if "author_Donor_PMI" in obs.columns else float("nan")
    return {
        "condition": cond,
        "sex": _normalize_sex(obs["sex"].iloc[0]) if "sex" in obs.columns else "Unknown",
        "age": age,
        "pmi": pmi,
    }


# ---------------------------------------------------------------------------
# Per-dataset orchestration
# ---------------------------------------------------------------------------
DATASET_PLAN = [
    # (dataset_name, h5ad_path, donor_col, celltype_col, celltypes, meta_fn)
    ("smajic",
     CACHE_IN / "smajic_vascular.h5ad",
     "donor_id", "vascular_class",
     ["endothelial", "pericyte"],
     _meta_smajic),
    ("agarwal",
     CACHE_IN / "agarwal_vascular.h5ad",
     "donor_id", "vascular_class",
     ["endothelial", "pericyte"],
     _meta_agarwal),
    ("gse178265",
     CACHE_IN / "gse178265_vascular.h5ad",
     "donor_id", "vascular_class",
     ["endothelial"],  # pericytes deferred to KISTI
     _meta_gse178265),
]


def process_dataset(
    name: str,
    path: Path,
    donor_col: str,
    celltype_col: str,
    celltypes: list[str],
    meta_fn,
) -> tuple[dict[str, Path], list[dict], dict]:
    """Process one h5ad. Returns (parquet_paths, metadata_rows, drop_summary)."""
    _print_section(f"[{name}] {path}")
    adata = ad.read_h5ad(path)
    print(f"  shape: {adata.shape}")
    print(f"  vascular_class counts: "
          f"{adata.obs[celltype_col].value_counts().to_dict()}")

    parquet_paths: dict[str, Path] = {}
    metadata_rows: list[dict] = []
    drop_summary: dict[str, list[tuple[str, int]]] = {}

    for ct in celltypes:
        df, kept, dropped = _aggregate_donor_celltype(
            adata, donor_col, celltype_col, ct
        )
        out = CACHE_OUT / f"pseudobulk_{name}_{ct}.parquet"
        df.to_parquet(out)
        size_kb = out.stat().st_size / 1024
        parquet_paths[ct] = out
        drop_summary[ct] = dropped
        print(f"  -> {out.name}: {df.shape[0]} genes x {df.shape[1]} donors "
              f"({size_kb:.1f} KB)")
        if kept:
            print(f"     kept donors (n_cells): "
                  f"{ {d: kept[d] for d in sorted(kept)} }")
        if dropped:
            print(f"     dropped (<{MIN_CELLS_PER_DONOR_CELLTYPE} cells): "
                  f"{sorted(dropped)}")

        # Build metadata rows for kept donors only.
        for donor, n_cells in sorted(kept.items()):
            md = meta_fn(adata, donor)
            metadata_rows.append({
                "dataset": name,
                "donor_id": donor,
                "cell_type": ct,
                "condition": md["condition"],
                "sex": md["sex"],
                "age": md["age"],
                "pmi": md["pmi"],
                "n_cells_aggregated": int(n_cells),
            })

    return parquet_paths, metadata_rows, drop_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    CACHE_OUT.mkdir(parents=True, exist_ok=True)

    all_metadata: list[dict] = []
    all_parquets: list[Path] = []
    all_drops: dict[tuple[str, str], list[tuple[str, int]]] = {}

    for name, path, donor_col, ct_col, cts, meta_fn in DATASET_PLAN:
        paths, rows, drops = process_dataset(
            name, path, donor_col, ct_col, cts, meta_fn
        )
        all_metadata.extend(rows)
        all_parquets.extend(paths.values())
        for ct, dropped in drops.items():
            all_drops[(name, ct)] = dropped

    # Combined metadata table.
    meta_df = pd.DataFrame(all_metadata, columns=[
        "dataset", "donor_id", "cell_type", "condition",
        "sex", "age", "pmi", "n_cells_aggregated",
    ])
    meta_path = CACHE_OUT / "pseudobulk_metadata.parquet"
    meta_df.to_parquet(meta_path, index=False)

    # ---- Summary ----------------------------------------------------------
    _print_section("Summary: parquet files")
    for p in all_parquets + [meta_path]:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:50s}  {size_kb:8.1f} KB")

    _print_section("Summary: rows per pseudobulk parquet")
    for p in all_parquets:
        df = pd.read_parquet(p)
        print(f"  {p.name:50s}  genes={df.shape[0]:6d}  donors={df.shape[1]:3d}  "
              f"donor_ids={list(df.columns)}")

    _print_section("Summary: donors dropped (<30 cells)")
    any_dropped = False
    for (ds, ct), dropped in sorted(all_drops.items()):
        if dropped:
            any_dropped = True
            print(f"  {ds}/{ct}: {sorted(dropped)}")
    if not any_dropped:
        print("  (none)")

    _print_section("Summary: metadata table")
    print(f"  rows: {len(meta_df)}")
    print(f"  columns: {list(meta_df.columns)}")
    print(meta_df.to_string(index=False))

    _print_section("Done")


if __name__ == "__main__":
    main()
