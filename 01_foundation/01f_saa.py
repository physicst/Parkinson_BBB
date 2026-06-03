"""Load PPMI CSF α-synuclein Seed Amplification Assay results.

SAA file uses CLINICAL_EVENT (not EVENT_ID); rename for consistency.

Output schema:
  PATNO (str), EVENT_ID (str), SAA_Status (str: 'Positive'/'Negative'/'Indeterminate'/'Unknown'),
  SAA_Type (str), TYPE (str: typically 'Cerebrospinal Fluid').
"""
from __future__ import annotations
import pandas as pd

from lib.config import load_config
from lib.ppmi_ids import normalize_patno


def load_saa() -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_csv(cfg["clinical"]["saa"])

    # PPMI uses 'Inconclusive'; map to canonical 'Indeterminate' per output schema.
    status = df["SAA_Status"].fillna("Unknown").astype(str).replace({"Inconclusive": "Indeterminate"})

    out = pd.DataFrame({
        "PATNO": normalize_patno(df["PATNO"]),
        "EVENT_ID": df["CLINICAL_EVENT"].astype(str).str.strip(),
        "SAA_Status": status,
        "SAA_Type": df["SAA_Type"].fillna("").astype(str),
        "Sample_Type": df["TYPE"].fillna("").astype(str),
    })
    out = out.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="last").reset_index(drop=True)

    valid_status = {"Positive", "Negative", "Indeterminate", "Unknown"}
    bad = set(out["SAA_Status"].unique()) - valid_status
    assert not bad, f"Unexpected SAA_Status values: {bad}"
    assert out["PATNO"].notna().all(), "SAA has null PATNO"

    return out


if __name__ == "__main__":
    df = load_saa()
    print(f"SAA rows: {len(df)}")
    print(f"Unique patients: {df['PATNO'].nunique()}")
    print(f"\nSAA_Status distribution:\n{df['SAA_Status'].value_counts(dropna=False)}")
