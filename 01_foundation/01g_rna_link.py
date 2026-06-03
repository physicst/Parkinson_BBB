"""Load PPMI RNA-seq metadata IR3 to link RNA samples → (PATNO, EVENT_ID).

Output schema (one row per RNA-seq sample):
  PATNO (str), EVENT_ID (str), RNA_HudAlphaID (str),
  RNA_QCflag (str: 'pass'/'fail'), RNA_RIN (float),
  RNA_uniquely_mapped_pct (float), RNA_total_reads (Int64).
"""
from __future__ import annotations
import pandas as pd

from lib.config import load_config
from lib.ppmi_ids import normalize_patno


def load_rna_link() -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_csv(cfg["rna_seq"]["metadata"])

    out = pd.DataFrame({
        "PATNO": normalize_patno(df["PATNO"]),
        "EVENT_ID": df["CLINICAL_EVENT"].astype(str).str.strip(),
        "RNA_HudAlphaID": df["HudAlphaID"].astype(str).str.strip(),
        "RNA_QCflag": df["QCflagIR3"].astype(str).str.strip().str.lower(),
        "RNA_RIN": pd.to_numeric(df["RIN Value"], errors="coerce"),
        "RNA_uniquely_mapped_pct": pd.to_numeric(df["uniquely_mapped_percent"], errors="coerce"),
        "RNA_total_reads": pd.to_numeric(df["total_reads"], errors="coerce").astype("Int64"),
    })

    # If a patient has multiple RNA samples for same EVENT_ID, prefer pass + highest mapping
    out["_rank"] = (
        (out["RNA_QCflag"] == "pass").astype(int) * 1000
        + out["RNA_uniquely_mapped_pct"].fillna(0)
    )
    out = (
        out.sort_values(["PATNO", "EVENT_ID", "_rank"], ascending=[True, True, False])
           .drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")
           .drop(columns="_rank")
           .reset_index(drop=True)
    )

    assert out["PATNO"].notna().all()
    return out


if __name__ == "__main__":
    df = load_rna_link()
    print(f"RNA samples (after dedup): {len(df)}")
    print(f"  pass: {(df['RNA_QCflag'] == 'pass').sum()}")
    print(f"  fail: {(df['RNA_QCflag'] == 'fail').sum()}")
    print(f"Unique patients: {df['PATNO'].nunique()}")
    print(f"Unique EVENT_IDs: {df['EVENT_ID'].nunique()}")
