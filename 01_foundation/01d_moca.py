"""Load PPMI Montreal Cognitive Assessment.

Output schema:
  PATNO (str), EVENT_ID (str), INFODT (datetime), MoCA_Total (Int64).
"""
from __future__ import annotations
import pandas as pd

from lib.config import load_config
from lib.ppmi_ids import normalize_patno, parse_ppmi_date


def load_moca() -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_csv(cfg["clinical"]["moca"])

    out = pd.DataFrame({
        "PATNO": normalize_patno(df["PATNO"]),
        "EVENT_ID": df["EVENT_ID"].astype(str).str.strip(),
        "INFODT": parse_ppmi_date(df["INFODT"]),
        "MoCA_Total": pd.to_numeric(df["MCATOT"], errors="coerce").astype("Int64"),
    })
    out = out.dropna(subset=["MoCA_Total"])
    out = out.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="last").reset_index(drop=True)

    assert (out["MoCA_Total"] >= 0).all() and (out["MoCA_Total"] <= 30).all(), \
        "MoCA out of range [0,30]"
    assert out["PATNO"].notna().all(), "MoCA has null PATNO"
    return out


if __name__ == "__main__":
    df = load_moca()
    print(f"MoCA rows: {len(df)}")
    print(f"Unique patients: {df['PATNO'].nunique()}")
    print(f"\nMoCA_Total stats:\n{df['MoCA_Total'].describe()}")
