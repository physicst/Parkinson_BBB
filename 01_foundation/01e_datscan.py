"""Load PPMI DaT-SCAN Quantitative SBR (Xing Core Lab).

Output schema:
  PATNO (str), EVENT_ID (str), DATSCAN_DATE (datetime),
  Striatum_R, Caudate_R, Putamen_R,
  Striatum_L, Caudate_L, Putamen_L (all float, ref CWM normalized).
  DATSCAN_Putamen_Worst (float)  -  min(Putamen_R, Putamen_L), the standard
  PPMI summary used in cross-cohort comparison.
"""
from __future__ import annotations
import pandas as pd

from lib.config import load_config
from lib.ppmi_ids import normalize_patno


def load_datscan() -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_csv(cfg["clinical"]["datscan_quant"])

    # Only keep analyzed scans
    df = df[df["DATSCAN_ANALYZED"].astype(str).str.strip().str.lower() == "yes"]

    out = pd.DataFrame({
        "PATNO": normalize_patno(df["PATNO"]),
        "EVENT_ID": df["EVENT_ID"].astype(str).str.strip(),
        "DATSCAN_DATE": pd.to_datetime(df["DATSCAN_DATE"], format="%m/%Y", errors="coerce"),
        "Striatum_R": pd.to_numeric(df["STRIATUM_REF_CWM"], errors="coerce"),
        "Caudate_R": pd.to_numeric(df["CAUDATE_REF_CWM"], errors="coerce"),
        "Putamen_R": pd.to_numeric(df["PUTAMEN_REF_CWM"], errors="coerce"),
        "Striatum_L": pd.to_numeric(df["STRIATUM_L_REF_CWM"], errors="coerce"),
        "Caudate_L": pd.to_numeric(df["CAUDATE_L_REF_CWM"], errors="coerce"),
        "Putamen_L": pd.to_numeric(df["PUTAMEN_L_REF_CWM"], errors="coerce"),
    })

    out["DATSCAN_Putamen_Worst"] = out[["Putamen_R", "Putamen_L"]].min(axis=1)

    out = out.dropna(subset=["Putamen_R", "Putamen_L"])
    # A handful of PPMI scans (~0.03%) report negative reference-CWM values,
    # which are physically implausible artifacts (severe-loss scans below
    # reference noise). Treat as failed QC and drop.
    out = out[out["DATSCAN_Putamen_Worst"] > 0]
    out = out.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="last").reset_index(drop=True)

    # Sanity: SBR ratios are typically 0.3 (severe loss) to 3.5 (healthy)
    assert (out["DATSCAN_Putamen_Worst"] > 0).all(), "Non-positive SBR"
    assert (out["DATSCAN_Putamen_Worst"] < 10).all(), "SBR > 10 implausible"
    assert out["PATNO"].notna().all(), "DaT-SCAN has null PATNO"
    return out


if __name__ == "__main__":
    df = load_datscan()
    print(f"DaT-SCAN rows: {len(df)}")
    print(f"Unique patients: {df['PATNO'].nunique()}")
    print(f"\nPutamen_Worst stats:\n{df['DATSCAN_Putamen_Worst'].describe()}")
