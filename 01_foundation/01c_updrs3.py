"""Load PPMI MDS-UPDRS Part III; compute UPDRS3 total per visit.

PPMI convention: when both ON and OFF rows exist for a visit, prefer OFF
(more sensitive to disease state). PDSTATE column: 'OFF', 'ON', or blank.

NOTE on UPDRS3_Total: PPMI's MDS-UPDRS Part III CSV DOES include a
precomputed total column `NP3TOT`. PPMI also uses `101` as a sentinel for
"untestable due to medical reason" in individual items. Naively summing
all NP3* columns therefore (a) double-counts (items + NP3TOT) and (b)
inflates by thousands when a sentinel is present. Use `NP3TOT` directly
- it is PPMI's official total, set to NaN by PPMI when sentinel values
are present (~13% of rows are NaN'd). Visits where NP3TOT is missing
are dropped.

Output schema:
  PATNO (str), EVENT_ID (str), INFODT (datetime), UPDRS3_Total (float),
  UPDRS3_State (str: 'OFF'/'ON'/'NA'), PDTRTMNT (Int64).
"""
from __future__ import annotations
import pandas as pd

from lib.config import load_config
from lib.ppmi_ids import normalize_patno, parse_ppmi_date


def load_updrs3() -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_csv(cfg["clinical"]["updrs3"], low_memory=False)

    out = pd.DataFrame({
        "PATNO": normalize_patno(df["PATNO"]),
        "EVENT_ID": df["EVENT_ID"].astype(str).str.strip(),
        "INFODT": parse_ppmi_date(df["INFODT"]),
        "UPDRS3_Total": pd.to_numeric(df["NP3TOT"], errors="coerce"),
        "UPDRS3_State": df.get("PDSTATE", pd.Series([""] * len(df))).fillna("NA").astype(str),
        "PDTRTMNT": pd.to_numeric(df.get("PDTRTMNT"), errors="coerce").astype("Int64"),
    })

    # Drop rows with no UPDRS3 total
    out = out.dropna(subset=["UPDRS3_Total"])

    # When both OFF and ON rows exist for same (PATNO, EVENT_ID), prefer OFF
    state_priority = {"OFF": 0, "NA": 1, "": 1, "ON": 2}
    out["_priority"] = out["UPDRS3_State"].map(state_priority).fillna(3)
    out = (
        out.sort_values(["PATNO", "EVENT_ID", "_priority"])
           .drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")
           .drop(columns="_priority")
           .reset_index(drop=True)
    )

    assert (out["UPDRS3_Total"] >= 0).all(), "Negative UPDRS3 total"
    assert (out["UPDRS3_Total"] <= 132).all(), "UPDRS3 total > theoretical max 132"
    assert out["PATNO"].notna().all(), "UPDRS3 has null PATNO"
    return out


if __name__ == "__main__":
    df = load_updrs3()
    print(f"UPDRS3 visits: {len(df)}")
    print(f"Unique patients: {df['PATNO'].nunique()}")
    print(f"\nState distribution:\n{df['UPDRS3_State'].value_counts(dropna=False)}")
    print(f"\nUPDRS3_Total stats:\n{df['UPDRS3_Total'].describe()}")
