"""Load PPMI Demographics + Cohort History; emit a baseline-per-PATNO frame.

Output schema (returned by load_demographics):
  PATNO (str), Sex (str: 'Male'/'Female'/'Unknown'),
  BIRTHDT (datetime), Hispanic (0/1), Race_White, Race_Black, Race_Asian,
  Race_Hawaiian_PI, Race_Native, Race_Other, Race_Unknown (each 0/1).
"""
from __future__ import annotations
import io
import zipfile
from pathlib import Path
import pandas as pd

from lib.config import load_config
from lib.ppmi_ids import normalize_patno, parse_ppmi_date

# PPMI Demographics SEX coding (per PPMI Data Dictionary):
#   0 = Female, 1 = Male
SEX_MAP = {0: "Female", 1: "Male", "0": "Female", "1": "Male"}


def _read_csv_from_zip(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            return pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"))


def load_demographics() -> pd.DataFrame:
    cfg = load_config()
    zip_path = Path(cfg["clinical"]["demographics_zip"])
    member = cfg["clinical"]["demographics_csv_in_zip"]
    df = _read_csv_from_zip(zip_path, member)

    # Many subjects have multiple SCREEN/TRANS rows; keep the earliest visit per PATNO.
    # Parse INFODT to datetime so sort is chronological (raw strings sort lexicographically,
    # which would put "02/2011" before "12/2010"). NaT sorts to end, so a missing INFODT
    # yields the second-row pick when the other row has a real date.
    df = df.assign(_infodt=parse_ppmi_date(df["INFODT"]))
    df = (
        df.sort_values(["PATNO", "_infodt"])
          .drop_duplicates(subset="PATNO", keep="first")
          .drop(columns="_infodt")
    )

    out = pd.DataFrame({
        "PATNO": normalize_patno(df["PATNO"]),
        "Sex": df["SEX"].map(SEX_MAP).fillna("Unknown"),
        "BIRTHDT": parse_ppmi_date(df["BIRTHDT"]),
        "Hispanic": df["HISPLAT"].fillna(0).astype(int),
        "Race_White": df["RAWHITE"].fillna(0).astype(int),
        "Race_Black": df["RABLACK"].fillna(0).astype(int),
        "Race_Asian": df["RAASIAN"].fillna(0).astype(int),
        "Race_Hawaiian_PI": df["RAHAWOPI"].fillna(0).astype(int),
        "Race_Native": df["RAINDALS"].fillna(0).astype(int),
        "Race_Other": df["RANOS"].fillna(0).astype(int),
        "Race_Unknown": df["RAUNKNOWN"].fillna(0).astype(int),
    })

    assert out["PATNO"].is_unique, "Demographics PATNO not unique after dedup"
    assert out["PATNO"].notna().all(), "Demographics has null PATNO"
    return out.reset_index(drop=True)


if __name__ == "__main__":
    df = load_demographics()
    print(f"Demographics rows: {len(df)}")
    print(df.head())
    print(f"\nSex distribution:\n{df['Sex'].value_counts(dropna=False)}")
