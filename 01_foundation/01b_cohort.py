"""Load PPMI Subject Cohort History; decode COHORT codes.

Output schema:
  PATNO (str), Cohort_at_Enrollment (str), Cohort_Current (str).
"""
from __future__ import annotations
import io
import zipfile
from pathlib import Path
import pandas as pd

from lib.config import load_config
from lib.ppmi_ids import normalize_patno

# PPMI cohort coding (per PPMI Data Dictionary; APPRDX = enrollment, COHORT = current).
# Reference: https://www.ppmi-info.org/study-design  (Data Dictionary download)
COHORT_MAP = {
    1: "PD",
    2: "HC",
    3: "SWEDD",
    4: "Prodromal",
    5: "Genetic_Cohort_PD",
    6: "Genetic_Cohort_Unaffected",
    7: "Genetic_Registry_PD",
    8: "Genetic_Registry_Unaffected",
}


def decode_cohort(s: pd.Series) -> pd.Series:
    """Map PPMI COHORT integer codes to human-readable strings.
    Unknown codes become 'Unknown'."""
    s_int = pd.to_numeric(s, errors="coerce").astype("Int64")
    return s_int.map(COHORT_MAP).fillna("Unknown").astype(str)


def load_cohort() -> pd.DataFrame:
    cfg = load_config()
    zip_path = Path(cfg["clinical"]["demographics_zip"])
    member = cfg["clinical"]["cohort_csv_in_zip"]
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"))

    out = pd.DataFrame({
        "PATNO": normalize_patno(df["PATNO"]),
        "Cohort_at_Enrollment": decode_cohort(df["APPRDX"]),
        "Cohort_Current": decode_cohort(df["COHORT"]),
    })

    # Subject_Cohort_History rows have no INFODT, but PPMI emits them in
    # chronological append order, so keep="last" selects the most recent
    # classification per subject.
    out = out.drop_duplicates(subset="PATNO", keep="last")
    assert out["PATNO"].is_unique, "Cohort PATNO not unique"
    assert out["PATNO"].notna().all(), "Cohort has null PATNO"
    return out.reset_index(drop=True)


if __name__ == "__main__":
    df = load_cohort()
    print(f"Cohort rows: {len(df)}")
    print(f"\nCurrent cohort distribution:\n{df['Cohort_Current'].value_counts(dropna=False)}")
    print(f"\nEnrollment cohort distribution:\n{df['Cohort_at_Enrollment'].value_counts(dropna=False)}")
