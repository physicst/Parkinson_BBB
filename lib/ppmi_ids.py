"""Helpers for normalizing PPMI IDs and dates across files."""
from __future__ import annotations
import pandas as pd


def normalize_patno(s: pd.Series) -> pd.Series:
    """Strip 'PPMI-' prefix, cast to string, strip whitespace.
    Olink files use 'PPMI-3004', clinical files use '3004'."""
    return (
        s.astype(str)
         .str.strip()
         .str.removeprefix("PPMI-")
         .str.strip()
    )


def parse_ppmi_date(s: pd.Series) -> pd.Series:
    """PPMI INFODT and BIRTHDT are typically MM/YYYY (e.g., '02/2011').
    Returns pandas datetime at first day of the given month, NaT for blanks."""
    return pd.to_datetime(s.astype(str).str.strip(), format="%m/%Y", errors="coerce")
