"""Load Google Keyword Planner stats and aggregate to keyword-set level."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_processing import clean_keyword_series as _clean_keyword


def load_gkp_keyword_stats(gkp_dir: str | Path) -> pd.DataFrame:
    """Load the newest Saved Keywords Stats CSV under gkp_dir."""
    gkp_path = Path(gkp_dir)
    if not gkp_path.exists():
        return pd.DataFrame()

    files = list(gkp_path.glob("Saved Keyword* Stats*.csv"))
    if not files:
        return pd.DataFrame()

    gkp_file = max(files, key=lambda f: f.stat().st_mtime)
    for encoding in ("utf-8", "utf-16"):
        try:
            gkp_df = pd.read_csv(gkp_file, sep="\t", encoding=encoding)
            break
        except Exception:
            gkp_df = None
    if gkp_df is None:
        gkp_df = pd.read_csv(gkp_file)

    gkp_df.columns = gkp_df.columns.astype(str).str.strip()
    if "Keyword" not in gkp_df.columns:
        return pd.DataFrame()

    gkp_df = gkp_df.rename(columns={"Keyword": "keyword"})
    gkp_df["keyword"] = _clean_keyword(gkp_df["keyword"])

    search_cols = [c for c in gkp_df.columns if c.startswith("Searches:")]
    if search_cols:
        last_col = sorted(
            search_cols,
            key=lambda h: (
                int(re.search(r"(\d{4})$", h.replace("Searches: ", "")).group(1))
                if re.search(r"(\d{4})$", h.replace("Searches: ", ""))
                else 0
            ),
        )[-1]
        gkp_df["last_month_searches"] = pd.to_numeric(
            gkp_df[last_col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    else:
        gkp_df["last_month_searches"] = np.nan

    comp_col = "Competition (indexed value)"
    if comp_col in gkp_df.columns:
        gkp_df["competition_index"] = pd.to_numeric(
            gkp_df[comp_col].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        )
    else:
        gkp_df["competition_index"] = np.nan

    low_col = "Top of page bid (low range)"
    if low_col in gkp_df.columns:
        gkp_df["bid_low"] = pd.to_numeric(
            gkp_df[low_col].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )
    else:
        gkp_df["bid_low"] = np.nan

    return gkp_df[
        ["keyword", "last_month_searches", "competition_index", "bid_low"]
    ].drop_duplicates(subset=["keyword"])


_GKP_BASE_COLS = ("last_month_searches", "competition_index", "bid_low")
_GKP_AGG_SUFFIXES = ("mean", "std", "p90")


def gkp_aggregate_column_names() -> list[str]:
    """Set-level GKP columns: mean, std, and p90 for each base stat."""
    return [f"{base}_{suffix}" for base in _GKP_BASE_COLS for suffix in _GKP_AGG_SUFFIXES]


def _gkp_aggregate_row(sub: pd.DataFrame) -> dict[str, float]:
    """Mean, std, and p90 for each GKP stat over keyword rows in ``sub``."""
    nan = float("nan")
    out: dict[str, float] = {}
    for base in _GKP_BASE_COLS:
        vals = pd.to_numeric(sub[base], errors="coerce").dropna()
        out[f"{base}_mean"] = float(vals.mean()) if len(vals) else nan
        out[f"{base}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else nan
        out[f"{base}_p90"] = float(np.percentile(vals, 90)) if len(vals) else nan
    return out


def aggregate_gkp_to_keyword_sets(
    keyword_sets: pd.DataFrame,
    gkp_kw: pd.DataFrame,
    *,
    positive_col: str = "positive_keywords",
) -> pd.DataFrame:
    """
    keyword_sets: keyword_set_id, positive_keywords (semicolon-separated).
    Returns one row per keyword_set_id with mean/std/p90 GKP aggregates.
    """
    empty_stats = {col: np.nan for col in gkp_aggregate_column_names()}
    if gkp_kw.empty:
        base = keyword_sets[["keyword_set_id"]].drop_duplicates().copy()
        for col, val in empty_stats.items():
            base[col] = val
        return base

    gkp_map = gkp_kw.set_index("keyword")
    rows = []
    for _, row in keyword_sets.iterrows():
        set_id = row["keyword_set_id"]
        raw = row.get(positive_col, "")
        if pd.isna(raw) or not str(raw).strip():
            keywords: list[str] = []
        else:
            keywords = [k.strip().lower() for k in str(raw).split(";") if k.strip()]

        if not keywords:
            rows.append({"keyword_set_id": set_id, **empty_stats})
            continue

        sub = gkp_map.reindex(keywords)
        rows.append({"keyword_set_id": set_id, **_gkp_aggregate_row(sub)})
    return pd.DataFrame(rows)
