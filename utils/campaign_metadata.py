"""Campaign metadata helpers (keyword-day index for summary building)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from utils.data_processing import _clean_campaign, _clean_match_type, clean_keyword_series as _clean_keyword


def read_keyword_day_index(path: Path) -> dict[str, list[tuple[str, str, str]]]:
    """Return keyword-day tuples by campaign from a kw-day-panel CSV."""
    df = pd.read_csv(path)
    required = {"date", "keyword", "campaign", "match_type"}
    missing = required - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"kw-day-panel is missing required column(s): {missing_cols}")

    df = df.copy()
    df["campaign"] = _clean_campaign(df["campaign"])
    df["keyword"] = _clean_keyword(df["keyword"])
    df["match_type"] = _clean_match_type(df["match_type"])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    keywords_by_campaign: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for row in df.itertuples(index=False):
        campaign = str(row.campaign) if pd.notna(row.campaign) else ""
        day = str(row.date) if pd.notna(row.date) else ""
        keyword = str(row.keyword) if pd.notna(row.keyword) else ""
        match_type = str(row.match_type) if pd.notna(row.match_type) else ""
        if campaign and day and keyword and match_type:
            keywords_by_campaign[campaign].append((day, keyword, match_type))

    return keywords_by_campaign
