"""Decision variables and constraints for campaign MILP.

Segment = ``region / match_types``. Candidates come from ``segment-keyword-candidates.csv``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from campaign_opt.schema import CampaignOptConfig


def historical_budget_bounds(
    panel: pd.DataFrame,
    segments: list[str],
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for seg in segments:
        sub = panel[panel["segment"] == seg]["daily_budget"].dropna()
        if sub.empty:
            bounds[seg] = (0.0, 500.0)
        else:
            bounds[seg] = (float(sub.min()), float(sub.max()))
    return bounds


def build_segment_list(candidates: pd.DataFrame) -> list[str]:
    return sorted(candidates["segment"].unique().tolist())


def candidates_by_segment(candidates: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for seg, grp in candidates.groupby("segment"):
        out[str(seg)] = grp["keyword_set_id"].astype(str).tolist()
    return out


def parse_regional_order(constraints: dict[str, Any]) -> list[str]:
    return list(constraints.get("regional_order") or [])


def region_of_segment(segment: str) -> str:
    return segment.split(" / ")[0].strip()
