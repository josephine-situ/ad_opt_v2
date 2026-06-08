"""Decision variables and constraints for campaign MILP.

Segment = ``region / match_types``. Candidates come from ``segment-keyword-candidates.csv``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def panel_before_date(
    panel: pd.DataFrame,
    before: pd.Timestamp,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Rows strictly before ``before`` (walk-forward slice for optimizer gating)."""
    cutoff = pd.Timestamp(before).normalize()
    dates = pd.to_datetime(panel[date_col])
    return panel.loc[dates < cutoff].copy()


def optimizer_gating_panel(
    panel: pd.DataFrame,
    planning_date: pd.Timestamp,
) -> pd.DataFrame:
    """Campaign panel slice for observed-min budget floors in the MILP (no future leakage)."""
    return panel_before_date(panel, planning_date)


def observed_min_daily_budget(
    panel: pd.DataFrame,
    segments: list[str],
) -> dict[str, float]:
    """
    Smallest historical ``daily_budget`` per segment (configured cap).

    Used to zero optimizer predictions below observed spend levels. Empty
    segment history returns ``0.0`` (no floor).
    """
    mins: dict[str, float] = {}
    for seg in segments:
        sub = panel[panel["segment"] == seg]["daily_budget"].dropna()
        if sub.empty:
            mins[seg] = 0.0
        else:
            mins[seg] = float(sub.min())
    return mins


def historical_budget_bounds(
    panel: pd.DataFrame,
    segments: list[str],
) -> dict[str, tuple[float, float]]:
    """
    Per-segment budget bounds from historical ``daily_budget``.

    When more than one distinct spend level was observed, bounds are
    ``[min, max]``. Zero is allowed as the lower bound only when a single
    budget level was observed (or there is no panel history).
    """
    bounds: dict[str, tuple[float, float]] = {}
    for seg in segments:
        sub = panel[panel["segment"] == seg]["daily_budget"].dropna()
        if sub.empty:
            bounds[seg] = (0.0, 500.0)
            continue
        hi = float(sub.max())
        if sub.nunique() <= 1:
            bounds[seg] = (0.0, hi)
        else:
            bounds[seg] = (float(sub.min()), hi)
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


def parse_allowed_match_types(constraints: dict[str, Any]) -> list[str] | None:
    raw = constraints.get("allowed_match_types")
    if not raw:
        return None
    return [str(v) for v in raw]


def parse_excluded_regions(constraints: dict[str, Any]) -> list[str]:
    raw = constraints.get("excluded_regions")
    if not raw:
        return []
    return [str(v) for v in raw]


def filter_candidates_by_region(
    candidates: pd.DataFrame,
    excluded_regions: list[str] | None = None,
) -> pd.DataFrame:
    """Drop keyword-set candidate rows for excluded regions."""
    if not excluded_regions:
        return candidates
    excluded = set(excluded_regions)
    if "region" in candidates.columns:
        return candidates[~candidates["region"].isin(excluded)].copy()
    if "segment" in candidates.columns:
        mask = ~candidates["segment"].map(region_of_segment).isin(excluded)
        return candidates[mask].copy()
    return candidates


def apply_candidate_region_policy(
    candidates: pd.DataFrame,
    constraints: dict[str, Any],
) -> pd.DataFrame:
    return filter_candidates_by_region(candidates, parse_excluded_regions(constraints))


def region_of_segment(segment: str) -> str:
    return segment.split(" / ")[0].strip()


def actual_campaign_budget_total(
    panel: pd.DataFrame,
    date: pd.Timestamp,
    *,
    excluded_regions: list[str] | None = None,
) -> float:
    """
    Sum configured ``daily_budget`` caps across all panel rows on ``date``.

    One row per segment (Broad + Phrase; Exact, etc.) is summed so the MILP cap
    matches total configured spend that day. ``region_actual_lookup`` still uses
    regional medians for reference columns in plan-vs-actual only.
    """
    day = panel[panel["date"] == pd.Timestamp(date).normalize()].copy()
    if day.empty:
        raise ValueError(f"No panel rows on {date.date()}")
    if "region" not in day.columns and "segment" in day.columns:
        day["region"] = day["segment"].map(region_of_segment)
    if excluded_regions:
        day = day[~day["region"].isin(excluded_regions)]
    if day.empty:
        raise ValueError(f"No panel rows on {date.date()} after region exclusions")
    budget = pd.to_numeric(day["daily_budget"], errors="coerce")
    if budget.isna().all():
        raise ValueError(f"daily_budget missing on {date.date()}")
    return float(budget.sum())
