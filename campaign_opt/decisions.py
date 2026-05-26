"""Decision variables and constraints for campaign MILP.

Segment = ``region / match_types``. Candidates come from ``segment-keyword-candidates.csv``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


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


def median_budgets_by_segment(panel: pd.DataFrame, segments: list[str]) -> dict[str, float]:
    """Historical median daily budget per segment (for stage-1 set selection)."""
    out: dict[str, float] = {}
    for seg in segments:
        sub = panel[panel["segment"] == seg]["daily_budget"].dropna()
        out[seg] = float(sub.median()) if len(sub) else 50.0
    return out


def scale_budgets_to_cap(budgets: dict[str, float], total_budget: float) -> dict[str, float]:
    """Scale segment budgets proportionally so their sum does not exceed total_budget."""
    total = sum(budgets.values())
    if total <= total_budget or total <= 0:
        return dict(budgets)
    scale = total_budget / total
    return {seg: val * scale for seg, val in budgets.items()}


def segment_conversion_rates(
    panel: pd.DataFrame,
    segments: list[str],
    *,
    conv_col: str = "all_conv",
) -> dict[str, float]:
    """Historical conversions per budget dollar for each segment (pooled over panel)."""
    rates: dict[str, float] = {}
    for seg in segments:
        sub = panel[panel["segment"] == seg]
        if sub.empty or conv_col not in sub.columns:
            rates[seg] = 0.0
            continue
        conv = float(sub[conv_col].fillna(0).sum())
        budget = float(sub["daily_budget"].fillna(0).sum())
        rates[seg] = conv / budget if budget > 0 else 0.0
    return rates


def budgets_proportional_to_conversion_rates(
    panel: pd.DataFrame,
    segments: list[str],
    total_budget: float,
    *,
    conv_col: str = "all_conv",
) -> dict[str, float]:
    """
    Allocate ``total_budget`` across segments in proportion to historical conv/$.

    Maximizes predicted conversions when marginal return is constant at the
    segment's pooled conversion rate.
    """
    rates = segment_conversion_rates(panel, segments, conv_col=conv_col)
    positive = {seg: max(rates.get(seg, 0.0), 0.0) for seg in segments}
    total_rate = sum(positive.values())
    if total_rate <= 0 or total_budget <= 0:
        share = total_budget / len(segments) if segments else 0.0
        return {seg: share for seg in segments}
    return {seg: total_budget * positive[seg] / total_rate for seg in segments}
