"""Partial pooling helpers for sparse segment budget slopes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def segment_budget_level_counts(df: pd.DataFrame, segment_col: str = "segment") -> pd.Series:
    return (
        df.groupby(segment_col)["daily_budget"]
        .nunique()
        .rename("n_budget_levels")
    )


def shrink_segment_slopes(
    segment_slopes: pd.Series,
    *,
    global_slope: float,
    min_levels: int = 3,
    weight: float = 0.5,
) -> pd.Series:
    """
    Blend segment slopes toward global_slope when n_budget_levels < min_levels.
    weight=1 -> full shrink to global; weight=0 -> keep segment estimate.
    """
    out = segment_slopes.copy()
    for seg, slope in segment_slopes.items():
        if pd.isna(slope):
            out[seg] = global_slope
        else:
            out[seg] = (1 - weight) * slope + weight * global_slope
    return out
