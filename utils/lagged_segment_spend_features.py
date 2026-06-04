"""Causal lagged segment spend / cap features (not same-day cost or budget)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

LAGGED_WINDOWS = ("last", "r7d", "r14d", "r30d")
LAGGED_SPEND_VARS = ("cost", "budget", "eff_cost", "eff_budget")

_WINDOW_DAYS = {"r7d": 7, "r14d": 14, "r30d": 30}
_COST_FLOOR = 0.01
_BUDGET_FLOOR = 0.01


def lagged_segment_column_name(window: str, var: str) -> str:
    """e.g. ``hist_seg_cost_last``, ``hist_seg_eff_cost_r7d_mean``."""
    if window == "last":
        return f"hist_seg_{var}_last"
    return f"hist_seg_{var}_{window}_mean"


def all_lagged_segment_column_names(
    *,
    windows: Iterable[str] = LAGGED_WINDOWS,
    vars_: Iterable[str] = LAGGED_SPEND_VARS,
) -> list[str]:
    cols: list[str] = []
    for var in vars_:
        for window in windows:
            if window == "last":
                cols.append(lagged_segment_column_name("last", var))
            elif window in _WINDOW_DAYS:
                cols.append(lagged_segment_column_name(window, var))
    return list(dict.fromkeys(cols))


def _causal_stats_on_series(
    dates: np.ndarray,
    values: np.ndarray,
    as_of: np.datetime64,
) -> dict[str, float]:
    pos = int(np.searchsorted(dates, as_of, side="left"))
    if pos == 0:
        return {w: float("nan") for w in LAGGED_WINDOWS}

    def _roll(days: int) -> float:
        start = as_of - np.timedelta64(days, "D")
        i0 = int(np.searchsorted(dates, start, side="left"))
        sl = values[i0:pos]
        return float(np.mean(sl)) if len(sl) else float("nan")

    return {
        "last": float(values[pos - 1]),
        "r7d": _roll(7),
        "r14d": _roll(14),
        "r30d": _roll(30),
    }


def _enrich_segment_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    dates = g["date"].to_numpy(dtype="datetime64[ns]")
    cost = pd.to_numeric(g["cost"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    budget = pd.to_numeric(g["daily_budget"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if "conv_scaled_clicks" in g.columns:
        conv = pd.to_numeric(g["conv_scaled_clicks"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    elif "clicks" in g.columns:
        conv = pd.to_numeric(g["clicks"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        conv = np.zeros(len(g), dtype=float)

    eff_cost = conv / np.clip(cost, _COST_FLOOR, None)
    eff_budget = conv / np.clip(budget, _BUDGET_FLOOR, None)

    series_map = {
        "cost": cost,
        "budget": budget,
        "eff_cost": eff_cost,
        "eff_budget": eff_budget,
    }

    n = len(g)
    col_data: dict[str, np.ndarray] = {}
    for var, vals in series_map.items():
        for window in LAGGED_WINDOWS:
            col = lagged_segment_column_name(window, var)
            out = np.full(n, np.nan)
            for i, as_of in enumerate(dates):
                stats = _causal_stats_on_series(dates, vals, as_of)
                out[i] = stats[window]
            col_data[col] = out

    for col, arr in col_data.items():
        g[col] = arr
    return g


def add_lagged_segment_spend_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add causal lagged segment ``cost``, ``daily_budget``, and segment-level efficiency.

    Uses only rows strictly before each ``date`` within ``segment`` (previous observed day
    or calendar rolling mean over past 7/14/30d). Same-day ``cost`` / ``daily_budget`` are
    never copied into these columns.
    """
    required = {"segment", "date", "cost", "daily_budget"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns for lagged spend features: {sorted(missing)}")

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in all_lagged_segment_column_names():
        out[col] = np.nan

    parts = [_enrich_segment_group(g) for _, g in out.groupby("segment", sort=False)]
    return pd.concat(parts, ignore_index=True).sort_values(["date", "segment"]).reset_index(drop=True)
