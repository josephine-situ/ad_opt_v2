"""Historical keyword efficiency features for campaign-day modeling (causal lookback)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from utils.campaign_features import (
    MATCH_TYPE_LIST_COLS,
    _keywords_from_list_column,
    add_conversion_scaled_clicks_target,
    add_segment_column,
    data_paths,
    load_campaign_day_panel,
    load_course_conv_per_click_rates,
    load_keyword_sets_for_features,
)
from utils.keyword_allowlist import normalize_keyword
from utils.keyword_candidates import _segment_allowed_match_types

EFFICIENCY_WINDOWS = ("last", "r7d", "r14d", "r30d")
EFFICIENCY_DENOMS = ("cost", "budget")
# ``mean`` / ``std``: aggregate per-keyword *time-mean* efficiency across keywords.
# ``vol``: mean across keywords of per-keyword *temporal* std of daily efficiency in the window.
EFFICIENCY_STATS = ("mean", "std", "vol")
EFFICIENCY_POOLS = ("union", "broad", "phrase", "exact")

_WINDOW_DAYS = {"r7d": 7, "r14d": 14, "r30d": 30}
_COST_FLOOR = 0.01
_BUDGET_FLOOR = 0.01


def efficiency_column_name(
    window: str,
    pool: str,
    stat: str,
    denom: str,
) -> str:
    """Column name: ``hist_kw_eff_{window}_{pool}_{stat}_{denom}``."""
    return f"hist_kw_eff_{window}_{pool}_{stat}_{denom}"


def all_efficiency_column_names(
    *,
    windows: Iterable[str] = EFFICIENCY_WINDOWS,
    pools: Iterable[str] = EFFICIENCY_POOLS,
    stats: Iterable[str] = EFFICIENCY_STATS,
    denoms: Iterable[str] = EFFICIENCY_DENOMS,
) -> list[str]:
    cols: list[str] = []
    for window in windows:
        for pool in pools:
            for stat in stats:
                for denom in denoms:
                    cols.append(efficiency_column_name(window, pool, stat, denom))
    return list(dict.fromkeys(cols))


def _kw_value_column(window: str, denom: str, *, temporal: str = "mean") -> str:
    """Per-keyword scalar: ``mean`` = time mean; ``vol`` = time std in window."""
    if temporal == "vol":
        return f"{window}_vol_{denom}"
    return f"{window}_{denom}"


def _keyword_pools_for_set(row: pd.Series) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = {}
    union: set[str] = set()
    for mt, col in MATCH_TYPE_LIST_COLS.items():
        keys = [normalize_keyword(k) for k in _keywords_from_list_column(row.get(col))]
        pools[mt.lower()] = keys
        union.update(keys)
    if "positive_keywords" in row.index:
        for k in _keywords_from_list_column(row.get("positive_keywords")):
            union.add(normalize_keyword(k))
    pools["union"] = sorted(union)
    return pools


def _segment_budget_lookup(course: str) -> pd.DataFrame:
    panel = load_campaign_day_panel(course)
    panel = add_segment_column(panel)
    panel["date"] = pd.to_datetime(panel["date"])
    budget = panel[["date", "region", "match_types", "daily_budget"]].drop_duplicates()
    budget["daily_budget"] = pd.to_numeric(budget["daily_budget"], errors="coerce")
    return budget


def _load_keyword_day_efficiency(course: str, budget: pd.DataFrame) -> pd.DataFrame:
    paths = data_paths(course)
    kw_path = paths["processed"] / "kw-day-panel.csv"
    if not kw_path.exists():
        return pd.DataFrame()

    kw = pd.read_csv(kw_path)
    kw["date"] = pd.to_datetime(kw["date"])
    kw["kw_key"] = kw["keyword"].astype(str).map(normalize_keyword)
    kw["match_type"] = kw["match_type"].astype(str).str.strip().str.title()
    kw["region"] = kw["region"].astype(str)

    rates = load_course_conv_per_click_rates(course)
    kw = add_conversion_scaled_clicks_target(
        kw.rename(columns={"match_type": "match_types"}),
        target_col="_conv_scaled",
        rates=rates,
    ).rename(columns={"match_types": "match_type"})
    kw["cost"] = pd.to_numeric(kw["cost"], errors="coerce").fillna(0.0)
    kw["eff_cost"] = kw["_conv_scaled"] / kw["cost"].clip(lower=_COST_FLOOR)

    seg_rows = []
    for _, row in budget.iterrows():
        mts = [m.strip().title() for m in str(row["match_types"]).replace(";", " ").split() if m.strip()]
        for mt in mts:
            seg_rows.append(
                {
                    "date": row["date"],
                    "region": row["region"],
                    "match_type": mt,
                    "daily_budget": row["daily_budget"],
                }
            )
    seg_budget_mt = pd.DataFrame(seg_rows).drop_duplicates(subset=["date", "region", "match_type"])
    kw = kw.merge(seg_budget_mt, on=["date", "region", "match_type"], how="left")
    kw["eff_budget"] = kw["_conv_scaled"] / kw["daily_budget"].clip(lower=_BUDGET_FLOOR)
    return kw[["date", "region", "match_type", "kw_key", "eff_cost", "eff_budget"]]


def _causal_stats_for_group(g: pd.DataFrame, panel_dates: np.ndarray) -> pd.DataFrame:
    g = g.sort_values("date")
    dates = g["date"].to_numpy(dtype="datetime64[ns]")
    eff_cost = g["eff_cost"].to_numpy(dtype=float)
    eff_budget = g["eff_budget"].to_numpy(dtype=float)

    rows: list[dict[str, object]] = []
    for as_of in panel_dates:
        as_of64 = np.datetime64(as_of, "ns")
        pos = int(np.searchsorted(dates, as_of64, side="left"))
        if pos == 0:
            row: dict[str, object] = {"as_of": as_of}
            for w in EFFICIENCY_WINDOWS:
                for d in ("cost", "budget"):
                    row[_kw_value_column(w, d)] = np.nan
                    row[_kw_value_column(w, d, temporal="vol")] = np.nan
            rows.append(row)
            continue

        def _roll_mean(days: int, col: np.ndarray) -> float:
            start = as_of64 - np.timedelta64(days, "D")
            i0 = int(np.searchsorted(dates, start, side="left"))
            sl = col[i0:pos]
            return float(np.mean(sl)) if len(sl) else float("nan")

        def _roll_vol(days: int, col: np.ndarray) -> float:
            start = as_of64 - np.timedelta64(days, "D")
            i0 = int(np.searchsorted(dates, start, side="left"))
            sl = col[i0:pos]
            if len(sl) < 2:
                return float("nan")
            return float(np.std(sl, ddof=1))

        rows.append(
            {
                "as_of": as_of,
                "last_cost": float(eff_cost[pos - 1]),
                "last_budget": float(eff_budget[pos - 1]),
                "last_vol_cost": float("nan"),
                "last_vol_budget": float("nan"),
                "r7d_cost": _roll_mean(7, eff_cost),
                "r7d_budget": _roll_mean(7, eff_budget),
                "r7d_vol_cost": _roll_vol(7, eff_cost),
                "r7d_vol_budget": _roll_vol(7, eff_budget),
                "r14d_cost": _roll_mean(14, eff_cost),
                "r14d_budget": _roll_mean(14, eff_budget),
                "r14d_vol_cost": _roll_vol(14, eff_cost),
                "r14d_vol_budget": _roll_vol(14, eff_budget),
                "r30d_cost": _roll_mean(30, eff_cost),
                "r30d_budget": _roll_mean(30, eff_budget),
                "r30d_vol_cost": _roll_vol(30, eff_cost),
                "r30d_vol_budget": _roll_vol(30, eff_budget),
            }
        )

    out = pd.DataFrame(rows)
    out["region"] = g["region"].iloc[0]
    out["match_type"] = g["match_type"].iloc[0]
    out["kw_key"] = g["kw_key"].iloc[0]
    return out


def _precompute_keyword_causal_stats(kw: pd.DataFrame, panel_dates: pd.Series) -> pd.DataFrame:
    if kw.empty:
        return pd.DataFrame()
    dates_arr = np.sort(panel_dates.unique().to_numpy(dtype="datetime64[ns]"))
    parts = [_causal_stats_for_group(g, dates_arr) for _, g in kw.groupby(["region", "match_type", "kw_key"], sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _build_set_keyword_membership(sets: pd.DataFrame) -> pd.DataFrame:
    """Long table: keyword_set_id, pool, kw_key, match_type (one row per set keyword × MT)."""
    rows: list[dict[str, object]] = []
    for _, row in sets.iterrows():
        kid = str(row["keyword_set_id"])
        pools = _keyword_pools_for_set(row)
        for pool, keys in pools.items():
            if pool == "union":
                for kw_key in keys:
                    for mt in ("Broad", "Phrase", "Exact"):
                        if kw_key in pools.get(mt.lower(), []):
                            rows.append(
                                {
                                    "keyword_set_id": kid,
                                    "pool": "union",
                                    "kw_key": kw_key,
                                    "match_type": mt,
                                }
                            )
            else:
                for kw_key in keys:
                    rows.append(
                        {
                            "keyword_set_id": kid,
                            "pool": pool,
                            "kw_key": kw_key,
                            "match_type": pool.title(),
                        }
                    )
    return pd.DataFrame(rows)


def _aggregate_pool_stats(values: pd.Series, stat: str) -> float:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return float("nan")
    if stat == "mean":
        return float(np.mean(arr))
    if len(arr) < 2:
        return float("nan")
    return float(np.std(arr, ddof=1))


def build_keyword_efficiency_features_for_panel(
    panel: pd.DataFrame,
    course: str,
    *,
    keyword_sets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Attach causal historical keyword-efficiency columns to a modeling panel.

    Efficiency = conv-scaled clicks / cost or / segment daily_budget (fixed segment rates).
    Lookback: previous observed day (``last``) or rolling mean / temporal std over 7/14/30d.
    Aggregates across keywords: ``mean`` / ``std`` of per-keyword time-means; ``vol`` = mean of
    per-keyword temporal std (efficiency volatility in the window).
    """
    required = {"keyword_set_id", "date", "region"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns for efficiency features: {sorted(missing)}")

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    if "match_types" not in out.columns and "segment" in out.columns:
        parts = out["segment"].astype(str).str.split(" / ", n=1, expand=True)
        out["match_types"] = parts[1]

    for col in all_efficiency_column_names():
        out[col] = np.nan

    budget = _segment_budget_lookup(course)
    kw = _load_keyword_day_efficiency(course, budget)
    if kw.empty:
        return out

    kw_stats = _precompute_keyword_causal_stats(kw, out["date"])
    sets = keyword_sets if keyword_sets is not None else load_keyword_sets_for_features(course)
    members = _build_set_keyword_membership(sets)
    if members.empty:
        return out

    out["_row"] = np.arange(len(out))
    base_cols = ["_row", "date", "region", "match_types", "keyword_set_id"]
    base = out[base_cols].copy()

    long = base.merge(members, on="keyword_set_id", how="inner")
    long = long.merge(
        kw_stats,
        left_on=["date", "region", "match_type", "kw_key"],
        right_on=["as_of", "region", "match_type", "kw_key"],
        how="left",
    )

    allowed_rows: list[dict[str, object]] = []
    for ridx, row in out.iterrows():
        for mt in _segment_allowed_match_types(row):
            allowed_rows.append({"_row": ridx, "match_type": mt})
    allowed = pd.DataFrame(allowed_rows)
    long = long.merge(allowed, on=["_row", "match_type"], how="inner")

    for window in EFFICIENCY_WINDOWS:
        for pool in EFFICIENCY_POOLS:
            for stat in EFFICIENCY_STATS:
                for denom in EFFICIENCY_DENOMS:
                    if stat == "vol":
                        src = _kw_value_column(window, denom, temporal="vol")
                    else:
                        src = _kw_value_column(window, denom, temporal="mean")
                    sub = long[long["pool"] == pool]
                    if sub.empty or src not in sub.columns:
                        continue
                    agg = sub.groupby("_row")[src].apply(
                        lambda s, st=stat: _aggregate_pool_stats(s, "mean" if st == "vol" else st)
                    )
                    out_col = efficiency_column_name(window, pool, stat, denom)
                    out.loc[agg.index, out_col] = agg.values

    return out.drop(columns=["_row"], errors="ignore")


def merge_keyword_efficiency_features(panel: pd.DataFrame, course: str) -> pd.DataFrame:
    """Drop existing efficiency columns and recompute."""
    col_names = all_efficiency_column_names()
    base = panel.drop(columns=[c for c in col_names if c in panel.columns], errors="ignore")
    return build_keyword_efficiency_features_for_panel(base, course)
