"""Tests for causal keyword efficiency context features."""

from __future__ import annotations

import pandas as pd

from utils.keyword_efficiency_features import (
    _causal_stats_for_group,
    all_efficiency_column_names,
    build_keyword_efficiency_features_for_panel,
    efficiency_column_name,
)


def test_efficiency_column_naming():
    assert efficiency_column_name("last", "union", "mean", "cost") == "hist_kw_eff_last_union_mean_cost"
    assert efficiency_column_name("r7d", "union", "vol", "cost") == "hist_kw_eff_r7d_union_vol_cost"
    assert len(all_efficiency_column_names()) == 4 * 4 * 3 * 2


def test_temporal_volatility_is_std_over_observed_days_in_window():
    kw_hist = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-06", "2024-01-08"]),
            "region": ["R"] * 4,
            "match_type": ["Broad"] * 4,
            "kw_key": ["alpha"] * 4,
            "eff_cost": [1.0, 5.0, 3.0, 9.0],
            "eff_budget": [1.0, 5.0, 3.0, 9.0],
        }
    )
    from utils.keyword_efficiency_features import _causal_stats_for_group

    stats = _causal_stats_for_group(
        kw_hist,
        pd.to_datetime(["2024-01-10"]).to_numpy(dtype="datetime64[ns]"),
    )
    # Observed days in [2024-01-03, 2024-01-10): eff_cost 5, 3, 9
    assert abs(stats.iloc[0]["r7d_vol_cost"] - pd.Series([5.0, 3.0, 9.0]).std(ddof=1)) < 1e-9


def test_causal_last_uses_only_prior_days():
    kw_hist = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10"]),
            "region": ["R"] * 3,
            "match_type": ["Broad"] * 3,
            "kw_key": ["alpha"] * 3,
            "eff_cost": [1.0, 3.0, 9.0],
            "eff_budget": [1.0, 3.0, 9.0],
        }
    )
    stats = _causal_stats_for_group(
        kw_hist,
        pd.to_datetime(["2024-01-10"]).to_numpy(dtype="datetime64[ns]"),
    )
    assert stats.iloc[0]["last_cost"] == 3.0

    sets = pd.DataFrame(
        [
            {
                "keyword_set_id": "ks1",
                "broad_keywords": "alpha",
                "phrase_keywords": "",
                "exact_keywords": "",
                "positive_keywords": "alpha",
            }
        ]
    )
    panel = pd.DataFrame(
        [
            {
                "date": "2024-01-10",
                "region": "R",
                "match_types": "Broad",
                "keyword_set_id": "ks1",
                "daily_budget": 100.0,
            }
        ]
    )

    from utils import keyword_efficiency_features as mod

    def _fake_kw(_course, _budget):
        return kw_hist

    def _fake_stats(kw, panel_dates):
        return _precompute_from_kw(kw, panel_dates)

    def _precompute_from_kw(kw, panel_dates):
        dates_arr = pd.to_datetime(panel_dates).unique().to_numpy(dtype="datetime64[ns]")
        return _causal_stats_for_group(kw, dates_arr)

    orig_kw = mod._load_keyword_day_efficiency
    orig_stats = mod._precompute_keyword_causal_stats
    orig_budget = mod._segment_budget_lookup
    mod._load_keyword_day_efficiency = _fake_kw
    mod._precompute_keyword_causal_stats = _fake_stats
    mod._segment_budget_lookup = lambda _c: pd.DataFrame()
    try:
        out = build_keyword_efficiency_features_for_panel(
            panel, "sys_think", keyword_sets=sets
        )
    finally:
        mod._load_keyword_day_efficiency = orig_kw
        mod._precompute_keyword_causal_stats = orig_stats
        mod._segment_budget_lookup = orig_budget

    col = efficiency_column_name("last", "union", "mean", "cost")
    assert out.loc[0, col] == 3.0
