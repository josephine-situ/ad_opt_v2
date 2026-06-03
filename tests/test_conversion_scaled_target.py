"""Tests for conversion-scaled clicks modeling target."""

from __future__ import annotations

import pandas as pd

from utils.campaign_features import (
    add_conversion_scaled_clicks_target,
    add_version_run_features,
    compute_segment_conv_per_click_rates,
)


def test_conv_scaled_clicks_uses_region_match_scaling():
    panel = pd.DataFrame(
        {
            "region": ["USA", "USA", "A", "A"],
            "match_types": ["Broad", "Broad", "Phrase; Exact", "Phrase; Exact"],
            "clicks": [100.0, 50.0, 60.0, 40.0],
            "all_conv": [10.0, 4.0, 3.0, 1.0],
        }
    )
    out = add_conversion_scaled_clicks_target(panel)
    usa_rate = (10.0 + 4.0) / (100.0 + 50.0)
    a_rate = (3.0 + 1.0) / (60.0 + 40.0)
    assert out.loc[0, "conv_scaled_clicks"] == panel.loc[0, "clicks"] * usa_rate
    assert out.loc[3, "conv_scaled_clicks"] == panel.loc[3, "clicks"] * a_rate


def test_conv_scaled_clicks_falls_back_to_clicks_without_all_conv():
    panel = pd.DataFrame(
        {
            "region": ["USA"],
            "match_types": ["Broad"],
            "clicks": [12.0],
        }
    )
    out = add_conversion_scaled_clicks_target(panel)
    assert out.loc[0, "conv_scaled_clicks"] == 12.0


def test_conv_scaled_clicks_uses_fixed_rates_not_subset_panel():
    """Rates from ``rates=`` must not change when only a tail of rows is scored."""
    full = pd.DataFrame(
        {
            "region": ["USA", "USA", "USA"],
            "match_types": ["Broad", "Broad", "Broad"],
            "clicks": [100.0, 100.0, 100.0],
            "all_conv": [10.0, 10.0, 0.0],
        }
    )
    rates = compute_segment_conv_per_click_rates(full)
    tail = full.iloc[[2]].copy()
    out = add_conversion_scaled_clicks_target(tail, rates=rates)
    expected_rate = (10.0 + 10.0 + 0.0) / (100.0 + 100.0 + 100.0)
    assert out.iloc[0]["conv_scaled_clicks"] == 100.0 * expected_rate


def test_days_since_version_start_from_summary():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-10", "2024-01-15"]),
            "campaign_version": [1, 2],
            "region": ["USA", "USA"],
            "match_types": ["Broad", "Broad"],
        }
    )
    summary = pd.DataFrame(
        {
            "campaign_version": [1, 2],
            "start_date": ["2024-01-01", "2024-01-12"],
            "keyword_set_id": ["ks1", "ks2"],
        }
    )
    out = add_version_run_features(panel, summary)
    assert out.loc[0, "days_since_version_start"] == 9.0
    assert out.loc[1, "days_since_version_start"] == 3.0
