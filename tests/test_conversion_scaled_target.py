"""Tests for conversion-scaled clicks modeling target."""

from __future__ import annotations

import pandas as pd

from utils.campaign_features import (
    add_conversion_scaled_clicks_target,
    compute_segment_conv_per_click_rates,
    export_segment_conv_per_click_rates,
    load_course_conv_per_click_rates,
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


def test_load_course_conv_per_click_rates_reads_export(monkeypatch, tmp_path):
    panel = pd.DataFrame(
        {
            "region": ["USA", "USA", "A"],
            "match_types": ["Broad", "Broad", "Phrase; Exact"],
            "clicks": [100.0, 50.0, 40.0],
            "all_conv": [10.0, 4.0, 1.0],
        }
    )
    course = "test_course"
    processed = tmp_path / course / "data" / "processed"
    processed.mkdir(parents=True)
    export_segment_conv_per_click_rates(panel, processed / "segment-conv-per-click-rates.csv")

    def _data_paths(course_name: str) -> dict:
        base = tmp_path / course_name / "data"
        return {"processed": base / "processed", "gkp": base / "gkp", "cache": base / "cache"}

    monkeypatch.setattr("utils.campaign_features.data_paths", _data_paths)

    rates = load_course_conv_per_click_rates(course)
    expected = compute_segment_conv_per_click_rates(panel)
    pd.testing.assert_frame_equal(
        rates.reset_index(drop=True),
        expected[["region", "match_types", "conv_per_click"]].reset_index(drop=True),
    )
