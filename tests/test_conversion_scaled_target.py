"""Tests for conversion-scaled clicks modeling target."""

from __future__ import annotations

import pandas as pd

from utils.campaign_features import add_conversion_scaled_clicks_target


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

