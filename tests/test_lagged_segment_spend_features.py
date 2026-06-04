"""Tests for causal lagged segment spend features."""

from __future__ import annotations

import pandas as pd

from utils.lagged_segment_spend_features import (
    add_lagged_segment_spend_features,
    lagged_segment_column_name,
)


def test_lagged_cost_last_is_previous_observed_day_not_same_day():
    panel = pd.DataFrame(
        [
            {
                "segment": "R / Broad",
                "date": "2024-01-01",
                "cost": 10.0,
                "daily_budget": 100.0,
                "conv_scaled_clicks": 5.0,
            },
            {
                "segment": "R / Broad",
                "date": "2024-01-05",
                "cost": 30.0,
                "daily_budget": 100.0,
                "conv_scaled_clicks": 15.0,
            },
            {
                "segment": "R / Broad",
                "date": "2024-01-10",
                "cost": 90.0,
                "daily_budget": 100.0,
                "conv_scaled_clicks": 45.0,
            },
        ]
    )
    out = add_lagged_segment_spend_features(panel)
    col = lagged_segment_column_name("last", "cost")
    assert out.loc[out["date"] == "2024-01-01", col].isna().all()
    assert out.loc[out["date"] == "2024-01-05", col].iloc[0] == 10.0
    assert out.loc[out["date"] == "2024-01-10", col].iloc[0] == 30.0
