"""Tests for segment budget bounds."""

from __future__ import annotations

import pandas as pd

from utils.decisions import historical_budget_bounds


def test_historical_budget_bounds_allows_zero_for_single_level():
    panel = pd.DataFrame(
        {
            "segment": ["A / Broad"] * 3,
            "daily_budget": [13.5, 13.5, 13.5],
        }
    )
    lo, hi = historical_budget_bounds(panel, ["A / Broad"])["A / Broad"]
    assert lo == 0.0
    assert hi == 13.5


def test_historical_budget_bounds_caps_at_historical_max_for_multiple_levels():
    panel = pd.DataFrame(
        {
            "segment": ["A / Broad", "A / Broad", "A / Broad"],
            "daily_budget": [13.5, 20.0, 13.5],
        }
    )
    lo, hi = historical_budget_bounds(panel, ["A / Broad"])["A / Broad"]
    assert lo == 0.0
    assert hi == 20.0


def test_historical_budget_bounds_empty_panel_fallback():
    panel = pd.DataFrame({"segment": [], "daily_budget": []})
    lo, hi = historical_budget_bounds(panel, ["B / Broad"])["B / Broad"]
    assert lo == 0.0
    assert hi == 500.0
