"""Tests for segment budget bounds."""

from __future__ import annotations

import pandas as pd

from campaign_opt.decisions import historical_budget_bounds


def test_historical_budget_bounds_zero_lower_bound():
    panel = pd.DataFrame(
        {
            "segment": ["A / Broad", "A / Broad", "A / Broad"],
            "daily_budget": [13.5, 20.0, 13.5],
        }
    )
    lo, hi = historical_budget_bounds(panel, ["A / Broad"])["A / Broad"]
    assert lo == 0.0
    assert hi == 20.0
