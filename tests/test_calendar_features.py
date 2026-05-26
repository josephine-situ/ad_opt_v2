"""Tests for extended calendar columns."""

from __future__ import annotations

import pandas as pd

from utils.date_features import add_calendar_features, add_month_cycle_features


def test_add_month_and_cycle_features():
    df = pd.DataFrame({"date": pd.to_datetime(["2025-09-15"]), "region": ["USA"]})
    out = add_calendar_features(df, course="sys_think")
    assert out.iloc[0]["month"] == "Sep"
    assert out.iloc[0]["season"] == "Fall"
    assert -1.0 <= float(out.iloc[0]["month_sin"]) <= 1.0
    assert -1.0 <= float(out.iloc[0]["month_cos"]) <= 1.0


def test_month_cycle_smooth():
    dates = pd.to_datetime([f"2025-{m:02d}-01" for m in range(1, 13)])
    cycle = add_month_cycle_features(dates)
    assert len(cycle) == 12
    assert cycle["month_sin"].std() > 0.5
