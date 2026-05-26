"""Tests for budget diagnostic helpers."""

from __future__ import annotations

import pandas as pd

from campaign_opt.budget_diagnostics import (
    bivariate_budget_slopes,
    within_keyword_set_budget_slopes,
)


def test_bivariate_budget_slopes():
    df = pd.DataFrame(
        {
            "segment": ["A / Broad"] * 4,
            "daily_budget": [10.0, 10.0, 20.0, 20.0],
            "clicks": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = bivariate_budget_slopes(df, ["clicks"])
    all_row = out[out["scope"] == "all"].iloc[0]
    assert all_row["slope_budget"] == 0.2


def test_within_set_identifiability():
    df = pd.DataFrame(
        {
            "segment": ["A / Broad", "A / Broad", "A / Broad", "B / Exact"],
            "keyword_set_id": ["ks1", "ks1", "ks2", "ks3"],
            "daily_budget": [10.0, 20.0, 15.0, 30.0],
            "clicks": [1.0, 3.0, 2.0, 5.0],
            "cost": [5.0, 10.0, 7.0, 20.0],
        }
    )
    out = within_keyword_set_budget_slopes(df, ["clicks"])
    ks1 = out[(out["keyword_set_id"] == "ks1")].iloc[0]
    assert ks1["identifiable"] is True
    assert ks1["slope_budget"] == 0.2
    ks2 = out[(out["keyword_set_id"] == "ks2")].iloc[0]
    assert ks2["identifiable"] is False
