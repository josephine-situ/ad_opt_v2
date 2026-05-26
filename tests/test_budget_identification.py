"""Tests for budget identification helpers."""

from __future__ import annotations

import pandas as pd

from campaign_opt.budget_identification import (
    build_cell_fixed_effects_design,
    filter_identifiable_rows,
    identifiability_report,
    pooled_within_cell_budget_slopes,
)
from campaign_opt.schema import CampaignOptConfig


def _mini_config(target: str = "clicks") -> CampaignOptConfig:
    return CampaignOptConfig(
        exp_name="test",
        course="sys_think",
        target=target,
        context_features={"calendar": ["season"]},
    )


def test_identifiability_same_set_multiple_budgets():
    df = pd.DataFrame(
        {
            "segment": ["A / Broad"] * 4,
            "keyword_set_id": ["ks1", "ks1", "ks1", "ks2"],
            "daily_budget": [10.0, 20.0, 20.0, 30.0],
            "clicks": [1.0, 2.0, 2.5, 3.0],
            "season": ["Fall", "Fall", "Winter", "Fall"],
        }
    )
    report = identifiability_report(df)
    assert report["n_cells"] == 2
    assert report["n_identifiable_cells"] == 1
    assert report["rows_in_identifiable_cells"] == 3

    filtered = filter_identifiable_rows(df)
    assert len(filtered) == 3


def test_cell_fe_design_uses_budget_column():
    df = pd.DataFrame(
        {
            "segment": ["A / Broad"] * 3,
            "keyword_set_id": ["ks1"] * 3,
            "daily_budget": [10.0, 20.0, 30.0],
            "clicks": [1.0, 2.0, 3.0],
            "season": ["Fall", "Fall", "Winter"],
        }
    )
    design = build_cell_fixed_effects_design(df, _mini_config(), identifiable_only=True)
    assert "daily_budget" in design.x_columns
    assert design.n_rows_used == 3
    assert design.n_cells == 1


def test_pooled_within_cell_slopes():
    df = pd.DataFrame(
        {
            "segment": ["A / Broad"] * 4 + ["B / Exact"] * 2,
            "keyword_set_id": ["ks1", "ks1", "ks1", "ks1", "ks2", "ks2"],
            "daily_budget": [10.0, 20.0, 30.0, 30.0, 5.0, 10.0],
            "clicks": [1.0, 2.0, 3.0, 3.0, 1.0, 2.0],
        }
    )
    pooled = pooled_within_cell_budget_slopes(df, "clicks")
    assert len(pooled) == 2
    a_row = pooled[pooled["segment"] == "A / Broad"].iloc[0]
    assert a_row["pooled_slope_budget"] == 0.1
