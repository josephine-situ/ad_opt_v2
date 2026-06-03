"""Tests for MILP-linear design matrix (region + match decomposition)."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from campaign_opt.coefficients import (
    coeffs_from_linear_milp_design,
    context_contribution,
    refresh_static_context_lift,
)
from campaign_opt.linear_design import (
    CONTEXT_MISSING_NUMERIC_FILL,
    build_linear_milp_design_matrix,
    segment_intercept_from_model,
    segment_slope_from_model,
)
from campaign_opt.schema import CampaignOptConfig, ModelPolicy


def _minimal_config() -> CampaignOptConfig:
    return CampaignOptConfig(
        exp_name="t",
        course="sys_think",
        target="clicks",
        context_features={"calendar": ["is_weekend"]},
        model_policy=ModelPolicy(),
    )


def test_linear_design_uses_region_and_match_not_segment_dummies():
    df = pd.DataFrame(
        {
            "segment": ["USA / Broad", "USA / Exact", "A / Phrase; Exact"],
            "daily_budget": [10.0, 20.0, 15.0],
            "clicks": [5.0, 8.0, 6.0],
            "is_weekend": [0, 1, 0],
        }
    )
    design = build_linear_milp_design_matrix(df, _minimal_config())
    cols = design.x_columns
    assert any(c.startswith("region_") for c in cols)
    assert "is_broad_match" in cols
    assert "budget_x_region_USA" in cols
    assert "budget_x_is_broad_match" in cols
    assert not any(c.startswith("seg_") for c in cols)


def test_segment_intercept_slope_match_decomposed_coefs():
    df = pd.DataFrame(
        {
            "segment": ["USA / Broad", "USA / Broad", "A / Exact"],
            "daily_budget": [10.0, 20.0, 15.0],
            "clicks": [5.0, 10.0, 7.0],
            "is_weekend": [0, 0, 0],
        }
    )
    config = _minimal_config()
    design = build_linear_milp_design_matrix(df, config)
    model = Ridge(alpha=1.0)
    model.fit(design.X.values, design.y)

    seg = "USA / Broad"
    alpha = segment_intercept_from_model(model, design.x_columns, seg)
    beta = segment_slope_from_model(model, design.x_columns, seg)
    eval_df = pd.DataFrame(
        {
            "segment": [seg],
            "daily_budget": [12.0],
            "clicks": [0.0],
            "is_weekend": [0],
        }
    )
    eval_design = build_linear_milp_design_matrix(eval_df, config, columns=design.x_columns)
    pred_manual = alpha + beta * 12.0 + float(model.coef_[design.x_columns.index("is_weekend")] * 0)
    pred_model = float(model.predict(eval_design.X.values)[0])
    assert pred_manual == pytest.approx(pred_model, rel=1e-6)


def test_coeffs_export_static_context_lift_key():
    df = pd.DataFrame(
        {
            "segment": ["USA / Broad", "USA / Broad"],
            "keyword_set_id": ["ks1", "ks2"],
            "daily_budget": [10.0, 20.0],
            "clicks": [5.0, 10.0],
            "embed_cohesion": [0.5, 0.7],
            "is_weekend": [0, 0],
        }
    )
    config = CampaignOptConfig(
        exp_name="t",
        course="sys_think",
        target="clicks",
        context_features={
            "calendar": ["is_weekend"],
            "keyword_set_static": ["embed_cohesion"],
        },
        model_policy=ModelPolicy(),
    )
    design = build_linear_milp_design_matrix(df, config)
    model = Ridge(alpha=1.0)
    model.fit(design.X.values, design.y)
    coeffs = coeffs_from_linear_milp_design(model, design, config)
    assert "static_context_lift" in coeffs
    assert "keyword_set_effect" not in coeffs
    assert "USA / Broad" in coeffs["segment_intercept"]


def test_missing_gkp_zero_fill_matches_design_matrix_and_milp_lift():
    df = pd.DataFrame(
        {
            "segment": ["USA / Broad", "USA / Broad"],
            "keyword_set_id": ["ks1", "ks2"],
            "daily_budget": [10.0, 20.0],
            "clicks": [5.0, 10.0],
            "last_month_searches_mean": [100.0, float("nan")],
            "is_weekend": [0, 0],
        }
    )
    config = CampaignOptConfig(
        exp_name="t",
        course="sys_think",
        target="clicks",
        context_features={
            "calendar": ["is_weekend"],
            "gkp_set": ["last_month_searches_mean"],
        },
        model_policy=ModelPolicy(),
    )
    design = build_linear_milp_design_matrix(df, config)
    assert design.X["last_month_searches_mean"].tolist() == [100.0, CONTEXT_MISSING_NUMERIC_FILL]

    model = Ridge(alpha=1.0)
    model.fit(design.X.values, design.y)
    context_coefs = {
        col: float(model.coef_[i])
        for i, col in enumerate(design.x_columns)
        if col == "last_month_searches_mean"
    }
    row_missing = df.iloc[1]
    assert context_contribution(row_missing, context_coefs, ["last_month_searches_mean"]) == pytest.approx(
        context_coefs["last_month_searches_mean"] * CONTEXT_MISSING_NUMERIC_FILL
    )

    coeffs = coeffs_from_linear_milp_design(model, design, config)
    candidates = pd.DataFrame({"keyword_set_id": ["ks1", "ks2"], "segment": ["USA / Broad"] * 2})
    set_features = pd.DataFrame(
        {
            "keyword_set_id": ["ks1", "ks2"],
            "last_month_searches_mean": [100.0, float("nan")],
        }
    )
    refreshed = refresh_static_context_lift(coeffs, config, candidates, set_features)
    assert refreshed["static_context_lift"]["ks2"] == pytest.approx(
        context_coefs["last_month_searches_mean"] * CONTEXT_MISSING_NUMERIC_FILL
    )
