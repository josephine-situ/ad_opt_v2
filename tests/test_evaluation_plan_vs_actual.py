"""Plan vs actual uses panel campaign budgets (daily_budget), not cost or plan padding."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from utils.evaluation import compare_plan_and_actual, plan_vs_actual_row_metrics
from utils.campaign_config import CampaignOptConfig, EvaluationConfig, ModelPolicy, ValidationConfig


@pytest.fixture
def tiny_config():
    return CampaignOptConfig(
        exp_name="test",
        course="sys_think",
        target="clicks",
        model_policy=ModelPolicy(candidates=["ridge"], validation=ValidationConfig(cv_folds=2)),
        evaluation=EvaluationConfig(),
        context_features={"calendar": ["is_weekend"], "keyword_set_static": [], "gkp_set": []},
    )


def test_market_actuals_use_panel_campaigns_not_zero_pad(
    tiny_config, synthetic_sys_think_data, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    from utils.modeling_prep import prepare_modeling_data
    from utils.evaluation import fit_ensemble
    from utils.campaign_features import build_keyword_set_feature_table

    df = prepare_modeling_data(tiny_config)
    ensemble = fit_ensemble(df, tiny_config)
    set_features = build_keyword_set_feature_table(tiny_config.course)
    train = df[df["date"] < df["date"].max()].copy()
    day = df["date"].max()
    day_df = df[df["date"] == day].copy()

    plan = pd.DataFrame(
        {
            "segment": ["USA / Broad", "B / Phrase; Exact"],
            "daily_budget": [40.0, 60.0],
            "keyword_set_id": ["ks_0", "ks_1"],
        }
    )
    comp = compare_plan_and_actual(
        ensemble, plan, day_df, train, tiny_config, pd.Timestamp(day), set_features
    )
    assert not comp.empty
    assert set(comp["row_kind"]) == {"plan", "market"}

    market = comp[comp["row_kind"] == "market"]
    assert market["actual_model_lift"].notna().all()
    assert (market["daily_budget"] > 0).any()

    metrics = plan_vs_actual_row_metrics(comp, tiny_config.target)
    assert metrics["actual_model_lift_total"] == pytest.approx(float(market["actual_model_lift"].sum()))
    assert metrics["act_budget_total"] == pytest.approx(float(market["daily_budget"].sum()))

    plan_rows = comp[comp["row_kind"] == "plan"]
    assert plan_rows["actual_budget"].notna().all()
    assert "campaign_budget" in market.columns
    assert market["campaign_budget"].equals(market["daily_budget"])
    assert metrics.get("pred_lift_total") == pytest.approx(float(plan_rows["pred_lift"].sum()))


def test_actual_decisions_reject_missing_daily_budget():
    from utils.evaluation import actual_decisions_by_segment

    day_df = pd.DataFrame(
        {
            "segment": ["A / Broad"],
            "cost": [100.0],
            "keyword_set_id": ["ks1"],
        }
    )
    with pytest.raises(ValueError, match="daily_budget"):
        actual_decisions_by_segment(day_df)
