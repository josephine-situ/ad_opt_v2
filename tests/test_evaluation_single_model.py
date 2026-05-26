"""Tests for single-model (optimizer) plan-vs-actual helpers."""

from __future__ import annotations

from campaign_opt.evaluation import optimizer_winner_name
from campaign_opt.schema import CampaignOptConfig, ModelPolicy


def test_optimizer_winner_name_prefers_config():
    config = CampaignOptConfig(
        model_policy=ModelPolicy(optimizer_winner="xgboost"),
    )
    assert optimizer_winner_name(config, {"winner": "power_level"}) == "xgboost"
