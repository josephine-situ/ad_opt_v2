"""Tests for tournament-winner mismatch warnings."""

from __future__ import annotations

from campaign_opt.modeling import (
    configured_evaluation_model_name,
    warn_if_not_tournament_winner,
)
from campaign_opt.schema import CampaignOptConfig, EvaluationConfig, ModelPolicy


def test_warn_if_not_tournament_winner_skips_match(capsys):
    warn_if_not_tournament_winner("xgboost", {"winner": "xgboost"}, role="Optimizer")
    assert capsys.readouterr().out == ""


def test_warn_if_not_tournament_winner_prints_mismatch(capsys):
    warn_if_not_tournament_winner(
        "ensemble_ridge_xgb",
        {"winner": "xgboost"},
        role="Optimizer",
    )
    out = capsys.readouterr().out
    assert "[Warn]" in out
    assert "ensemble_ridge_xgb" in out
    assert "xgboost" in out
    assert "Optimizer" in out


def test_configured_evaluation_model_name_single_vs_ensemble():
    single = CampaignOptConfig(
        exp_name="default",
        course="sys_think",
        model_policy=ModelPolicy(optimizer_winner="xgboost"),
        evaluation=EvaluationConfig(use_ensemble=False),
    )
    ensemble = CampaignOptConfig(
        exp_name="default",
        course="sys_think",
        model_policy=ModelPolicy(optimizer_winner="xgboost"),
        evaluation=EvaluationConfig(use_ensemble=True),
    )
    assert configured_evaluation_model_name(single) == "xgboost"
    assert configured_evaluation_model_name(ensemble) == "ensemble"
