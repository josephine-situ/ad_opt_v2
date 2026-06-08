"""Tests for modeling warnings and config-derived model names."""

from __future__ import annotations

from campaign_opt.evaluation import optimizer_winner_name
from campaign_opt.modeling import (
    POOR_R2_THRESHOLD,
    configured_evaluation_model_name,
    warn_if_not_tournament_winner,
    warn_if_poor_r2,
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


def test_optimizer_winner_name_from_config():
    config = CampaignOptConfig(
        exp_name="default",
        course="sys_think",
        model_policy=ModelPolicy(optimizer_winner="xgboost"),
    )
    assert optimizer_winner_name(config) == "xgboost"


def test_warn_if_poor_r2_skips_good_and_nan(capsys):
    warn_if_poor_r2(0.5, scope="holdout")
    warn_if_poor_r2(float("nan"), scope="holdout")
    assert capsys.readouterr().out == ""


def test_warn_if_poor_r2_prints_for_low_r2(capsys):
    warn_if_poor_r2(0.1, scope="CV", label="ridge")
    out = capsys.readouterr().out
    assert "[Warn]" in out
    assert "Poor CV R²=0.1000" in out
    assert f"< {POOR_R2_THRESHOLD}" in out
    assert "(ridge)" in out


def test_warn_if_poor_r2_respects_custom_threshold(capsys):
    warn_if_poor_r2(0.4, scope="holdout", threshold=0.5)
    out = capsys.readouterr().out
    assert "[Warn]" in out
    assert "< 0.5" in out
