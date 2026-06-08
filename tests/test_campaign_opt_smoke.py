"""Smoke tests with minimal synthetic campaign data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.coefficients import export_linear_solver_coeffs
from utils.campaign_config import CampaignOptConfig, ModelPolicy, load_config
from utils.modeling_prep import prepare_modeling_data, train_holdout_split
from utils.modeling import run_tournament
from utils.optimize import _resolve_backend, require_optimizer_winner


def test_optimizer_winner_resolves_tree_embed_backend():
    config = CampaignOptConfig(
        course="sys_think",
        model_policy=ModelPolicy(optimizer_winner="xgboost"),
    )
    manifest = {"winner": "ridge", "backend": "linear"}
    assert require_optimizer_winner(config) == "xgboost"
    assert _resolve_backend(config, manifest) == "tree_embed"


def test_config_load():
    cfg = load_config("sys_think")
    assert cfg.course == "sys_think"
    assert cfg.target == "conv_scaled_clicks"


def test_tournament_on_synthetic(monkeypatch, synthetic_sys_think_data):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    config = load_config("sys_think")
    config.model_policy.candidates = ["ridge"]
    config.model_policy.validation.holdout_days = 30

    df = prepare_modeling_data(config)

    if df.empty:
        pytest.skip("empty panel")

    config.target = "clicks"
    train, holdout = train_holdout_split(df, config.model_policy.validation.holdout_days)
    if len(holdout) < 5:
        pytest.skip("insufficient holdout rows")

    winner, metrics, manifest = run_tournament(train, holdout, config)
    assert winner.name == "ridge"
    assert "ridge" in metrics


def test_linear_coeffs_export(monkeypatch, synthetic_sys_think_data, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    config = load_config("sys_think")
    df = prepare_modeling_data(config)
    config.target = "clicks"
    coeffs = export_linear_solver_coeffs(df, config, tmp_path / "c.json")
    assert "segment_budget_slope" in coeffs
    assert "context_feature_coefs" in coeffs
    assert "static_context_lift" in coeffs
    saved = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert "keyword_set_effect" not in saved
