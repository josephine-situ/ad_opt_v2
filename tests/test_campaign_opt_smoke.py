"""Smoke tests with minimal synthetic campaign data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from campaign_opt.coefficients import export_linear_solver_coeffs
from campaign_opt.features import prepare_modeling_data, train_holdout_split
from campaign_opt.modeling import run_tournament
from campaign_opt.optimize import _resolve_backend, require_optimizer_winner
from campaign_opt.schema import CampaignOptConfig, ModelPolicy, load_campaign_config


def test_optimizer_winner_resolves_tree_embed_backend():
    config = CampaignOptConfig(
        exp_name="t",
        course="sys_think",
        model_policy=ModelPolicy(optimizer_winner="xgboost"),
    )
    manifest = {"winner": "power_level", "backend": "piecewise_linear"}
    assert require_optimizer_winner(config) == "xgboost"
    assert _resolve_backend(config, manifest) == "tree_embed"


def test_config_load():
    path = Path("sys_think/opt_results/campaign/default/campaign_config.json")
    if not path.exists():
        pytest.skip("default config missing")
    cfg = load_campaign_config(path)
    assert cfg.course == "sys_think"


def test_tournament_on_synthetic(monkeypatch, synthetic_sys_think_data):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    config_path = Path("sys_think/opt_results/campaign/default/campaign_config.json")
    if not config_path.exists():
        pytest.skip("config missing")
    config = load_campaign_config(config_path)
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
    config_path = Path("sys_think/opt_results/campaign/default/campaign_config.json")
    if not config_path.exists():
        pytest.skip("config missing")
    config = load_campaign_config(config_path)
    df = prepare_modeling_data(config)
    config.target = "clicks"
    coeffs = export_linear_solver_coeffs(df, config, tmp_path / "c.json")
    assert "segment_budget_slope" in coeffs
    assert "context_feature_coefs" in coeffs
    assert "static_context_lift" in coeffs
    saved = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert "keyword_set_effect" not in saved
