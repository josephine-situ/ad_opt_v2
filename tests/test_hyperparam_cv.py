"""Tests for hyperparameter CV tuning."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.modeling_prep import prepare_modeling_data, train_holdout_split
from utils.cv import iter_param_grid, tune_hyperparams
from utils.modeling import fit_ridge, model_feature_overview_lines
from utils.campaign_config import CampaignOptConfig, ModelPolicy, ValidationConfig
from utils.train_specs import DEFAULT_HYPERPARAM_GRIDS


def test_iter_param_grid():
    grid = {"alpha": [0.1, 1.0], "beta": [2, 3]}
    combos = iter_param_grid(grid)
    assert len(combos) == 4
    assert {"alpha": 0.1, "beta": 2} in combos


def test_tune_ridge_alpha(monkeypatch, synthetic_sys_think_data):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    config = CampaignOptConfig(
        exp_name="test",
        course="sys_think",
        target="clicks",
        model_policy=ModelPolicy(
            candidates=["ridge"],
            validation=ValidationConfig(cv_folds=2, tune_hyperparams=True),
        ),
    )
    df = prepare_modeling_data(config)
    if df.empty:
        pytest.skip("empty panel")
    train, holdout = train_holdout_split(df, 30)
    feature_cols: list[str] = []
    best, cv = tune_hyperparams("ridge", fit_ridge, train, config, feature_cols, n_folds=2)
    assert "alpha" in best
    assert best["alpha"] in DEFAULT_HYPERPARAM_GRIDS["ridge"]["alpha"]
    assert cv["cv_rmse_levels"] < float("inf")


def test_model_feature_overview_ridge(monkeypatch, synthetic_sys_think_data):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    config = CampaignOptConfig(exp_name="test", course="sys_think", target="clicks")
    df = prepare_modeling_data(config)
    if df.empty:
        pytest.skip("empty panel")
    train, holdout = train_holdout_split(df, 30)
    res = fit_ridge(train, holdout, config, [])
    lines = model_feature_overview_lines(res)
    assert lines
    joined = " ".join(lines)
    assert "budget slope" in joined or "context" in joined
