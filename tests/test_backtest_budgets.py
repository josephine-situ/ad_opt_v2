"""Tests for budget helpers and backtest optimizer alignment."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from campaign_opt.backtest import load_fit_manifest, run_daily_backtest
from campaign_opt.decisions import (
    budgets_proportional_to_conversion_rates,
    segment_conversion_rates,
)
from campaign_opt.schema import CampaignOptConfig, EvaluationConfig, ModelPolicy


def test_segment_conversion_rates():
    panel = pd.DataFrame(
        {
            "segment": ["A / Broad", "A / Broad", "B / Phrase; Exact", "B / Phrase; Exact"],
            "daily_budget": [10.0, 20.0, 5.0, 15.0],
            "all_conv": [1.0, 2.0, 3.0, 1.0],
        }
    )
    rates = segment_conversion_rates(panel, ["A / Broad", "B / Phrase; Exact"])
    assert rates["A / Broad"] == 0.1
    assert rates["B / Phrase; Exact"] == 0.2


def test_budgets_proportional_to_conversion_rates():
    panel = pd.DataFrame(
        {
            "segment": ["A / Broad", "A / Broad", "B / Phrase; Exact"],
            "daily_budget": [10.0, 10.0, 20.0],
            "all_conv": [2.0, 2.0, 8.0],
        }
    )
    segments = ["A / Broad", "B / Phrase; Exact"]
    budgets = budgets_proportional_to_conversion_rates(panel, segments, total_budget=100.0)
    assert abs(sum(budgets.values()) - 100.0) < 1e-9
    assert budgets["B / Phrase; Exact"] > budgets["A / Broad"]


def test_load_fit_manifest_missing_without_optimizer_winner(tmp_path: Path):
    config = CampaignOptConfig(exp_name="t", course="sys_think")
    with pytest.raises(FileNotFoundError, match="optimizer_winner"):
        load_fit_manifest(config)


def test_optimizer_manifest_from_config_without_fit():
    config = CampaignOptConfig(
        exp_name="t2",
        course="sys_think",
        model_policy=ModelPolicy(
            optimizer_winner="xgboost",
            optimizer_backend="tree_embed",
        ),
    )
    manifest = load_fit_manifest(config)
    assert manifest["winner"] == "xgboost"
    assert manifest["backend"] == "tree_embed"


def test_daily_backtest_uses_production_manifest_not_fixed_budgets(tmp_path: Path):
    manifest = {"winner": "xgboost", "backend": "tree_embed", "best_hyperparams": {}}

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"] * 59 + ["2025-01-02"]),
            "segment": ["A / Broad"] * 60,
            "daily_budget": [10.0] * 60,
            "clicks": [1.0] * 60,
            "region": ["A"] * 60,
            "match_types": ["Broad"] * 60,
        }
    )
    candidates = pd.DataFrame(
        {
            "segment": ["A / Broad"],
            "region": ["A"],
            "match_types": ["Broad"],
            "keyword_set_id": ["ks1"],
        }
    )
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
            "region": ["A"],
            "match_types": ["Broad"],
            "daily_budget": [10.0],
            "segment": ["A / Broad"],
        }
    )

    captured: dict = {}

    def fake_optimizer(_config, opt_manifest, _train, _candidates, _panel, **kwargs):
        captured["manifest"] = opt_manifest
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "segment": ["A / Broad"],
                "daily_budget": [50.0],
                "keyword_set_id": ["ks1"],
            }
        )

    config = CampaignOptConfig(
        exp_name="t",
        course="sys_think",
        evaluation=EvaluationConfig(use_ensemble=False),
    )
    exp_dir = config.exp_dir()
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with patch("campaign_opt.backtest.run_optimizer", side_effect=fake_optimizer):
        run_daily_backtest(
            config,
            df,
            candidates,
            panel,
            start=pd.Timestamp("2025-01-02"),
            end=pd.Timestamp("2025-01-02"),
            total_budget=100.0,
            out_dir=tmp_path,
            refit_each_day=True,
        )

    assert captured.get("fixed_budgets") is None
    assert captured["manifest"]["winner"] == "xgboost"
