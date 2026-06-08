"""Tests for budget helpers and backtest optimizer alignment."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from utils.modeling_prep import load_fit_manifest
from utils.two_stage_plan import optimize_budgets_for_day
from utils.decisions import actual_campaign_budget_total
from utils.campaign_config import CampaignOptConfig, ModelPolicy


def test_actual_campaign_budget_total():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-01"]),
            "region": ["USA", "A", "B"],
            "match_types": ["Broad", "Broad", "Broad"],
            "daily_budget": [200.0, 100.0, 50.0],
        }
    )
    total = actual_campaign_budget_total(panel, pd.Timestamp("2025-01-01"))
    assert total == 350.0
    total_excl = actual_campaign_budget_total(
        panel, pd.Timestamp("2025-01-01"), excluded_regions=["B"]
    )
    assert total_excl == 300.0


def test_actual_campaign_budget_total_sums_segments_not_regional_median():
    """Two segments per region: cap is sum of segment budgets, not median(region)."""
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-08"] * 2),
            "region": ["A", "A"],
            "segment": ["A / Broad", "A / Phrase; Exact"],
            "match_types": ["Broad", "Phrase; Exact"],
            "daily_budget": [13.53, 50.0],
        }
    )
    assert actual_campaign_budget_total(panel, pd.Timestamp("2025-01-08")) == pytest.approx(63.53)


def test_load_fit_manifest_requires_optimizer_winner():
    config = CampaignOptConfig(
        exp_name="t",
        course="sys_think",
        model_policy=ModelPolicy(optimizer_winner=None),
    )
    with pytest.raises(ValueError, match="optimizer_winner"):
        load_fit_manifest(config)


def test_optimizer_manifest_requires_fit_artifacts(tmp_path: Path):
    config = CampaignOptConfig(
        course="sys_think",
        model_policy=ModelPolicy(
            optimizer_winner="xgboost",
        ),
    )
    config.prod_dir = lambda base=None: tmp_path
    config.exp_dir = config.prod_dir
    with pytest.raises(FileNotFoundError, match="model_manifest.json"):
        load_fit_manifest(config)


def test_stage2_budget_opt_tunes_hyperparams_daily(tmp_path: Path):
    manifest = {
        "winner": "xgboost",
        "backend": "tree_embed",
        "best_hyperparams": {},
        "feature_cols": ["day_of_week"],
    }
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
            "date": pd.to_datetime(["2025-01-02"]),
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
                "daily_budget": [25.0],
                "keyword_set_id": ["ks1"],
            }
        )

    config = CampaignOptConfig(
        exp_name="t",
        course="sys_think",
        model_policy=ModelPolicy(optimizer_winner="xgboost"),
    )
    exp_dir = config.exp_dir(tmp_path)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with patch("utils.two_stage_plan.run_optimizer", side_effect=fake_optimizer):
        plan = optimize_budgets_for_day(
            config,
            manifest,
            df,
            candidates,
            panel,
            planning_date=pd.Timestamp("2025-01-02"),
            total_budget=100.0,
            fixed_keyword_sets={"A / Broad": "ks1"},
            output_dir=tmp_path,
        )

    assert captured["tune_optimizer"] is True
    assert captured["fixed_keyword_sets"] == {"A / Broad": "ks1"}
    assert plan.iloc[0]["daily_budget"] == 25.0
