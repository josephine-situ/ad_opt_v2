"""Tests for budget helpers and backtest optimizer alignment."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from campaign_opt.backtest import load_fit_manifest, run_daily_backtest
from campaign_opt.decisions import (
    actual_campaign_budget_total,
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


def test_load_fit_manifest_requires_optimizer_winner():
    config = CampaignOptConfig(exp_name="t", course="sys_think")
    with pytest.raises(ValueError, match="optimizer_winner"):
        load_fit_manifest(config)


def test_optimizer_manifest_requires_fit_artifacts(tmp_path: Path):
    config = CampaignOptConfig(
        exp_name="t2",
        course="sys_think",
        model_policy=ModelPolicy(
            optimizer_winner="xgboost",
            optimizer_backend="tree_embed",
        ),
    )
    with pytest.raises(FileNotFoundError, match="model_manifest.json"):
        load_fit_manifest(config)


def test_daily_backtest_uses_production_manifest_not_fixed_budgets(tmp_path: Path):
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
        model_policy=ModelPolicy(optimizer_winner="xgboost"),
        evaluation=EvaluationConfig(use_ensemble=False),
    )
    exp_dir = config.exp_dir()
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    comp = pd.DataFrame(
        {
            "pred_lift": [1.0],
            "actual_model_lift": [0.8],
            "observed_clicks": [3.0],
            "row_kind": ["plan"],
        }
    )
    with (
        patch("campaign_opt.backtest._fit_evaluation_model", return_value=MagicMock()),
        patch("campaign_opt.backtest.run_optimizer", side_effect=fake_optimizer),
        patch("campaign_opt.backtest.compare_plan_and_actual", return_value=comp),
    ):
        summary = run_daily_backtest(
            config,
            df,
            candidates,
            panel,
            start=pd.Timestamp("2025-01-02"),
            end=pd.Timestamp("2025-01-02"),
            total_budget=100.0,
            out_dir=tmp_path,
        )

    assert captured.get("fixed_budgets") is None
    assert captured["tune_optimizer"] is True
    assert captured["manifest"]["winner"] == "xgboost"
    assert len(summary) == 1
    assert summary.iloc[0]["plan_budget_total"] == 50.0
    assert (tmp_path / "daily_backtest_summary.csv").exists()


def test_daily_backtest_writes_plan_vs_actual_without_ensemble(tmp_path: Path):
    manifest = {
        "winner": "power_level",
        "backend": "tree_embed",
        "best_hyperparams": {},
        "feature_cols": ["day_of_week"],
    }
    comp = pd.DataFrame(
        {
            "segment": ["A / Broad"],
            "pred_lift": [1.0],
            "actual_model_lift": [0.8],
            "observed_clicks": [3.0],
        }
    )

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"] * 59 + ["2025-01-02"]),
            "segment": ["A / Broad"] * 60,
            "daily_budget": [10.0] * 60,
            "clicks": [1.0] * 60,
            "keyword_set_id": ["ks1"] * 60,
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
            "keyword_set_id": ["ks1"],
        }
    )

    config = CampaignOptConfig(
        exp_name="t",
        course="sys_think",
        model_policy=ModelPolicy(
            optimizer_winner="xgboost",
            optimizer_backend="tree_embed",
        ),
        evaluation=EvaluationConfig(use_ensemble=False),
    )
    exp_dir = config.exp_dir()
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    fake_plan = pd.DataFrame(
        {
            "segment": ["A / Broad"],
            "daily_budget": [50.0],
            "keyword_set_id": ["ks1"],
        }
    )

    with (
        patch("campaign_opt.backtest.run_optimizer", return_value=fake_plan),
        patch("campaign_opt.backtest.fit_single_model_evaluation") as mock_fit,
        patch("campaign_opt.backtest.compare_plan_and_actual", return_value=comp) as mock_cmp,
        patch("campaign_opt.backtest.save_evaluation_model"),
        patch("campaign_opt.backtest.eval_pipeline_holdout", return_value=None),
    ):
        mock_eval = MagicMock()
        mock_eval.members = [MagicMock(pipeline=object())]
        mock_fit.return_value = mock_eval
        summary = run_daily_backtest(
            config,
            df,
            candidates,
            panel,
            start=pd.Timestamp("2025-01-02"),
            end=pd.Timestamp("2025-01-02"),
            total_budget=100.0,
            out_dir=tmp_path,
        )

    mock_fit.assert_called_once()
    mock_cmp.assert_called_once()
    assert summary.iloc[0]["pred_lift_total"] == 1.0
    plan_vs_path = tmp_path / "plans" / "20250102" / "plan_vs_actual.csv"
    assert plan_vs_path.exists()
