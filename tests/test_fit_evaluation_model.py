"""Tests for shared production/backtest evaluation ensemble fitting."""

from __future__ import annotations

import pytest

from campaign_opt.evaluation import evaluation_ensemble_weights
from campaign_opt.schema import CampaignOptConfig, EvaluationConfig, ModelPolicy


def test_evaluation_ensemble_weights_from_cv_rmse():
    config = CampaignOptConfig(
        exp_name="t",
        course="sys_think",
        model_policy=ModelPolicy(
            candidates=["ridge", "power_log", "power_level", "random_forest", "xgboost", "ensemble"],
        ),
        evaluation=EvaluationConfig(weight_by_cv_rmse=True),
    )
    metrics = {
        "ridge": {"cv_rmse_levels": 2.80},
        "power_log": {"cv_rmse_levels": 2.75},
        "power_level": {"cv_rmse_levels": 2.64},
        "random_forest": {"cv_rmse_levels": 2.83},
        "xgboost": {"cv_rmse_levels": 2.73},
    }
    weights = evaluation_ensemble_weights(config, metrics)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["power_level"] == max(weights.values())
    assert weights["power_level"] == pytest.approx(0.208, abs=0.01)
