"""Tests for ensemble level prediction and gating."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.evaluation import (
    build_segment_decision_rows,
    evaluation_ensemble_weights,
    fit_ensemble,
)
from utils.decisions import observed_min_daily_budget
from utils.optimizer_prediction import predict_levels_optimizer
from utils.campaign_config import CampaignOptConfig, EvaluationConfig, ModelPolicy, ValidationConfig


@pytest.fixture
def tiny_config():
    return CampaignOptConfig(
        exp_name="test",
        course="sys_think",
        target="clicks",
        model_policy=ModelPolicy(candidates=["ridge"], validation=ValidationConfig(cv_folds=2)),
        evaluation=EvaluationConfig(),
        context_features={
            "calendar": ["is_weekend"],
            "keyword_set_static": [],
            "gkp_set": [],
        },
    )


def test_gated_level_zero_when_below_observed_min(tiny_config, synthetic_sys_think_data, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    from utils.modeling_prep import prepare_modeling_data

    tiny_config.evaluation.apply_observed_budget_floor = True
    df = prepare_modeling_data(tiny_config)
    ensemble = fit_ensemble(df, tiny_config)
    panel = df.copy()
    seg = str(df["segment"].iloc[0])
    bmin = observed_min_daily_budget(panel, [seg])[seg]
    if bmin <= 0:
        pytest.skip("synthetic panel has no positive observed min for gating test")

    set_feats = pd.DataFrame(
        {"keyword_set_id": df["keyword_set_id"].unique(), "embed_cohesion": 0.5}
    )
    planning_date = pd.Timestamp(df["date"].max())
    kid = df.groupby("segment")["keyword_set_id"].first().loc[seg]
    dec = pd.DataFrame(
        {"segment": [seg], "daily_budget": [0.0], "keyword_set_id": [kid]}
    )
    rows = build_segment_decision_rows(
        dec, planning_date, set_feats, tiny_config.course, ensemble.feature_cols
    )
    raw = ensemble.predict_levels(rows)
    gated = predict_levels_optimizer(ensemble, rows, panel, tiny_config)
    assert float(raw[0]) >= 0
    assert float(gated[0]) == 0.0


def test_evaluation_ensemble_weights_from_cv_rmse():
    config = CampaignOptConfig(
        exp_name="t",
        course="sys_think",
        model_policy=ModelPolicy(
            candidates=["ridge", "random_forest", "xgboost", "ensemble"],
        ),
        evaluation=EvaluationConfig(weight_by_cv_rmse=True),
    )
    metrics = {
        "ridge": {"cv_rmse_levels": 2.64},
        "random_forest": {"cv_rmse_levels": 2.83},
        "xgboost": {"cv_rmse_levels": 2.73},
    }
    weights = evaluation_ensemble_weights(config, metrics)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["ridge"] == max(weights.values())
    assert weights["ridge"] > weights["xgboost"] > weights["random_forest"]
