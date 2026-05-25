"""Tests for ensemble incremental prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from campaign_opt.evaluation import (
    EnsembleModel,
    build_baseline_rows,
    build_segment_decision_rows,
    fit_ensemble,
)
from campaign_opt.schema import CampaignOptConfig, EvaluationConfig, ModelPolicy, ValidationConfig


@pytest.fixture
def tiny_config():
    return CampaignOptConfig(
        exp_name="test",
        course="sys_think",
        target="clicks",
        model_policy=ModelPolicy(candidates=["ridge"], validation=ValidationConfig(cv_folds=2)),
        evaluation=EvaluationConfig(baseline_budget=0.0),
        context_features={
            "calendar": ["is_weekend"],
            "keyword_set_static": [],
            "gkp_set": [],
        },
    )


def test_incremental_zero_budget_baseline(tiny_config, synthetic_course, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    from tests.conftest import copy_synthetic_to_repo

    copy_synthetic_to_repo(synthetic_course, root)
    from campaign_opt.features import prepare_modeling_data

    df = prepare_modeling_data(tiny_config)
    ensemble = fit_ensemble(df, tiny_config)
    set_feats = pd.DataFrame(
        {
            "keyword_set_id": df["keyword_set_id"].unique(),
            "embed_cohesion": 0.5,
        }
    )
    segments = df["segment"].unique()[:2]
    baseline_sets = df.groupby("segment")["keyword_set_id"].first()
    planning_date = pd.Timestamp(df["date"].max())

    baseline_rows = build_baseline_rows(
        list(segments),
        baseline_sets,
        planning_date,
        set_feats,
        tiny_config.course,
        ensemble.feature_cols,
        0.0,
    )
    f0 = ensemble.predict_levels(baseline_rows)
    assert np.all(f0 >= 0)

    dec = pd.DataFrame(
        {
            "segment": segments,
            "daily_budget": [50.0, 80.0],
            "keyword_set_id": baseline_sets.loc[segments].values,
        }
    )
    dec_rows = build_segment_decision_rows(
        dec, planning_date, set_feats, tiny_config.course, ensemble.feature_cols
    )
    lift = ensemble.predict_incremental(dec_rows, baseline_rows)
    assert len(lift) == len(segments)
