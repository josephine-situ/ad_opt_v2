"""Tests for exponential recency sample weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from campaign_opt.recency_weights import recency_sample_weights
from campaign_opt.schema import CampaignOptConfig, ModelPolicy, ValidationConfig


def test_recency_weights_none_for_disabled():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "y": [1, 2, 3]})
    assert recency_sample_weights(df, half_life_days=None) is None
    assert recency_sample_weights(df, half_life_days=0) is None


def test_recency_weights_recent_heavier():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-12-01"]),
            "y": [1, 2, 3],
        }
    )
    w = recency_sample_weights(df, half_life_days=90.0)
    assert w is not None
    assert w.shape == (3,)
    assert np.isclose(w.mean(), 1.0)
    assert w[-1] > w[0]


def test_recency_half_life_from_config():
    cfg = CampaignOptConfig(
        exp_name="t",
        course="sys_think",
        model_policy=ModelPolicy(
            validation=ValidationConfig(recency_half_life_days=180.0),
        ),
    )
    from campaign_opt.recency_weights import recency_half_life_days

    assert recency_half_life_days(cfg) == 180.0

    cfg2 = CampaignOptConfig(
        exp_name="t",
        course="sys_think",
        model_policy=ModelPolicy(validation=ValidationConfig(recency_half_life_days=None)),
    )
    assert recency_half_life_days(cfg2) is None
