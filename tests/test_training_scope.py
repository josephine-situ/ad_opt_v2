"""Training panel scope filter and mean-baseline tournament candidate."""

from __future__ import annotations

import pandas as pd
import pytest

from utils.modeling_prep import filter_training_scope, prepare_modeling_data
from utils.modeling import fit_mean_baseline, is_mean_baseline_candidate
from utils.campaign_config import CampaignOptConfig, ModelPolicy
from utils.campaign_features import (
    SEGMENT_BROAD_MATCH_COL,
    add_segment_match_type_indicators,
    is_broad_match_campaign,
)


def test_is_broad_match_campaign():
    assert is_broad_match_campaign("Broad")
    assert not is_broad_match_campaign("Phrase; Exact")
    assert not is_broad_match_campaign("Broad; Phrase; Exact")
    assert not is_broad_match_campaign("Exact")


def test_filter_training_scope():
    df = pd.DataFrame(
        {
            "segment": [
                "USA / Broad",
                "USA / Phrase; Exact",
                "C / Broad",
                "USA / Broad; Phrase; Exact",
            ],
            "daily_budget": [1.0, 2.0, 3.0, 4.0],
        }
    )
    config = CampaignOptConfig(
        exp_name="t",
        course="sys_think",
        constraints={
            "allowed_match_types": ["Broad", "Phrase; Exact"],
            "excluded_regions": ["C"],
        },
        model_policy=ModelPolicy(),
    )
    out = filter_training_scope(df, config)
    assert list(out["segment"]) == ["USA / Broad", "USA / Phrase; Exact"]


def test_add_segment_match_type_indicators_binary():
    df = pd.DataFrame(
        {
            "segment": ["USA / Broad", "A / Phrase; Exact", "USA / Broad; Phrase; Exact"],
        }
    )
    out = add_segment_match_type_indicators(df)
    assert out.iloc[0][SEGMENT_BROAD_MATCH_COL] == 1
    assert out.iloc[1][SEGMENT_BROAD_MATCH_COL] == 0
    assert out.iloc[2][SEGMENT_BROAD_MATCH_COL] == 0


def test_fit_mean_baseline_holdout():
    config = CampaignOptConfig(
        exp_name="t",
        course="sys_think",
        target="clicks",
        model_policy=ModelPolicy(),
    )
    train = pd.DataFrame({"clicks": [10.0, 20.0, 30.0]})
    holdout = pd.DataFrame({"clicks": [40.0, 50.0]})
    res = fit_mean_baseline(train, holdout, config, [])
    assert is_mean_baseline_candidate(res.name)
    assert res.extra["train_mean"] == pytest.approx(20.0)
    assert res.holdout_rmse == pytest.approx(25.495097568, rel=1e-5)
    assert res.holdout_r2 < 0.0


def test_prepare_modeling_data_filters_with_default_config():
    path = pytest.importorskip("pathlib").Path(
        "sys_think/opt_results/campaign/default/campaign_config.json"
    )
    if not path.exists():
        pytest.skip("default config missing")
    from utils.campaign_config import load_campaign_config

    config = load_campaign_config(path)
    df = prepare_modeling_data(config)
    segments = set(df["segment"].astype(str))
    assert not any(s.startswith("C /") for s in segments)
    assert "Broad; Phrase; Exact" not in " ".join(segments)
    assert segments <= {
        "USA / Broad",
        "USA / Phrase; Exact",
        "A / Broad",
        "A / Phrase; Exact",
        "B / Broad",
        "B / Phrase; Exact",
    }
