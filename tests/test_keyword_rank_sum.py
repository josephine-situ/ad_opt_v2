"""Tests for rank-sum keyword selection."""

from __future__ import annotations

import pandas as pd

from utils.keyword_candidates import (
    _top_keywords_by_composite,
    _top_keywords_by_rank_sum,
    _top_keywords_for_match_type_slice,
)


def test_top_conv_rank_sum_prefers_balanced_keywords():
    sub = pd.DataFrame(
        [
            {"keyword": "high vol low eff", "all_conv": 100, "cost": 100},
            {"keyword": "low vol high eff", "all_conv": 1, "cost": 0.01},
            {"keyword": "balanced", "all_conv": 50, "cost": 1},
        ]
    )
    out = _top_keywords_for_match_type_slice(
        sub,
        top_n=2,
        volume_col="all_conv",
        require_positive_volume=True,
    )
    assert out == ["balanced", "high vol low eff"]


def test_rank_sum_tiebreaks_alphabetically():
    picked = _top_keywords_by_rank_sum(
        ["b", "a", "c"],
        {"a": 10, "b": 10, "c": 1},
        {"a": 10, "b": 10, "c": 1},
        top_n=2,
    )
    assert picked == ["a", "b"]


def test_composite_rank_sum_orders_by_summed_ranks():
    import numpy as np

    emb_map = {
        "near": np.array([1.0, 0.0]),
        "far": np.array([0.0, 1.0]),
        "mid": np.array([0.7, 0.7]),
    }
    anchors = np.array([[1.0, 0.0]])
    assert _top_keywords_by_composite(
        ["near", "far", "mid"], emb_map, anchors, set_size=2
    ) == ["near", "far"]
