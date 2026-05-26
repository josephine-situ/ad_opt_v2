"""Tests for excluding regions from keyword-set candidates."""

from __future__ import annotations

import pandas as pd

from campaign_opt.decisions import (
    apply_candidate_region_policy,
    filter_candidates_by_region,
    parse_excluded_regions,
)


def test_parse_excluded_regions():
    assert parse_excluded_regions({}) == []
    assert parse_excluded_regions({"excluded_regions": ["C"]}) == ["C"]


def test_filter_candidates_by_region():
    df = pd.DataFrame(
        {
            "segment": ["USA / Broad", "C / Broad"],
            "region": ["USA", "C"],
            "keyword_set_id": ["ks1", "ks2"],
        }
    )
    out = filter_candidates_by_region(df, ["C"])
    assert len(out) == 1
    assert out.iloc[0]["region"] == "USA"


def test_apply_candidate_region_policy_from_segment():
    df = pd.DataFrame({"segment": ["A / Broad", "C / Phrase; Exact"], "keyword_set_id": ["a", "c"]})
    out = apply_candidate_region_policy(df, {"excluded_regions": ["C"]})
    assert list(out["keyword_set_id"]) == ["a"]
