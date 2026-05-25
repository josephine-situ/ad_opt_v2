"""Tests for segment decomposition into region + match-type indicators."""

from __future__ import annotations

import pandas as pd

from utils.campaign_features import (
    TREE_SEGMENT_FEATURE_COLS,
    add_segment_match_type_indicators,
    parse_match_types,
)


def test_parse_match_types():
    assert parse_match_types("Broad; Phrase; Exact") == {"Broad", "Phrase", "Exact"}
    assert parse_match_types("Phrase; Exact") == {"Phrase", "Exact"}
    assert parse_match_types("Broad") == {"Broad"}


def test_add_segment_match_type_indicators_from_segment():
    df = pd.DataFrame({"segment": ["USA / Broad; Phrase; Exact", "C / Phrase; Exact"]})
    out = add_segment_match_type_indicators(df)
    assert list(TREE_SEGMENT_FEATURE_COLS) == ["region", "has_broad", "has_phrase", "has_exact"]
    usa = out.iloc[0]
    assert usa["region"] == "USA"
    assert usa["has_broad"] == 1
    assert usa["has_phrase"] == 1
    assert usa["has_exact"] == 1
    c_row = out.iloc[1]
    assert c_row["region"] == "C"
    assert c_row["has_broad"] == 0
    assert c_row["has_phrase"] == 1
    assert c_row["has_exact"] == 1


def test_add_segment_match_type_indicators_from_columns():
    df = pd.DataFrame({"region": ["A"], "match_types": ["Broad"]})
    out = add_segment_match_type_indicators(df)
    assert out.iloc[0]["has_broad"] == 1
    assert out.iloc[0]["has_phrase"] == 0
    assert out.iloc[0]["has_exact"] == 0
