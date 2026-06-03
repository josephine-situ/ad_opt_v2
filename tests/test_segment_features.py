"""Tests for segment decomposition into region + broad-match indicator."""

from __future__ import annotations

import pandas as pd

from utils.campaign_features import (
    SEGMENT_BROAD_MATCH_COL,
    TREE_SEGMENT_FEATURE_COLS,
    add_segment_match_type_indicators,
    parse_match_types,
)


def test_parse_match_types():
    assert parse_match_types("Broad; Phrase; Exact") == {"Broad", "Phrase", "Exact"}
    assert parse_match_types("Phrase; Exact") == {"Phrase", "Exact"}
    assert parse_match_types("Broad") == {"Broad"}


def test_add_segment_match_type_indicators_from_segment():
    df = pd.DataFrame({"segment": ["USA / Broad", "C / Phrase; Exact"]})
    out = add_segment_match_type_indicators(df)
    assert list(TREE_SEGMENT_FEATURE_COLS) == ["region", SEGMENT_BROAD_MATCH_COL]
    assert out.iloc[0]["region"] == "USA"
    assert out.iloc[0][SEGMENT_BROAD_MATCH_COL] == 1
    assert out.iloc[1]["region"] == "C"
    assert out.iloc[1][SEGMENT_BROAD_MATCH_COL] == 0


def test_add_segment_match_type_indicators_from_columns():
    df = pd.DataFrame({"region": ["A"], "match_types": ["Broad"]})
    out = add_segment_match_type_indicators(df)
    assert out.iloc[0][SEGMENT_BROAD_MATCH_COL] == 1
