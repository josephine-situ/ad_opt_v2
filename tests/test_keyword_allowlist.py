"""Tests for enrollment keyword allowlist filtering."""

from __future__ import annotations

import pandas as pd

from utils.keyword_allowlist import (
    filter_keyword_list,
    filter_keyword_sets_dataframe,
    load_enrollment_keyword_allowlist,
    normalize_keyword,
)


def test_normalize_keyword_strips_brackets():
    assert normalize_keyword('[mit systems thinking]') == "mit systems thinking"


def test_filter_keyword_list_respects_allowlist():
    allowlist = {"mit systems thinking", "systems thinking course"}
    out = filter_keyword_list(
        ["MIT systems thinking", "random keyword", "systems thinking course"],
        allowlist,
    )
    assert len(out) == 2
    assert normalize_keyword(out[0]) in allowlist


def test_filter_keyword_sets_dataframe_drops_empty_sets():
    allowlist = {"systems thinking course"}
    sets = pd.DataFrame(
        [
            {
                "keyword_set_id": "ks_a",
                "broad_keywords": "systems thinking course; other",
                "phrase_keywords": "",
                "exact_keywords": "",
            },
            {
                "keyword_set_id": "ks_b",
                "broad_keywords": "only outside allowlist",
                "phrase_keywords": "",
                "exact_keywords": "",
            },
        ]
    )
    filtered = filter_keyword_sets_dataframe(sets, allowlist)
    assert list(filtered["keyword_set_id"]) == ["ks_a"]
    assert "systems thinking course" in filtered.iloc[0]["broad_keywords"]
    assert "other" not in filtered.iloc[0]["broad_keywords"]


def test_load_enrollment_keyword_allowlist_sys_think():
    allowlist = load_enrollment_keyword_allowlist("sys_think")
    assert allowlist is not None
    assert "what is system thinking" in allowlist
    assert len(allowlist) >= 60
