"""Tests for enrollment keyword allowlist filtering."""

from __future__ import annotations

import pandas as pd
import pytest

from utils.data_processing import clean_keyword_text
from utils.keyword_allowlist import (
    enrollment_allowlist_keywords,
    filter_keyword_list,
    filter_keyword_sets_dataframe,
    load_enrollment_keyword_allowlist,
    load_enrollment_keyword_allowlist_ordered,
)
from utils.paths import gkp_dir, require_enrollment_allowlist


def test_clean_keyword_text_strips_brackets():
    assert clean_keyword_text("[mit systems thinking]") == "mit systems thinking"


def test_clean_keyword_text_collapses_whitespace():
    assert clean_keyword_text("system  dynamics") == "system dynamics"
    assert clean_keyword_text("system  thinking  training") == "system thinking training"


def test_clean_keyword_text_strips_ads_artifacts():
    assert clean_keyword_text('[MIT systems thinking]') == "mit systems thinking"
    assert clean_keyword_text('"phrase match"') == "phrase match"
    assert clean_keyword_text("broad +match") == "broad match"


def test_clean_keyword_text_strict_rejects_non_ascii():
    assert clean_keyword_text("café marketing", strict=True) is None
    assert clean_keyword_text("systems thinking", strict=True) == "systems thinking"


def test_enrollment_allowlist_ordered_collapses_whitespace():
    ordered = load_enrollment_keyword_allowlist_ordered("sys_think")
    assert not any("  " in kw for kw in ordered)


def test_filter_keyword_list_respects_allowlist():
    allowlist = {"mit systems thinking", "systems thinking course"}
    out = filter_keyword_list(
        ["mit systems thinking", "random keyword", "systems thinking course"],
        allowlist,
    )
    assert len(out) == 2
    assert out[0] in allowlist


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


def test_enrollment_allowlist_keywords_uses_panel_keywords():
    allowlist = {"mit systems thinking", "systems thinking course"}
    kw_day = pd.DataFrame(
        [
            {"region": "B", "match_type": "Broad", "keyword": "mit systems thinking"},
            {"region": "USA", "match_type": "Broad", "keyword": "other"},
        ]
    )
    row = pd.Series({"region": "B", "match_types": "Broad"})
    out = enrollment_allowlist_keywords(allowlist, kw_day, row)
    assert out == ["mit systems thinking", "systems thinking course"]


def test_enrollment_allowlist_keywords_respects_priority_order():
    allowlist = {"mit systems thinking", "systems thinking course", "other kw"}
    kw_day = pd.DataFrame()
    row = pd.Series({"region": "B", "match_types": "Broad"})
    order = ["systems thinking course", "mit systems thinking", "other kw"]
    out = enrollment_allowlist_keywords(
        allowlist, kw_day, row, allowlist_order=order
    )
    assert out == ["systems thinking course", "mit systems thinking", "other kw"]


def test_load_enrollment_keyword_allowlist_sys_think():
    allowlist = load_enrollment_keyword_allowlist("sys_think")
    assert allowlist is not None
    assert "what is system thinking" in allowlist
    # 56 unique after collapsing internal whitespace (xlsx had duplicate spellings).
    assert len(allowlist) == 56


def test_require_enrollment_allowlist_finds_file():
    path = require_enrollment_allowlist("sys_think")
    assert path.is_file()
    assert "Enrollments" in path.name or "Keywords" in path.name
    assert path.parent == gkp_dir("sys_think")


def test_require_enrollment_allowlist_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "utils.paths.gkp_dir",
        lambda _course="sys_think": tmp_path / "empty_gkp",
    )
    with pytest.raises(FileNotFoundError, match="Required enrollment allowlist not found"):
        require_enrollment_allowlist("sys_think")
