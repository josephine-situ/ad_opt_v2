"""Tests for per-match-type panel ranking and synthetic list columns."""

from __future__ import annotations

import pandas as pd

from utils.keyword_candidates import (
    _append_synthetic_set,
    _apply_embedding_variant_by_match_type,
    _build_segment_panel_maps,
    _keywords_from_panel_by_match_type,
    _match_type_lists_to_columns,
    _union_keywords_from_match_types,
)


def _segment_row(region: str = "USA", match_types: str = "Broad; Phrase; Exact") -> pd.Series:
    return pd.Series({"region": region, "match_types": match_types})


def test_ranks_keywords_within_each_match_type_separately():
    kw_day = pd.DataFrame(
        [
            {"region": "USA", "keyword": "broad only", "match_type": "Broad", "all_conv": 100, "cost": 1},
            {"region": "USA", "keyword": "exact winner", "match_type": "Exact", "all_conv": 50, "cost": 1},
            {"region": "USA", "keyword": "broad only", "match_type": "Exact", "all_conv": 1, "cost": 1},
        ]
    )
    row = _segment_row()
    panel_by_mt, canonical, _keys_union = _build_segment_panel_maps(kw_day, row, None)
    by_mt = _keywords_from_panel_by_match_type(
        kw_day,
        row,
        top_n=5,
        volume_col="all_conv",
        allowed_keywords=None,
        panel_keys_by_mt=panel_by_mt,
        require_positive_volume=True,
    )
    assert by_mt["Broad"] == ["broad only"]
    assert set(by_mt["Exact"]) == {"broad only", "exact winner"}
    assert by_mt["Exact"][0] == "exact winner"
    assert by_mt["Phrase"] == []


def test_positive_keywords_is_union_of_match_type_lists():
    by_mt = {
        "Broad": ["alpha"],
        "Phrase": ["beta"],
        "Exact": ["alpha"],
    }
    cols = _match_type_lists_to_columns(by_mt, positive_col="positive_keywords")
    assert cols["broad_keywords"] == "alpha"
    assert cols["phrase_keywords"] == "beta"
    assert cols["exact_keywords"] == "alpha"
    assert set(cols["positive_keywords"].split("; ")) == {"alpha", "beta"}


def test_embedding_variant_ranks_per_match_type():
    import numpy as np

    emb_map = {
        "alpha": np.array([1.0, 0.0]),
        "beta": np.array([0.9, 0.1]),
        "gamma": np.array([0.0, 1.0]),
    }
    anchors = np.array([[1.0, 0.0]])
    row = _segment_row(match_types="Broad; Exact")
    panel_by_mt = {
        "Broad": {"alpha", "beta"},
        "Exact": {"gamma"},
    }
    canonical = {"alpha": "alpha", "beta": "beta", "gamma": "gamma"}
    keys_in_segment = {"alpha", "beta", "gamma"}

    def _pick_all(pool, _emb, _anchors, *, set_size: int) -> list[str]:
        return pool[:set_size]

    by_mt = _apply_embedding_variant_by_match_type(
        row,
        panel_by_mt,
        canonical,
        emb_map,
        anchors,
        top_n=2,
        selector=_pick_all,
        allowlist_keys_ordered=["alpha", "beta", "gamma"],
        enrollment_canonical=["alpha", "beta", "gamma"],
        segment="USA / Broad; Exact",
        source="synthetic_semantic_n2",
        variant_label="semantic",
        allowlist={"alpha", "beta", "gamma"},
    )
    assert by_mt["Broad"] == ["alpha", "beta"]
    # Exact has one intersection keyword; second slot padded from enrollment allowlist.
    assert by_mt["Exact"] == ["gamma", "alpha"]


def test_append_synthetic_set_writes_per_match_type_columns():
    synthetic_sets: list[dict] = []
    synth_cand: list[dict] = []
    row = _segment_row(match_types="Broad; Exact")
    by_mt = {"Broad": ["kw a"], "Exact": ["kw b"]}
    _append_synthetic_set(
        synthetic_sets=synthetic_sets,
        synth_cand=synth_cand,
        segment="USA / Broad; Exact",
        row=row,
        new_id="synthetic_test",
        by_match_type=by_mt,
        source="synthetic_top_conv_n10",
        positive_col="positive_keywords",
    )
    assert len(synthetic_sets) == 1
    rec = synthetic_sets[0]
    assert rec["broad_keywords"] == "kw a"
    assert rec["exact_keywords"] == "kw b"
    assert not rec["phrase_keywords"].strip()
    assert _union_keywords_from_match_types(by_mt) == ["kw a", "kw b"]
