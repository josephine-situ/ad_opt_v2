"""Tests for unified keyword-set counting and deduplication."""

from __future__ import annotations

import pandas as pd

from utils.campaign_features import (
    count_unique_keywords_in_set,
    keyword_set_content_fingerprint,
    keywords_from_keyword_set_row,
)
from utils.keyword_candidates import (
    _append_synthetic_set,
    _frozenset_union,
    _match_type_lists_to_columns,
)


def test_count_unique_keywords_prefers_match_type_columns():
    row = {
        "broad_keywords": "alpha; beta",
        "phrase_keywords": "alpha",
        "positive_keywords": "ignored keyword",
    }
    assert keywords_from_keyword_set_row(row) == ("alpha", "beta")
    assert count_unique_keywords_in_set(row) == 2


def test_match_type_columns_deduplicate_within_lists():
    by_mt = {"Broad": ["foo", "foo", "bar"], "Phrase": [], "Exact": []}
    cols = _match_type_lists_to_columns(by_mt, positive_col="positive_keywords")
    assert cols["broad_keywords"] == "bar; foo"
    assert count_unique_keywords_in_set(cols) == 2


def test_emit_synthetic_set_reuses_existing_fingerprint():
    synthetic_sets: list[dict] = []
    synth_cand: list[dict] = []
    row = pd.Series({"region": "A", "match_types": "Broad"})
    by_mt = {"Broad": ["systems thinking course"], "Phrase": [], "Exact": []}
    fp = _frozenset_union(by_mt)
    fingerprint_to_id = {fp: "existing_set"}

    def emit(by_mt_local: dict[str, list[str]], source: str) -> None:
        fp_local = _frozenset_union(by_mt_local)
        if fp_local in fingerprint_to_id:
            synth_cand.append(
                {
                    "segment": "A / Broad",
                    "region": "A",
                    "match_types": "Broad",
                    "keyword_set_id": fingerprint_to_id[fp_local],
                    "source": source,
                }
            )
            return
        _append_synthetic_set(
            synthetic_sets=synthetic_sets,
            synth_cand=synth_cand,
            segment="A / Broad",
            row=row,
            new_id="new_unused",
            by_match_type=by_mt_local,
            source=source,
            positive_col="positive_keywords",
        )
        fingerprint_to_id[fp_local] = "new_unused"

    emit(by_mt, "synthetic_semantic_n10")
    assert synthetic_sets == []
    assert len(synth_cand) == 1
    assert synth_cand[0]["keyword_set_id"] == "existing_set"


def test_keyword_set_fingerprint_ignores_positive_col_when_match_types_present():
    row = {
        "broad_keywords": "alpha",
        "positive_keywords": "beta; gamma",
    }
    assert keyword_set_content_fingerprint(row) == frozenset({"alpha"})
