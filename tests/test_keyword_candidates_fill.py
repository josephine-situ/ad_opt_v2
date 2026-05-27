"""Tests for top_conv allowlist padding."""

from __future__ import annotations

from utils.keyword_candidates import _fill_pool_to_top_n


def test_fill_pool_to_top_n_keeps_converters_then_pads_allowlist():
    pool = ["Conv A"]
    ranked = ["conv a", "top enroll", "second enroll"]
    segment = ["Conv A", "Top Enroll", "second enroll"]
    out = _fill_pool_to_top_n(
        pool,
        top_n=3,
        allowlist_ranked=ranked,
        segment_allowlist=segment,
    )
    assert out == ["Conv A", "Top Enroll", "second enroll"]


def test_fill_pool_to_top_n_does_not_exceed_top_n():
    pool = ["a", "b"]
    ranked = ["a", "b", "c", "d", "e"]
    out = _fill_pool_to_top_n(
        pool,
        top_n=3,
        allowlist_ranked=ranked,
        segment_allowlist=["a", "b", "c", "d", "e"],
    )
    assert len(out) == 3
