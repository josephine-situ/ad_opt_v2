"""Tests for tree-embed budget scaling and leaf filtering."""

from __future__ import annotations

import numpy as np

from campaign_opt.backends.tree_embedding import (
    _intervals_overlap,
    _leaf_budget_interval,
    _processed_budget_bounds,
)


def test_processed_budget_bounds_matches_standard_scaler():
    mean, scale = 50.0, 10.0
    lo, hi = 20.0, 120.0
    min_proc, max_proc = _processed_budget_bounds(lo, hi, mean, scale)
    assert min_proc == (lo - mean) / scale
    assert max_proc == (hi - mean) / scale


def test_processed_budget_bounds_no_scale():
    min_proc, max_proc = _processed_budget_bounds(5.0, 15.0, 3.0, 0.0)
    assert min_proc == 2.0
    assert max_proc == 12.0


def test_lt_split_bound_allows_negative_processed_budget():
    """Centered scaler: thr - eps must not be clamped to 0 (unlike ad_opt with_mean=False)."""
    thr = -0.25
    bound = thr - 1e-5
    assert bound < 0.0


def test_leaf_budget_interval_overlap():
    leaf = _leaf_budget_interval([("lt", 0.5), ("ge", -1.0)])
    assert leaf is not None
    assert _intervals_overlap(leaf, (-0.8, 0.2))


def test_leaf_budget_interval_empty():
    assert _leaf_budget_interval([("lt", -1.0), ("ge", 0.5)]) is None
