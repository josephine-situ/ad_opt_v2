"""Tests for segment-keyword-candidates verification."""

from __future__ import annotations

import pandas as pd

from utils.keyword_candidates import DEFAULT_TOP_N_VALUES, verify_segment_keyword_candidates


def test_verify_sys_think_candidates_passes():
    issues = verify_segment_keyword_candidates("sys_think", top_n_values=DEFAULT_TOP_N_VALUES)
    assert issues == [], issues


def test_verify_detects_missing_top_conv_cap():
    candidates = pd.DataFrame(
        [
            {"segment": "USA / Broad", "keyword_set_id": "ks_a", "source": "historical"},
            {"segment": "USA / Broad", "keyword_set_id": "syn_b", "source": "synthetic_top_conv_n10"},
        ]
    )
    extended = pd.DataFrame(
        [
            {
                "keyword_set_id": "ks_a",
                "positive_keywords": "systems thinking course",
                "broad_keywords": "systems thinking course",
                "phrase_keywords": "",
                "exact_keywords": "",
            },
            {
                "keyword_set_id": "syn_b",
                "positive_keywords": "systems thinking course",
                "broad_keywords": "systems thinking course",
                "phrase_keywords": "",
                "exact_keywords": "",
            },
        ]
    )
    issues = verify_segment_keyword_candidates(
        "sys_think",
        candidates,
        extended,
        top_n_values=(10, 20, 40),
    )
    assert any("synthetic_top_conv_n20" in issue for issue in issues)
    assert any("synthetic_top_conv_n40" in issue for issue in issues)
