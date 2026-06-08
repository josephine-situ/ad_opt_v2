"""Tests for tree preprocessor context encoding."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer

from campaign_opt.linear_design import split_context_columns_by_dtype
from campaign_opt.training_matrix import build_preprocessor, prep_xy


def test_split_context_columns_by_dtype():
    df = pd.DataFrame(
        {
            "embed_cohesion": [0.1, 0.2],
            "season": ["Fall", "Winter"],
            "n_broad": [1, 2],
        }
    )
    numeric, categorical = split_context_columns_by_dtype(
        df, ["embed_cohesion", "season", "n_broad"]
    )
    assert numeric == ["embed_cohesion", "n_broad"]
    assert categorical == ["season"]


def test_tree_preprocessor_keeps_numeric_context_columns():
    df = pd.DataFrame(
        {
            "region": ["A", "B"],
            "match_types": ["Broad", "Exact"],
            "is_broad_match": [1, 0],
            "daily_budget": [10.0, 20.0],
            "embed_cohesion": [0.38, 0.41],
            "season": ["Fall", "Fall"],
            "clicks": [1.0, 2.0],
            "segment": ["A / Broad", "B / Exact"],
        }
    )
    feature_cols = ["embed_cohesion", "season"]
    prep: ColumnTransformer = build_preprocessor(feature_cols, df)
    X, _ = prep_xy(df, "clicks", feature_cols)
    prep.fit(X)
    out_names = list(prep.get_feature_names_out())
    assert "ctx_num__embed_cohesion" in out_names
    assert not any(name.startswith("ctx_cat__embed_cohesion") for name in out_names)
    assert any("season" in name for name in out_names)
