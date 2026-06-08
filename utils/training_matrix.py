"""Shared feature-matrix helpers for model training and prediction.

Functions here are used across modeling, evaluation, SHAP, tree embedding,
and recency weighting -- extracted from ``modeling.py`` so they have a stable
public API instead of underscore-prefixed names imported cross-module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.linear_design import split_context_columns_by_dtype
from utils.campaign_features import (
    SEGMENT_BROAD_MATCH_COL,
    TREE_SEGMENT_FEATURE_COLS,
    add_segment_match_type_indicators,
)


def ensure_tree_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add segment/match-type indicator columns if missing."""
    if all(col in df.columns for col in TREE_SEGMENT_FEATURE_COLS):
        return df
    return add_segment_match_type_indicators(df)


def training_subframe(
    df: pd.DataFrame,
    target: str,
    *,
    y_col: str | None = None,
) -> pd.DataFrame:
    """Drop rows missing the target, budget, or region, and ensure segment features."""
    y_name = y_col or target
    return ensure_tree_segment_features(
        df.dropna(subset=[y_name, "daily_budget", "region"]).copy()
    )


def prep_xy(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    *,
    y_col: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build (X, y) for sklearn pipelines: segment indicators + budget + context features."""
    y_name = y_col or target
    sub = training_subframe(df, target, y_col=y_col)
    numeric_ctx, _ = split_context_columns_by_dtype(sub, feature_cols)
    for col in numeric_ctx:
        sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)
    use_cols = [*TREE_SEGMENT_FEATURE_COLS, "daily_budget", *feature_cols]
    return sub[use_cols], sub[y_name].astype(float).values


def build_preprocessor(feature_cols: list[str], sample: pd.DataFrame) -> ColumnTransformer:
    """ColumnTransformer for budget + region + match-type + context features."""
    ctx_numeric, ctx_cat = split_context_columns_by_dtype(sample, feature_cols)
    transformers: list[tuple[str, Any, list[str]]] = [
        ("num", StandardScaler(), ["daily_budget"]),
        (
            "region",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["region"],
        ),
        ("match", "passthrough", [SEGMENT_BROAD_MATCH_COL]),
    ]
    if ctx_numeric:
        transformers.append(("ctx_num", "passthrough", ctx_numeric))
    if ctx_cat:
        transformers.append(
            (
                "ctx_cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ctx_cat,
            )
        )
    return ColumnTransformer(transformers)
