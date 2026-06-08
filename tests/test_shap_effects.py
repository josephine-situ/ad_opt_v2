"""Tests for optional SHAP effect summaries."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from campaign_opt.training_matrix import build_preprocessor, prep_xy
from campaign_opt.shap_effects import compute_mean_shap_effects, shap_available


def test_compute_mean_shap_effects_without_shap():
    df = pd.DataFrame(
        {
            "region": ["A", "B"],
            "match_types": ["Broad", "Exact"],
            "is_broad_match": [1, 0],
            "daily_budget": [10.0, 20.0],
            "clicks": [1.0, 2.0],
            "segment": ["A / Broad", "B / Exact"],
        }
    )
    feature_cols: list[str] = []
    X, y = prep_xy(df, "clicks", feature_cols)
    pipe = Pipeline(
        [
            ("prep", build_preprocessor(feature_cols, df)),
            ("model", RandomForestRegressor(n_estimators=5, max_depth=2, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    if not shap_available():
        assert compute_mean_shap_effects(pipe, df, "clicks", feature_cols) is None
        return

    effects = compute_mean_shap_effects(pipe, df, "clicks", feature_cols)
    assert effects is not None
    assert effects
    assert all(isinstance(v, float) for v in effects.values())


@pytest.mark.skipif(not shap_available(), reason="shap extra not installed")
def test_compute_mean_shap_effects_tree_model():
    df = pd.DataFrame(
        {
            "region": ["A", "B", "A", "B"],
            "match_types": ["Broad", "Exact", "Broad", "Exact"],
            "is_broad_match": [1, 0, 1, 0],
            "daily_budget": [10.0, 20.0, 12.0, 18.0],
            "embed_cohesion": [0.38, 0.41, 0.39, 0.42],
            "clicks": [1.0, 2.0, 1.5, 2.5],
            "segment": ["A / Broad", "B / Exact", "A / Broad", "B / Exact"],
        }
    )
    feature_cols = ["embed_cohesion"]
    X, y = prep_xy(df, "clicks", feature_cols)
    pipe = Pipeline(
        [
            ("prep", build_preprocessor(feature_cols, df)),
            ("model", RandomForestRegressor(n_estimators=5, max_depth=2, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    effects = compute_mean_shap_effects(pipe, df, "clicks", feature_cols)
    assert effects is not None
    assert "embed_cohesion" in effects
    assert not any(k.startswith("embed_cohesion_0.") for k in effects)
