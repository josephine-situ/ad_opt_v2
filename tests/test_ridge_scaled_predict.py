"""Ridge MILP design: scaled fit must match predict on raw holdout rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import r2_score

from campaign_opt.features import prepare_modeling_data
from campaign_opt.linear_design import (
    build_linear_milp_design_matrix,
    fit_linear_milp_ridge,
    scale_milp_design_matrix,
)
from campaign_opt.modeling import _level_metrics
from campaign_opt.schema import default_config_path, load_campaign_config


def test_scaled_ridge_holdout_r2_beats_raw_on_sys_think():
    config = load_campaign_config(default_config_path("sys_think", "default"))
    df = prepare_modeling_data(config)
    holdout_days = config.model_policy.validation.holdout_days
    cutoff = df["date"].max() - __import__("pandas").Timedelta(days=holdout_days - 1)
    train = df[df["date"] < cutoff].copy()
    holdout = df[df["date"] >= cutoff].copy()

    train_d = build_linear_milp_design_matrix(train, config)
    hold_d = build_linear_milp_design_matrix(holdout, config, columns=train_d.x_columns)

    artifact = fit_linear_milp_ridge(train_d, config, alpha=0.01)
    assert artifact.uses_scaled_fit()

    pred = np.clip(artifact.predict_design_frame(holdout), 0, None)
    m = _level_metrics(hold_d.y, pred)
    assert m["holdout_r2_levels"] > 0.2

    from sklearn.linear_model import Ridge

    raw = Ridge(alpha=0.01)
    raw.fit(train_d.X.values, train_d.y)
    pred_raw = np.clip(raw.predict(hold_d.X.values), 0, None)
    assert m["holdout_r2_levels"] > r2_score(hold_d.y, pred_raw)


def test_scaled_train_predict_matches_scaled_matrix():
    config = load_campaign_config(default_config_path("sys_think", "default"))
    df = prepare_modeling_data(config).head(400)
    design = build_linear_milp_design_matrix(df, config)
    artifact = fit_linear_milp_ridge(design, config, alpha=1.0)
    from campaign_opt.linear_design import ridge_numeric_scale_column_names

    scale_cols = ridge_numeric_scale_column_names(
        design.sub, design.context_cols, design.x_columns
    )
    Xs, _ = scale_milp_design_matrix(design.X, scale_cols, scaler=artifact.scaler)
    direct = artifact.model.predict(Xs.values)
    via_frame = artifact.predict_design_frame(design.sub)
    np.testing.assert_allclose(direct, via_frame, rtol=1e-5, atol=1e-5)
