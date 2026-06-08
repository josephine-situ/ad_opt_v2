"""Ridge MILP design: scaled fit must match predict on the design matrix."""

from __future__ import annotations

import numpy as np

from campaign_opt.features import prepare_modeling_data
from campaign_opt.linear_design import (
    build_linear_milp_design_matrix,
    fit_linear_milp_ridge,
    ridge_numeric_scale_column_names,
    scale_milp_design_matrix,
)
from campaign_opt.schema import default_config_path, load_campaign_config


def test_scaled_train_predict_matches_scaled_matrix():
    config = load_campaign_config(default_config_path("sys_think", "default"))
    df = prepare_modeling_data(config).head(400)
    design = build_linear_milp_design_matrix(df, config)
    artifact = fit_linear_milp_ridge(design, config, alpha=1.0)
    scale_cols = ridge_numeric_scale_column_names(
        design.sub, design.context_cols, design.x_columns
    )
    Xs, _ = scale_milp_design_matrix(design.X, scale_cols, scaler=artifact.scaler)
    direct = artifact.model.predict(Xs.values)
    via_frame = artifact.predict_design_frame(design.sub)
    np.testing.assert_allclose(direct, via_frame, rtol=1e-5, atol=1e-5)
