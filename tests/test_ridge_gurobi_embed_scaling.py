"""Gurobi ridge linear embed matches LinearMilpRidgeModel with scaled fit."""

from __future__ import annotations

import gurobipy as gp
import numpy as np
import pandas as pd
import pytest

from campaign_opt.backends.milp_core import _eval_gurobi_expr
from campaign_opt.backends.tree_embed import (
    _build_candidate_feature_rows,
    _embed_linear_candidate_predictions,
)
from campaign_opt.decisions import apply_candidate_region_policy, historical_budget_bounds
from campaign_opt.linear_design import LinearMilpRidgeModel
from campaign_opt.modeling import fit_ridge_full, refit_optimizer_model
from campaign_opt.schema import default_config_path, load_campaign_config
from utils.campaign_features import add_segment_column, build_keyword_set_feature_table, load_campaign_day_panel


@pytest.mark.parametrize("budget", [0.0, 35.7, 120.0])
def test_embed_linear_matches_predict_design_frame(budget: float):
    config_path = default_config_path("default")
    if not config_path.exists():
        pytest.skip("sys_think config not present")
    config = load_campaign_config(config_path)
    from campaign_opt.features import prepare_modeling_data

    df = prepare_modeling_data(config).head(800)
    artifact = fit_ridge_full(df, config, hyperparams={"alpha": 10.0})
    assert isinstance(artifact, LinearMilpRidgeModel)
    assert artifact.uses_scaled_fit()

    candidates = apply_candidate_region_policy(
        pd.read_csv(f"data/{config.course}/processed/segment-keyword-candidates.csv"),
        config.constraints,
    )
    planning_date = pd.Timestamp(df["date"].max())
    set_features = build_keyword_set_feature_table(config.course)
    embed_rows, keys = _build_candidate_feature_rows(
        candidates, config, planning_date, set_features
    )
    panel = add_segment_column(load_campaign_day_panel(config.course))
    segments = sorted({seg for seg, _ in keys})
    bounds = historical_budget_bounds(panel, segments)

    model = gp.Model("test_linear_embed")
    model.setParam("OutputFlag", 0)
    x_vars = {
        seg: model.addVar(lb=bounds[seg][0], ub=bounds[seg][1], name=f"x_{seg}")
        for seg in segments
    }
    model.update()
    linear_pred = _embed_linear_candidate_predictions(
        artifact, embed_rows, keys, x_vars
    )

    i = 0
    seg, kid = keys[i]
    embedded = _eval_gurobi_expr(linear_pred[(seg, kid)], {x_vars[seg]: budget})
    assert embedded is not None

    row = embed_rows.iloc[i : i + 1].copy()
    row["daily_budget"] = budget
    row[config.target] = 0.0
    sklearn_level = float(artifact.predict_design_frame(row)[0])
    assert embedded == pytest.approx(sklearn_level, rel=1e-5, abs=1e-4)
