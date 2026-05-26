"""ridge_xgb_embed level predictions must match EnsembleModel.predict_levels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from campaign_opt.backends.tree_embed import (
    _blend_ridge_xgb_level,
    _build_candidate_feature_rows,
    _embed_candidate_predictions,
    _embed_linear_candidate_predictions,
    _probe_ridge_xgb_embed_on_embed_rows,
)
from campaign_opt.coefficients import ridge_embed_coeffs
from campaign_opt.decisions import apply_candidate_region_policy, build_segment_list, historical_budget_bounds
from campaign_opt.evaluation import EnsembleModel
from campaign_opt.features import prepare_modeling_data, train_holdout_split
from campaign_opt.modeling import refit_optimizer_model
from campaign_opt.schema import default_config_path, load_campaign_config
from utils.campaign_features import (
    add_segment_column,
    build_keyword_set_feature_table,
    load_campaign_day_panel,
)


def test_ridge_xgb_embed_matches_ensemble_levels():
    config_path = default_config_path("sys_think", "default")
    if not config_path.exists():
        return
    config = load_campaign_config(config_path)
    import json

    manifest = json.loads((config.exp_dir() / "model_manifest.json").read_text(encoding="utf-8"))
    df = prepare_modeling_data(config)
    train, holdout = train_holdout_split(df, config.model_policy.validation.holdout_days)
    production = (
        pd.concat([train, holdout], ignore_index=True).sort_values("date")
        if len(holdout)
        else train
    )
    panel = add_segment_column(load_campaign_day_panel(config.course))
    candidates = apply_candidate_region_policy(
        pd.read_csv(Path("data") / config.course / "processed" / "segment-keyword-candidates.csv"),
        config.constraints,
    )
    planning_date = pd.Timestamp(production["date"].max())
    ensemble = refit_optimizer_model("ensemble_ridge_xgb", production, config, manifest)
    assert isinstance(ensemble, EnsembleModel)

    set_features = build_keyword_set_feature_table(config.course)
    embed_rows, keys = _build_candidate_feature_rows(
        candidates, config, planning_date, set_features
    )
    segments = build_segment_list(candidates)
    bounds = historical_budget_bounds(panel, segments)
    ridge = next(m for m in ensemble.members if m.name == "ridge")
    xgb = next(m for m in ensemble.members if m.name == "xgboost")
    total_w = sum(m.weight for m in ensemble.members) or 1.0
    w_ridge = ridge.weight / total_w
    w_xgb = xgb.weight / total_w
    coeffs = ridge_embed_coeffs(
        ridge.pipeline,
        production,
        config,
        candidates,
        set_features,
        planning_date,
        segments,
    )

    import gurobipy as gp

    model = gp.Model("test_ridge_xgb")
    model.setParam("OutputFlag", 0)
    x_vars = {
        seg: model.addVar(lb=bounds[seg][0], ub=bounds[seg][1], name=f"x_{seg}")
        for seg in segments
    }
    linear_pred = _embed_linear_candidate_predictions(coeffs, keys, x_vars)
    tree_pred = _embed_candidate_predictions(
        model, xgb.pipeline, embed_rows, keys, x_vars, bounds, config
    )
    _blend_ridge_xgb_level(linear_pred, tree_pred, w_ridge=w_ridge, w_xgb=w_xgb)
    _probe_ridge_xgb_embed_on_embed_rows(
        ensemble,
        embed_rows,
        keys,
        coeffs,
        xgb.pipeline,
        bounds,
        config,
        w_ridge=w_ridge,
        w_xgb=w_xgb,
        budgets=[0.0, 50.0, 120.0],
        tol=1e-4,
    )
