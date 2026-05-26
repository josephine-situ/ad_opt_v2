"""Tree embed must match sklearn at fixed budgets (regression for leaf tightening)."""

from __future__ import annotations

from pathlib import Path

import gurobipy as gp
import joblib
import numpy as np
import pandas as pd
from gurobipy import GRB

from campaign_opt.backends.tree_embed import _build_candidate_feature_rows
from campaign_opt.backends.tree_embedding import (
    _booster_leaf_nodes_at_budgets,
    _budget_affine,
    _tighten_allowed_leaf_nodes,
    embed_tree_prediction,
    get_tree_path_sets,
)
from campaign_opt.decisions import (
    apply_candidate_region_policy,
    build_segment_list,
    historical_budget_bounds,
)
from campaign_opt.modeling import _prep_xy
from campaign_opt.optimize import _tree_embed_model_path
from campaign_opt.schema import default_config_path, load_campaign_config
from utils.campaign_features import (
    add_segment_column,
    build_keyword_set_feature_table,
    get_context_feature_columns,
    load_campaign_day_panel,
)


def _milp_level_at_budget(
    pipeline,
    *,
    feature_row: pd.DataFrame,
    x_proc: np.ndarray,
    budget: float,
    budget_lo: float,
    budget_hi: float,
    allowed: dict[int, set[int]] | None,
) -> float:
    tree_paths, base_or_n, kind = get_tree_path_sets(pipeline)
    budget_idx, budget_mean, budget_scale = _budget_affine(pipeline)
    model = gp.Model("embed_check")
    model.setParam("OutputFlag", 0)
    x = model.addVar(lb=budget_lo, ub=budget_hi, name="x")
    x.lb = float(budget)
    x.ub = float(budget)
    pred_var = embed_tree_prediction(
        model,
        tree_paths=tree_paths,
        x_proc_row=x_proc,
        budget_var=x,
        budget_lo=budget_lo,
        budget_hi=budget_hi,
        budget_idx=budget_idx,
        budget_mean=budget_mean,
        budget_scale=budget_scale,
        model_kind=kind,
        base_or_n_trees=base_or_n,
        name_prefix="emb",
        allowed_leaf_nodes=allowed,
    )
    model.optimize()
    assert model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL)
    return float(pred_var.X)


def test_embed_matches_sklearn_at_low_budget_after_leaf_tighten():
    """A / Phrase; Exact / ks_0013 must be feasible and exact at budget=0."""
    from campaign_opt.features import prepare_modeling_data, train_holdout_split

    if not default_config_path("sys_think", "default").exists():
        return
    config = load_campaign_config(default_config_path("sys_think", "default"))
    out_dir = config.exp_dir()
    manifest_path = out_dir / "model_manifest.json"
    if not manifest_path.exists():
        return
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    model_path = _tree_embed_model_path(
        config, manifest, production, out_dir, out_dir / "winner_model.joblib"
    )
    pipeline = joblib.load(model_path)
    planning_date = pd.Timestamp(production["date"].max())
    set_features = build_keyword_set_feature_table(config.course)
    embed_rows, keys = _build_candidate_feature_rows(
        candidates, config, planning_date, set_features
    )
    feature_cols = get_context_feature_columns(config.context_features)
    target = config.target
    bounds = historical_budget_bounds(panel, build_segment_list(candidates))

    seg, kid = "A / Phrase; Exact", "ks_0013"
    i = next(j for j, (s, k) in enumerate(keys) if s == seg and str(k) == kid)
    budget_lo, budget_hi = bounds[seg]

    def feature_row_at(budget: float) -> pd.DataFrame:
        row = embed_rows.iloc[i : i + 1].copy()
        row["daily_budget"] = budget
        row[target] = 0.0
        X, _ = _prep_xy(row, target, feature_cols)
        return X

    X0 = feature_row_at(0.0)
    x_proc = np.asarray(pipeline[:-1].transform(X0), dtype=float).ravel()
    allowed = _tighten_allowed_leaf_nodes(
        _booster_leaf_nodes_at_budgets(
            pipeline, feature_row_at(budget_lo), target, feature_cols, budget_lo, budget_hi
        ),
        pipeline,
        feature_row_at((budget_lo + budget_hi) / 2),
        target,
        feature_cols,
        budget_lo,
        budget_hi,
    )
    sk = float(pipeline.predict(feature_row_at(0.0))[0])
    milp = _milp_level_at_budget(
        pipeline,
        feature_row=feature_row_at(0.0),
        x_proc=x_proc,
        budget=0.0,
        budget_lo=budget_lo,
        budget_hi=budget_hi,
        allowed=allowed,
    )
    assert abs(sk - milp) < 1e-5
