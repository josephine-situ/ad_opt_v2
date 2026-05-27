"""Tree embed must match sklearn at all budget points (regression for leaf pruning bug)."""

from __future__ import annotations

from pathlib import Path

import gurobipy as gp
import joblib
import numpy as np
import pandas as pd
from gurobipy import GRB

from campaign_opt.backends.tree_embed import _build_candidate_feature_rows
from campaign_opt.backends.tree_embedding import (
    _budget_affine,
    _raw_budget_breakpoints_from_trees,
    embed_tree_prediction,
    get_tree_path_sets,
)
from campaign_opt.decisions import (
    apply_candidate_region_policy,
    build_segment_list,
    historical_budget_bounds,
)
from campaign_opt.modeling import _prep_xy
from campaign_opt.optimize import _fit_and_save_embed_model
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
    x_proc: np.ndarray,
    budget: float,
    budget_lo: float,
    budget_hi: float,
) -> float:
    tree_paths, base_or_n, kind = get_tree_path_sets(pipeline)
    budget_idx, budget_mean, budget_scale = _budget_affine(pipeline)
    model = gp.Model("embed_check")
    model.setParam("OutputFlag", 0)
    x = model.addVar(lb=float(budget), ub=float(budget), name="x")
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
    )
    model.optimize()
    assert model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL)
    return float(pred_var.X)


def test_embed_matches_sklearn_at_tree_thresholds():
    """Embedding must be exact at every budget — especially at tree split thresholds."""
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
    model_path = _fit_and_save_embed_model(
        config, manifest, production, out_dir, tune=False
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

    tree_paths, base_or_n, kind = get_tree_path_sets(pipeline)
    budget_idx, budget_mean, budget_scale = _budget_affine(pipeline)

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

    breakpoints = _raw_budget_breakpoints_from_trees(
        tree_paths, np.asarray(x_proc, dtype=np.float32),
        budget_idx, budget_mean, budget_scale, budget_lo, budget_hi,
    )
    eps = max(1e-6, (budget_hi - budget_lo) * 1e-9)
    test_budgets = {budget_lo, budget_hi, (budget_lo + budget_hi) / 2}
    for bp in breakpoints:
        test_budgets.add(bp)
        test_budgets.add(max(budget_lo, bp - eps))
        test_budgets.add(min(budget_hi, bp + eps))

    max_diff = 0.0
    for budget in sorted(test_budgets):
        sk = float(pipeline.predict(feature_row_at(budget))[0])
        milp = _milp_level_at_budget(
            pipeline,
            x_proc=x_proc,
            budget=budget,
            budget_lo=budget_lo,
            budget_hi=budget_hi,
        )
        diff = abs(sk - milp)
        max_diff = max(max_diff, diff)
        assert diff < 1e-4, (
            f"Embed != sklearn at budget={budget:.4f}: "
            f"sklearn={sk:.6f}, milp={milp:.6f}, diff={diff:.6g}"
        )
    print(f"  max embed vs sklearn diff: {max_diff:.2e} across {len(test_budgets)} budget points")
