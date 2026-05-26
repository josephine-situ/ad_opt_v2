"""Exact tree-embedded Gurobi MILP for RF / XGB winners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gurobipy as gp
import joblib
import numpy as np
import pandas as pd
from gurobipy import GRB

from campaign_opt.backends.milp_core import historical_budget_bounds, solve_campaign_milp
from campaign_opt.backends.tree_embedding import (
    _booster_leaf_nodes_at_budgets,
    _budget_affine,
    _tighten_allowed_leaf_nodes,
    embed_tree_prediction,
    get_tree_path_sets,
)
from campaign_opt.decisions import build_segment_list, candidates_by_segment, region_of_segment
from campaign_opt.evaluation import build_segment_decision_rows
from campaign_opt.modeling import _prep_xy
from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import (
    add_segment_match_type_indicators,
    build_keyword_set_feature_table,
    get_context_feature_columns,
)
from utils.date_features import calendar_vector_for_date


def _build_candidate_feature_rows(
    candidates: pd.DataFrame,
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    feature_cols = get_context_feature_columns(config.context_features)
    k_map = candidates_by_segment(candidates)
    set_feats = set_features.set_index("keyword_set_id")
    rows: list[dict] = []
    keys: list[tuple[str, str]] = []

    for seg in sorted(k_map.keys()):
        region = region_of_segment(seg)
        cal = calendar_vector_for_date(planning_date, region, config.course)
        for kid in k_map[seg]:
            kid = str(kid)
            row: dict = {
                "segment": seg,
                "daily_budget": 0.0,
                "keyword_set_id": kid,
                **cal,
            }
            if kid in set_feats.index:
                for col in feature_cols:
                    if col in set_feats.columns:
                        row[col] = set_feats.loc[kid, col]
            rows.append(row)
            keys.append((seg, kid))

    out = pd.DataFrame(rows)
    out = add_segment_match_type_indicators(out)
    for col in feature_cols:
        if col not in out.columns:
            out[col] = np.nan
    return out, keys


def _create_decision_vars(
    model: gp.Model,
    segments: list[str],
    k_map: dict[str, list[str]],
    bounds: dict[str, tuple[float, float]],
) -> tuple[dict[str, Any], dict[tuple[str, str], Any]]:
    x_vars: dict[str, Any] = {}
    y_vars: dict[tuple[str, str], Any] = {}
    for seg in segments:
        lo, hi = bounds[seg]
        x_vars[seg] = model.addVar(lb=lo, ub=hi, name=f"x_{seg}")
        for k in k_map.get(seg, []):
            y_vars[(seg, k)] = model.addVar(vtype=GRB.BINARY, name=f"y_{seg}_{k}")
        model.addConstr(
            gp.quicksum(y_vars[(seg, k)] for k in k_map.get(seg, [])) == 1,
            name=f"one_set_{seg}",
        )
    return x_vars, y_vars


def _embed_candidate_predictions(
    model: gp.Model,
    pipeline,
    embed_rows: pd.DataFrame,
    keys: list[tuple[str, str]],
    x_vars: dict[str, Any],
    bounds: dict[str, tuple[float, float]],
    config: CampaignOptConfig,
) -> dict[tuple[str, str], Any]:
    feature_cols = get_context_feature_columns(config.context_features)
    target = config.target
    probe = embed_rows.copy()
    probe[target] = 0.0
    X_raw, _ = _prep_xy(probe, target, feature_cols)
    X_proc = pipeline[:-1].transform(X_raw)

    base_preds = pipeline.predict(X_raw)
    tree_paths, base_or_n, kind = get_tree_path_sets(pipeline)
    budget_idx, budget_mean, budget_scale = _budget_affine(pipeline)
    allowed_by_key: dict[tuple[str, str], dict[int, set[int]]] | None = None
    if kind == "xgboost":
        allowed_by_key = {}
        for i, _pred in enumerate(base_preds):
            seg, kid = keys[i]
            budget_lo, budget_hi = bounds[seg]
            allowed_by_key[(seg, kid)] = _tighten_allowed_leaf_nodes(
                _booster_leaf_nodes_at_budgets(
                    pipeline,
                    X_raw.iloc[i : i + 1],
                    target,
                    feature_cols,
                    budget_lo,
                    budget_hi,
                ),
                pipeline,
                X_raw.iloc[i : i + 1],
                target,
                feature_cols,
                budget_lo,
                budget_hi,
            )

    pred_by_key: dict[tuple[str, str], Any] = {}
    for i, pred in enumerate(base_preds):
        if pred < 0:
            continue
        seg, kid = keys[i]
        budget_lo, budget_hi = bounds[seg]
        pred_by_key[(seg, kid)] = embed_tree_prediction(
            model,
            tree_paths=tree_paths,
            x_proc_row=np.asarray(X_proc[i]).ravel(),
            budget_var=x_vars[seg],
            budget_lo=budget_lo,
            budget_hi=budget_hi,
            budget_idx=budget_idx,
            budget_mean=budget_mean,
            budget_scale=budget_scale,
            model_kind=kind,
            base_or_n_trees=base_or_n,
            name_prefix=f"emb_{seg.replace(' ', '_')}_{kid}",
            allowed_leaf_nodes=(
                allowed_by_key.get((seg, kid)) if allowed_by_key is not None else None
            ),
        )
    if not pred_by_key:
        raise RuntimeError("All candidate rows have negative prediction at budget=0")
    return pred_by_key


def _disable_pruned_keyword_sets(
    model: gp.Model,
    y_vars: dict[tuple[str, str], Any],
    k_map: dict[str, list[str]],
    pred_by_key: dict[tuple[str, str], Any],
) -> None:
    """Keyword sets pruned at budget=0 cannot be selected."""
    for seg, kids in k_map.items():
        for k in kids:
            if (seg, str(k)) not in pred_by_key:
                model.addConstr(y_vars[(seg, k)] == 0, name=f"pruned_{seg}_{k}")
        if not any((seg, str(k)) in pred_by_key for k in kids):
            raise RuntimeError(f"Segment {seg!r}: all keyword sets pruned at budget=0")


def _apply_fixed_decisions(
    model: gp.Model,
    segments: list[str],
    k_map: dict[str, list[str]],
    x_vars: dict[str, Any],
    y_vars: dict[tuple[str, str], Any],
    *,
    fixed_keyword_sets: dict[str, str] | None,
    fixed_budgets: dict[str, float] | None,
) -> None:
    if fixed_budgets:
        for seg in segments:
            if seg in fixed_budgets:
                b = float(fixed_budgets[seg])
                x_vars[seg].lb = b
                x_vars[seg].ub = b
    if fixed_keyword_sets:
        for seg in segments:
            chosen = str(fixed_keyword_sets.get(seg, ""))
            for k in k_map.get(seg, []):
                fix_val = 1.0 if str(k) == chosen else 0.0
                y_vars[(seg, k)].lb = fix_val
                y_vars[(seg, k)].ub = fix_val


def _warn_if_milp_external_pred_mismatch(plan: pd.DataFrame, *, tol: float = 0.01) -> None:
    """Warn when embedded MILP predictions disagree with the sklearn pipeline."""
    if "milp_pred" not in plan.columns or "external_model_pred" not in plan.columns:
        return
    milp = pd.to_numeric(plan["milp_pred"], errors="coerce")
    ext = pd.to_numeric(plan["external_model_pred"], errors="coerce")
    diff = (milp - ext).abs()
    valid = diff.notna()
    if not valid.any():
        return
    max_diff = float(diff[valid].max())
    if max_diff <= tol:
        return
    worst_idx = diff[valid].idxmax()
    row = plan.loc[worst_idx]
    print(
        f"[Warn] MILP vs external model prediction mismatch: "
        f"max|milp_pred - external_model_pred| = {max_diff:.6g} > {tol} "
        f"(segment={row['segment']!r}, milp_pred={row['milp_pred']}, "
        f"external_model_pred={row['external_model_pred']})"
    )


def _external_incremental_pred_by_segment(
    plan: pd.DataFrame,
    pipeline: Any,
    train: pd.DataFrame,
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
) -> pd.DataFrame:
    """Independent model check: same plan features, returning f(plan) and f(plan)-f(0)."""
    feature_cols = get_context_feature_columns(config.context_features)
    target = config.target
    plan_dec = plan[["segment", "daily_budget", "keyword_set_id"]].copy()
    plan_dec["segment"] = plan_dec["segment"].astype(str)
    plan_dec["keyword_set_id"] = plan_dec["keyword_set_id"].astype(str)
    plan_dec["daily_budget"] = pd.to_numeric(plan_dec["daily_budget"], errors="coerce")
    segments = plan_dec["segment"].tolist()
    _ = train  # train no longer needed for plan-matched f(0)

    decision_rows = build_segment_decision_rows(
        plan_dec, planning_date, set_features, config.course, feature_cols
    )
    baseline_rows = decision_rows.copy()
    baseline_rows["daily_budget"] = 0.0
    if target not in decision_rows.columns:
        decision_rows[target] = 0.0
    if target not in baseline_rows.columns:
        baseline_rows[target] = 0.0
    if "region" not in decision_rows.columns:
        decision_rows["region"] = decision_rows["segment"].astype(str).map(region_of_segment)
    if "region" not in baseline_rows.columns:
        baseline_rows["region"] = baseline_rows["segment"].astype(str).map(region_of_segment)
    X_dec, _ = _prep_xy(decision_rows, target, feature_cols)
    X_zero, _ = _prep_xy(baseline_rows, target, feature_cols)
    pred_dec = np.asarray(pipeline.predict(X_dec), dtype=float)
    pred_zero = np.asarray(pipeline.predict(X_zero), dtype=float)
    return pd.DataFrame(
        {
            "segment": segments,
            "external_model_pred": pred_dec,
            "pred_over_base": pred_dec - pred_zero,
        }
    )


def solve_tree_embed_campaign_milp(
    config: CampaignOptConfig,
    model_path: Path,
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    total_budget: float,
    output_dir: Path,
    planning_date: pd.Timestamp,
    time_limit: int = 600,
    write_outputs: bool = True,
    fixed_keyword_sets: dict[str, str] | None = None,
    fixed_budgets: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Embed winner tree model exactly in Gurobi (ad_opt-style leaf formulation)."""
    pipeline = joblib.load(model_path)
    set_features = build_keyword_set_feature_table(config.course)
    embed_rows, keys = _build_candidate_feature_rows(
        candidates, config, planning_date, set_features
    )

    segments = build_segment_list(candidates)
    k_map = candidates_by_segment(candidates)
    bounds = historical_budget_bounds(panel, segments)

    model = gp.Model("campaign_tree_embed")
    model.setParam("OutputFlag", 1)
    model.setParam("TimeLimit", time_limit)
    x_vars, y_vars = _create_decision_vars(model, segments, k_map, bounds)
    _apply_fixed_decisions(
        model, segments, k_map, x_vars, y_vars,
        fixed_keyword_sets=fixed_keyword_sets,
        fixed_budgets=fixed_budgets,
    )
    pred_by_key = _embed_candidate_predictions(
        model, pipeline, embed_rows, keys, x_vars, bounds, config
    )
    _disable_pruned_keyword_sets(model, y_vars, k_map, pred_by_key)

    def segment_predictor(seg: str, x_var: Any, y_vars_map: dict, k_map_local: dict) -> Any:
        expr = gp.LinExpr()
        for k in k_map_local.get(seg, []):
            key = (seg, str(k))
            if key in pred_by_key:
                expr += pred_by_key[key] * y_vars_map[(seg, k)]
        return expr

    plan = solve_campaign_milp(
        config,
        candidates,
        panel,
        segment_predictor,
        total_budget=total_budget,
        output_dir=output_dir,
        model_name="campaign_tree_embed",
        time_limit=time_limit,
        write_outputs=write_outputs,
        model=model,
        x_vars=x_vars,
        y_vars=y_vars,
        fixed_keyword_sets=fixed_keyword_sets,
        fixed_budgets=fixed_budgets,
        train=train,
    )
    ext_pred = _external_incremental_pred_by_segment(
        plan, pipeline, train, config, planning_date, set_features
    )
    plan = plan.drop(columns=["external_model_pred", "pred_over_base"], errors="ignore")
    plan = plan.merge(ext_pred, on="segment", how="left")
    _warn_if_milp_external_pred_mismatch(plan)
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        plan.to_csv(output_dir / "campaign_plan.csv", index=False)
    return plan
