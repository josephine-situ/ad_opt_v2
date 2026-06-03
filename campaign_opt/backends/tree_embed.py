"""Exact tree-embedded Gurobi MILP for RF / XGB winners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gurobipy as gp
import joblib
import numpy as np
import pandas as pd
from gurobipy import GRB

from campaign_opt.backends.milp_core import (
    _predicted_value,
    historical_budget_bounds,
    solve_campaign_milp,
)
from campaign_opt.backends.tree_embedding import (
    _budget_affine,
    _raw_budget_breakpoints_from_trees,
    embed_tree_prediction,
    get_tree_path_sets,
)
from campaign_opt.coefficients import ridge_embed_coeffs
from campaign_opt.decisions import build_segment_list, candidates_by_segment, region_of_segment
from campaign_opt.evaluation import (
    EnsembleModel,
    baseline_levels_for_candidate_sets,
    build_baseline_rows_for_decisions,
    build_segment_decision_rows,
)
from campaign_opt.linear_design import LinearMilpRidgeModel, build_linear_milp_design_matrix
from campaign_opt.modeling import _prep_xy
from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import (
    add_segment_match_type_indicators,
    build_keyword_set_feature_table,
    get_context_feature_columns,
    version_run_vector_for_date,
)
from utils.date_features import calendar_vector_for_date


def _build_candidate_feature_rows(
    candidates: pd.DataFrame,
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
    *,
    panel: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    feature_cols = get_context_feature_columns(config.context_features)
    k_map = candidates_by_segment(candidates)
    set_feats = set_features.set_index("keyword_set_id")
    rows: list[dict] = []
    keys: list[tuple[str, str]] = []

    for seg in sorted(k_map.keys()):
        region = region_of_segment(seg)
        cal = calendar_vector_for_date(planning_date, region, config.course)
        regime = version_run_vector_for_date(
            planning_date, course=config.course, segment=seg, panel=panel
        )
        for kid in k_map[seg]:
            kid = str(kid)
            row: dict = {
                "segment": seg,
                "region": region,
                "daily_budget": 0.0,
                "keyword_set_id": kid,
                **cal,
                **regime,
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


def _relax_bounds_for_feasibility(
    bounds: dict[str, tuple[float, float]],
    segments: list[str],
    total_budget: float,
) -> dict[str, tuple[float, float]]:
    """Set lower bounds to 0 when sum(lo) > total_budget would cause infeasibility."""
    lo_sum = sum(bounds[seg][0] for seg in segments)
    if lo_sum <= total_budget:
        return bounds
    print(
        f"[Warn] Relaxing budget lower bounds: sum(lo)={lo_sum:.1f} > "
        f"total_budget={total_budget:.1f}; setting all lb=0"
    )
    return {seg: (0.0, bounds[seg][1]) for seg in bounds}


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


def _linear_level_affine_from_embed_row(
    ridge_artifact: LinearMilpRidgeModel,
    embed_row: pd.Series,
    segment: str,
    *,
    budget_unit: float = 1.0,
) -> tuple[float, float]:
    """
    Affine ridge level on raw ``daily_budget`` for one candidate embed row.

    ``level0 + slope * budget`` matches ``predict_design_frame`` (includes scaling).
    Calendar and static context for that keyword set are in ``level0``.
    """
    target = ridge_artifact.config.target
    rows = []
    for budget in (0.0, float(budget_unit)):
        row = embed_row.copy()
        row["segment"] = segment
        row["daily_budget"] = budget
        if target not in row.index:
            row[target] = 0.0
        rows.append(row)
    preds = ridge_artifact.predict_design_frame(pd.DataFrame(rows))
    level0 = float(preds[0])
    slope = (float(preds[1]) - level0) / float(budget_unit) if budget_unit else 0.0
    return level0, slope


def _embed_linear_candidate_predictions(
    ridge_artifact: LinearMilpRidgeModel,
    embed_rows: pd.DataFrame,
    keys: list[tuple[str, str]],
    x_vars: dict[str, Any],
    *,
    budget_unit: float = 1.0,
) -> dict[tuple[str, str], Any]:
    """Per-(segment, keyword_set) ridge level: intercept + slope * budget (scaled-fit aware)."""
    pred_by_key: dict[tuple[str, str], Any] = {}
    for i, (seg, kid) in enumerate(keys):
        level0, slope = _linear_level_affine_from_embed_row(
            ridge_artifact,
            embed_rows.iloc[i],
            seg,
            budget_unit=budget_unit,
        )
        pred_by_key[(seg, str(kid))] = level0 + slope * x_vars[seg]
    return pred_by_key


def _blend_ridge_xgb_level(
    linear_pred: dict[tuple[str, str], Any],
    tree_pred: dict[tuple[str, str], Any],
    *,
    w_ridge: float,
    w_xgb: float,
) -> dict[tuple[str, str], Any]:
    pred_by_key: dict[tuple[str, str], Any] = {}
    for key, tree_expr in tree_pred.items():
        lin_expr = linear_pred.get(key)
        if lin_expr is None:
            continue
        pred_by_key[key] = w_ridge * lin_expr + w_xgb * tree_expr
    return pred_by_key


def _embed_candidate_predictions(
    model: gp.Model,
    pipeline,
    embed_rows: pd.DataFrame,
    keys: list[tuple[str, str]],
    x_vars: dict[str, Any],
    bounds: dict[str, tuple[float, float]],
    config: CampaignOptConfig,
    *,
    name_suffix: str = "",
) -> dict[tuple[str, str], Any]:
    """Per-(segment, keyword_set) tree level via ad_opt-style big-M leaf embedding."""
    feature_cols = get_context_feature_columns(config.context_features)
    target = config.target
    probe = embed_rows.copy()
    if target not in probe.columns:
        probe[target] = 0.0
    X_raw, _ = _prep_xy(probe, target, feature_cols)
    X_proc = pipeline[:-1].transform(X_raw)
    base_preds = pipeline.predict(X_raw)
    tree_paths, base_or_n, kind = get_tree_path_sets(pipeline)
    budget_idx, budget_mean, budget_scale = _budget_affine(pipeline)
    pred_by_key: dict[tuple[str, str], Any] = {}
    for i, pred in enumerate(base_preds):
        if pred < 0:
            continue
        seg, kid = keys[i]
        budget_lo, budget_hi = bounds[seg]
        pred_by_key[(seg, str(kid))] = embed_tree_prediction(
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
            name_prefix=f"emb_{seg.replace(' ', '_')}_{kid}{name_suffix}",
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
    max_diff = _max_milp_external_level_diff(plan)
    if max_diff is None or max_diff <= tol:
        return
    milp = pd.to_numeric(plan["milp_pred"], errors="coerce")
    ext = pd.to_numeric(plan["external_model_pred"], errors="coerce")
    diff = (milp - ext).abs()
    worst_idx = diff.idxmax()
    row = plan.loc[worst_idx]
    print(
        f"[Warn] MILP vs external model prediction mismatch: "
        f"max|milp_pred - external_model_pred| = {max_diff:.6g} > {tol} "
        f"(segment={row['segment']!r}, milp_pred={row['milp_pred']}, "
        f"external_model_pred={row['external_model_pred']})"
    )


def _max_milp_external_level_diff(plan: pd.DataFrame) -> float | None:
    if "milp_pred" not in plan.columns or "external_model_pred" not in plan.columns:
        return None
    milp = pd.to_numeric(plan["milp_pred"], errors="coerce")
    ext = pd.to_numeric(plan["external_model_pred"], errors="coerce")
    valid = milp.notna() & ext.notna()
    if not valid.any():
        return None
    return float((milp[valid] - ext[valid]).abs().max())


def _plan_decision_frame(plan: pd.DataFrame) -> pd.DataFrame:
    plan_dec = plan[["segment", "daily_budget", "keyword_set_id"]].copy()
    plan_dec["segment"] = plan_dec["segment"].astype(str)
    plan_dec["keyword_set_id"] = plan_dec["keyword_set_id"].astype(str)
    plan_dec["daily_budget"] = pd.to_numeric(plan_dec["daily_budget"], errors="coerce")
    return plan_dec


def _max_milp_ensemble_level_diff(
    plan: pd.DataFrame,
    ensemble: EnsembleModel,
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
) -> tuple[float, pd.DataFrame, np.ndarray, np.ndarray]:
    feature_cols = get_context_feature_columns(config.context_features)
    target = config.target
    plan_dec = _plan_decision_frame(plan)
    decision_rows = build_segment_decision_rows(
        plan_dec, planning_date, set_features, config.course, feature_cols
    )
    if target not in decision_rows.columns:
        decision_rows[target] = 0.0
    expected = ensemble.predict_levels(decision_rows)
    milp = np.clip(pd.to_numeric(plan["milp_pred"], errors="coerce").values, 0, None)
    diff = np.abs(milp - expected)
    max_diff = float(np.nanmax(diff)) if len(diff) else 0.0
    return max_diff, plan_dec, milp, expected


def warn_milp_matches_ensemble_levels(
    plan: pd.DataFrame,
    ensemble: EnsembleModel,
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
    *,
    tol: float = 0.05,
) -> float:
    """Warn if embedded level predictions differ from ``EnsembleModel.predict_levels``."""
    max_diff, plan_dec, milp, expected = _max_milp_ensemble_level_diff(
        plan, ensemble, config, planning_date, set_features
    )
    if max_diff > tol:
        worst = int(np.nanargmax(np.abs(milp - expected)))
        print(
            f"[Warn] MILP level pred != ensemble predict_levels: max diff {max_diff:.6g} > {tol} "
            f"(segment={plan_dec.iloc[worst]['segment']!r}, "
            f"milp={milp[worst]:.6g}, ensemble={expected[worst]:.6g})"
        )
    return max_diff


def assert_milp_matches_ensemble_levels(
    plan: pd.DataFrame,
    ensemble: EnsembleModel,
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
    *,
    tol: float = 1e-4,
) -> None:
    """Raise if embedded level predictions differ from ``EnsembleModel.predict_levels``."""
    max_diff, plan_dec, milp, expected = _max_milp_ensemble_level_diff(
        plan, ensemble, config, planning_date, set_features
    )
    if max_diff > tol:
        worst = int(np.nanargmax(np.abs(milp - expected)))
        raise AssertionError(
            f"MILP level pred != ensemble predict_levels: max diff {max_diff:.6g} > {tol} "
            f"(segment={plan_dec.iloc[worst]['segment']!r}, "
            f"milp={milp[worst]:.6g}, ensemble={expected[worst]:.6g})"
        )


def warn_milp_matches_ensemble_incremental(
    plan: pd.DataFrame,
    ensemble: EnsembleModel,
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
    *,
    tol: float = 0.05,
) -> float:
    """Warn if MILP / external incremental lift differs from ``predict_incremental_raw``."""
    feature_cols = get_context_feature_columns(config.context_features)
    target = config.target
    plan_dec = _plan_decision_frame(plan)
    decision_rows = build_segment_decision_rows(
        plan_dec, planning_date, set_features, config.course, feature_cols
    )
    baseline_rows = build_baseline_rows_for_decisions(
        plan_dec,
        planning_date,
        set_features,
        config.course,
        feature_cols,
        float(config.evaluation.baseline_budget),
    )
    if target not in decision_rows.columns:
        decision_rows[target] = 0.0
    if target not in baseline_rows.columns:
        baseline_rows[target] = 0.0
    expected = ensemble.predict_incremental_raw(decision_rows, baseline_rows)
    if "pred_over_base" not in plan.columns:
        print("[Warn] plan has no pred_over_base column for incremental validation")
        return 0.0
    got = pd.to_numeric(plan["pred_over_base"], errors="coerce").values
    valid = np.isfinite(got) & np.isfinite(expected)
    if not valid.any():
        print("[Warn] no finite pred_over_base values to validate")
        return 0.0
    diff = np.abs(got[valid] - expected[valid])
    max_diff = float(np.max(diff))
    if max_diff > tol:
        worst = int(np.where(valid)[0][int(np.argmax(diff))])
        print(
            f"[Warn] MILP incremental != ensemble predict_incremental_raw: max diff {max_diff:.6g} > {tol} "
            f"(segment={plan_dec.iloc[worst]['segment']!r}, "
            f"got={got[worst]:.6g}, expected={expected[worst]:.6g})"
        )
    return max_diff


def assert_milp_matches_ensemble_incremental(
    plan: pd.DataFrame,
    ensemble: EnsembleModel,
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
    *,
    tol: float = 1e-4,
) -> None:
    """Raise if MILP / external incremental lift differs from ``predict_incremental_raw``."""
    feature_cols = get_context_feature_columns(config.context_features)
    target = config.target
    plan_dec = _plan_decision_frame(plan)
    decision_rows = build_segment_decision_rows(
        plan_dec, planning_date, set_features, config.course, feature_cols
    )
    baseline_rows = build_baseline_rows_for_decisions(
        plan_dec,
        planning_date,
        set_features,
        config.course,
        feature_cols,
        float(config.evaluation.baseline_budget),
    )
    if target not in decision_rows.columns:
        decision_rows[target] = 0.0
    if target not in baseline_rows.columns:
        baseline_rows[target] = 0.0
    expected = ensemble.predict_incremental_raw(decision_rows, baseline_rows)
    if "pred_over_base" not in plan.columns:
        raise AssertionError("plan has no pred_over_base column for incremental validation")
    got = pd.to_numeric(plan["pred_over_base"], errors="coerce").values
    valid = np.isfinite(got) & np.isfinite(expected)
    if not valid.any():
        raise AssertionError("no finite pred_over_base values to validate")
    diff = np.abs(got[valid] - expected[valid])
    max_diff = float(np.max(diff))
    if max_diff > tol:
        worst = int(np.where(valid)[0][int(np.argmax(diff))])
        raise AssertionError(
            f"MILP incremental != ensemble predict_incremental_raw: max diff {max_diff:.6g} > {tol} "
            f"(segment={plan_dec.iloc[worst]['segment']!r}, "
            f"got={got[worst]:.6g}, expected={expected[worst]:.6g})"
        )


def warn_milp_matches_ensemble_plan(
    plan: pd.DataFrame,
    ensemble: EnsembleModel,
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
    *,
    level_tol: float = 0.05,
    incremental_tol: float = 0.05,
) -> None:
    """Warn when MILP levels / incremental differ from the optimizer ensemble (tree big-M embed)."""
    warn_milp_matches_ensemble_levels(
        plan, ensemble, config, planning_date, set_features, tol=level_tol
    )
    warn_milp_matches_ensemble_incremental(
        plan, ensemble, config, planning_date, set_features, tol=incremental_tol
    )
    _warn_if_milp_external_pred_mismatch(plan, tol=level_tol)


def assert_milp_matches_ensemble_plan(
    plan: pd.DataFrame,
    ensemble: EnsembleModel,
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
    *,
    level_tol: float = 1e-4,
    incremental_tol: float = 1e-4,
) -> None:
    """Level and incremental predictions vs the optimizer ensemble (strict; tests)."""
    assert_milp_matches_ensemble_levels(
        plan, ensemble, config, planning_date, set_features, tol=level_tol
    )
    assert_milp_matches_ensemble_incremental(
        plan, ensemble, config, planning_date, set_features, tol=incremental_tol
    )
    max_diff = _max_milp_external_level_diff(plan)
    if max_diff is not None and max_diff > level_tol:
        milp = pd.to_numeric(plan["milp_pred"], errors="coerce")
        ext = pd.to_numeric(plan["external_model_pred"], errors="coerce")
        worst_idx = (milp - ext).abs().idxmax()
        raise AssertionError(
            f"milp_pred != external_model_pred: max diff {max_diff:.6g} > {level_tol} "
            f"(segment={plan.loc[worst_idx]['segment']!r})"
        )


def _probe_ridge_xgb_embed_on_embed_rows(
    ensemble: EnsembleModel,
    embed_rows: pd.DataFrame,
    keys: list[tuple[str, str]],
    ridge_artifact: LinearMilpRidgeModel,
    xgb_pipeline: Any,
    bounds: dict[str, tuple[float, float]],
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
    *,
    w_ridge: float,
    w_xgb: float,
    budgets: list[float] | None = None,
    tol: float = 1e-4,
) -> None:
    """Check embedded ridge+XGB levels match ``EnsembleModel.predict_levels``."""
    feature_cols = get_context_feature_columns(config.context_features)
    target = config.target

    max_diff = 0.0
    for i, (seg, kid) in enumerate(keys):
        lo, hi = bounds[seg]
        probe_budgets = budgets if budgets is not None else [0.0, (lo + hi) / 2, hi]
        row = embed_rows.iloc[i]
        level0, slope = _linear_level_affine_from_embed_row(ridge_artifact, row, seg)

        def feature_row_at(budget: float) -> pd.DataFrame:
            r = embed_rows.iloc[i : i + 1].copy()
            r["daily_budget"] = budget
            if target not in r.columns:
                r[target] = 0.0
            X, _ = _prep_xy(r, target, feature_cols)
            return X

        for budget in probe_budgets:
            dec = pd.DataFrame(
                [
                    {
                        "segment": seg,
                        "keyword_set_id": str(kid),
                        "daily_budget": float(budget),
                    }
                ]
            )
            dec_rows = build_segment_decision_rows(
                dec, planning_date, set_features, config.course, feature_cols
            )
            if target not in dec_rows.columns:
                dec_rows[target] = 0.0
            expected = float(ensemble.predict_levels(dec_rows)[0])
            ridge_p = float(ridge_artifact.predict_design_frame(dec_rows)[0])
            lin_p = level0 + slope * float(budget)
            sk_tree = float(xgb_pipeline.predict(feature_row_at(budget))[0])
            blend = max(0.0, w_ridge * ridge_p + w_xgb * sk_tree)
            max_diff = max(
                max_diff,
                abs(expected - blend),
                abs(ridge_p - lin_p),
            )
    if max_diff > tol:
        print(
            f"[Warn] ridge_xgb embed probe mismatch: max diff = {max_diff:.6g} > {tol}"
        )


def _diagnose_milp_vs_sklearn_at_solved_budgets(
    plan: pd.DataFrame,
    ensemble: EnsembleModel,
    embed_rows: pd.DataFrame,
    keys: list[tuple[str, str]],
    ridge_artifact: LinearMilpRidgeModel,
    xgb_pipeline: Any,
    bounds: dict[str, tuple[float, float]],
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    w_ridge: float,
    w_xgb: float,
) -> None:
    """Post-solve: compare embed-path sklearn vs validation-path ensemble at solved budgets."""
    feature_cols = get_context_feature_columns(config.context_features)
    target = config.target
    key_to_idx = {k: i for i, k in enumerate(keys)}

    for _, row in plan.iterrows():
        seg = str(row["segment"])
        kid = str(row["keyword_set_id"])
        budget = float(row["daily_budget"])
        milp_pred = float(row["milp_pred"])
        key = (seg, kid)
        if key not in key_to_idx:
            continue
        idx = key_to_idx[key]

        # Embed-path: XGB prediction at solved budget
        r = embed_rows.iloc[idx: idx + 1].copy()
        r["daily_budget"] = budget
        if target not in r.columns:
            r[target] = 0.0
        X_embed, _ = _prep_xy(r, target, feature_cols)
        sk_xgb_embed = float(xgb_pipeline.predict(X_embed)[0])

        # Embed-path: ridge prediction at solved budget
        ridge_embed = float(ridge_artifact.predict_design_frame(
            pd.DataFrame([{**embed_rows.iloc[idx].to_dict(), "daily_budget": budget, target: 0.0}])
        )[0])

        blend_embed = w_ridge * ridge_embed + w_xgb * sk_xgb_embed
        from campaign_opt.decisions import observed_min_daily_budget
        from campaign_opt.optimizer_prediction import apply_observed_budget_floor

        mins = observed_min_daily_budget(panel, [seg])
        floor_blend = float(
            apply_observed_budget_floor(
                np.array([blend_embed]),
                np.array([budget]),
                np.array([seg]),
                mins,
                budget_atol=float(config.evaluation.budget_floor_atol),
            )[0]
        )

        # Validation-path: ensemble prediction at solved budget
        dec = pd.DataFrame([{"segment": seg, "keyword_set_id": kid, "daily_budget": budget}])
        dec_rows = build_segment_decision_rows(
            dec, planning_date, set_features, config.course, feature_cols
        )
        if target not in dec_rows.columns:
            dec_rows[target] = 0.0
        ensemble_pred = float(ensemble.predict_levels(dec_rows)[0])

        # Validation-path: per-component
        X_val, _ = _prep_xy(dec_rows, target, feature_cols)
        sk_xgb_val = float(xgb_pipeline.predict(X_val)[0])
        ridge_val = float(ridge_artifact.predict_design_frame(dec_rows)[0])

        embed_vs_val = abs(blend_embed - ensemble_pred)
        xgb_diff = abs(sk_xgb_embed - sk_xgb_val)
        ridge_diff = abs(ridge_embed - ridge_val)
        milp_vs_floor = abs(milp_pred - floor_blend)

        if embed_vs_val > 0.01 or milp_vs_floor > 0.01:
            print(
                f"[Diag] {seg} kid={kid} budget={budget:.2f}\n"
                f"  milp_pred={milp_pred:.6f}  blend_embed={blend_embed:.6f}  "
                f"floor_blend={floor_blend:.6f}  ensemble_val={ensemble_pred:.6f}\n"
                f"  xgb: embed={sk_xgb_embed:.6f} val={sk_xgb_val:.6f} diff={xgb_diff:.6g}\n"
                f"  ridge: embed={ridge_embed:.6f} val={ridge_val:.6f} diff={ridge_diff:.6g}\n"
                f"  milp_vs_floor_blend={milp_vs_floor:.6g}  embed_vs_validation={embed_vs_val:.6g}"
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
    baseline_rows = build_baseline_rows_for_decisions(
        plan_dec,
        planning_date,
        set_features,
        config.course,
        feature_cols,
        float(config.evaluation.baseline_budget),
    )
    if target not in decision_rows.columns:
        decision_rows[target] = 0.0
    if target not in baseline_rows.columns:
        baseline_rows[target] = 0.0
    if "region" not in decision_rows.columns:
        decision_rows["region"] = decision_rows["segment"].astype(str).map(region_of_segment)
    if "region" not in baseline_rows.columns:
        baseline_rows["region"] = baseline_rows["segment"].astype(str).map(region_of_segment)

    if isinstance(pipeline, EnsembleModel):
        pred_dec = pipeline.predict_levels(decision_rows)
        pred_zero = pipeline.predict_levels(baseline_rows)
    else:
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


def solve_ridge_xgb_embed_campaign_milp(
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
    """Blend ridge (linear MILP coeffs) and XGB (exact tree embed) in the objective."""
    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, EnsembleModel):
        raise TypeError(f"Expected EnsembleModel at {model_path}, got {type(pipeline).__name__}")
    ridge_member = next((m for m in pipeline.members if m.name == "ridge"), None)
    xgb_member = next((m for m in pipeline.members if m.name == "xgboost"), None)
    if ridge_member is None or xgb_member is None:
        raise RuntimeError("ensemble_ridge_xgb requires fitted ridge and xgboost members")

    total_w = sum(m.weight for m in pipeline.members) or 1.0
    w_ridge = ridge_member.weight / total_w
    w_xgb = xgb_member.weight / total_w

    set_features = build_keyword_set_feature_table(config.course)
    embed_rows, keys = _build_candidate_feature_rows(
        candidates, config, planning_date, set_features, panel=panel
    )
    segments = build_segment_list(candidates)
    k_map = candidates_by_segment(candidates)
    bounds = historical_budget_bounds(panel, segments)
    bounds = _relax_bounds_for_feasibility(bounds, segments, total_budget)

    ridge_artifact = ridge_member.pipeline
    if not isinstance(ridge_artifact, LinearMilpRidgeModel):
        raise TypeError("ridge member must be a LinearMilpRidgeModel")
    coeffs = ridge_embed_coeffs(
        ridge_artifact,
        train,
        config,
        candidates,
        set_features,
        planning_date,
        segments,
    )
    if write_outputs and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "ridge_embed_coeffs.json", "w", encoding="utf-8") as f:
            json.dump(coeffs, f, indent=2)

    model = gp.Model("campaign_ridge_xgb_embed")
    model.setParam("OutputFlag", 1)
    model.setParam("TimeLimit", time_limit)
    x_vars, y_vars = _create_decision_vars(model, segments, k_map, bounds)
    _apply_fixed_decisions(
        model, segments, k_map, x_vars, y_vars,
        fixed_keyword_sets=fixed_keyword_sets,
        fixed_budgets=fixed_budgets,
    )
    linear_pred = _embed_linear_candidate_predictions(
        ridge_artifact, embed_rows, keys, x_vars
    )
    tree_pred = _embed_candidate_predictions(
        model, xgb_member.pipeline, embed_rows, keys, x_vars, bounds, config
    )
    pred_by_key = _blend_ridge_xgb_level(
        linear_pred, tree_pred, w_ridge=w_ridge, w_xgb=w_xgb
    )
    baseline_budget = float(config.evaluation.baseline_budget)
    _probe_ridge_xgb_embed_on_embed_rows(
        pipeline,
        embed_rows,
        keys,
        ridge_artifact,
        xgb_member.pipeline,
        bounds,
        config,
        planning_date,
        set_features,
        w_ridge=w_ridge,
        w_xgb=w_xgb,
        budgets=None,
        tol=1e-4,
    )
    _disable_pruned_keyword_sets(model, y_vars, k_map, pred_by_key)

    baseline_level_by_key = baseline_levels_for_candidate_sets(
        pipeline, k_map, config, planning_date, set_features, baseline_budget=baseline_budget
    )

    # Compute valid level_ub from actual model predictions so the McCormick
    # gating variable bound never clips the true prediction.
    feature_cols = get_context_feature_columns(config.context_features)
    target = config.target
    level_ub_overrides: dict[str, float] = {}
    for i, (seg, kid) in enumerate(keys):
        if (seg, str(kid)) not in pred_by_key:
            continue
        lo, hi = bounds[seg]
        probe_budgets = [lo, hi, (lo + hi) / 2]
        # Also check at tree breakpoints where the XGB prediction can jump.
        r0 = embed_rows.iloc[i: i + 1].copy()
        r0["daily_budget"] = 0.0
        if target not in r0.columns:
            r0[target] = 0.0
        X0, _ = _prep_xy(r0, target, feature_cols)
        x_proc_0 = np.asarray(xgb_member.pipeline[:-1].transform(X0), dtype=np.float32).ravel()
        tree_paths, _, _ = get_tree_path_sets(xgb_member.pipeline)
        budget_idx, budget_mean, budget_scale = _budget_affine(xgb_member.pipeline)
        bps = _raw_budget_breakpoints_from_trees(
            tree_paths, x_proc_0, budget_idx, budget_mean, budget_scale, lo, hi
        )
        probe_budgets.extend(bps)

        max_blend = 0.0
        for b in probe_budgets:
            b = max(lo, min(hi, float(b)))
            r = embed_rows.iloc[i: i + 1].copy()
            r["daily_budget"] = b
            if target not in r.columns:
                r[target] = 0.0
            X_b, _ = _prep_xy(r, target, feature_cols)
            xgb_b = float(xgb_member.pipeline.predict(X_b)[0])
            ridge_b = float(ridge_artifact.predict_design_frame(
                pd.DataFrame([{**embed_rows.iloc[i].to_dict(), "daily_budget": b, target: 0.0}])
            )[0])
            max_blend = max(max_blend, w_ridge * ridge_b + w_xgb * xgb_b)
        cur = level_ub_overrides.get(seg, 0.0)
        level_ub_overrides[seg] = max(cur, max_blend * 1.1)

    def segment_predictor(seg: str, x_var: Any, y_vars_map: dict, k_map_local: dict) -> Any:
        seg_pred = model.addVar(lb=-GRB.INFINITY, name=f"seg_pred_{seg.replace(' ', '_')}")
        for k in k_map_local.get(seg, []):
            key = (seg, str(k))
            if key not in pred_by_key:
                continue
            model.addGenConstrIndicator(
                y_vars_map[(seg, k)], 1,
                seg_pred - pred_by_key[key], GRB.EQUAL, 0.0,
                name=f"ind_{seg.replace(' ', '_')}_{k}",
            )
        return seg_pred

    plan = solve_campaign_milp(
        config,
        candidates,
        panel,
        segment_predictor,
        total_budget=total_budget,
        output_dir=output_dir,
        model_name="campaign_ridge_xgb_embed",
        time_limit=time_limit,
        write_outputs=write_outputs,
        model=model,
        x_vars=x_vars,
        y_vars=y_vars,
        fixed_keyword_sets=fixed_keyword_sets,
        fixed_budgets=fixed_budgets,
        train=train,
        baseline_level_by_key=baseline_level_by_key,
        level_ub_overrides=level_ub_overrides,
    )
    # Post-solve: re-probe embedding at actual solved budgets to localize discrepancy.
    _diagnose_milp_vs_sklearn_at_solved_budgets(
        plan, pipeline, embed_rows, keys, ridge_artifact, xgb_member.pipeline,
        bounds, config, planning_date, set_features, panel,
        w_ridge=w_ridge, w_xgb=w_xgb,
    )
    # pred_over_base from Gurobi can differ slightly from sklearn on tree embed; use ensemble lift.
    ext_pred = _external_incremental_pred_by_segment(
        plan, pipeline, train, config, planning_date, set_features
    )
    plan = plan.drop(columns=["external_model_pred", "pred_over_base"], errors="ignore")
    plan = plan.merge(ext_pred, on="segment", how="left")
    warn_milp_matches_ensemble_plan(
        plan, pipeline, config, planning_date, set_features, level_tol=0.05
    )
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        plan.to_csv(output_dir / "campaign_plan.csv", index=False)
    return plan


def solve_ridge_xgb_embed_multiday_campaign_milp(
    config: CampaignOptConfig,
    model_path: Path,
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    total_budget: float,
    output_dir: Path,
    planning_dates: list[pd.Timestamp],
    time_limit: int = 600,
    write_outputs: bool = True,
    fixed_keyword_sets: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Multi-day ridge+XGB embed MILP: shared keyword set selection, per-day budgets.

    Multi-day variant of :func:`solve_ridge_xgb_embed_campaign_milp`: shared keyword-set
    binaries, per-day budgets, regional order, and observed-budget floor on predictions
    (after keyword-set indicators; same semantics as single-day ``solve_campaign_milp``).
    Used as Stage 1 of the two-stage backtest.
    """
    from campaign_opt.decisions import parse_regional_order

    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, EnsembleModel):
        raise TypeError(f"Expected EnsembleModel at {model_path}, got {type(pipeline).__name__}")
    ridge_member = next((m for m in pipeline.members if m.name == "ridge"), None)
    xgb_member = next((m for m in pipeline.members if m.name == "xgboost"), None)
    if ridge_member is None or xgb_member is None:
        raise RuntimeError("ensemble_ridge_xgb requires fitted ridge and xgboost members")

    total_w = sum(m.weight for m in pipeline.members) or 1.0
    w_ridge = ridge_member.weight / total_w
    w_xgb = xgb_member.weight / total_w

    set_features = build_keyword_set_feature_table(config.course)
    segments = build_segment_list(candidates)
    k_map = candidates_by_segment(candidates)
    bounds = historical_budget_bounds(panel, segments)
    bounds = _relax_bounds_for_feasibility(bounds, segments, total_budget)
    feature_cols = get_context_feature_columns(config.context_features)
    target = config.target

    ridge_artifact = ridge_member.pipeline
    if not isinstance(ridge_artifact, LinearMilpRidgeModel):
        raise TypeError("ridge member must be a LinearMilpRidgeModel")

    n_days = len(planning_dates)
    print(
        f"[Info] Multi-day ridge+XGB embed MILP: {n_days} days, "
        f"{len(segments)} segments, budget_cap={total_budget}"
    )

    # --- Compute level_ub per segment (probe model at budget range + tree breakpoints) ---
    level_ub_overrides: dict[str, float] = {}
    tree_paths_probe, _, _ = get_tree_path_sets(xgb_member.pipeline)
    budget_idx_probe, budget_mean_probe, budget_scale_probe = _budget_affine(xgb_member.pipeline)
    # Use first day's embed rows for probing (calendar differences are small)
    embed_rows_probe, keys_probe = _build_candidate_feature_rows(
        candidates, config, pd.Timestamp(planning_dates[0]), set_features, panel=panel
    )
    for i, (seg, kid) in enumerate(keys_probe):
        lo, hi = bounds[seg]
        probe_budgets = [lo, hi, (lo + hi) / 2]
        r0 = embed_rows_probe.iloc[i: i + 1].copy()
        r0["daily_budget"] = 0.0
        if target not in r0.columns:
            r0[target] = 0.0
        X0, _ = _prep_xy(r0, target, feature_cols)
        x_proc_0 = np.asarray(
            xgb_member.pipeline[:-1].transform(X0), dtype=np.float32
        ).ravel()
        bps = _raw_budget_breakpoints_from_trees(
            tree_paths_probe, x_proc_0, budget_idx_probe,
            budget_mean_probe, budget_scale_probe, lo, hi
        )
        probe_budgets.extend(bps)
        max_blend = 0.0
        for b in probe_budgets:
            b = max(lo, min(hi, float(b)))
            r = embed_rows_probe.iloc[i: i + 1].copy()
            r["daily_budget"] = b
            if target not in r.columns:
                r[target] = 0.0
            X_b, _ = _prep_xy(r, target, feature_cols)
            xgb_b = float(xgb_member.pipeline.predict(X_b)[0])
            ridge_b = float(ridge_artifact.predict_design_frame(
                pd.DataFrame([{**embed_rows_probe.iloc[i].to_dict(), "daily_budget": b, target: 0.0}])
            )[0])
            max_blend = max(max_blend, w_ridge * ridge_b + w_xgb * xgb_b)
        cur = level_ub_overrides.get(seg, 0.0)
        level_ub_overrides[seg] = max(cur, max_blend * 1.1)

    # --- Baseline levels for incremental objective ---
    baseline_budget = float(config.evaluation.baseline_budget)
    # Average baseline across all planning dates
    baseline_level_by_key: dict[tuple[str, str], float] = {}
    for t_idx, plan_date in enumerate(planning_dates):
        day_baselines = baseline_levels_for_candidate_sets(
            pipeline, k_map, config, pd.Timestamp(plan_date), set_features,
            baseline_budget=baseline_budget,
        )
        for key, val in day_baselines.items():
            baseline_level_by_key[key] = baseline_level_by_key.get(key, 0.0) + val

    # --- Build Gurobi model ---
    model = gp.Model("campaign_ridge_xgb_embed_multiday")
    model.setParam("OutputFlag", 1)
    model.setParam("TimeLimit", time_limit)

    # Keyword set selection: shared across all days
    y_vars: dict[tuple[str, str], Any] = {}
    for seg in segments:
        for k in k_map.get(seg, []):
            y_vars[(seg, k)] = model.addVar(vtype=GRB.BINARY, name=f"y_{seg}_{k}")
        model.addConstr(
            gp.quicksum(y_vars[(seg, k)] for k in k_map.get(seg, [])) == 1,
            name=f"one_set_{seg}",
        )
    if fixed_keyword_sets:
        for seg in segments:
            chosen = str(fixed_keyword_sets.get(seg, ""))
            for k in k_map.get(seg, []):
                fix_val = 1.0 if str(k) == chosen else 0.0
                y_vars[(seg, k)].lb = fix_val
                y_vars[(seg, k)].ub = fix_val

    # Per-day budget variables
    x_day_vars: dict[tuple[str, int], Any] = {}
    for t in range(n_days):
        for seg in segments:
            lo, hi = bounds[seg]
            x_day_vars[(seg, t)] = model.addVar(
                lb=lo, ub=hi, name=f"x_{seg}_d{t}"
            )
        model.addConstr(
            gp.quicksum(x_day_vars[(seg, t)] for seg in segments) <= total_budget,
            name=f"budget_day_{t}",
        )

    # Regional order constraints (per day): e.g. USA >= A >= B
    regional_order = parse_regional_order(config.constraints)
    if len(regional_order) >= 2:
        for t in range(n_days):
            region_spend: dict[str, Any] = {}
            for region in regional_order:
                segs = [s for s in segments if region_of_segment(s) == region]
                if segs:
                    region_spend[region] = gp.quicksum(x_day_vars[(s, t)] for s in segs)
            for i in range(len(regional_order) - 1):
                r_hi, r_lo = regional_order[i], regional_order[i + 1]
                if r_hi in region_spend and r_lo in region_spend:
                    model.addConstr(
                        region_spend[r_hi] >= region_spend[r_lo],
                        name=f"order_{r_hi}_{r_lo}_d{t}",
                    )

    # --- Per-day embeddings ---
    day_pred_exprs: list[dict[tuple[str, str], Any]] = []

    for t, plan_date in enumerate(planning_dates):
        plan_date = pd.Timestamp(plan_date)
        embed_rows_t, keys_t = _build_candidate_feature_rows(
            candidates, config, plan_date, set_features, panel=panel
        )
        x_vars_t = {seg: x_day_vars[(seg, t)] for seg in segments}

        linear_pred_t = _embed_linear_candidate_predictions(
            ridge_artifact, embed_rows_t, keys_t, x_vars_t
        )
        tree_pred_t = _embed_candidate_predictions(
            model, xgb_member.pipeline, embed_rows_t, keys_t, x_vars_t, bounds, config,
            name_suffix=f"_d{t}",
        )
        blended_t = _blend_ridge_xgb_level(
            linear_pred_t, tree_pred_t, w_ridge=w_ridge, w_xgb=w_xgb
        )
        day_pred_exprs.append(blended_t)

    # A keyword set is fully pruned only if missing on ALL days
    always_pruned: set[tuple[str, str]] = set()
    for seg in segments:
        for k in k_map.get(seg, []):
            key = (seg, str(k))
            if all(key not in day_pred_exprs[t] for t in range(n_days)):
                always_pruned.add(key)
                model.addConstr(y_vars[(seg, k)] == 0, name=f"pruned_{seg}_{k}")
        if not any(
            (seg, str(k)) not in always_pruned for k in k_map.get(seg, [])
        ):
            raise RuntimeError(f"Segment {seg!r}: all keyword sets pruned on every day")

    # --- Per-(seg, day) prediction variable with indicator gating ---
    seg_day_preds: dict[tuple[str, int], Any] = {}
    for t in range(n_days):
        blended_t = day_pred_exprs[t]
        for seg in segments:
            seg_pred = model.addVar(
                lb=-GRB.INFINITY, name=f"seg_pred_{seg.replace(' ', '_')}_d{t}"
            )
            for k in k_map.get(seg, []):
                key = (seg, str(k))
                if key in blended_t:
                    model.addGenConstrIndicator(
                        y_vars[(seg, k)], 1,
                        seg_pred - blended_t[key], GRB.EQUAL, 0.0,
                        name=f"ind_{seg.replace(' ', '_')}_{k}_d{t}",
                    )
                else:
                    model.addGenConstrIndicator(
                        y_vars[(seg, k)], 1,
                        seg_pred, GRB.EQUAL, 0.0,
                        name=f"ind_{seg.replace(' ', '_')}_{k}_d{t}_pruned",
                    )
            seg_day_preds[(seg, t)] = seg_pred

    # Observed-budget floor on the selected segment-day prediction (ridge+XGB above floor).
    if config.evaluation.apply_observed_budget_floor:
        from campaign_opt.backends.prediction_gating import (
            budget_big_m_from_bounds,
            gate_level_expr,
        )
        from campaign_opt.decisions import observed_min_daily_budget

        min_budgets = observed_min_daily_budget(panel, segments)
        budget_atol = float(config.evaluation.budget_floor_atol)
        for t in range(n_days):
            for seg in segments:
                bmin = float(min_budgets.get(seg, 0.0))
                if bmin <= 0.0:
                    continue
                lo, hi = bounds[seg]
                level_ub = float(level_ub_overrides.get(seg, 1.0))
                m_b = budget_big_m_from_bounds(lo, hi)
                safe = str(seg).replace(" ", "_").replace("/", "_")
                seg_day_preds[(seg, t)] = gate_level_expr(
                    model,
                    seg_day_preds[(seg, t)],
                    x_day_vars[(seg, t)],
                    budget_min=bmin,
                    level_ub=level_ub,
                    budget_big_m=m_b,
                    name_prefix=f"gate_{safe}_d{t}",
                    budget_atol=budget_atol,
                )

    # --- Objective ---
    objective_mode = str(config.evaluation.objective or "incremental").strip().lower()
    level_sum = gp.quicksum(
        seg_day_preds[(seg, t)] for seg in segments for t in range(n_days)
    )

    if objective_mode == "incremental":
        baseline_for_obj = baseline_level_by_key
        if config.evaluation.apply_observed_budget_floor:
            from campaign_opt.backends.prediction_gating import (
                apply_gated_baseline_levels,
            )
            from campaign_opt.decisions import observed_min_daily_budget

            min_budgets = observed_min_daily_budget(panel, segments)
            baseline_for_obj = apply_gated_baseline_levels(
                baseline_level_by_key, baseline_budget, min_budgets
            )
        baseline_terms = []
        for seg in segments:
            for k in k_map.get(seg, []):
                key = (seg, str(k))
                if key not in y_vars:
                    continue
                f0 = float(baseline_for_obj.get(key, 0.0))
                baseline_terms.append(f0 * y_vars[key])
        if baseline_terms:
            objective_expr = level_sum - gp.quicksum(baseline_terms)
        else:
            objective_expr = level_sum
    else:
        objective_expr = level_sum

    penalty = float((config.constraints or {}).get("budget_tiebreak_penalty", 1e-8))
    if penalty > 0:
        budget_sum = gp.quicksum(
            x_day_vars[(seg, t)] for seg in segments for t in range(n_days)
        )
        model.setObjective(objective_expr - penalty * budget_sum, GRB.MAXIMIZE)
    else:
        model.setObjective(objective_expr, GRB.MAXIMIZE)

    model.update()
    print(
        f"[Info] Multi-day MILP ({objective_mode}): "
        f"{model.NumVars} vars, {model.NumConstrs} constrs "
        f"({model.NumIntVars} integer)",
        flush=True,
    )
    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        status_name = {
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INFEASIBLE_OR_UNBOUNDED",
            GRB.UNBOUNDED: "UNBOUNDED",
        }.get(int(model.Status), str(model.Status))
        raise RuntimeError(
            f"Multi-day Gurobi solve failed: {status_name} ({int(model.Status)})"
        )

    # --- Extract solution ---
    has_solution = model.SolCount > 0
    rows = []
    for seg in segments:
        if fixed_keyword_sets and seg in fixed_keyword_sets:
            chosen_k = str(fixed_keyword_sets[seg])
        elif has_solution:
            chosen_k = next(
                (k for k in k_map.get(seg, []) if y_vars[(seg, k)].X > 0.5),
                None,
            )
        else:
            chosen_k = None

        day_budgets = []
        for t in range(n_days):
            day_budgets.append(
                float(x_day_vars[(seg, t)].X) if has_solution else 0.0
            )
        avg_budget = sum(day_budgets) / n_days if n_days else 0.0

        rows.append({
            "segment": seg,
            "region": region_of_segment(seg),
            "daily_budget": avg_budget,
            "keyword_set_id": chosen_k,
            "n_planning_days": n_days,
            "day_budgets": json.dumps(day_budgets),
        })

    plan = pd.DataFrame(rows)

    if write_outputs and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plan.to_csv(output_dir / "campaign_plan.csv", index=False)
        status_payload: dict[str, Any] = {
            "status": int(model.Status),
            "obj_val": float(model.ObjVal) if has_solution else None,
            "objective": objective_mode,
            "n_days": n_days,
            "n_segments": len(segments),
            "total_budget_per_day": total_budget,
        }
        if has_solution and len(plan):
            bud = pd.to_numeric(plan["daily_budget"], errors="coerce")
            status_payload["avg_daily_budget_sum"] = float(bud.sum())
            status_payload["tiebreak_penalty"] = penalty
        with open(output_dir / "solver_status.json", "w", encoding="utf-8") as f:
            json.dump(status_payload, f, indent=2)

    print(
        f"[Info] Multi-day MILP solved: status={model.Status}, "
        f"obj={model.ObjVal if has_solution else 'N/A'}"
    )
    return plan


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
    """Embed winner tree model in Gurobi via big-M leaf formulation."""
    pipeline = joblib.load(model_path)
    set_features = build_keyword_set_feature_table(config.course)
    embed_rows, keys = _build_candidate_feature_rows(
        candidates, config, planning_date, set_features, panel=panel
    )

    segments = build_segment_list(candidates)
    k_map = candidates_by_segment(candidates)
    bounds = historical_budget_bounds(panel, segments)
    bounds = _relax_bounds_for_feasibility(bounds, segments, total_budget)

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

    # Compute valid level_ub from sklearn predictions across budget range.
    feature_cols = get_context_feature_columns(config.context_features)
    target = config.target
    level_ub_overrides: dict[str, float] = {}
    tree_paths_ub, _, _ = get_tree_path_sets(pipeline)
    budget_idx_ub, budget_mean_ub, budget_scale_ub = _budget_affine(pipeline)
    for i, (seg, kid) in enumerate(keys):
        if (seg, str(kid)) not in pred_by_key:
            continue
        lo, hi = bounds[seg]
        r0 = embed_rows.iloc[i: i + 1].copy()
        r0["daily_budget"] = 0.0
        if target not in r0.columns:
            r0[target] = 0.0
        X0, _ = _prep_xy(r0, target, feature_cols)
        x_proc_0 = np.asarray(pipeline[:-1].transform(X0), dtype=np.float32).ravel()
        bps = _raw_budget_breakpoints_from_trees(
            tree_paths_ub, x_proc_0, budget_idx_ub, budget_mean_ub, budget_scale_ub, lo, hi
        )
        probe_budgets = [lo, hi, (lo + hi) / 2, *bps]
        max_pred = 0.0
        for b in probe_budgets:
            b = max(lo, min(hi, float(b)))
            r = embed_rows.iloc[i: i + 1].copy()
            r["daily_budget"] = b
            if target not in r.columns:
                r[target] = 0.0
            X_b, _ = _prep_xy(r, target, feature_cols)
            max_pred = max(max_pred, float(pipeline.predict(X_b)[0]))
        cur = level_ub_overrides.get(seg, 0.0)
        level_ub_overrides[seg] = max(cur, max_pred * 1.1)

    baseline_budget = float(config.evaluation.baseline_budget)
    baseline_level_by_key = baseline_levels_for_candidate_sets(
        pipeline, k_map, config, planning_date, set_features, baseline_budget=baseline_budget
    )

    def segment_predictor(seg: str, x_var: Any, y_vars_map: dict, k_map_local: dict) -> Any:
        seg_pred = model.addVar(lb=-GRB.INFINITY, name=f"seg_pred_{seg.replace(' ', '_')}")
        for k in k_map_local.get(seg, []):
            key = (seg, str(k))
            if key not in pred_by_key:
                continue
            model.addGenConstrIndicator(
                y_vars_map[(seg, k)], 1,
                seg_pred - pred_by_key[key], GRB.EQUAL, 0.0,
                name=f"ind_{seg.replace(' ', '_')}_{k}",
            )
        return seg_pred

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
        baseline_level_by_key=baseline_level_by_key,
        level_ub_overrides=level_ub_overrides,
    )
    ext_pred = _external_incremental_pred_by_segment(
        plan, pipeline, train, config, planning_date, set_features
    )
    plan = plan.drop(columns=["external_model_pred", "pred_over_base"], errors="ignore")
    plan = plan.merge(ext_pred, on="segment", how="left")
    if isinstance(pipeline, EnsembleModel):
        warn_milp_matches_ensemble_plan(
            plan, pipeline, config, planning_date, set_features, level_tol=0.05
        )
    else:
        _warn_if_milp_external_pred_mismatch(plan, tol=0.05)
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        plan.to_csv(output_dir / "campaign_plan.csv", index=False)
    return plan
