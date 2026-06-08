"""Shared Gurobi MILP setup for all campaign optimizer backends."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from utils.backends.prediction_gating import gate_pred_vars_if_enabled
from utils.decisions import (
    build_segment_list,
    candidates_by_segment,
    historical_budget_bounds,
    parse_regional_order,
    region_of_segment,
)
from utils.campaign_config import CampaignOptConfig

# Type: (segment, x_var, y_vars dict, k_map) -> Gurobi linear expression for predicted target
SegmentPredictor = Callable[[str, Any, dict[tuple[str, str], Any], dict[str, list[str]]], Any]


def _predicted_value(expr: Any) -> float | None:
    try:
        return float(expr.getValue())
    except Exception:
        try:
            return float(expr.X)
        except Exception:
            return None


def _eval_gurobi_expr(expr: Any, var_values: dict[Any, float]) -> float | None:
    """Evaluate a Gurobi expression at explicit variable values (no .X mutation)."""
    try:
        if isinstance(expr, (int, float)):
            return float(expr)
        # LinExpr path
        if hasattr(expr, "getVar"):
            total = float(expr.getConstant())
            for i in range(expr.size()):
                v = expr.getVar(i)
                if v not in var_values:
                    return None
                total += float(expr.getCoeff(i)) * float(var_values.get(v, 0.0))
            return total
        # QuadExpr path
        if hasattr(expr, "getLinExpr") and hasattr(expr, "getVar1") and hasattr(expr, "getVar2"):
            total = 0.0
            lin = expr.getLinExpr()
            total += float(lin.getConstant())
            for i in range(lin.size()):
                v = lin.getVar(i)
                if v not in var_values:
                    return None
                total += float(lin.getCoeff(i)) * float(var_values.get(v, 0.0))
            for i in range(expr.size()):
                v1 = expr.getVar1(i)
                v2 = expr.getVar2(i)
                if v1 not in var_values or v2 not in var_values:
                    return None
                c = float(expr.getCoeff(i))
                total += c * float(var_values.get(v1, 0.0)) * float(var_values.get(v2, 0.0))
            return total
        # Constant-like fallback
        return float(expr)
    except Exception:
        return None


def _gurobi_debug_pred(
    model: gp.Model,
    seg: str,
    pred_var: Any,
    x_vars: dict[str, Any],
    y_vars: dict[tuple[str, str], Any],
    k_map: dict[str, list[str]],
    *,
    coeffs: dict[str, Any] | None = None,
) -> float | None:
    """Debug-only solver prediction f(decision) to validate embedded objective expressions."""
    if not model.SolCount:
        return None
    decision_budget = float(x_vars[seg].X)
    decision_vals: dict[Any, float] = {x_vars[seg]: decision_budget}
    for k in k_map.get(seg, []):
        decision_vals[y_vars[(seg, k)]] = float(y_vars[(seg, k)].X)
    f_dec = _eval_gurobi_expr(pred_var, decision_vals)
    if f_dec is not None:
        return float(f_dec)
    f_dec = _predicted_value(pred_var)
    if f_dec is not None:
        return float(f_dec)
    return None


def _budget_tiebreak_penalty(config: CampaignOptConfig) -> float:
    """Per-dollar penalty on summed daily budgets; breaks ties on predicted target."""
    raw = (config.constraints or {}).get("budget_tiebreak_penalty", 1e-8)
    penalty = float(raw)
    if penalty < 0:
        raise ValueError("constraints.budget_tiebreak_penalty must be non-negative")
    return penalty


def _add_regional_order_constraints(
    model: gp.Model,
    config: CampaignOptConfig,
    segments: list[str],
    x_vars: dict[str, Any],
) -> None:
    """Enforce USA >= A >= B (or custom order from config)."""
    regional_order = parse_regional_order(config.constraints)
    if len(regional_order) < 2:
        return
    region_spend: dict[str, Any] = {}
    for region in regional_order:
        segs = [s for s in segments if region_of_segment(s) == region]
        if segs:
            region_spend[region] = gp.quicksum(x_vars[s] for s in segs)
    for i in range(len(regional_order) - 1):
        r_hi, r_lo = regional_order[i], regional_order[i + 1]
        if r_hi in region_spend and r_lo in region_spend:
            model.addConstr(region_spend[r_hi] >= region_spend[r_lo], name=f"order_{r_hi}_{r_lo}")


def solve_campaign_milp(
    config: CampaignOptConfig,
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    segment_predictor: SegmentPredictor,
    *,
    total_budget: float,
    output_dir: Path | None = None,
    model_name: str = "campaign_milp",
    time_limit: int = 600,
    write_outputs: bool = True,
    model: gp.Model | None = None,
    x_vars: dict[str, Any] | None = None,
    y_vars: dict[tuple[str, str], Any] | None = None,
    fixed_keyword_sets: dict[str, str] | None = None,
    fixed_budgets: dict[str, float] | None = None,
    planning_dates: list[pd.Timestamp] | None = None,
    segment_predictors_by_date: list[SegmentPredictor] | None = None,
    train: pd.DataFrame | None = None,
    solver_coeffs: dict[str, Any] | None = None,
    planning_calendar_offsets: dict[tuple[str, int], float] | None = None,
    level_ub_overrides: dict[str, float] | None = None,
    gating_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Single entry point for segment budget + keyword-set MILPs.

    Each backend only supplies how predicted target is built per segment
    (linear budget slope, embedded trees, etc.).

    Maximizes ``sum_s f_s(plan)`` (total predicted target), minus an optional
    budget tie-break. When ``apply_observed_budget_floor`` is true, each ``f_s``
    is zero below the segment's minimum observed ``daily_budget``.

    Pass ``model``, ``x_vars``, and ``y_vars`` when a backend adds constraints
    (e.g. exact tree embedding) before the objective is set.
    """
    segments = build_segment_list(candidates)
    k_map = candidates_by_segment(candidates)
    bounds = historical_budget_bounds(panel, segments)

    if model is None:
        model = gp.Model(model_name)
        model.setParam("OutputFlag", 1)
        model.setParam("TimeLimit", time_limit)

    if x_vars is None or y_vars is None:
        x_vars = x_vars or {}
        y_vars = y_vars or {}
        for seg in segments:
            if seg not in x_vars:
                lo, hi = bounds[seg]
                if fixed_budgets and seg in fixed_budgets:
                    b = float(fixed_budgets[seg])
                    x_vars[seg] = model.addVar(lb=b, ub=b, name=f"x_{seg}")
                else:
                    x_vars[seg] = model.addVar(lb=lo, ub=hi, name=f"x_{seg}")
            if not fixed_keyword_sets:
                for k in k_map.get(seg, []):
                    key = (seg, k)
                    if key not in y_vars:
                        y_vars[key] = model.addVar(vtype=GRB.BINARY, name=f"y_{seg}_{k}")
                model.addConstr(
                    gp.quicksum(y_vars[(seg, k)] for k in k_map.get(seg, [])) == 1,
                    name=f"one_set_{seg}",
                )
            else:
                chosen = str(fixed_keyword_sets.get(seg, ""))
                for k in k_map.get(seg, []):
                    key = (seg, k)
                    if key not in y_vars:
                        fix_val = 1.0 if str(k) == chosen else 0.0
                        y_vars[key] = model.addVar(
                            lb=fix_val, ub=fix_val, vtype=GRB.BINARY, name=f"y_{seg}_{k}"
                        )

    pred_vars: dict[str, Any] = {}
    if segment_predictors_by_date:
        for seg in segments:
            terms = [
                segment_predictors_by_date[i](seg, x_vars[seg], y_vars, k_map)
                for i in range(len(segment_predictors_by_date))
            ]
            pred_vars[seg] = gp.quicksum(terms) if len(terms) > 1 else terms[0]
    else:
        for seg in segments:
            pred_vars[seg] = segment_predictor(seg, x_vars[seg], y_vars, k_map)

    n_planning_days = len(segment_predictors_by_date) if segment_predictors_by_date else 1
    min_budgets = gate_pred_vars_if_enabled(
        model,
        config,
        pred_vars,
        x_vars,
        segments,
        k_map,
        bounds,
        panel,
        solver_coeffs=solver_coeffs,
        n_planning_days=n_planning_days,
        calendar_offsets=planning_calendar_offsets,
        level_ub_overrides=level_ub_overrides,
        gating_panel=gating_panel,
    )

    model.addConstr(gp.quicksum(x_vars[s] for s in segments) <= total_budget, name="total_budget")
    _add_regional_order_constraints(model, config, segments, x_vars)

    level_sum = gp.quicksum(pred_vars[s] for s in segments)
    penalty = _budget_tiebreak_penalty(config)
    if penalty > 0:
        model.setObjective(
            level_sum - penalty * gp.quicksum(x_vars[s] for s in segments),
            GRB.MAXIMIZE,
        )
    else:
        model.setObjective(level_sum, GRB.MAXIMIZE)
    tb = f" − {penalty}×Σ budget" if penalty > 0 else ""
    print(f"[Info] MILP objective: Σ segment level{tb}", flush=True)
    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        status_name = {
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INFEASIBLE_OR_UNBOUNDED",
            GRB.UNBOUNDED: "UNBOUNDED",
        }.get(int(model.Status), str(model.Status))
        raise RuntimeError(
            f"Gurobi solve failed with status {status_name} ({int(model.Status)}). "
            "No solution available for campaign_plan."
        )

    if write_outputs and output_dir is not None:
        output_dir = Path(output_dir)
        if config.debug_write_lp:
            model.write(str(output_dir / f"{model_name}.lp"))

    rows = []
    has_solution = model.SolCount > 0
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
        budget_raw = float(x_vars[seg].X) if has_solution else None
        budget_out = round(budget_raw, 2) if budget_raw is not None else None
        milp_raw = _predicted_value(pred_vars[seg])
        milp_out = milp_raw
        if (
            config.evaluation.apply_observed_budget_floor
            and budget_raw is not None
            and milp_raw is not None
            and min_budgets.get(seg, 0.0) > 0.0
        ):
            from utils.optimizer_prediction import apply_observed_budget_floor

            milp_out = float(
                apply_observed_budget_floor(
                    np.array([milp_raw]),
                    np.array([budget_raw]),
                    np.array([seg]),
                    min_budgets,
                    budget_atol=float(config.evaluation.budget_floor_atol),
                )[0]
            )
        rows.append(
            {
                "segment": seg,
                "region": region_of_segment(seg),
                "daily_budget": budget_out,
                "keyword_set_id": chosen_k,
                "milp_pred": milp_out,
                "external_model_pred": None,
                "n_planning_days": n_planning_days,
            }
        )
    plan = pd.DataFrame(rows)

    if write_outputs and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        plan.to_csv(output_dir / "campaign_plan.csv", index=False)
        status_payload: dict[str, Any] = {
            "status": int(model.Status),
            "obj_val": model.ObjVal if model.SolCount else None,
            "objective": "levels",
        }
        if model.SolCount and len(plan):
            milp = pd.to_numeric(plan["milp_pred"], errors="coerce")
            bud = pd.to_numeric(plan["daily_budget"], errors="coerce")
            level_sum_val = float(milp.sum())
            budget_sum = float(bud.sum())
            status_payload["predicted_level_sum"] = level_sum_val
            status_payload["budget_sum"] = budget_sum
            status_payload["tiebreak_penalty"] = penalty
            status_payload["tiebreak_term"] = penalty * budget_sum
            if penalty > 0 and model.ObjVal is not None:
                status_payload["obj_val_as_level_sum_minus_tiebreak"] = float(
                    model.ObjVal
                ) + penalty * budget_sum
        with open(output_dir / "solver_status.json", "w", encoding="utf-8") as f:
            json.dump(status_payload, f, indent=2)
    return plan


def make_linear_segment_predictor(
    coeffs: dict[str, Any],
    *,
    calendar_offsets: dict[tuple[str, int], float] | None = None,
    date_index: int = 0,
) -> SegmentPredictor:
    """Linear: alpha_s + beta_s * budget + calendar + sum_k y_sk * static_context_lift_k."""
    seg_slope = coeffs.get("segment_budget_slope", {})
    seg_intercept = coeffs.get("segment_intercept", {})
    set_lift = coeffs.get("static_context_lift") or coeffs.get("keyword_set_effect", {})
    cal_offset = float(coeffs.get("calendar_offset", 0.0))

    def _predict(seg: str, x_var: Any, y_vars: dict, k_map: dict[str, list[str]]) -> Any:
        beta = float(seg_slope.get(seg, seg_slope.get(str(seg), 0.0)))
        alpha = float(seg_intercept.get(seg, seg_intercept.get(str(seg), 0.0)))
        cal = cal_offset
        if calendar_offsets is not None:
            cal = float(calendar_offsets.get((seg, date_index), cal_offset))
        expr = alpha + beta * x_var + cal
        for k in k_map.get(seg, []):
            lift = float(set_lift.get(str(k), 0.0))
            expr += lift * y_vars[(seg, k)]
        return expr

    return _predict


def make_linear_segment_predictors_for_dates(
    coeffs: dict[str, Any],
    calendar_offsets: dict[tuple[str, int], float],
    n_dates: int,
) -> list[SegmentPredictor]:
    """One segment predictor per planning date (for summed multi-day objectives)."""
    return [
        make_linear_segment_predictor(coeffs, calendar_offsets=calendar_offsets, date_index=i)
        for i in range(n_dates)
    ]

