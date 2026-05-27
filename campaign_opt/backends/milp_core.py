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

from campaign_opt.decisions import (
    build_segment_list,
    candidates_by_segment,
    historical_budget_bounds,
    parse_regional_order,
    region_of_segment,
)
from campaign_opt.schema import CampaignOptConfig

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


def _sos2_weights_at_x(knots: list[float], x: float) -> list[float]:
    """Piecewise-linear (SOS2) weights for budget ``x`` on ``knots``."""
    xs = [float(k) for k in knots]
    x = float(x)
    n = len(xs)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    if x <= xs[0]:
        w = [0.0] * n
        w[0] = 1.0
        return w
    if x >= xs[-1]:
        w = [0.0] * n
        w[-1] = 1.0
        return w
    for i in range(n - 1):
        lo, hi = xs[i], xs[i + 1]
        if lo <= x <= hi:
            span = hi - lo
            t = (x - lo) / span if span > 0 else 0.0
            w = [0.0] * n
            w[i] = 1.0 - t
            w[i + 1] = t
            return w
    w = [0.0] * n
    w[-1] = 1.0
    return w


def _piecewise_lambda_values(
    model: gp.Model,
    seg: str,
    budget: float,
    pw_seg: dict[str, list[float]],
) -> dict[Any, float]:
    weights = _sos2_weights_at_x(pw_seg["knots"], budget)
    out: dict[Any, float] = {}
    prefix = f"lam_{seg}["
    for v in model.getVars():
        vn = v.VarName
        if not vn.startswith(prefix):
            continue
        idx = int(vn[len(prefix) : -1])
        if idx < len(weights):
            out[v] = weights[idx]
    return out


def _baseline_var_values_for_pred(
    seg: str,
    x_vars: dict[str, Any],
    y_vars: dict[tuple[str, str], Any],
    k_map: dict[str, list[str]],
    baseline_k: str,
    baseline_budget: float,
    model: gp.Model,
    coeffs: dict[str, Any] | None,
) -> dict[Any, float]:
    values: dict[Any, float] = {x_vars[seg]: float(baseline_budget)}
    for k in k_map.get(seg, []):
        values[y_vars[(seg, k)]] = 1.0 if str(k) == str(baseline_k) else 0.0
    pw = (coeffs or {}).get("piecewise_budget")
    if pw and seg in pw:
        values.update(_piecewise_lambda_values(model, seg, baseline_budget, pw[seg]))
    return values


def _baseline_keyword_sets_for_milp(
    train: pd.DataFrame | None,
    segments: list[str],
    k_map: dict[str, list[str]],
) -> dict[str, str]:
    """Modal train keyword set per segment for f(0), matching ensemble evaluation."""
    if train is not None and not train.empty:
        from campaign_opt.evaluation import baseline_keyword_sets

        ref = baseline_keyword_sets(train)
        return {
            str(seg): str(ref.get(seg, k_map.get(seg, [""])[0]))
            for seg in segments
        }
    return {seg: str(k_map.get(seg, [""])[0]) for seg in segments}


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
    pw = (coeffs or {}).get("piecewise_budget")
    if pw and seg in pw:
        decision_vals.update(_piecewise_lambda_values(model, seg, decision_budget, pw[seg]))
    f_dec = _eval_gurobi_expr(pred_var, decision_vals)
    if f_dec is not None:
        return float(f_dec)
    # Fallback for non-LinExpr predictors.
    f_dec = _predicted_value(pred_var)
    if f_dec is not None:
        return float(f_dec)
    return None


def _gurobi_incremental_pred(
    model: gp.Model,
    seg: str,
    pred_var: Any,
    x_vars: dict[str, Any],
    y_vars: dict[tuple[str, str], Any],
    k_map: dict[str, list[str]],
    baseline_k: str,
    baseline_budget: float,
    *,
    coeffs: dict[str, Any] | None = None,
) -> float | None:
    """f(decision) - f(0) in solver space (kept for compatibility helpers)."""
    if not model.SolCount:
        return None
    f_dec = _gurobi_debug_pred(
        model,
        seg,
        pred_var,
        x_vars,
        y_vars,
        k_map,
        coeffs=coeffs,
    )
    baseline_vals = _baseline_var_values_for_pred(
        seg,
        x_vars,
        y_vars,
        k_map,
        baseline_k,
        baseline_budget,
        model,
        coeffs,
    )
    f0 = _eval_gurobi_expr(pred_var, baseline_vals)
    if f0 is None:
        # Gurobi 13+ disallows mutating Var.X; legacy fallback for non-LinExpr predictors.
        saved_x = x_vars[seg].X
        saved_y = {k: y_vars[(seg, k)].X for k in k_map.get(seg, [])}
        try:
            x_vars[seg].X = float(baseline_budget)
            for k in k_map.get(seg, []):
                y_vars[(seg, k)].X = 1.0 if str(k) == str(baseline_k) else 0.0
            f0 = _predicted_value(pred_var)
        except AttributeError:
            f0 = None
        finally:
            try:
                x_vars[seg].X = saved_x
                for k, val in saved_y.items():
                    y_vars[(seg, k)].X = val
            except AttributeError:
                pass
    if f_dec is None or f0 is None:
        return None
    return float(f_dec) - float(f0)


def _plan_incremental_pred(
    model: gp.Model,
    seg: str,
    pred_var: Any,
    chosen_k: str | None,
    x_vars: dict[str, Any],
    y_vars: dict[tuple[str, str], Any],
    k_map: dict[str, list[str]],
    baseline_budget: float,
    *,
    baseline_level_by_key: dict[tuple[str, str], float] | None = None,
    coeffs: dict[str, Any] | None = None,
) -> float | None:
    """Incremental lift for the solved plan; tree backends use precomputed baseline levels."""
    if baseline_level_by_key is not None and chosen_k is not None:
        f_dec = _predicted_value(pred_var)
        key = (seg, str(chosen_k))
        f0 = baseline_level_by_key.get(key, baseline_level_by_key.get((str(seg), str(chosen_k))))
        if f_dec is not None and f0 is not None:
            return float(f_dec) - float(f0)
    return _optimizer_incremental_pred(
        model,
        seg,
        pred_var,
        x_vars,
        y_vars,
        k_map,
        str(chosen_k),
        baseline_budget,
        coeffs=coeffs,
    )


def _optimizer_incremental_pred(
    model: gp.Model,
    seg: str,
    pred_var: Any,
    x_vars: dict[str, Any],
    y_vars: dict[tuple[str, str], Any],
    k_map: dict[str, list[str]],
    baseline_k: str,
    baseline_budget: float,
    *,
    coeffs: dict[str, Any] | None = None,
) -> float | None:
    """f(plan) - f(0) where f(plan) is read directly from optimized expression value."""
    if not model.SolCount:
        return None
    f_dec = _predicted_value(pred_var)
    baseline_vals = _baseline_var_values_for_pred(
        seg,
        x_vars,
        y_vars,
        k_map,
        baseline_k,
        baseline_budget,
        model,
        coeffs,
    )
    f0 = _eval_gurobi_expr(pred_var, baseline_vals)
    if f0 is None:
        # Gurobi 13+ disallows mutating Var.X; legacy fallback for non-LinExpr predictors.
        saved_x = x_vars[seg].X
        saved_y = {k: y_vars[(seg, k)].X for k in k_map.get(seg, [])}
        try:
            x_vars[seg].X = float(baseline_budget)
            for k in k_map.get(seg, []):
                y_vars[(seg, k)].X = 1.0 if str(k) == str(baseline_k) else 0.0
            f0 = _predicted_value(pred_var)
        except AttributeError:
            f0 = None
        finally:
            try:
                x_vars[seg].X = saved_x
                for k, val in saved_y.items():
                    y_vars[(seg, k)].X = val
            except AttributeError:
                pass
    if f_dec is None or f0 is None:
        return None
    return float(f_dec) - float(f0)


def _budget_tiebreak_penalty(config: CampaignOptConfig) -> float:
    """Per-dollar penalty on summed daily budgets; breaks ties on predicted target."""
    raw = (config.constraints or {}).get("budget_tiebreak_penalty", 1e-8)
    penalty = float(raw)
    if penalty < 0:
        raise ValueError("constraints.budget_tiebreak_penalty must be non-negative")
    return penalty


def _baseline_budget(config: CampaignOptConfig) -> float:
    return float(config.evaluation.baseline_budget)


def _piecewise_budget_level_at(pw_seg: dict[str, list[float]], budget: float) -> float:
    weights = _sos2_weights_at_x(pw_seg["knots"], budget)
    vals = pw_seg["values"]
    return float(sum(weights[i] * float(vals[i]) for i in range(len(weights))))


def baseline_levels_from_coeffs(
    coeffs: dict[str, Any],
    segments: list[str],
    k_map: dict[str, list[str]],
    baseline_budget: float,
    *,
    n_planning_days: int = 1,
    calendar_offsets: dict[tuple[str, int], float] | None = None,
) -> dict[tuple[str, str], float]:
    """
    f_k(baseline_budget) per (segment, keyword_set) for incremental MILP objectives.

    Matches evaluation: same keyword set at ``baseline_budget`` (default 0).
    """
    pw = coeffs.get("piecewise_budget")
    set_lift = coeffs.get("static_context_lift") or coeffs.get("keyword_set_effect", {})
    seg_slope = coeffs.get("segment_budget_slope", {})
    seg_intercept = coeffs.get("segment_intercept", {})
    cal_offset = float(coeffs.get("calendar_offset", 0.0))
    n_days = max(1, int(n_planning_days))
    out: dict[tuple[str, str], float] = {}

    for seg in segments:
        beta = float(seg_slope.get(seg, seg_slope.get(str(seg), 0.0)))
        alpha = float(seg_intercept.get(seg, seg_intercept.get(str(seg), 0.0)))
        for k in k_map.get(seg, []):
            kid = str(k)
            lift = float(set_lift.get(kid, 0.0))
            total = 0.0
            for day_idx in range(n_days):
                cal = cal_offset
                if calendar_offsets is not None:
                    cal = float(calendar_offsets.get((seg, day_idx), cal_offset))
                if pw is not None and seg in pw:
                    total += _piecewise_budget_level_at(pw[seg], baseline_budget) + cal + lift
                else:
                    total += alpha + beta * float(baseline_budget) + cal + lift
            out[(seg, kid)] = total
    return out


def _incremental_objective_expr(
    pred_vars: dict[str, Any],
    y_vars: dict[tuple[str, str], Any],
    segments: list[str],
    k_map: dict[str, list[str]],
    baseline_level_by_key: dict[tuple[str, str], float],
) -> Any:
    """Sum_s f_s(plan) - sum_{s,k} y_sk * f_k(baseline)."""
    level_sum = gp.quicksum(pred_vars[s] for s in segments)
    baseline_terms = []
    for seg in segments:
        for k in k_map.get(seg, []):
            key = (seg, str(k))
            if key not in y_vars:
                continue
            f0 = float(baseline_level_by_key.get(key, 0.0))
            baseline_terms.append(f0 * y_vars[key])
    if not baseline_terms:
        return level_sum
    return level_sum - gp.quicksum(baseline_terms)


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
    baseline_level_by_key: dict[tuple[str, str], float] | None = None,
    planning_calendar_offsets: dict[tuple[str, int], float] | None = None,
) -> pd.DataFrame:
    """
    Single entry point for segment budget + keyword-set MILPs.

    Each backend only supplies how predicted target is built per segment
    (linear budget slope, piecewise budget curve, embedded trees, etc.).

    The objective maximizes incremental lift
    ``sum_s f_s(plan) - sum_{s,k} y_sk * f_k(baseline_budget)`` (same keyword set at
  ``evaluation.baseline_budget``), minus an optional budget tie-break.

    Pass ``model``, ``x_vars``, and ``y_vars`` when a backend adds constraints
    (e.g. exact tree embedding) before the objective is set.

    Tree backends must pass ``baseline_level_by_key`` (per-candidate level at baseline).
    Linear/piecewise backends may omit it; levels are derived from ``solver_coeffs``.
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

    model.addConstr(gp.quicksum(x_vars[s] for s in segments) <= total_budget, name="total_budget")
    _add_regional_order_constraints(model, config, segments, x_vars)

    n_planning_days = len(segment_predictors_by_date) if segment_predictors_by_date else 1
    baseline_budget = _baseline_budget(config)
    if baseline_level_by_key is None:
        if solver_coeffs is None:
            raise ValueError(
                "incremental MILP objective requires baseline_level_by_key or solver_coeffs"
            )
        baseline_level_by_key = baseline_levels_from_coeffs(
            solver_coeffs,
            segments,
            k_map,
            baseline_budget,
            n_planning_days=n_planning_days,
            calendar_offsets=planning_calendar_offsets,
        )

    pred_lift = _incremental_objective_expr(
        pred_vars, y_vars, segments, k_map, baseline_level_by_key
    )
    penalty = _budget_tiebreak_penalty(config)
    if penalty > 0:
        model.setObjective(
            pred_lift - penalty * gp.quicksum(x_vars[s] for s in segments),
            GRB.MAXIMIZE,
        )
    else:
        model.setObjective(pred_lift, GRB.MAXIMIZE)
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
        rows.append(
            {
                "segment": seg,
                "region": region_of_segment(seg),
                "daily_budget": x_vars[seg].X if model.SolCount else None,
                "keyword_set_id": chosen_k,
                "milp_pred": _predicted_value(pred_vars[seg]),
                "pred_over_base": _plan_incremental_pred(
                    model,
                    seg,
                    pred_vars[seg],
                    chosen_k,
                    x_vars,
                    y_vars,
                    k_map,
                    baseline_budget,
                    baseline_level_by_key=baseline_level_by_key,
                    coeffs=solver_coeffs,
                ),
                "external_model_pred": None,
                "n_planning_days": n_planning_days,
            }
        )
    plan = pd.DataFrame(rows)

    if write_outputs and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        plan.to_csv(output_dir / "campaign_plan.csv", index=False)
        with open(output_dir / "solver_status.json", "w", encoding="utf-8") as f:
            json.dump(
                {"status": int(model.Status), "obj_val": model.ObjVal if model.SolCount else None},
                f,
            )
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


def make_piecewise_segment_predictor(
    coeffs: dict[str, Any],
    panel: pd.DataFrame,
    segments: list[str],
    n_knots: int,
    model: gp.Model,
) -> SegmentPredictor:
    """Must use the same ``model`` instance passed to ``solve_campaign_milp``."""
    """Piecewise-linear budget curve per segment (SOS2-style lambda weights)."""
    pw = coeffs.get("piecewise_budget")
    if pw is None:
        pw = _build_piecewise_budget(panel, segments, coeffs, n_knots)

    set_lift = coeffs.get("static_context_lift") or coeffs.get("keyword_set_effect", {})
    cal_offset = float(coeffs.get("calendar_offset", 0.0))
    # Store per-segment lambda vars on the model object via closure
    lam_vars: dict[str, Any] = {}

    def _predict(seg: str, x_var: Any, y_vars: dict, k_map: dict[str, list[str]]) -> Any:
        knots = np.array(pw[seg]["knots"])
        vals = np.array(pw[seg]["values"])
        n = len(knots)
        if seg not in lam_vars:
            lam_vars[seg] = model.addVars(n, lb=0, ub=1, name=f"lam_{seg}")
        lam = lam_vars[seg]
        model.addConstr(gp.quicksum(lam[i] for i in range(n)) == 1, name=f"conv_{seg}")
        model.addConstr(gp.quicksum(lam[i] * knots[i] for i in range(n)) == x_var, name=f"budget_pw_{seg}")
        expr = gp.quicksum(lam[i] * vals[i] for i in range(n)) + cal_offset
        for k in k_map.get(seg, []):
            expr += float(set_lift.get(str(k), 0.0)) * y_vars[(seg, k)]
        return expr

    return _predict


def _build_piecewise_budget(
    panel: pd.DataFrame,
    segments: list[str],
    coeffs: dict[str, Any],
    n_knots: int,
) -> dict[str, dict[str, list[float]]]:
    pw: dict[str, dict[str, list[float]]] = {}
    for seg in segments:
        sub = panel[panel["segment"] == seg]["daily_budget"].dropna()
        lo, hi = (float(sub.min()), float(sub.max())) if len(sub) >= 2 else (1.0, 100.0)
        knots = np.linspace(lo, hi, n_knots)
        slope = float(coeffs.get("segment_budget_slope", {}).get(seg, 0.01))
        a = max(float(coeffs.get("segment_intercept", {}).get(seg, 0.0)), 1e-6)
        b = 0.85 if slope > 0 else 0.5
        vals = a * np.power(np.clip(knots, 1e-6, None), b)
        pw[seg] = {"knots": knots.tolist(), "values": vals.tolist()}
    return pw
