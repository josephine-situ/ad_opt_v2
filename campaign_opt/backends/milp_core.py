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
) -> pd.DataFrame:
    """
    Single entry point for segment budget + keyword-set MILPs.

    Each backend only supplies how predicted target is built per segment
    (linear budget slope, piecewise budget curve, etc.).
    """
    segments = build_segment_list(candidates)
    k_map = candidates_by_segment(candidates)
    bounds = historical_budget_bounds(panel, segments)

    if model is None:
        model = gp.Model(model_name)
        model.setParam("OutputFlag", 1)
        model.setParam("TimeLimit", time_limit)

    x_vars: dict[str, Any] = {}
    y_vars: dict[tuple[str, str], Any] = {}
    pred_vars: dict[str, Any] = {}

    # Decision vars: one budget per segment, one keyword set per segment
    for seg in segments:
        lo, hi = bounds[seg]
        x_vars[seg] = model.addVar(lb=lo, ub=hi, name=f"x_{seg}")
        for k in k_map.get(seg, []):
            y_vars[(seg, k)] = model.addVar(vtype=GRB.BINARY, name=f"y_{seg}_{k}")
        model.addConstr(
            gp.quicksum(y_vars[(seg, k)] for k in k_map.get(seg, [])) == 1,
            name=f"one_set_{seg}",
        )
        pred_vars[seg] = segment_predictor(seg, x_vars[seg], y_vars, k_map)

    model.addConstr(gp.quicksum(x_vars[s] for s in segments) <= total_budget, name="total_budget")
    _add_regional_order_constraints(model, config, segments, x_vars)
    model.setObjective(gp.quicksum(pred_vars[s] for s in segments), GRB.MAXIMIZE)
    model.optimize()

    if write_outputs and output_dir is not None:
        output_dir = Path(output_dir)
        if config.debug_write_lp:
            model.write(str(output_dir / f"{model_name}.lp"))

    rows = []
    for seg in segments:
        chosen_k = next(
            (k for k in k_map.get(seg, []) if y_vars[(seg, k)].X > 0.5),
            None,
        )
        rows.append(
            {
                "segment": seg,
                "region": region_of_segment(seg),
                "daily_budget": x_vars[seg].X if model.SolCount else None,
                "keyword_set_id": chosen_k,
                "predicted_target": model.getValue(pred_vars[seg]) if model.SolCount else None,
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


def make_linear_segment_predictor(coeffs: dict[str, Any]) -> SegmentPredictor:
    """Linear: alpha_s + beta_s * budget + calendar + sum_k y_sk * set_lift_k."""
    seg_slope = coeffs.get("segment_budget_slope", {})
    seg_intercept = coeffs.get("segment_intercept", {})
    set_effect = coeffs.get("keyword_set_effect", {})
    cal_offset = float(coeffs.get("calendar_offset", 0.0))

    def _predict(seg: str, x_var: Any, y_vars: dict, k_map: dict[str, list[str]]) -> Any:
        beta = float(seg_slope.get(seg, seg_slope.get(str(seg), 0.0)))
        alpha = float(seg_intercept.get(seg, seg_intercept.get(str(seg), 0.0)))
        expr = alpha + beta * x_var + cal_offset
        for k in k_map.get(seg, []):
            lift = float(set_effect.get(str(k), 0.0))
            expr += lift * y_vars[(seg, k)]
        return expr

    return _predict


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

    set_effect = coeffs.get("keyword_set_effect", {})
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
            expr += float(set_effect.get(str(k), 0.0)) * y_vars[(seg, k)]
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
