"""McCormick gating of MILP level predictions below observed minimum budget."""

from __future__ import annotations

from typing import Any

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

from campaign_opt.decisions import observed_min_daily_budget
from campaign_opt.schema import CampaignOptConfig


def budget_big_m_from_bounds(budget_lo: float, budget_hi: float) -> float:
    """Budget-span Big-M (matches tree embed processed-span padding, raw dollars)."""
    span = max(float(budget_hi) - float(budget_lo), 1.0)
    return span * 1.1 + 1.0


def linear_segment_level_ub(
    seg: str,
    k_map: dict[str, list[str]],
    coeffs: dict[str, Any],
    budget_hi: float,
    *,
    n_planning_days: int = 1,
    calendar_offsets: dict[tuple[str, int], float] | None = None,
) -> float:
    """Conservative upper bound on linear/piecewise level for one segment."""
    seg_slope = coeffs.get("segment_budget_slope", {})
    seg_intercept = coeffs.get("segment_intercept", {})
    set_lift = coeffs.get("static_context_lift") or coeffs.get("keyword_set_effect", {})
    cal_offset = abs(float(coeffs.get("calendar_offset", 0.0)))
    if calendar_offsets is not None:
        cal_offset = max(
            (abs(float(calendar_offsets.get((seg, i), cal_offset))) for i in range(n_planning_days)),
            default=cal_offset,
        )
    beta = abs(float(seg_slope.get(seg, seg_slope.get(str(seg), 0.0))))
    alpha = abs(float(seg_intercept.get(seg, seg_intercept.get(str(seg), 0.0))))
    hi = float(budget_hi)
    lifts = [abs(float(set_lift.get(str(k), 0.0))) for k in k_map.get(seg, [])]
    max_lift = max(lifts) if lifts else 0.0
    return alpha + beta * hi + cal_offset * max(1, n_planning_days) + max_lift


def panel_segment_level_ub(
    seg: str,
    panel: pd.DataFrame,
    target: str,
    *,
    config_max: float | None = None,
) -> float:
    """Panel-based level upper bound (tree backends without solver coeffs on solve path)."""
    sub = panel[panel["segment"] == seg]
    if sub.empty or target not in sub.columns:
        ub = 1.0
    else:
        vals = pd.to_numeric(sub[target], errors="coerce").fillna(0.0)
        ub = max(float(vals.max()) * 1.25, 1.0)
    if config_max is not None:
        ub = min(ub, float(config_max))
    return ub


def segment_level_ub(
    seg: str,
    k_map: dict[str, list[str]],
    coeffs: dict[str, Any] | None,
    budget_hi: float,
    panel: pd.DataFrame,
    config: CampaignOptConfig,
    *,
    n_planning_days: int = 1,
    calendar_offsets: dict[tuple[str, int], float] | None = None,
) -> float:
    """Resolve McCormick ``level_ub`` for one segment."""
    cap = config.evaluation.max_level_ub
    if coeffs is not None:
        ub = linear_segment_level_ub(
            seg,
            k_map,
            coeffs,
            budget_hi,
            n_planning_days=n_planning_days,
            calendar_offsets=calendar_offsets,
        )
    else:
        ub = panel_segment_level_ub(seg, panel, config.target, config_max=cap)
    if cap is not None:
        ub = min(ub, float(cap))
    return max(ub, 1e-6)


def apply_gated_baseline_levels(
    baseline_level_by_key: dict[tuple[str, str], float],
    baseline_budget: float,
    min_by_segment: dict[str, float],
) -> dict[tuple[str, str], float]:
    """Zero baseline levels when baseline budget is below observed min for that segment."""
    out: dict[tuple[str, str], float] = {}
    for (seg, kid), level in baseline_level_by_key.items():
        bmin = float(min_by_segment.get(str(seg), min_by_segment.get(seg, 0.0)))
        if float(baseline_budget) < bmin:
            out[(seg, kid)] = 0.0
        else:
            out[(seg, kid)] = float(level)
    return out


def gate_level_expr(
    model: gp.Model,
    raw_expr: Any,
    x_var: Any,
    *,
    budget_min: float,
    level_ub: float,
    budget_big_m: float,
    name_prefix: str,
    budget_atol: float = 0.01,
) -> Any:
    """
    McCormick envelope: gated level is 0 when ``x_var < budget_min``, else ``raw_expr``.

    When ``budget_min <= 0``, returns ``raw_expr`` unchanged.
    """
    if float(budget_min) <= 0.0:
        return raw_expr
    ub = max(float(level_ub), 1e-6)
    m_b = max(float(budget_big_m), 1.0)
    atol = max(float(budget_atol), 0.0)
    active = model.addVar(vtype=GRB.BINARY, name=f"{name_prefix}_active")
    gated = model.addVar(lb=0.0, ub=ub, name=f"{name_prefix}_gated")
    # Match sklearn floor: active when budget >= budget_min - atol (cent tolerance).
    model.addConstr(
        x_var >= float(budget_min) - atol - m_b * (1 - active),
        name=f"{name_prefix}_act",
    )
    model.addConstr(gated <= ub * active, name=f"{name_prefix}_g_ub1")
    model.addConstr(gated <= raw_expr, name=f"{name_prefix}_g_ub2")
    model.addConstr(gated >= raw_expr - ub * (1 - active), name=f"{name_prefix}_g_lb")
    return gated


def gate_pred_vars_if_enabled(
    model: gp.Model,
    config: CampaignOptConfig,
    pred_vars: dict[str, Any],
    x_vars: dict[str, Any],
    segments: list[str],
    k_map: dict[str, list[str]],
    bounds: dict[str, tuple[float, float]],
    panel: pd.DataFrame,
    *,
    solver_coeffs: dict[str, Any] | None = None,
    n_planning_days: int = 1,
    calendar_offsets: dict[tuple[str, int], float] | None = None,
) -> dict[str, float]:
    """
    Optionally replace ``pred_vars`` with gated expressions.

    Returns ``observed_min_daily_budget`` map (for baseline gating).
    """
    mins = observed_min_daily_budget(panel, segments)
    if not config.evaluation.apply_observed_budget_floor:
        return mins
    for seg in segments:
        bmin = mins.get(seg, 0.0)
        if bmin <= 0.0:
            continue
        lo, hi = bounds[seg]
        level_ub = segment_level_ub(
            seg,
            k_map,
            solver_coeffs,
            hi,
            panel,
            config,
            n_planning_days=n_planning_days,
            calendar_offsets=calendar_offsets,
        )
        m_b = budget_big_m_from_bounds(lo, hi)
        safe = str(seg).replace(" ", "_").replace("/", "_")
        pred_vars[seg] = gate_level_expr(
            model,
            pred_vars[seg],
            x_vars[seg],
            budget_min=bmin,
            level_ub=level_ub,
            budget_big_m=m_b,
            name_prefix=f"gate_{safe}",
            budget_atol=float(config.evaluation.budget_floor_atol),
        )
    return mins
