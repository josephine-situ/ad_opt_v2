"""Piecewise-linear budget MILP backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gurobipy as gp
import pandas as pd

from campaign_opt.backends.linear import _prepare_linear_coeffs, solve_linear_campaign_milp
from campaign_opt.backends.milp_core import (
    _build_piecewise_budget,
    make_piecewise_segment_predictor,
    solve_campaign_milp,
)
from campaign_opt.decisions import build_segment_list
from campaign_opt.schema import CampaignOptConfig


def build_piecewise_coeffs(
    panel: pd.DataFrame,
    segments: list[str],
    coeffs: dict[str, Any],
    n_knots: int,
) -> dict[str, Any]:
    out = dict(coeffs)
    out["piecewise_budget"] = _build_piecewise_budget(panel, segments, coeffs, n_knots)
    return out


def solve_piecewise_campaign_milp(
    config: CampaignOptConfig,
    coeffs: dict[str, Any],
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    total_budget: float,
    output_dir: Path,
    time_limit: int = 600,
    write_outputs: bool = True,
    fixed_keyword_sets: dict[str, str] | None = None,
    fixed_budgets: dict[str, float] | None = None,
    planning_dates: list[pd.Timestamp] | None = None,
    train: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if planning_dates and len(planning_dates) > 1:
        return solve_linear_campaign_milp(
            config=config,
            coeffs=coeffs,
            candidates=candidates,
            panel=panel,
            total_budget=total_budget,
            output_dir=output_dir,
            time_limit=time_limit,
            write_outputs=write_outputs,
            fixed_keyword_sets=fixed_keyword_sets,
            fixed_budgets=fixed_budgets,
            planning_dates=planning_dates,
            train=train,
        )
    segments = build_segment_list(candidates)
    enriched = build_piecewise_coeffs(panel, segments, _prepare_linear_coeffs(config, coeffs, candidates), config.piecewise_budget_knots)

    try:
        # Piecewise lambdas are attached to this model inside the predictor closure
        model = gp.Model("campaign_pw")
        model.setParam("OutputFlag", 1)
        model.setParam("TimeLimit", time_limit)
        predictor = make_piecewise_segment_predictor(
            enriched, panel, segments, config.piecewise_budget_knots, model
        )
        return solve_campaign_milp(
            config,
            candidates,
            panel,
            predictor,
            total_budget=total_budget,
            output_dir=output_dir,
            model_name="campaign_piecewise",
            time_limit=time_limit,
            write_outputs=write_outputs,
            model=model,
            fixed_keyword_sets=fixed_keyword_sets,
            fixed_budgets=fixed_budgets,
            train=train,
            solver_coeffs=enriched,
        )
    except Exception as exc:
        print(f"[Warn] Piecewise solve failed ({exc}); falling back to linear.")
        return solve_linear_campaign_milp(
            config,
            enriched,
            candidates,
            panel,
            total_budget=total_budget,
            output_dir=output_dir,
            time_limit=time_limit,
            write_outputs=write_outputs,
            fixed_keyword_sets=fixed_keyword_sets,
            fixed_budgets=fixed_budgets,
            planning_dates=planning_dates,
            train=train,
        )
