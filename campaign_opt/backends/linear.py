"""Linear MILP backend (thin wrapper over milp_core)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from campaign_opt.backends.milp_core import (
    make_linear_segment_predictor,
    make_linear_segment_predictors_for_dates,
    solve_campaign_milp,
)
from campaign_opt.coefficients import calendar_offsets_for_planning, refresh_static_context_lift
from campaign_opt.decisions import build_segment_list
from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import build_keyword_set_feature_table


def _prepare_linear_coeffs(
    config: CampaignOptConfig,
    coeffs: dict[str, Any],
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    set_features = build_keyword_set_feature_table(config.course)
    return refresh_static_context_lift(coeffs, config, candidates, set_features)


def solve_linear_campaign_milp(
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
    segments = build_segment_list(candidates)
    dates = [pd.Timestamp(d) for d in planning_dates] if planning_dates else None
    coeffs = _prepare_linear_coeffs(config, coeffs, candidates)
    segment_predictor = make_linear_segment_predictor(coeffs)
    segment_predictors_by_date = None

    if dates and len(dates) > 1:
        if train is None:
            raise ValueError("train is required when planning_dates has more than one date")
        cal_offsets = calendar_offsets_for_planning(train, config, dates, segments)
        segment_predictors_by_date = make_linear_segment_predictors_for_dates(
            coeffs, cal_offsets, len(dates)
        )

    return solve_campaign_milp(
        config,
        candidates,
        panel,
        segment_predictor,
        total_budget=total_budget,
        output_dir=output_dir,
        model_name="campaign_linear",
        time_limit=time_limit,
        write_outputs=write_outputs,
        fixed_keyword_sets=fixed_keyword_sets,
        fixed_budgets=fixed_budgets,
        planning_dates=dates,
        segment_predictors_by_date=segment_predictors_by_date,
        train=train,
        solver_coeffs=coeffs,
    )
