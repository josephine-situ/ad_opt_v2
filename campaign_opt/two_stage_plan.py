"""Shared two-stage planning: stage-1 keyword sets, stage-2 daily budgets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from campaign_opt.features import train_before_date
from campaign_opt.optimize import run_optimizer
from campaign_opt.schema import CampaignOptConfig


def select_keyword_sets_for_window(
    config: CampaignOptConfig,
    manifest: dict,
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    total_budget: float,
    output_dir: Path,
) -> tuple[dict[str, str], pd.DataFrame]:
    """
    Stage 1: multi-day MILP over ``[window_start, window_end]``.

    Trains the optimizer on rows with ``date < window_start`` and picks one keyword
    set per segment for the full window (same path as ``--strategy two_stage`` backtest).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    window_start = pd.Timestamp(window_start).normalize()
    window_end = pd.Timestamp(window_end).normalize()
    dates = pd.date_range(window_start, window_end, freq="D")

    min_train = config.model_policy.validation.min_train_rows
    train = train_before_date(df, window_start)
    if len(train) < min_train:
        raise RuntimeError(
            f"Insufficient train rows before {window_start.date()}: "
            f"{len(train)} < {min_train}"
        )

    set_plan = run_optimizer(
        config,
        manifest,
        train,
        candidates,
        panel,
        total_budget=total_budget,
        output_dir=output_dir,
        planning_dates=list(dates),
        write_outputs=True,
        tune_optimizer=True,
    )
    fixed_keyword_sets = {
        str(row["segment"]): str(row["keyword_set_id"])
        for _, row in set_plan.iterrows()
        if pd.notna(row.get("keyword_set_id"))
    }
    if not fixed_keyword_sets:
        raise RuntimeError("Stage 1 produced no keyword_set_id assignments")

    with open(output_dir / "fixed_keyword_sets.json", "w", encoding="utf-8") as f:
        json.dump(fixed_keyword_sets, f, indent=2)
    set_plan.to_csv(output_dir / "keyword_set_plan.csv", index=False)
    return fixed_keyword_sets, set_plan


def optimize_budgets_for_day(
    config: CampaignOptConfig,
    manifest: dict,
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    planning_date: pd.Timestamp,
    total_budget: float,
    fixed_keyword_sets: dict[str, str],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Stage 2: single-day MILP with fixed keyword sets.

    Trains on ``date < planning_date`` (walk-forward) with time-series CV
    hyperparameter search on that slice (same as daily backtest).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    planning_date = pd.Timestamp(planning_date).normalize()

    min_train = config.model_policy.validation.min_train_rows
    train = train_before_date(df, planning_date)
    if len(train) < min_train:
        raise RuntimeError(
            f"Insufficient train rows before {planning_date.date()}: "
            f"{len(train)} < {min_train}"
        )

    plan = run_optimizer(
        config,
        manifest,
        train,
        candidates,
        panel,
        total_budget=total_budget,
        output_dir=output_dir,
        planning_date=planning_date,
        fixed_keyword_sets=fixed_keyword_sets,
        write_outputs=True,
        tune_optimizer=True,
    )
    plan.to_csv(output_dir / "campaign_plan.csv", index=False)
    return plan
