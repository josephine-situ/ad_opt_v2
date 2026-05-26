"""Two-stage walk-forward backtest: fix keyword sets for period, re-optimize budgets weekly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from campaign_opt.backtest import (
    _fit_evaluation_model,
    optimizer_manifest_for_backtest,
)
from campaign_opt.evaluation import (
    compare_plan_and_actual_week,
    plan_vs_actual_row_metrics,
    week_planning_dates,
    week_starts_in_window,
)
from campaign_opt.features import train_before_date
from campaign_opt.optimize import require_optimizer_winner, run_optimizer
from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import build_keyword_set_feature_table


def run_two_stage_backtest(
    config: CampaignOptConfig,
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    total_budget: float,
    out_dir: Path,
    budget_cadence: str = "W-MON",
) -> pd.DataFrame:
    """
    Stage 1: pick keyword set per segment at ``start`` (optimizer with walk-forward train).
    Stage 2: each week, walk-forward train + linear multi-day MILP for budgets (sets fixed).
    Evaluation uses one full-panel model for all scoring.
    """
    out_dir = Path(out_dir)
    plans_dir = out_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    set_features = build_keyword_set_feature_table(config.course)
    opt_manifest = optimizer_manifest_for_backtest(config)
    eval_model = _fit_evaluation_model(config, df, opt_manifest, out_dir)

    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    min_train = config.model_policy.validation.min_train_rows

    # --- Stage 1: period keyword-set selection ---
    stage1_dir = out_dir / "stage1_keyword_sets"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    train0 = train_before_date(df, start)
    if len(train0) < min_train:
        raise RuntimeError(
            f"Insufficient train rows before {start.date()}: {len(train0)} < {min_train}"
        )

    print(
        f"  [stage1] keyword-set optimize at {start.date()} "
        f"(window {start.date()} → {end.date()})"
    )
    set_plan = run_optimizer(
        config,
        opt_manifest,
        train0,
        candidates,
        panel,
        total_budget=total_budget,
        output_dir=stage1_dir,
        planning_date=start,
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
    with open(out_dir / "fixed_keyword_sets.json", "w", encoding="utf-8") as f:
        json.dump(fixed_keyword_sets, f, indent=2)
    set_plan.to_csv(stage1_dir / "keyword_set_plan.csv", index=False)

    # --- Stage 2: weekly budget (linear multi-day MILP; sets fixed) ---
    week_starts = week_starts_in_window(start, end, freq=budget_cadence)
    weekly_rows: list[dict[str, Any]] = []

    for week_start in week_starts:
        week_start = pd.Timestamp(week_start).normalize()
        week_dates = week_planning_dates(week_start, start, end)
        if not week_dates:
            raise RuntimeError(f"No planning dates in week starting {week_start.date()}")

        train = train_before_date(df, week_start)
        if len(train) < min_train:
            raise RuntimeError(
                f"Insufficient train rows before week {week_start.date()}: "
                f"{len(train)} < {min_train}"
            )

        week_dir = plans_dir / week_start.strftime("%Y%m%d")
        week_dir.mkdir(parents=True, exist_ok=True)

        holdout = df[df["date"].isin(week_dates)]
        if holdout.empty:
            raise RuntimeError(f"No modeling-panel rows in week starting {week_start.date()}")

        plan = run_optimizer(
            config,
            opt_manifest,
            train,
            candidates,
            panel,
            total_budget=total_budget,
            output_dir=week_dir,
            planning_dates=week_dates,
            fixed_keyword_sets=fixed_keyword_sets,
            write_outputs=True,
            tune_optimizer=False,
        )
        plan["week_start"] = week_start.date().isoformat()
        plan["week_end"] = week_dates[-1].date().isoformat()
        plan["n_planning_days"] = len(week_dates)
        plan.to_csv(week_dir / "campaign_plan.csv", index=False)

        plan_budget = pd.to_numeric(plan["daily_budget"], errors="coerce").fillna(0.0)
        week_row: dict[str, Any] = {
            "week_start": week_start.date().isoformat(),
            "week_end": week_dates[-1].date().isoformat(),
            "n_days": len(week_dates),
            "optimizer_winner": require_optimizer_winner(config),
            "n_segments": len(plan),
            "plan_budget_total": float(plan_budget.sum()),
            "n_segments_zero_budget": int((plan_budget <= 0).sum()),
        }

        weekly_comp, daily_comp = compare_plan_and_actual_week(
            eval_model,
            plan,
            df,
            df,
            config,
            week_dates,
            set_features,
        )
        if weekly_comp.empty:
            raise RuntimeError(f"No weekly plan_vs_actual rows for week {week_start.date()}")
        weekly_comp.to_csv(week_dir / "plan_vs_actual_weekly.csv", index=False)
        week_row.update(plan_vs_actual_row_metrics(weekly_comp, config.target))
        week_row["n_segments"] = len(weekly_comp)
        if not daily_comp.empty:
            daily_comp.to_csv(week_dir / "plan_vs_actual_daily.csv", index=False)

        weekly_rows.append(week_row)
        print(f"  [{week_start.date()}] done — week days={len(week_dates)}, segments={len(plan)}")

    summary = pd.DataFrame(weekly_rows)
    summary.to_csv(out_dir / "weekly_backtest_summary.csv", index=False)
    with open(out_dir / "weekly_backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "strategy": "two_stage",
                "start": str(start.date()),
                "end": str(end.date()),
                "n_weeks": len(summary),
                "fixed_keyword_sets": fixed_keyword_sets,
                "mean_rmse_model_lift": float(summary["rmse_pred_vs_actual_model_lift"].mean())
                if "rmse_pred_vs_actual_model_lift" in summary.columns and len(summary)
                else None,
            },
            f,
            indent=2,
        )
    return summary
