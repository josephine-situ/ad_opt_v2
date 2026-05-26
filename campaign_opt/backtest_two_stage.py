"""Two-stage walk-forward backtest: fix keyword sets for period, re-optimize budgets weekly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from campaign_opt.backtest import _cv_rmse_weights, _load_holdout_metrics, optimizer_manifest_for_backtest
from campaign_opt.evaluation import (
    compare_plan_and_actual_week,
    fit_ensemble,
    fit_single_model_evaluation,
    optimizer_winner_name,
    plan_vs_actual_row_metrics,
    save_ensemble,
    save_evaluation_model,
    week_planning_dates,
    week_starts_in_window,
)
from campaign_opt.features import train_before_date
from campaign_opt.modeling import run_tournament
from campaign_opt.optimize import run_optimizer
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
    refit_each_week: bool = True,
    budget_cadence: str = "W-MON",
) -> pd.DataFrame:
    """
    Stage 1: pick keyword set per segment over [start, end] (multi-day linear MILP).
    Stage 2: each week, walk-forward train; re-optimize budgets with fixed keyword sets.

    Stage 1 picks keyword sets with the configured optimizer backend at ``start``
    (same as production). Stage 2 weekly budget re-optimization uses a multi-day
    linear MILP with those sets fixed (tree_embed does not support summed multi-day objectives).
    """
    out_dir = Path(out_dir)
    plans_dir = out_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    set_features = build_keyword_set_feature_table(config.course)
    opt_manifest = optimizer_manifest_for_backtest(config)

    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()

    # --- Stage 1: period keyword-set selection ---
    stage1_dir = out_dir / "stage1_keyword_sets"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    train0 = train_before_date(df, start)
    if len(train0) < 50:
        raise RuntimeError(f"Insufficient train rows before {start.date()} ({len(train0)})")

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
        model_path=stage1_dir / "winner_model.joblib",
        planning_date=start,
        write_outputs=True,
    )
    fixed_keyword_sets = {
        str(row["segment"]): str(row["keyword_set_id"])
        for _, row in set_plan.iterrows()
        if pd.notna(row.get("keyword_set_id"))
    }
    with open(out_dir / "fixed_keyword_sets.json", "w", encoding="utf-8") as f:
        json.dump(fixed_keyword_sets, f, indent=2)
    set_plan.to_csv(stage1_dir / "keyword_set_plan.csv", index=False)

    # --- Stage 2: weekly budget + set scoring ---
    week_starts = week_starts_in_window(start, end, freq=budget_cadence)
    weekly_rows: list[dict[str, Any]] = []
    static_eval_model = None
    static_metrics: dict[str, dict[str, float]] = _load_holdout_metrics(config)
    eval_winner = optimizer_winner_name(config, opt_manifest)

    if not config.evaluation.use_ensemble and eval_winner:
        dmin = pd.to_datetime(df["date"]).min().date()
        dmax = pd.to_datetime(df["date"]).max().date()
        print(
            f"Fitting evaluation model {eval_winner!r} on full panel: "
            f"{len(df)} rows ({dmin} → {dmax})"
        )
        static_eval_model = fit_single_model_evaluation(df, config, opt_manifest, model_name=eval_winner)
        save_evaluation_model(static_eval_model, out_dir / f"evaluation_{eval_winner}.joblib")

    for week_start in week_starts:
        week_start = pd.Timestamp(week_start).normalize()
        week_dates = week_planning_dates(week_start, start, end)
        if not week_dates:
            continue

        train = train_before_date(df, week_start)
        if len(train) < 50:
            print(f"  [{week_start.date()}] skip — insufficient train rows ({len(train)})")
            continue

        week_dir = plans_dir / week_start.strftime("%Y%m%d")
        week_dir.mkdir(parents=True, exist_ok=True)

        holdout = df[df["date"].isin(week_dates)]
        if holdout.empty:
            print(f"  [{week_start.date()}] skip — no panel rows in week")
            continue

        metrics_table: dict[str, dict[str, float]] = static_metrics
        if config.evaluation.use_ensemble and refit_each_week:
            _, metrics_table, _ = run_tournament(train, holdout, config)
            static_metrics = metrics_table

        plan = run_optimizer(
            config,
            opt_manifest,
            train,
            candidates,
            panel,
            total_budget=total_budget,
            output_dir=week_dir,
            model_path=week_dir / "winner_model.joblib",
            planning_dates=week_dates,
            fixed_keyword_sets=fixed_keyword_sets,
            write_outputs=True,
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
            "winner": opt_manifest.get("winner"),
            "backend": opt_manifest.get("backend"),
            "n_segments": len(plan),
            "plan_budget_total": float(plan_budget.sum()),
            "n_segments_zero_budget": int((plan_budget <= 0).sum()),
        }

        eval_model = None
        if config.evaluation.use_ensemble:
            if refit_each_week or static_eval_model is None:
                weights = (
                    _cv_rmse_weights(metrics_table)
                    if config.evaluation.weight_by_cv_rmse and metrics_table
                    else None
                )
                print(f"  [{week_start.date()}] fitting ensemble on {len(train)} rows...")
                eval_model = fit_ensemble(
                    train,
                    config,
                    member_weights=weights,
                    member_hyperparams=opt_manifest.get("best_hyperparams"),
                )
                save_ensemble(eval_model, week_dir / "ensemble_model.joblib")
                static_eval_model = eval_model
            else:
                eval_model = static_eval_model
        elif eval_winner:
            eval_model = static_eval_model

        if eval_model is not None:
            weekly_comp, daily_comp = compare_plan_and_actual_week(
                eval_model,
                plan,
                df,
                df,
                config,
                week_dates,
                set_features,
            )
            if not weekly_comp.empty:
                weekly_comp.to_csv(week_dir / "plan_vs_actual_weekly.csv", index=False)
                week_row.update(plan_vs_actual_row_metrics(weekly_comp, config.target))
                week_row["n_segments"] = len(weekly_comp)
            if not daily_comp.empty:
                daily_comp.to_csv(week_dir / "plan_vs_actual_daily.csv", index=False)
        elif not config.evaluation.use_ensemble:
            print(
                f"  [{week_start.date()}] skip plan_vs_actual — "
                f"evaluation model {eval_winner!r} not configured"
            )

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
