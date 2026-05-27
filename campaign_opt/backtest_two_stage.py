"""Two-stage walk-forward backtest: fix keyword sets for full window, re-optimize budgets daily."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from campaign_opt.backtest import optimizer_manifest_for_backtest
from campaign_opt.decisions import actual_campaign_budget_total, parse_excluded_regions
from campaign_opt.evaluation import (
    compare_plan_and_actual,
    fit_evaluation_model,
    plan_vs_actual_row_metrics,
)
from campaign_opt.features import train_before_date
from campaign_opt.modeling import eval_pipeline_holdout
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
    use_actual_budget: bool = False,
) -> pd.DataFrame:
    """
    Stage 1: Multi-day ridge+xgb MILP over all days in [start, end].
             Picks ONE keyword set per segment for the whole window.
             Budgets can vary per day.  Model trained on data before ``start``.

    Stage 2: Fix keyword sets from Stage 1.  Each day t in [start, end],
             retrain ridge+xgb on data available up to day t, then single-day
             MILP to optimize that day's budget allocation.

    When ``use_actual_budget=True``, each day's budget constraint is the actual
    configured daily budget from the panel (matching the daily backtest behaviour).
    Stage 1 uses the budget for the first day as its per-day cap.
    """
    out_dir = Path(out_dir)
    plans_dir = out_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    set_features = build_keyword_set_feature_table(config.course)
    opt_manifest = optimizer_manifest_for_backtest(config)
    eval_model = fit_evaluation_model(config, df, opt_manifest, out_dir)
    feature_cols = opt_manifest["feature_cols"]

    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    min_train = config.model_policy.validation.min_train_rows
    dates = pd.date_range(start, end, freq="D")
    excluded_regions = parse_excluded_regions(config.constraints)

    # Resolve per-day budget cap (Stage 1 uses the fixed total_budget for its
    # per-day constraint since the multi-day MILP already allows different budgets
    # per day within that cap).
    stage1_budget = total_budget
    if use_actual_budget:
        stage1_budget = actual_campaign_budget_total(
            panel, dates[0], excluded_regions=excluded_regions
        )

    # ─── Stage 1: keyword-set selection over full window ───────────────────
    stage1_dir = out_dir / "stage1_keyword_sets"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    train0 = train_before_date(df, start)
    if len(train0) < min_train:
        raise RuntimeError(
            f"Insufficient train rows before {start.date()}: {len(train0)} < {min_train}"
        )

    print(
        f"  [stage1] Multi-day keyword-set optimize "
        f"(window {start.date()} → {end.date()}, {len(dates)} days)"
    )
    set_plan = run_optimizer(
        config,
        opt_manifest,
        train0,
        candidates,
        panel,
        total_budget=stage1_budget,
        output_dir=stage1_dir,
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
    with open(out_dir / "fixed_keyword_sets.json", "w", encoding="utf-8") as f:
        json.dump(fixed_keyword_sets, f, indent=2)
    set_plan.to_csv(stage1_dir / "keyword_set_plan.csv", index=False)
    print(
        f"  [stage1] Done — {len(fixed_keyword_sets)} segments assigned keyword sets"
    )

    # ─── Stage 2: daily budget optimization (keyword sets fixed) ───────────
    daily_rows: list[dict[str, Any]] = []

    for opt_date in dates:
        opt_date = pd.Timestamp(opt_date)
        train = train_before_date(df, opt_date)
        if len(train) < min_train:
            raise RuntimeError(
                f"Insufficient train rows before {opt_date.date()}: "
                f"{len(train)} < {min_train}"
            )

        day_dir = plans_dir / opt_date.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        holdout = df[df["date"] == opt_date]
        if holdout.empty:
            raise RuntimeError(f"No modeling-panel rows on {opt_date.date()}")

        day_budget = (
            actual_campaign_budget_total(panel, opt_date, excluded_regions=excluded_regions)
            if use_actual_budget
            else total_budget
        )

        plan = run_optimizer(
            config,
            opt_manifest,
            train,
            candidates,
            panel,
            total_budget=day_budget,
            output_dir=day_dir,
            planning_date=opt_date,
            fixed_keyword_sets=fixed_keyword_sets,
            write_outputs=True,
            tune_optimizer=False,
        )
        plan["opt_date"] = opt_date.date().isoformat()
        plan.to_csv(day_dir / "campaign_plan.csv", index=False)

        plan_budget = pd.to_numeric(plan["daily_budget"], errors="coerce").fillna(0.0)
        day_row: dict[str, Any] = {
            "opt_date": opt_date.date().isoformat(),
            "optimizer_winner": require_optimizer_winner(config),
            "n_segments": len(plan),
            "total_budget": day_budget,
            "plan_budget_total": float(plan_budget.sum()),
            "n_segments_zero_budget": int((plan_budget <= 0).sum()),
        }

        if not config.evaluation.use_ensemble and eval_model.members:
            ho_metrics = eval_pipeline_holdout(
                eval_model.members[0].pipeline, holdout, config, feature_cols
            )
            if ho_metrics:
                day_row.update(ho_metrics)

        comp = compare_plan_and_actual(
            eval_model,
            plan,
            holdout,
            df,
            config,
            opt_date,
            set_features,
            scoring_panel=train,
        )
        comp.to_csv(day_dir / "plan_vs_actual.csv", index=False)
        day_row.update(plan_vs_actual_row_metrics(comp, config.target))

        daily_rows.append(day_row)
        budget_note = (
            f"budget=${day_row['plan_budget_total']:.1f} "
            f"(cap=${day_budget:.1f}, "
            f"{day_row['n_segments_zero_budget']}/{day_row['n_segments']} segments at $0)"
        )
        print(f"  [stage2 {opt_date.date()}] done — segments={len(plan)}, {budget_note}")

    summary = pd.DataFrame(daily_rows)
    summary.to_csv(out_dir / "daily_backtest_summary.csv", index=False)
    summary_payload: dict[str, Any] = {
        "strategy": "two_stage",
        "start": str(start.date()),
        "end": str(end.date()),
        "n_days": len(summary),
        "fixed_keyword_sets": fixed_keyword_sets,
        "budget_mode": "actual" if use_actual_budget else "fixed",
        "mean_rmse_model_lift": float(summary["rmse_pred_vs_actual_model_lift"].mean())
        if "rmse_pred_vs_actual_model_lift" in summary.columns and len(summary)
        else None,
    }
    if "holdout_r2" in summary.columns and len(summary):
        summary_payload["mean_holdout_r2"] = float(summary["holdout_r2"].mean())
        summary_payload["mean_holdout_rmse"] = float(summary["holdout_rmse"].mean())
    if "plan_budget_total" in summary.columns and len(summary):
        summary_payload["mean_plan_budget_total"] = float(summary["plan_budget_total"].mean())
    with open(out_dir / "daily_backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    return summary
