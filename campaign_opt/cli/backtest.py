#!/usr/bin/env python3
"""Two-stage walk-forward campaign backtest over a date range."""

from __future__ import annotations

import argparse

import pandas as pd

from campaign_opt.backtest_analysis import analyze_backtest_run, backtest_window_dir, save_backtest_config
from campaign_opt.backtest_two_stage import run_two_stage_backtest
from campaign_opt.cli.course_arg import add_course_arg
from campaign_opt.pipeline_inputs import load_planning_inputs
from campaign_opt.schema import default_config_path, load_campaign_config
from config import COURSE_CONFIG
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage walk-forward campaign backtest.")
    add_course_arg(parser)
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--start", required=True, help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Backtest end date YYYY-MM-DD")
    parser.add_argument(
        "--day",
        default="",
        help="Run a single day only (sets start=end=day; for Slurm array tasks)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="After backtest, run performance summary (analyze-backtest)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip backtest; only summarize existing outputs in the window dir",
    )
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument(
        "--use-actual-budget",
        action="store_true",
        help="Use configured daily budget caps from the panel for each backtest day",
    )
    parser.add_argument(
        "--optimizer-backend",
        default=None,
        help="Override model_policy.optimizer_backend (e.g. linear when Gurobi license is size-limited)",
    )
    parser.add_argument(
        "--optimizer-winner",
        default=None,
        help="Override model_policy.optimizer_winner for the MILP optimizer",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip keyword-set selection; load fixed_keyword_sets.json from a prior run",
    )
    args = parser.parse_args()

    config_path = default_config_path(args.course, args.exp_name) if not args.config else args.config
    config = load_campaign_config(config_path)
    if args.optimizer_backend:
        config.model_policy.optimizer_backend = args.optimizer_backend
    if args.optimizer_winner:
        config.model_policy.optimizer_winner = args.optimizer_winner
    if args.skip_stage1 and config.backtest.strategy != "two_stage":
        parser.error("--skip-stage1 requires two_stage backtest strategy")

    start = pd.Timestamp(args.day or args.start)
    end = pd.Timestamp(args.day or args.end)
    out_dir = backtest_window_dir(args.course, args.exp_name, args.start, args.end)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_tee_logging(log_file=None, default_log_prefix="backtest_two_stage")
    print(f"Backtest window: {start.date()} → {end.date()} (strategy=two_stage)")

    save_backtest_config(
        out_dir,
        {
            "course": config.course,
            "exp_name": args.exp_name,
            "start_day": args.start,
            "end_day": args.end,
            "strategy": "two_stage",
            "target": config.target,
            "budget_mode": "actual" if args.use_actual_budget else "fixed",
            "optimizer_winner": config.model_policy.optimizer_winner,
            "optimizer_backend": config.model_policy.optimizer_backend,
            "skip_stage1": args.skip_stage1,
            "total_budget": args.budget
            or float(COURSE_CONFIG[config.course].get("campaign_budget", 400.0)),
        },
    )

    if args.analyze_only:
        result = analyze_backtest_run(out_dir, target=config.target)
        print(f"Analysis complete: {result.get('backtest_summary')}")
        return

    df, panel, candidates = load_planning_inputs(config)
    if args.use_actual_budget and args.budget is not None:
        print("[Warn] --budget ignored when --use-actual-budget is set")
    total_budget = args.budget or float(COURSE_CONFIG[config.course].get("campaign_budget", 400.0))

    summary = run_two_stage_backtest(
        config,
        df,
        candidates,
        panel,
        start=start,
        end=end,
        total_budget=total_budget,
        out_dir=out_dir,
        use_actual_budget=args.use_actual_budget,
        skip_stage1=args.skip_stage1,
    )
    print(f"Finished {len(summary)} days. Summary: {out_dir / 'daily_backtest_summary.csv'}")

    if args.analyze:
        result = analyze_backtest_run(out_dir, target=config.target)
        print(f"Analysis: {result.get('backtest_summary')}")


if __name__ == "__main__":
    main()
