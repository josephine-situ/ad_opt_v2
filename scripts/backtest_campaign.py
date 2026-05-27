#!/usr/bin/env python3
"""
Walk-forward campaign backtest over a date range.

Default (``--strategy daily``): one full optimize per calendar day.
Two-stage (``--strategy two_stage``): fix keyword sets for the period, re-optimize
budgets weekly (Mon–Sun sum of daily predictions).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.backtest import run_daily_backtest
from campaign_opt.backtest_analysis import analyze_backtest_run, backtest_window_dir, save_backtest_config
from campaign_opt.backtest_two_stage import run_two_stage_backtest
from campaign_opt.decisions import (
    apply_candidate_region_policy,
    parse_allowed_match_types,
    parse_excluded_regions,
)
from campaign_opt.features import prepare_modeling_data
from campaign_opt.schema import default_config_path, load_campaign_config
from config import COURSE_CONFIG
from utils.campaign_features import add_segment_column, load_campaign_day_panel
from utils.tee_logging import setup_tee_logging


def _load_backtest_inputs(config, course: str):
    df = prepare_modeling_data(config)
    if config.target not in df.columns or df[config.target].isna().all():
        print(f"[Warn] target={config.target} missing; using clicks")
        config.target = "clicks"

    panel = add_segment_column(load_campaign_day_panel(config.course))
    allowed_match_types = parse_allowed_match_types(config.constraints)
    excluded_regions = parse_excluded_regions(config.constraints)
    from utils.keyword_candidates import ensure_segment_keyword_candidates

    cand_path = ensure_segment_keyword_candidates(
        config.course,
        allowed_match_types=allowed_match_types,
        excluded_regions=excluded_regions or None,
    )
    candidates = apply_candidate_region_policy(pd.read_csv(cand_path), config.constraints)
    return df, panel, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward campaign backtest.")
    parser.add_argument("--course", default="sys_think")
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
        help="After backtest, run performance summary (analyze_backtest_results)",
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
        "--strategy",
        choices=("daily", "two_stage"),
        default=None,
        help="Backtest mode: daily (default) or two_stage (fixed keyword sets + weekly budgets)",
    )
    parser.add_argument(
        "--keyword-set-horizon",
        default=None,
        help="For two_stage: period (default) — keyword sets chosen over full [start, end]",
    )
    parser.add_argument(
        "--budget-cadence",
        default=None,
        help="For two_stage: pandas offset alias for budget re-optimization (default W-MON)",
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
        help="For two_stage: skip keyword-set selection; load fixed_keyword_sets.json "
        "from the backtest output dir (from a prior run)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path(args.course, args.exp_name)
    config = load_campaign_config(config_path)
    if args.optimizer_backend:
        config.model_policy.optimizer_backend = args.optimizer_backend
    if args.optimizer_winner:
        config.model_policy.optimizer_winner = args.optimizer_winner
    strategy = args.strategy or config.backtest.strategy
    budget_cadence = args.budget_cadence or config.backtest.budget_cadence
    if args.skip_stage1 and strategy != "two_stage":
        parser.error("--skip-stage1 requires --strategy two_stage")

    start = pd.Timestamp(args.day or args.start)
    end = pd.Timestamp(args.day or args.end)
    out_dir = backtest_window_dir(config.course, args.exp_name, args.start, args.end)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_prefix = f"backtest_{strategy}_{config.course}"
    setup_tee_logging(log_file=None, default_log_prefix=log_prefix)

    print(f"Backtest window: {start.date()} → {end.date()} (strategy={strategy})")

    save_backtest_config(
        out_dir,
        {
            "course": config.course,
            "exp_name": args.exp_name,
            "start_day": args.start,
            "end_day": args.end,
            "strategy": strategy,
            "budget_cadence": budget_cadence,
            "target": config.target,
            "budget_mode": "actual" if args.use_actual_budget else "fixed",
            "optimizer_winner": config.model_policy.optimizer_winner,
            "optimizer_backend": config.model_policy.optimizer_backend,
            "skip_stage1": args.skip_stage1,
            "total_budget": args.budget
            or float(COURSE_CONFIG.get(config.course, {}).get("campaign_budget", 400.0)),
        },
    )

    if args.analyze_only:
        result = analyze_backtest_run(out_dir, target=config.target)
        print(f"Analysis complete: {result.get('backtest_summary')}")
        return

    df, panel, candidates = _load_backtest_inputs(config, args.course)
    if args.use_actual_budget and args.budget is not None:
        print("[Warn] --budget ignored when --use-actual-budget is set")
    total_budget = args.budget or float(COURSE_CONFIG.get(config.course, {}).get("campaign_budget", 400.0))

    if strategy == "two_stage":
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
    else:
        summary = run_daily_backtest(
            config,
            df,
            candidates,
            panel,
            start=start,
            end=end,
            total_budget=total_budget,
            out_dir=out_dir,
            use_actual_budget=args.use_actual_budget,
        )
        print(f"Finished {len(summary)} days. Summary: {out_dir / 'daily_backtest_summary.csv'}")

    if args.analyze:
        result = analyze_backtest_run(out_dir, target=config.target)
        print(f"Analysis: {result.get('backtest_summary')}")


if __name__ == "__main__":
    main()
