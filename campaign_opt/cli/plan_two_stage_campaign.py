#!/usr/bin/env python3
"""Production two-stage plan: fix keyword sets for a window, then optimize daily budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from campaign_opt.cli.course_arg import add_course_arg
from campaign_opt.pipeline_inputs import load_planning_inputs, optimizer_manifest_for_backtest
from campaign_opt.schema import default_config_path, load_campaign_config
from campaign_opt.two_stage_plan import optimize_budgets_for_day, select_keyword_sets_for_window
from config import COURSE_CONFIG
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage campaign plan (stage-1 keyword sets + stage-2 budgets)."
    )
    add_course_arg(parser)
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--window-start", required=True, help="Stage-1 window start YYYY-MM-DD")
    parser.add_argument("--window-end", required=True, help="Stage-1 window end YYYY-MM-DD")
    parser.add_argument(
        "--planning-date",
        default="",
        help="Stage-2 planning day YYYY-MM-DD (default: window-start)",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Reuse fixed_keyword_sets.json from output-dir",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Write stage outputs here (default: exp_dir/two_stage_plan)",
    )
    args = parser.parse_args()

    config_path = default_config_path(args.course, args.exp_name) if not args.config else args.config
    config = load_campaign_config(config_path)
    out_dir = Path(args.output_dir) if args.output_dir else config.exp_dir() / "two_stage_plan"
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_tee_logging(log_file=None, default_log_prefix="plan_two_stage")
    print(f"Config: {config_path}")
    print(f"Output: {out_dir}")

    manifest = optimizer_manifest_for_backtest(config)
    df, panel, candidates = load_planning_inputs(config)

    total_budget = args.budget or float(COURSE_CONFIG[config.course].get("campaign_budget", 400.0))
    window_start = pd.Timestamp(args.window_start)
    window_end = pd.Timestamp(args.window_end)
    planning_date = pd.Timestamp(args.planning_date) if args.planning_date else window_start

    stage1_dir = out_dir / "stage1_keyword_sets"
    if args.skip_stage1:
        fixed_path = out_dir / "fixed_keyword_sets.json"
        if not fixed_path.is_file():
            fixed_path = stage1_dir / "fixed_keyword_sets.json"
        if not fixed_path.is_file():
            raise FileNotFoundError(
                f"No fixed_keyword_sets.json under {out_dir}; run stage 1 first"
            )
        with open(fixed_path, encoding="utf-8") as f:
            fixed_keyword_sets = {str(k): str(v) for k, v in json.load(f).items() if v}
        print(f"[stage1] Skipped — loaded {len(fixed_keyword_sets)} sets from {fixed_path}")
    else:
        print(f"[stage1] Keyword-set selection {window_start.date()} → {window_end.date()}")
        fixed_keyword_sets, _ = select_keyword_sets_for_window(
            config,
            manifest,
            df,
            candidates,
            panel,
            window_start=window_start,
            window_end=window_end,
            total_budget=total_budget,
            output_dir=stage1_dir,
        )
        with open(out_dir / "fixed_keyword_sets.json", "w", encoding="utf-8") as f:
            json.dump(fixed_keyword_sets, f, indent=2)
        print(f"[stage1] Done — {len(fixed_keyword_sets)} segments")

    stage2_dir = out_dir / "stage2_budgets" / planning_date.strftime("%Y%m%d")
    print(f"[stage2] Budget optimize for {planning_date.date()}")
    plan = optimize_budgets_for_day(
        config,
        manifest,
        df,
        candidates,
        panel,
        planning_date=planning_date,
        total_budget=total_budget,
        fixed_keyword_sets=fixed_keyword_sets,
        output_dir=stage2_dir,
    )
    print(f"[stage2] Done — {len(plan)} segments, budget=${plan['daily_budget'].sum():.1f}")


if __name__ == "__main__":
    main()
