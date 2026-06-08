#!/usr/bin/env python3
"""Summarize campaign backtest performance from plan_vs_actual outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from campaign_opt.backtest_analysis import analyze_backtest_run, backtest_window_dir
from campaign_opt.schema import default_config_path, load_campaign_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize campaign backtest results.")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--start", default="", help="Backtest window start YYYY-MM-DD")
    parser.add_argument("--end", default="", help="Backtest window end YYYY-MM-DD")
    parser.add_argument(
        "--backtest-dir",
        default="",
        help="Explicit backtest output directory (overrides exp/start/end)",
    )
    parser.add_argument("--no-latex", action="store_true")
    args = parser.parse_args()

    if args.backtest_dir:
        backtest_dir = Path(args.backtest_dir)
        target = "clicks"
    else:
        if not args.start or not args.end:
            parser.error("Provide --start and --end, or --backtest-dir")
        config = load_campaign_config(default_config_path(args.exp_name))
        target = config.target
        backtest_dir = backtest_window_dir(args.exp_name, args.start, args.end)

    if not backtest_dir.exists():
        print(f"Backtest directory not found: {backtest_dir}")
        sys.exit(1)

    result = analyze_backtest_run(backtest_dir, target=target, write_latex=not args.no_latex)

    print(f"Backtest dir: {result['backtest_dir']}")
    print(f"Days evaluated: {result['n_days_evaluated']}")
    if result["missing_days"]:
        print(f"Missing days ({len(result['missing_days'])}): {', '.join(result['missing_days'])}")
    else:
        print("Missing days: none")
    for key in ("evaluation_results", "backtest_summary", "regional_breakdown", "backtest_summary_tex"):
        if result.get(key):
            print(f"  {key}: {result[key]}")
    if result.get("latex"):
        print("\n--- LaTeX preview ---\n")
        print(result["latex"])


if __name__ == "__main__":
    main()
