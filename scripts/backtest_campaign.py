#!/usr/bin/env python3
"""
Walk-forward daily backtest over a date range.

For each day t in [--start, --end]:
  - Train on campaign-days with date < t
  - Model tournament with time-series CV
  - Solve budget + keyword-set MILP for day t
  - Compare plan vs actuals on day t

Pattern mirrors ad_opt ``backtest_daily.py``.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.backtest import run_daily_backtest
from campaign_opt.features import prepare_modeling_data
from campaign_opt.schema import default_config_path, load_campaign_config
from config import COURSE_CONFIG
from utils.campaign_features import add_segment_column, load_campaign_day_panel
from utils.keyword_candidates import build_segment_candidates
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily walk-forward campaign backtest.")
    parser.add_argument("--course", default="sys_think")
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--start", required=True, help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Backtest end date YYYY-MM-DD")
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument(
        "--static-model",
        action="store_true",
        help="Fit model once on first day only (faster; more leakage risk)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path(args.course, args.exp_name)
    config = load_campaign_config(config_path)
    out_dir = config.exp_dir() / "backtest" / f"{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_tee_logging(log_file=None, default_log_prefix=f"backtest_daily_{config.course}")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    print(f"Backtest window: {start.date()} → {end.date()}")

    df = prepare_modeling_data(config)
    if config.target not in df.columns or df[config.target].isna().all():
        print(f"[Warn] target={config.target} missing; using clicks")
        config.target = "clicks"

    panel = add_segment_column(load_campaign_day_panel(config.course))
    cand_path = Path("data") / config.course / "processed" / "segment-keyword-candidates.csv"
    if not cand_path.exists():
        candidates, extended = build_segment_candidates(config.course)
        cand_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(cand_path, index=False)
        extended.to_csv(
            Path("data") / config.course / "processed" / "campaign-keyword-sets-extended.csv",
            index=False,
        )
    candidates = pd.read_csv(cand_path)

    total_budget = args.budget or float(COURSE_CONFIG.get(config.course, {}).get("campaign_budget", 400.0))

    summary = run_daily_backtest(
        config,
        df,
        candidates,
        panel,
        start=start,
        end=end,
        total_budget=total_budget,
        out_dir=out_dir,
        refit_each_day=not args.static_model,
    )
    print(f"Finished {len(summary)} days. Summary: {out_dir / 'daily_backtest_summary.csv'}")


if __name__ == "__main__":
    main()
