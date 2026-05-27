#!/usr/bin/env python3
"""Solve campaign budget + keyword-set MILP."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.decisions import (
    apply_candidate_region_policy,
    parse_allowed_match_types,
    parse_excluded_regions,
)
from campaign_opt.features import prepare_modeling_data, train_holdout_split
from campaign_opt.optimize import run_optimizer
from campaign_opt.schema import default_config_path, load_campaign_config
from config import COURSE_CONFIG
from utils.campaign_features import add_segment_column, load_campaign_day_panel
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize campaign budgets and keyword sets.")
    parser.add_argument("--course", default="sys_think")
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--planning-date", default="")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path(args.course, args.exp_name)
    config = load_campaign_config(config_path)
    out_dir = config.exp_dir()

    setup_tee_logging(log_file=None, default_log_prefix=f"optimize_{config.course}")

    manifest_path = out_dir / "model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}; run fit_response_models.py first.")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    df = prepare_modeling_data(config)
    holdout_days = config.model_policy.validation.holdout_days
    train, holdout = train_holdout_split(df, holdout_days)
    production = (
        pd.concat([train, holdout], ignore_index=True).sort_values("date")
        if len(holdout)
        else train
    )
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

    course_cfg = COURSE_CONFIG.get(config.course, {})
    total_budget = args.budget or float(course_cfg.get("campaign_budget", 400.0))
    planning_date = (
        pd.Timestamp(args.planning_date)
        if args.planning_date
        else pd.Timestamp(production["date"].max())
    )

    plan = run_optimizer(
        config,
        manifest,
        production,
        candidates,
        panel,
        total_budget=total_budget,
        output_dir=out_dir,
        planning_date=planning_date,
        tune_optimizer=False,
    )
    print(f"Optimization complete. Plan rows: {len(plan)}")
    print(plan.to_string(index=False))


if __name__ == "__main__":
    main()
