#!/usr/bin/env python3
"""Solve campaign budget + keyword-set MILP."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    train, _ = train_holdout_split(df, holdout_days)
    panel = add_segment_column(load_campaign_day_panel(config.course))

    cand_path = Path("data") / config.course / "processed" / "segment-keyword-candidates.csv"
    if not cand_path.exists():
        from utils.keyword_candidates import build_segment_candidates

        candidates, extended = build_segment_candidates(config.course)
        cand_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(cand_path, index=False)
        extended.to_csv(
            Path("data") / config.course / "processed" / "campaign-keyword-sets-extended.csv",
            index=False,
        )
    candidates = pd.read_csv(cand_path)

    course_cfg = COURSE_CONFIG.get(config.course, {})
    total_budget = args.budget or float(course_cfg.get("campaign_budget", 400.0))
    planning_date = (
        pd.Timestamp(args.planning_date)
        if args.planning_date
        else pd.Timestamp(train["date"].max())
    )

    plan = run_optimizer(
        config,
        manifest,
        train,
        candidates,
        panel,
        total_budget=total_budget,
        output_dir=out_dir,
        model_path=out_dir / "winner_model.joblib",
        planning_date=planning_date,
    )
    print(f"Optimization complete. Plan rows: {len(plan)}")
    print(plan.to_string(index=False))


if __name__ == "__main__":
    main()
