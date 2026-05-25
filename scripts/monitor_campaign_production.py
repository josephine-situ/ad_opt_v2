#!/usr/bin/env python3
"""Production monitoring: ensemble incremental lift vs actual decisions."""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.evaluation import compare_plan_and_actual, fit_ensemble, metrics_from_comparison
from campaign_opt.features import prepare_modeling_data, train_before_date
from campaign_opt.schema import default_config_path, load_campaign_config
from utils.campaign_features import add_segment_column, build_keyword_set_feature_table, load_campaign_day_panel
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor campaign plan vs actuals (ensemble lift).")
    parser.add_argument("--course", default="sys_think")
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--base-date", default="")
    parser.add_argument("--plan-file", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path(args.course, args.exp_name)
    config = load_campaign_config(config_path)

    setup_tee_logging(log_file=None, default_log_prefix=f"monitor_campaign_{config.course}")

    eval_date = pd.Timestamp(args.base_date) if args.base_date else pd.Timestamp(
        datetime.now() - timedelta(days=args.lag)
    )
    eval_date = pd.Timestamp(eval_date).normalize()

    plan_path = Path(args.plan_file) if args.plan_file else config.exp_dir() / "campaign_plan.csv"
    if not plan_path.exists():
        print(f"Error: plan not found: {plan_path}")
        sys.exit(1)

    plan = pd.read_csv(plan_path)
    df = prepare_modeling_data(config)
    train = train_before_date(df, eval_date)
    day_df = df[pd.to_datetime(df["date"]).dt.normalize() == eval_date]
    set_features = build_keyword_set_feature_table(config.course)

    ensemble_path = config.exp_dir() / "ensemble_model.joblib"
    if ensemble_path.exists():
        ensemble = joblib.load(ensemble_path)
    else:
        print("No saved ensemble; fitting on all pre-eval data...")
        ensemble = fit_ensemble(train, config)

    comp = compare_plan_and_actual(ensemble, plan, day_df, train, config, eval_date, set_features)
    metrics = metrics_from_comparison(comp, config.target)
    metrics["eval_date"] = str(eval_date.date())

    analysis_dir = Path("opt_results") / "analysis" / config.course
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_path = analysis_dir / f"production_report_{eval_date.date()}.csv"
    comp.to_csv(report_path, index=False)

    with open(analysis_dir / f"production_report_{eval_date.date()}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
