#!/usr/bin/env python3
"""Diagnose daily_budget cap vs outcomes; compare all_conv vs clicks models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.budget_diagnostics import print_budget_diagnostics_summary, run_budget_diagnostics
from campaign_opt.calendar_ablation import print_calendar_ablation_summary, run_calendar_ablation
from campaign_opt.features import prepare_modeling_data
from campaign_opt.schema import default_config_path, load_campaign_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Budget-cap response diagnostics and all_conv vs clicks model comparison. "
            "Uses daily_budget only (not cost)."
        )
    )
    parser.add_argument("--course", default="sys_think")
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument(
        "--targets",
        default="all_conv,clicks",
        help="Comma-separated targets to compare (default: all_conv,clicks)",
    )
    parser.add_argument("--out-dir", default="", help="Output directory (default: <exp>/diagnostics/budget)")
    parser.add_argument("--holdout-days", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--models",
        default="ridge,xgboost",
        help="Comma-separated models to compare per target",
    )
    parser.add_argument(
        "--calendar-ablation",
        action="store_true",
        help="Run calendar feature ablation (season vs month vs month sin/cos)",
    )
    parser.add_argument(
        "--ablation-target",
        default="",
        help="Target for calendar ablation (default: config target)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path(args.course, args.exp_name)
    config = load_campaign_config(config_path)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else config.exp_dir() / "diagnostics" / "budget"
    )

    df = prepare_modeling_data(config)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    targets = [t for t in targets if t in df.columns]
    if not targets:
        raise SystemExit("No requested targets found in modeling frame.")

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())

    print(f"Config: {config_path}")
    print(f"Rows: {len(df)}  date range: {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"Targets: {targets}")
    print(f"Writing diagnostics to {out_dir}")

    report = run_budget_diagnostics(
        df,
        config,
        out_dir,
        targets=targets,
        holdout_days=args.holdout_days,
        write_plots=not args.no_plots,
        models=models,
    )
    print_budget_diagnostics_summary(report)
    print(f"\nWrote {out_dir / 'budget_diagnostics.json'}")
    print(f"Wrote {out_dir / 'bivariate_slopes.csv'}")
    print(f"Wrote {out_dir / 'within_set_budget_slopes.csv'}")
    if report.get("plots"):
        print("Plots:")
        for p in report["plots"]:
            print(f"  {p}")

    if args.calendar_ablation:
        ablation_target = args.ablation_target.strip() or config.target
        ablation_dir = out_dir / "calendar_ablation"
        print(f"\nRunning calendar ablation (target={ablation_target}) -> {ablation_dir}")
        ablation = run_calendar_ablation(
            df,
            config,
            ablation_dir,
            target=ablation_target,
            models=models,
            holdout_days=args.holdout_days,
        )
        print_calendar_ablation_summary(ablation)
        print(f"Wrote {ablation_dir / 'calendar_ablation.csv'}")


if __name__ == "__main__":
    main()
