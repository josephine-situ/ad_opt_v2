#!/usr/bin/env python3
"""Ablation: exponential recency sample weights on shipped deduped baseline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.features import prepare_modeling_data
from campaign_opt.recency_ablation import (
    RECENCY_ABLATION_HALF_LIVES,
    print_recency_ablation_summary,
    run_recency_ablation,
)
from campaign_opt.schema import default_config_path, load_campaign_config
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recency sample-weight ablation (exp decay half-life grid, ridge + XGB)."
    )
    parser.add_argument("--course", default="sys_think")
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Default: <exp>/diagnostics/recency_ablation",
    )
    parser.add_argument("--holdout-days", type=int, default=None)
    parser.add_argument(
        "--models",
        default="ridge,xgboost",
        help="Comma-separated models (default: ridge,xgboost)",
    )
    parser.add_argument(
        "--target",
        default="",
        help="Target column (default: config target)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="CV-tune hyperparams per half-life",
    )
    parser.add_argument(
        "--half-lives",
        default="",
        help="Comma-separated half-lives in days; empty=365,180,90,45 (+ baseline uniform)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path(args.course, args.exp_name)
    config = load_campaign_config(config_path)
    base_out = (
        Path(args.out_dir)
        if args.out_dir
        else config.exp_dir() / "diagnostics" / "recency_ablation"
    )
    if args.tune:
        base_out = Path(str(base_out) + "_tuned")
    target = args.target.strip() or config.target
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())

    if args.half_lives.strip():
        half_lives: tuple[float | None, ...] = (None,)
        for part in args.half_lives.split(","):
            part = part.strip()
            if not part or part.lower() in ("none", "uniform", "baseline"):
                continue
            half_lives = half_lives + (float(part),)
    else:
        half_lives = RECENCY_ABLATION_HALF_LIVES

    setup_tee_logging(log_file=None, default_log_prefix=f"recency_ablation_{config.course}")

    print("Preparing modeling frame (shipped context spec)...")
    df = prepare_modeling_data(config)
    if target not in df.columns:
        raise SystemExit(f"Target {target!r} not in modeling frame.")

    print(f"Config: {config_path}")
    print(f"Rows: {len(df)}  dates: {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"Target: {target}  models: {models}  tune: {args.tune}")
    print(f"Half-lives: {half_lives}")

    report = run_recency_ablation(
        df,
        config,
        base_out,
        target=target,
        models=models,
        holdout_days=args.holdout_days,
        half_lives=half_lives,
        tune_models=args.tune,
    )
    print_recency_ablation_summary(report)
    print(f"Wrote {base_out / 'recency_ablation.csv'}")


if __name__ == "__main__":
    main()
