#!/usr/bin/env python3
"""Ablation: per-match-type keyword-set features vs baseline context features."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.features import prepare_modeling_data
from campaign_opt.match_type_ablation import (
    MATCH_TYPE_ABLATION_SPECS,
    SEMANTIC_MATCH_TYPE_ABLATION_SPECS,
    print_match_type_ablation_summary,
    run_match_type_ablation,
)
from campaign_opt.schema import default_config_path, load_campaign_config
from utils.campaign_features import merge_match_type_set_features
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Match-type keyword-set feature ablation.")
    parser.add_argument("--course", default="sys_think")
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--out-dir", default="", help="Default: <exp>/diagnostics/match_type_ablation")
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
        help="CV-tune hyperparams per spec (slower; recommended for ridge)",
    )
    parser.add_argument(
        "--spec-set",
        choices=("all", "semantic"),
        default="all",
        help="Ablation grid: all match-type features or semantic-only (default: all)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path(args.course, args.exp_name)
    config = load_campaign_config(config_path)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else config.exp_dir() / "diagnostics" / "match_type_ablation"
    )
    if args.spec_set == "semantic":
        out_dir = out_dir.parent / "semantic_match_type_ablation"
    if args.tune:
        out_dir = Path(str(out_dir) + "_tuned")
    target = args.target.strip() or config.target
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())

    setup_tee_logging(log_file=None, default_log_prefix=f"mt_ablation_{config.course}")

    df = prepare_modeling_data(config)
    if target not in df.columns:
        raise SystemExit(f"Target {target!r} not in modeling frame.")
    specs = (
        SEMANTIC_MATCH_TYPE_ABLATION_SPECS
        if args.spec_set == "semantic"
        else MATCH_TYPE_ABLATION_SPECS
    )
    print("Building per-match-type set features (counts, GKP, semantic embeddings)...")
    df = merge_match_type_set_features(df, config.course)

    print(f"Config: {config_path}")
    print(f"Rows: {len(df)}  dates: {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"Target: {target}  models: {models}  spec_set: {args.spec_set}")
    print(f"Output: {out_dir}")

    report = run_match_type_ablation(
        df,
        config,
        out_dir,
        target=target,
        models=models,
        holdout_days=args.holdout_days,
        tune_models=args.tune,
        specs=specs,
    )
    print_match_type_ablation_summary(report)
    print(f"\nWrote {out_dir / 'match_type_ablation.csv'}")


if __name__ == "__main__":
    main()
