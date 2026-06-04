#!/usr/bin/env python3
"""Ablation: historical keyword efficiency features (ridge + XGB, CV-focused)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.efficiency_ablation import (
    EFFICIENCY_ABLATION_SPECS,
    EFFICIENCY_SUBSET_ABLATION_SPECS,
    print_efficiency_ablation_summary,
    run_efficiency_ablation,
)

_ALL_EFFICIENCY_SPECS = {**EFFICIENCY_SUBSET_ABLATION_SPECS, **EFFICIENCY_ABLATION_SPECS}
from campaign_opt.features import prepare_modeling_data
from campaign_opt.schema import default_config_path, load_campaign_config
from utils.keyword_efficiency_features import merge_keyword_efficiency_features
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Historical keyword efficiency ablation (last / roll 7-14-30d, cost vs budget)."
    )
    parser.add_argument("--course", default="sys_think")
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Default: <exp>/diagnostics/efficiency_ablation",
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
        help="CV-tune hyperparams per spec",
    )
    parser.add_argument(
        "--spec-set",
        choices=("full", "subset", "both"),
        default="subset",
        help="full=original 20 specs; subset=focused singles/vol/combos (~45); both=run subset then full",
    )
    parser.add_argument(
        "--only-spec",
        default="",
        help="Comma-separated spec names only (baseline is always included), e.g. add_r7d_per_mt_mean_cost",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path(args.course, args.exp_name)
    config = load_campaign_config(config_path)
    base_out = (
        Path(args.out_dir)
        if args.out_dir
        else config.exp_dir() / "diagnostics" / "efficiency_ablation"
    )
    if args.tune:
        base_out = Path(str(base_out) + "_tuned")
    target = args.target.strip() or config.target
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())

    setup_tee_logging(log_file=None, default_log_prefix=f"eff_ablation_{config.course}")

    print("Preparing modeling frame...")
    df = prepare_modeling_data(config)
    if target not in df.columns:
        raise SystemExit(f"Target {target!r} not in modeling frame.")

    print("Building historical keyword efficiency features (causal lookback)...")
    df = merge_keyword_efficiency_features(df, config.course)

    only_names = [s.strip() for s in args.only_spec.split(",") if s.strip()]
    runs: list[tuple[str, dict, Path]] = []
    if only_names:
        missing = [n for n in only_names if n not in _ALL_EFFICIENCY_SPECS]
        if missing:
            raise SystemExit(f"Unknown spec(s): {missing}")
        label = only_names[0] if len(only_names) == 1 else "custom"
        specs = {"baseline": []}
        for name in only_names:
            specs[name] = _ALL_EFFICIENCY_SPECS[name]
        out_sub = base_out / label if len(only_names) == 1 else base_out / "custom"
        runs.append((label, specs, out_sub))
    else:
        if args.spec_set in ("subset", "both"):
            runs.append(
                (
                    "subset",
                    EFFICIENCY_SUBSET_ABLATION_SPECS,
                    base_out / "subset",
                )
            )
        if args.spec_set in ("full", "both"):
            runs.append(
                (
                    "full",
                    EFFICIENCY_ABLATION_SPECS,
                    base_out if args.spec_set == "full" else base_out / "full",
                )
            )

    print(f"Config: {config_path}")
    print(f"Rows: {len(df)}  dates: {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"Target: {target}  models: {models}  tune: {args.tune}")
    if only_names:
        print(f"Only specs: {only_names}")
        for name in only_names:
            print(f"  {name}: {_ALL_EFFICIENCY_SPECS[name]}")
    else:
        print(f"spec_set: {args.spec_set}")

    for label, specs, out_dir in runs:
        print(f"\n--- Running {label} ablation ({len(specs)} specs) -> {out_dir} ---")
        report = run_efficiency_ablation(
            df,
            config,
            out_dir,
            target=target,
            models=models,
            holdout_days=args.holdout_days,
            tune_models=args.tune,
            specs=specs,
        )
        report["title"] = f"Keyword efficiency ablation ({label})"
        print_efficiency_ablation_summary(report)
        print(f"Wrote {out_dir / 'efficiency_ablation.csv'}")


if __name__ == "__main__":
    main()
