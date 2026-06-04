#!/usr/bin/env python3
"""Compare keyword efficiency features vs lagged segment cost/budget (regime confound)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.features import prepare_modeling_data
from campaign_opt.schema import default_config_path, load_campaign_config
from campaign_opt.spend_regime_diagnostics import (
    build_spend_regime_frame,
    print_spend_regime_summary,
    run_spend_regime_diagnostics,
)
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Correlate hist_kw_eff_* with lagged segment cost/budget; "
            "ablate lagged cost vs keyword efficiency (ridge + XGB, CV)."
        )
    )
    parser.add_argument("--course", default="sys_think")
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Default: <exp>/diagnostics/spend_regime",
    )
    parser.add_argument("--holdout-days", type=int, default=None)
    parser.add_argument("--models", default="ridge,xgboost")
    parser.add_argument("--target", default="", help="Default: config target")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path(args.course, args.exp_name)
    config = load_campaign_config(config_path)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else config.exp_dir() / "diagnostics" / "spend_regime"
    )
    target = args.target.strip() or config.target
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())

    setup_tee_logging(log_file=None, default_log_prefix=f"spend_regime_{config.course}")

    print("Building modeling frame + efficiency + lagged segment spend...")
    panel = prepare_modeling_data(config)
    if target not in panel.columns:
        raise SystemExit(f"Target {target!r} not in panel.")
    df = build_spend_regime_frame(panel, config.course)

    print(f"Config: {config_path}")
    print(f"Rows: {len(df)}  target: {target}  output: {out_dir}")

    summary = run_spend_regime_diagnostics(
        df,
        config,
        out_dir,
        target=target,
        models=models,
        holdout_days=args.holdout_days,
    )
    print_spend_regime_summary(summary)
    print(f"\nWrote {out_dir / 'efficiency_vs_lagged_spend_corr.csv'}")
    print(f"Wrote {out_dir / 'spend_regime_ablation.csv'}")


if __name__ == "__main__":
    main()
