#!/usr/bin/env python3
"""Fit ensemble on all available campaign-day data for evaluation / monitoring."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.evaluation import fit_ensemble, save_ensemble
from campaign_opt.features import prepare_modeling_data
from campaign_opt.schema import default_config_path, load_campaign_config
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit full-data ensemble for incremental evaluation.")
    parser.add_argument("--course", default="sys_think")
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path(args.course, args.exp_name)
    config = load_campaign_config(config_path)
    out_dir = config.exp_dir()

    setup_tee_logging(log_file=None, default_log_prefix=f"fit_ensemble_{config.course}")

    df = prepare_modeling_data(config)
    if config.target not in df.columns or df[config.target].isna().all():
        config.target = "clicks"

    print(f"Fitting ensemble on {len(df)} rows (all data)...")
    ensemble = fit_ensemble(df, config, member_weights=None)
    path = out_dir / "ensemble_model.joblib"
    save_ensemble(ensemble, path)
    meta = {
        "n_members": len(ensemble.members),
        "members": [m.name for m in ensemble.members],
        "target": config.target,
        "baseline_budget": config.evaluation.baseline_budget,
    }
    with open(out_dir / "ensemble_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
