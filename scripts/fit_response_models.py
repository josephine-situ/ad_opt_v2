#!/usr/bin/env python3
"""Model tournament with level-scale holdout selection."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.features import prepare_modeling_data, train_holdout_split
from campaign_opt.modeling import run_tournament, save_manifest
from campaign_opt.schema import default_config_path, load_campaign_config
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit campaign response models.")
    parser.add_argument("--course", default="sys_think")
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path(args.course, args.exp_name)
    config = load_campaign_config(config_path)
    out_dir = config.exp_dir()

    setup_tee_logging(log_file=None, default_log_prefix=f"fit_models_{config.course}")

    df = prepare_modeling_data(config)
    if config.target not in df.columns or df[config.target].isna().all():
        if config.target == "all_conv":
            print("[Warn] all_conv missing; falling back to clicks for tournament.")
            config.target = "clicks"
        else:
            raise ValueError(f"Target {config.target} not available in modeling frame.")

    # Train / holdout split: holdout is for reporting; CV on train picks the winner
    holdout_days = config.model_policy.validation.holdout_days
    train, holdout = train_holdout_split(df, holdout_days)
    print(f"Train rows: {len(train)}, holdout rows: {len(holdout)}")
    print(f"Validation scheme: {config.model_policy.validation.scheme}")

    winner, metrics_table, manifest = run_tournament(train, holdout, config)
    manifest["config_path"] = str(config_path)

    save_manifest(manifest, winner, out_dir / "model_manifest.json")
    with open(out_dir / "holdout_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_table, f, indent=2)

    print(f"Winner: {winner.name} (backend={manifest['backend']})")


if __name__ == "__main__":
    main()
