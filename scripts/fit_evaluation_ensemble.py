#!/usr/bin/env python3
"""Fit ensemble on all available campaign-day data for evaluation / monitoring."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.evaluation import fit_evaluation_model
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
    manifest_path = out_dir / "model_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}; run fit_response_models.py first.")

    setup_tee_logging(log_file=None, default_log_prefix=f"fit_ensemble_{config.course}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    df = prepare_modeling_data(config)
    if config.target not in df.columns or df[config.target].isna().all():
        config.target = "clicks"

    fit_evaluation_model(config, df, manifest, out_dir)


if __name__ == "__main__":
    main()
