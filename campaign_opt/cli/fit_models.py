#!/usr/bin/env python3
"""Model tournament with level-scale holdout selection."""

from __future__ import annotations

import argparse
import json

from campaign_opt.evaluation import fit_evaluation_model
from campaign_opt.feature_artifacts import save_modeling_artifacts
from campaign_opt.features import prepare_modeling_data, train_holdout_split
from campaign_opt.modeling import (
    configured_evaluation_model_name,
    model_feature_overview_lines,
    print_tournament_metric_summary,
    run_tournament,
    save_manifest,
    warn_if_not_tournament_winner,
)
from campaign_opt.optimize import require_optimizer_winner
from campaign_opt.schema import default_config_path, load_campaign_config
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit campaign response models.")
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument(
        "--skip-evaluation-ensemble",
        action="store_true",
        help="Do not fit/save evaluation ensemble after tournament (default: fit when configured)",
    )
    args = parser.parse_args()

    config_path = default_config_path(args.exp_name) if not args.config else args.config
    config = load_campaign_config(config_path)
    out_dir = config.exp_dir()

    setup_tee_logging(log_file=None, default_log_prefix="fit_models")

    df = prepare_modeling_data(config)
    print(f"Modeling panel: {len(df)} rows, {df['segment'].nunique()} segments")
    if config.modeling_lookback_days:
        print(
            f"Modeling lookback: last {config.modeling_lookback_days} days "
            f"({df['date'].min().date()} to {df['date'].max().date()}, "
            f"{df['date'].nunique()} days, {len(df)} rows)"
        )
    if config.target not in df.columns or df[config.target].isna().all():
        if config.target == "all_conv":
            print("[Warn] all_conv missing; falling back to clicks for tournament.")
            config.target = "clicks"
        else:
            raise ValueError(f"Target {config.target} not available in modeling frame.")

    holdout_days = config.model_policy.validation.holdout_days
    train, holdout = train_holdout_split(df, holdout_days)
    print(f"Train rows: {len(train)}, holdout rows: {len(holdout)}")
    print(f"Validation scheme: {config.model_policy.validation.scheme}")

    artifact_paths = save_modeling_artifacts(out_dir, config, train, holdout)
    print(f"Saved feature artifacts under {out_dir / 'features'}")

    winner, metrics_table, manifest = run_tournament(train, holdout, config, export_dir=out_dir)
    manifest["config_path"] = str(config_path)
    manifest["feature_artifacts"] = artifact_paths

    save_manifest(manifest, winner, out_dir / "model_manifest.json")
    with open(out_dir / "holdout_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_table, f, indent=2)

    print(
        f"Winner: {winner.name} (backend={manifest['backend']}, "
        f"holdout RMSE={winner.holdout_rmse:.4f}, holdout R^2={winner.holdout_r2:.4f})"
    )
    if winner.cv_r2 is not None:
        print(f"  Winner CV R^2={winner.cv_r2:.4f} (CV RMSE={winner.cv_rmse:.4f})")
    print_tournament_metric_summary(metrics_table, winner_name=winner.name)

    warn_if_not_tournament_winner(require_optimizer_winner(config), manifest, role="Optimizer")
    warn_if_not_tournament_winner(
        configured_evaluation_model_name(config), manifest, role="Evaluation"
    )

    for line in model_feature_overview_lines(
        winner, shap_effects=manifest.get("shap_mean_effects")
    ):
        print(line)

    if not args.skip_evaluation_ensemble:
        print("\n--- Evaluation model (full panel) ---")
        fit_evaluation_model(config, df, manifest, out_dir)
    else:
        print("\nSkipped evaluation ensemble (--skip-evaluation-ensemble).")


if __name__ == "__main__":
    main()
