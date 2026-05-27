#!/usr/bin/env python3
"""Ablation: keyword-count features vs holdout R² in the model tournament."""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.features import prepare_modeling_data, train_holdout_split
from campaign_opt.modeling import run_tournament
from campaign_opt.schema import default_config_path, load_campaign_config

EMBED_COLS = [
    "embed_cohesion",
    "embed_dispersion",
    "embed_course_sim_mean",
    "embed_course_sim_p90",
]
COUNT_COLS = ("num_unique_keywords", "n_positive")


def _keyword_static_variants() -> dict[str, list[str]]:
    base_embed = list(EMBED_COLS)
    return {
        "baseline": base_embed + ["num_unique_keywords"],
        "no_count": list(base_embed),
        "n_positive": base_embed + ["n_positive"],
        "both_counts": base_embed + list(COUNT_COLS),
    }


def _config_for_variant(base_config, static_cols: list[str]):
    ctx = copy.deepcopy(base_config.context_features)
    gkp = list(ctx.get("gkp_set", []))
    ctx["keyword_set_static"] = static_cols
    ctx["gkp_set"] = gkp
    return replace(base_config, context_features=ctx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare keyword-count feature ablations.")
    parser.add_argument("--course", default="sys_think")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument(
        "--candidates",
        default="ridge,power_log,xgboost,ensemble_ridge_xgb",
        help="Comma-separated tournament candidates",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Disable hyperparameter tuning (faster; fixed defaults)",
    )
    args = parser.parse_args()

    config_path = default_config_path(args.course, args.exp_name)
    base = load_campaign_config(config_path)
    if args.no_tune:
        base = replace(
            base,
            model_policy=replace(base.model_policy, validation=replace(
                base.model_policy.validation, tune_hyperparams=False
            )),
        )
    base = replace(
        base,
        model_policy=replace(
            base.model_policy,
            candidates=[c.strip() for c in args.candidates.split(",") if c.strip()],
        ),
    )

    df = prepare_modeling_data(base)
    holdout_days = base.model_policy.validation.holdout_days
    train, holdout = train_holdout_split(df, holdout_days)
    print(f"Train={len(train)} holdout={len(holdout)} target={base.target}\n")

    rows: list[dict[str, object]] = []
    for label, static_cols in _keyword_static_variants().items():
        cfg = _config_for_variant(base, static_cols)
        print(f"=== {label}: keyword_set_static={static_cols}")
        winner, metrics, _ = run_tournament(train, holdout, cfg, export_dir=None)
        row: dict[str, object] = {"variant": label, "winner": winner.name, "winner_r2": winner.holdout_r2}
        for name, m in sorted(metrics.items()):
            r2 = m.get("holdout_r2_levels")
            if r2 is not None:
                row[f"r2_{name}"] = r2
        rows.append(row)
        print(f"  -> winner {winner.name} holdout R²={winner.holdout_r2:.4f}\n")

    print("Summary (holdout R² by variant / model):")
    for row in rows:
        parts = [f"{row['variant']}: winner={row['winner']} R²={row['winner_r2']:.4f}"]
        for key, val in sorted(row.items()):
            if key.startswith("r2_") and key != "r2_":
                parts.append(f"{key[3:]}={val:.4f}")
        print("  " + ", ".join(parts))


if __name__ == "__main__":
    main()
