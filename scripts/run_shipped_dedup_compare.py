#!/usr/bin/env python3
"""Compare shipped 20-feat vs deduplicated subset."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.feature_ablation import RECOMMENDED_CONTEXT, SHIPPED_CONTEXT, run_feature_ablation
from campaign_opt.features import prepare_modeling_data
from campaign_opt.schema import default_config_path, load_campaign_config
from utils.campaign_features import SHIPPED_DEDUPED_CONTEXT, merge_match_type_set_features

SPECS = {
    "shipped_baseline": SHIPPED_CONTEXT,
    "shipped_deduped": SHIPPED_DEDUPED_CONTEXT,
    "lean_union_gkp": RECOMMENDED_CONTEXT,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--models", default="xgboost,ridge")
    args = parser.parse_args()

    config = load_campaign_config(default_config_path("sys_think", "default"))
    if args.tune:
        val = replace(config.model_policy.validation, tune_hyperparams=True)
        policy = replace(config.model_policy, validation=val)
        config = replace(config, model_policy=policy)
    out_dir = config.exp_dir() / "diagnostics" / "feature_ablation" / "shipped_deduped"
    if args.tune:
        out_dir = Path(str(out_dir) + "_tuned")
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    df = merge_match_type_set_features(prepare_modeling_data(config), config.course)
    report = run_feature_ablation(
        df, config, out_dir, models=models, specs=SPECS, tune_models=args.tune
    )
    label = "tuned" if args.tune else "untuned"
    print(f"\n=== Shipped vs deduped ({label}) ===")
    for model in sorted({r["model"] for r in report["results"]}):
        print(f"\n  {model}:")
        for row in sorted(
            [r for r in report["results"] if r.get("model") == model and r.get("status") == "ok"],
            key=lambda r: r["spec"],
        ):
            gap = row["cv_r2"] - row["holdout_r2"]
            print(
                f"    {row['spec']:20s} n={row['n_features']:2d}  "
                f"cv_r2={row['cv_r2']:.3f}  ho_r2={row['holdout_r2']:.3f}  gap={gap:+.3f}"
            )
    print(f"\nWrote {out_dir / 'feature_ablation.csv'}")


if __name__ == "__main__":
    main()
