#!/usr/bin/env python3
"""Lean 7-feat config: union gkp_set vs per-match-type GKP (with optional --tune)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.feature_ablation import RECOMMENDED_CONTEXT, _ctx, run_feature_ablation
from campaign_opt.features import prepare_modeling_data
from campaign_opt.schema import default_config_path, load_campaign_config
from utils.campaign_features import MT_GKP_MEAN_NO_BID, merge_match_type_set_features
from utils.tee_logging import setup_tee_logging

MT_GKP_MEAN = list(MT_GKP_MEAN_NO_BID)

LEAN_GKP_SPECS = {
    "lean_union_gkp": RECOMMENDED_CONTEXT,
    "lean_mt_gkp_replace": _ctx(
        calendar=RECOMMENDED_CONTEXT["calendar"],
        keyword_set_static=RECOMMENDED_CONTEXT["keyword_set_static"],
        match_type_set=MT_GKP_MEAN,
    ),
    "lean_union_and_mt_gkp": _ctx(
        calendar=RECOMMENDED_CONTEXT["calendar"],
        keyword_set_static=RECOMMENDED_CONTEXT["keyword_set_static"],
        gkp_set=RECOMMENDED_CONTEXT["gkp_set"],
        match_type_set=MT_GKP_MEAN,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Lean config: union vs MT GKP ablation.")
    parser.add_argument("--course", default="sys_think")
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--holdout-days", type=int, default=None)
    parser.add_argument("--models", default="xgboost,ridge")
    parser.add_argument("--target", default="")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path(args.course, args.exp_name)
    config = load_campaign_config(config_path)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else config.exp_dir() / "diagnostics" / "feature_ablation" / "mt_gkp_lean"
    )
    if args.tune:
        out_dir = Path(str(out_dir) + "_tuned")
    target = args.target.strip() or config.target
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())

    setup_tee_logging(log_file=None, default_log_prefix=f"lean_gkp_ablation_{config.course}")

    df = prepare_modeling_data(config)
    if target not in df.columns:
        raise SystemExit(f"Target {target!r} not in modeling frame.")
    print("Building per-match-type set features (counts, GKP stats, semantic embeddings)...")
    df = merge_match_type_set_features(df, config.course)

    print(f"Config: {config_path}")
    print(f"Rows: {len(df)}  dates: {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"Target: {target}  models: {models}  tune: {args.tune}")
    print(f"Specs: {list(LEAN_GKP_SPECS)}  output: {out_dir}")

    report = run_feature_ablation(
        df,
        config,
        out_dir,
        target=target,
        models=models,
        holdout_days=args.holdout_days,
        specs=LEAN_GKP_SPECS,
        tune_models=args.tune,
    )
    # Patch summary baseline label for this focused comparison.
    for row in report.get("results", []):
        if row.get("spec") == "lean_union_gkp" and row.get("status") == "ok":
            row["_is_baseline"] = True
    print("\n=== Lean GKP ablation (baseline=lean_union_gkp) ===")
    rows = [r for r in report.get("results", []) if r.get("status") == "ok"]
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        base = next((r for r in sub if r["spec"] == "lean_union_gkp"), None)
        print(f"\n  {model}:")
        if base:
            print(
                f"    lean_union_gkp: cv_rmse={base['cv_rmse']:.3f} "
                f"cv_r2={base['cv_r2']:.3f} n={base['n_features']}"
            )
        for r in sorted(sub, key=lambda x: x["cv_rmse"]):
            d = ""
            if base:
                d = f"  d_cv={r['cv_rmse'] - base['cv_rmse']:+.3f}"
            print(
                f"      {r['spec']:28s}  n={r['n_features']:2d}  "
                f"cv_rmse={r['cv_rmse']:.3f}{d}  cv_r2={r['cv_r2']:.3f}"
            )
    print(f"\nWrote {out_dir / 'feature_ablation.csv'}")


if __name__ == "__main__":
    main()
