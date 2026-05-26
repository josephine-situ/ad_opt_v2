#!/usr/bin/env python3
"""Re-score an existing daily backtest with a full-panel ensemble evaluation model."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.backtest import _cv_rmse_weights, _load_holdout_metrics, optimizer_manifest_for_backtest
from campaign_opt.backtest_analysis import analyze_backtest_run, load_backtest_config
from campaign_opt.evaluation import (
    compare_plan_and_actual,
    fit_ensemble,
    plan_vs_actual_row_metrics,
    save_ensemble,
)
from campaign_opt.features import prepare_modeling_data
from campaign_opt.schema import default_config_path, load_campaign_config
from utils.campaign_features import build_keyword_set_feature_table
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score backtest plans with ensemble evaluation.")
    parser.add_argument("--course", default="sys_think")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--config", default="")
    parser.add_argument(
        "--src-backtest-dir",
        required=True,
        help="Existing backtest output (must contain plans/YYYYMMDD/campaign_plan.csv)",
    )
    parser.add_argument(
        "--dst-backtest-dir",
        default="",
        help="Output dir (default: <src>_ensemble)",
    )
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    src = Path(args.src_backtest_dir)
    dst = Path(args.dst_backtest_dir) if args.dst_backtest_dir else Path(str(src) + "_ensemble")
    if not src.exists():
        raise SystemExit(f"Source backtest dir not found: {src}")

    setup_tee_logging(log_file=None, default_log_prefix="rescore_ensemble")

    config_path = Path(args.config) if args.config else default_config_path(args.course, args.exp_name)
    config = load_campaign_config(config_path)
    config.evaluation.use_ensemble = True

    cfg = load_backtest_config(src)
    df = prepare_modeling_data(config)
    if config.target not in df.columns or df[config.target].isna().all():
        config.target = "clicks"

    opt_manifest = optimizer_manifest_for_backtest(config)
    set_features = build_keyword_set_feature_table(config.course)

    dst.mkdir(parents=True, exist_ok=True)
    plans_dst = dst / "plans"
    plans_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "backtest_config.json", dst / "backtest_config.json")
    with open(dst / "rescore_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "src": str(src),
                "evaluation_use_ensemble": True,
                "weight_by_cv_rmse": config.evaluation.weight_by_cv_rmse,
            },
            f,
            indent=2,
        )

    plan_dirs = sorted((src / "plans").glob("20*"))
    if not plan_dirs:
        raise SystemExit(f"No plan day folders under {src / 'plans'}")

    static_metrics = _load_holdout_metrics(config)
    weights = (
        _cv_rmse_weights(static_metrics)
        if config.evaluation.weight_by_cv_rmse
        else None
    )
    dmin = pd.to_datetime(df["date"]).min().date()
    dmax = pd.to_datetime(df["date"]).max().date()
    print(f"Fitting evaluation ensemble on full panel: {len(df)} rows ({dmin} → {dmax})")
    eval_model = fit_ensemble(
        df,
        config,
        member_weights=weights,
        member_hyperparams=opt_manifest.get("best_hyperparams"),
    )
    save_ensemble(eval_model, dst / "ensemble_model.joblib")

    daily_rows: list[dict] = []
    for day_dir in plan_dirs:
        plan_path = day_dir / "campaign_plan.csv"
        if not plan_path.exists():
            print(f"  [{day_dir.name}] skip — no campaign_plan.csv")
            continue

        out_day = plans_dst / day_dir.name
        opt_date = pd.Timestamp(day_dir.name)
        existing_comp = out_day / "plan_vs_actual.csv"
        if existing_comp.exists():
            print(f"  [{day_dir.name}] skip — already rescored")
            plan = pd.read_csv(plan_path)
            comp = pd.read_csv(existing_comp)
            day_row = {
                "opt_date": opt_date.date().isoformat(),
                "n_segments": len(plan),
                "plan_budget_total": float(
                    pd.to_numeric(plan["daily_budget"], errors="coerce").fillna(0).sum()
                ),
            }
            if not comp.empty:
                day_row.update(plan_vs_actual_row_metrics(comp, config.target))
            daily_rows.append(day_row)
            continue

        holdout = df[df["date"] == opt_date]
        if holdout.empty:
            print(f"  [{opt_date.date()}] skip — no holdout rows")
            continue

        plan = pd.read_csv(plan_path)
        out_day.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan_path, out_day / "campaign_plan.csv")
        if (day_dir / "optimizer_xgboost.joblib").exists():
            shutil.copy2(day_dir / "optimizer_xgboost.joblib", out_day / "optimizer_xgboost.joblib")

        comp = compare_plan_and_actual(
            eval_model, plan, holdout, df, config, opt_date, set_features
        )
        day_row = {
            "opt_date": opt_date.date().isoformat(),
            "n_segments": len(plan),
            "plan_budget_total": float(pd.to_numeric(plan["daily_budget"], errors="coerce").fillna(0).sum()),
        }
        if not comp.empty:
            comp.to_csv(out_day / "plan_vs_actual.csv", index=False)
            day_row.update(plan_vs_actual_row_metrics(comp, config.target))
        daily_rows.append(day_row)
        print(f"  [{opt_date.date()}] rescored — plan_budget=${day_row['plan_budget_total']:.1f}")

    summary = pd.DataFrame(daily_rows)
    summary.to_csv(dst / "daily_backtest_summary.csv", index=False)
    print(f"Wrote {len(summary)} days to {dst}")

    if args.analyze:
        result = analyze_backtest_run(dst, target=config.target)
        print(f"Analysis: {result.get('backtest_summary')}")


if __name__ == "__main__":
    main()
