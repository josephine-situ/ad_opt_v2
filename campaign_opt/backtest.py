"""Daily walk-forward backtest (one optimize per calendar day)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campaign_opt.evaluation import (
    compare_plan_and_actual,
    fit_ensemble,
    metrics_from_comparison,
    save_ensemble,
)
from campaign_opt.features import prepare_modeling_data, train_before_date
from campaign_opt.modeling import run_tournament, save_manifest
from campaign_opt.optimize import run_optimizer
from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import add_segment_column, build_keyword_set_feature_table, load_campaign_day_panel


def _cv_rmse_weights(metrics_table: dict[str, dict[str, float]]) -> dict[str, float]:
    """Inverse-CV-RMSE weights for ensemble members."""
    inv: dict[str, float] = {}
    for name, m in metrics_table.items():
        rmse = m.get("cv_rmse_levels") or m.get("holdout_rmse_levels") or float("inf")
        inv[name] = 1.0 / max(rmse, 1e-9)
    total = sum(inv.values()) or 1.0
    return {k: v / total for k, v in inv.items()}


def run_daily_backtest(
    config: CampaignOptConfig,
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    total_budget: float,
    out_dir: Path,
    refit_each_day: bool = True,
) -> pd.DataFrame:
    """
    For each day t in [start, end]:
      - Train on date < t; tournament picks MILP model
      - Fit ensemble on all train data for evaluation
      - Solve MILP for t
      - Compare plan vs actual via pred_lift = f(plan) - f(0), same ensemble
    """
    out_dir = Path(out_dir)
    plans_dir = out_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    set_features = build_keyword_set_feature_table(config.course)

    dates = pd.date_range(start, end, freq="D")
    daily_rows: list[dict[str, Any]] = []
    static_manifest: dict | None = None
    static_winner = None
    static_ensemble = None

    for opt_date in dates:
        opt_date = pd.Timestamp(opt_date)
        train = train_before_date(df, opt_date)
        if len(train) < 50:
            print(f"  [{opt_date.date()}] skip — insufficient train rows ({len(train)})")
            continue

        day_dir = plans_dir / opt_date.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        holdout = df[df["date"] == opt_date]
        if holdout.empty:
            print(f"  [{opt_date.date()}] skip — no panel rows on this date")
            continue

        metrics_table: dict = {}
        if refit_each_day or static_manifest is None:
            winner, metrics_table, manifest = run_tournament(train, holdout, config)
            save_manifest(manifest, winner, day_dir / "model_manifest.json")
            static_manifest = manifest
            static_winner = winner
        else:
            manifest = static_manifest
            metrics_table = manifest.get("holdout_metrics", {})
            import joblib

            joblib.dump(static_winner.pipeline, day_dir / "winner_model.joblib")

        plan = run_optimizer(
            config,
            manifest,
            train,
            candidates,
            panel,
            total_budget=total_budget,
            output_dir=day_dir,
            model_path=day_dir / "winner_model.joblib",
            planning_date=opt_date,
            write_outputs=True,
        )
        plan["opt_date"] = opt_date.date().isoformat()
        plan.to_csv(day_dir / "campaign_plan.csv", index=False)

        # Ensemble evaluation: f(decision) - f(0), same model for plan and actual
        if config.evaluation.use_ensemble:
            if refit_each_day or static_ensemble is None:
                weights = (
                    _cv_rmse_weights(metrics_table)
                    if config.evaluation.weight_by_cv_rmse and metrics_table
                    else None
                )
                print(f"  [{opt_date.date()}] fitting ensemble on {len(train)} rows...")
                ensemble = fit_ensemble(train, config, member_weights=weights)
                save_ensemble(ensemble, day_dir / "ensemble_model.joblib")
                static_ensemble = ensemble
            else:
                ensemble = static_ensemble

            comp = compare_plan_and_actual(
                ensemble,
                plan,
                holdout,
                train,
                config,
                opt_date,
                set_features,
            )
            comp.to_csv(day_dir / "plan_vs_actual.csv", index=False)
            mets = metrics_from_comparison(comp, config.target)
            daily_rows.append(
                {
                    "opt_date": opt_date.date().isoformat(),
                    "winner": manifest["winner"],
                    "backend": manifest["backend"],
                    "n_segments": len(comp),
                    "pred_lift_total": float(comp["pred_lift"].sum()),
                    "actual_model_lift_total": float(comp["actual_model_lift"].sum()),
                    "observed_total": float(comp[f"observed_{config.target}"].sum())
                    if f"observed_{config.target}" in comp.columns
                    else None,
                    **mets,
                }
            )
        print(f"  [{opt_date.date()}] done — segments={len(plan)}")

    summary = pd.DataFrame(daily_rows)
    summary.to_csv(out_dir / "daily_backtest_summary.csv", index=False)
    with open(out_dir / "daily_backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "start": str(start.date()),
                "end": str(end.date()),
                "n_days": len(summary),
                "mean_rmse_model_lift": float(summary["rmse_pred_vs_actual_model_lift"].mean())
                if "rmse_pred_vs_actual_model_lift" in summary.columns and len(summary)
                else None,
            },
            f,
            indent=2,
        )
    return summary
