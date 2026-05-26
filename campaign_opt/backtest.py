"""Daily walk-forward backtest (one optimize per calendar day)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from campaign_opt.evaluation import (
    compare_plan_and_actual,
    fit_ensemble,
    fit_single_model_evaluation,
    optimizer_winner_name,
    plan_vs_actual_row_metrics,
    save_ensemble,
    save_evaluation_model,
)
from campaign_opt.features import train_before_date
from campaign_opt.modeling import (
    _cv_rmse_member_weights,
    base_tournament_candidates,
    eval_pipeline_holdout,
)
from campaign_opt.optimize import require_optimizer_winner, run_optimizer
from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import build_keyword_set_feature_table


def load_fit_manifest(config: CampaignOptConfig) -> dict:
    """Alias for :func:`optimizer_manifest_for_backtest`."""
    return optimizer_manifest_for_backtest(config)


def optimizer_manifest_for_backtest(config: CampaignOptConfig) -> dict:
    """Load ``model_manifest.json``; required for backtest and optimizer metadata."""
    require_optimizer_winner(config)
    path = config.exp_dir() / "model_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run fit_response_models.py before backtest."
        )
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    if not manifest.get("feature_cols"):
        raise ValueError(f"{path} missing feature_cols")
    return manifest


def _load_holdout_metrics(config: CampaignOptConfig) -> dict[str, dict[str, float]]:
    path = config.exp_dir() / "holdout_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run fit_response_models.py before backtest."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _evaluation_ensemble_weights(
    config: CampaignOptConfig,
    metrics_table: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Inverse-CV-RMSE weights over base tournament members only."""
    member_names = base_tournament_candidates(config.model_policy.candidates)
    for name in member_names:
        rmse = (metrics_table.get(name) or {}).get("cv_rmse_levels")
        if rmse is None:
            raise ValueError(
                f"holdout_metrics.json entry {name!r} missing cv_rmse_levels "
                "(required when evaluation.weight_by_cv_rmse is true)"
            )
    return _cv_rmse_member_weights(metrics_table, member_names)


def _fit_evaluation_model(
    config: CampaignOptConfig,
    df: pd.DataFrame,
    opt_manifest: dict,
    out_dir: Path,
) -> Any:
    """Single full-panel model for plan-vs-actual scoring (not walk-forward)."""
    if not config.evaluation.use_ensemble:
        eval_name = optimizer_winner_name(config)
        print(f"Fitting evaluation model {eval_name!r} on full panel: {len(df)} rows")
        model = fit_single_model_evaluation(
            df,
            config,
            opt_manifest,
            model_name=eval_name,
        )
        save_evaluation_model(model, out_dir / f"evaluation_{eval_name}.joblib")
        return model

    static_metrics = _load_holdout_metrics(config)
    weights = (
        _evaluation_ensemble_weights(config, static_metrics)
        if config.evaluation.weight_by_cv_rmse
        else None
    )
    dmin = pd.to_datetime(df["date"]).min().date()
    dmax = pd.to_datetime(df["date"]).max().date()
    print(
        f"Fitting evaluation ensemble on full panel: "
        f"{len(df)} rows ({dmin} → {dmax})"
    )
    ensemble = fit_ensemble(
        df,
        config,
        member_weights=weights,
        member_hyperparams=opt_manifest.get("best_hyperparams"),
    )
    save_ensemble(ensemble, out_dir / "ensemble_model.joblib")
    return ensemble


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
) -> pd.DataFrame:
    """
    For each day t in [start, end]:
      - Fit optimizer on walk-forward train (date < t) with CV hyperparameter search,
        embed in MILP, and optimize budget + keyword set.
      - Score plan vs actual with one evaluation model fit on the full modeling panel.
    """
    out_dir = Path(out_dir)
    plans_dir = out_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    set_features = build_keyword_set_feature_table(config.course)
    opt_manifest = optimizer_manifest_for_backtest(config)
    eval_model = _fit_evaluation_model(config, df, opt_manifest, out_dir)
    feature_cols = opt_manifest["feature_cols"]

    dates = pd.date_range(start, end, freq="D")
    daily_rows: list[dict[str, Any]] = []
    min_train = config.model_policy.validation.min_train_rows

    for opt_date in dates:
        opt_date = pd.Timestamp(opt_date)
        train = train_before_date(df, opt_date)
        if len(train) < min_train:
            raise RuntimeError(
                f"Insufficient train rows before {opt_date.date()}: "
                f"{len(train)} < {min_train}"
            )

        day_dir = plans_dir / opt_date.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        holdout = df[df["date"] == opt_date]
        if holdout.empty:
            raise RuntimeError(f"No modeling-panel rows on {opt_date.date()}")

        plan = run_optimizer(
            config,
            opt_manifest,
            train,
            candidates,
            panel,
            total_budget=total_budget,
            output_dir=day_dir,
            planning_date=opt_date,
            write_outputs=True,
            tune_optimizer=True,
        )
        plan["opt_date"] = opt_date.date().isoformat()
        plan.to_csv(day_dir / "campaign_plan.csv", index=False)

        plan_budget = pd.to_numeric(plan["daily_budget"], errors="coerce").fillna(0.0)
        day_row: dict[str, Any] = {
            "opt_date": opt_date.date().isoformat(),
            "optimizer_winner": require_optimizer_winner(config),
            "backend": _resolve_backtest_backend(config),
            "n_segments": len(plan),
            "plan_budget_total": float(plan_budget.sum()),
            "n_segments_zero_budget": int((plan_budget <= 0).sum()),
        }

        if not config.evaluation.use_ensemble and eval_model.members:
            ho_metrics = eval_pipeline_holdout(
                eval_model.members[0].pipeline, holdout, config, feature_cols
            )
            if ho_metrics:
                day_row.update(ho_metrics)
                print(
                    f"  [{opt_date.date()}] eval holdout R²={ho_metrics['holdout_r2']:.4f} "
                    f"RMSE={ho_metrics['holdout_rmse']:.4f} "
                    f"(n={int(ho_metrics['n_holdout'])})"
                )

        comp = compare_plan_and_actual(
            eval_model,
            plan,
            holdout,
            df,
            config,
            opt_date,
            set_features,
        )
        comp.to_csv(day_dir / "plan_vs_actual.csv", index=False)
        day_row.update(plan_vs_actual_row_metrics(comp, config.target))

        daily_rows.append(day_row)
        budget_note = (
            f"budget=${day_row['plan_budget_total']:.1f} "
            f"({day_row['n_segments_zero_budget']}/{day_row['n_segments']} segments at $0)"
        )
        print(f"  [{opt_date.date()}] done — segments={len(plan)}, {budget_note}")

    summary = pd.DataFrame(daily_rows)
    summary.to_csv(out_dir / "daily_backtest_summary.csv", index=False)
    summary_payload: dict[str, Any] = {
        "start": str(start.date()),
        "end": str(end.date()),
        "n_days": len(summary),
        "mean_rmse_model_lift": float(summary["rmse_pred_vs_actual_model_lift"].mean())
        if "rmse_pred_vs_actual_model_lift" in summary.columns and len(summary)
        else None,
    }
    if "holdout_r2" in summary.columns and len(summary):
        summary_payload["mean_holdout_r2"] = float(summary["holdout_r2"].mean())
        summary_payload["mean_holdout_rmse"] = float(summary["holdout_rmse"].mean())
    if "plan_budget_total" in summary.columns and len(summary):
        summary_payload["mean_plan_budget_total"] = float(summary["plan_budget_total"].mean())
    with open(out_dir / "daily_backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    return summary


def _resolve_backtest_backend(config: CampaignOptConfig) -> str:
    from campaign_opt.optimize import _resolve_backend

    return _resolve_backend(config, {})
