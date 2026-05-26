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
from campaign_opt.modeling import eval_pipeline_holdout, run_tournament
from campaign_opt.optimize import run_optimizer
from campaign_opt.schema import CampaignOptConfig
from campaign_opt.train_specs import get_train_spec
from utils.campaign_features import build_keyword_set_feature_table, get_context_feature_columns


def load_fit_manifest(config: CampaignOptConfig) -> dict:
    """Alias for :func:`optimizer_manifest_for_backtest`."""
    return optimizer_manifest_for_backtest(config)


def optimizer_manifest_for_backtest(config: CampaignOptConfig) -> dict:
    """
  Manifest for ``run_optimizer`` during backtest.

  Uses ``model_manifest.json`` from ``fit_response_models.py`` when present.
  Otherwise requires ``model_policy.optimizer_winner`` (e.g. ``xgboost``) and
  builds a minimal manifest (default hyperparameters, no tournament winner).
    """
    path = config.exp_dir() / "model_manifest.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    winner = config.model_policy.optimizer_winner
    if not winner:
        raise FileNotFoundError(
            f"Missing {path} and model_policy.optimizer_winner is unset. "
            "Run fit_response_models.py or set optimizer_winner (e.g. xgboost) in campaign_config.json."
        )

    backend = config.model_policy.optimizer_backend
    if backend == "auto":
        spec = get_train_spec(winner)
        backend = spec.backend if spec else "tree_embed"

    print(
        f"[Info] No {path.name}; backtest optimizer uses {winner!r} / {backend!r} "
        f"from config (default hyperparameters)."
    )
    return {
        "winner": winner,
        "backend": backend,
        "target": config.target,
        "best_hyperparams": {},
        "feature_cols": get_context_feature_columns(config.context_features),
    }


def _load_holdout_metrics(config: CampaignOptConfig) -> dict[str, dict[str, float]]:
    path = config.exp_dir() / "holdout_metrics.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
      - Optimize on walk-forward train (date < t) via ``run_optimizer``
      - Score plan vs actual with a full-panel evaluation model (``use_ensemble: false``)
        or a walk-forward ensemble (``use_ensemble: true``)
    """
    out_dir = Path(out_dir)
    plans_dir = out_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    set_features = build_keyword_set_feature_table(config.course)
    opt_manifest = optimizer_manifest_for_backtest(config)

    dates = pd.date_range(start, end, freq="D")
    daily_rows: list[dict[str, Any]] = []
    static_eval_model = None
    static_metrics: dict[str, dict[str, float]] = _load_holdout_metrics(config)
    eval_winner = optimizer_winner_name(config, opt_manifest)
    feature_cols = opt_manifest.get("feature_cols") or get_context_feature_columns(
        config.context_features
    )

    if not config.evaluation.use_ensemble and eval_winner:
        dmin = pd.to_datetime(df["date"]).min().date()
        dmax = pd.to_datetime(df["date"]).max().date()
        print(
            f"Fitting evaluation model {eval_winner!r} on full panel: "
            f"{len(df)} rows ({dmin} → {dmax})"
        )
        static_eval_model = fit_single_model_evaluation(
            df,
            config,
            opt_manifest,
            model_name=eval_winner,
            feature_cols=feature_cols,
        )
        save_evaluation_model(static_eval_model, out_dir / f"evaluation_{eval_winner}.joblib")

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

        metrics_table: dict[str, dict[str, float]] = static_metrics
        if config.evaluation.use_ensemble and refit_each_day:
            _, metrics_table, _ = run_tournament(train, holdout, config)
            static_metrics = metrics_table

        plan = run_optimizer(
            config,
            opt_manifest,
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

        plan_budget = pd.to_numeric(plan["daily_budget"], errors="coerce").fillna(0.0)
        day_row: dict[str, Any] = {
            "opt_date": opt_date.date().isoformat(),
            "winner": opt_manifest.get("winner"),
            "backend": opt_manifest.get("backend"),
            "n_segments": len(plan),
            "plan_budget_total": float(plan_budget.sum()),
            "n_segments_zero_budget": int((plan_budget <= 0).sum()),
        }
        eval_model = None
        if config.evaluation.use_ensemble:
            if refit_each_day or static_eval_model is None:
                weights = (
                    _cv_rmse_weights(metrics_table)
                    if config.evaluation.weight_by_cv_rmse and metrics_table
                    else None
                )
                print(f"  [{opt_date.date()}] fitting ensemble on {len(train)} rows...")
                eval_model = fit_ensemble(
                    train,
                    config,
                    member_weights=weights,
                    member_hyperparams=opt_manifest.get("best_hyperparams"),
                )
                save_ensemble(eval_model, day_dir / "ensemble_model.joblib")
                static_eval_model = eval_model
            else:
                eval_model = static_eval_model
        elif eval_winner:
            eval_model = static_eval_model

        if eval_model is not None:
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
            if not comp.empty:
                comp.to_csv(day_dir / "plan_vs_actual.csv", index=False)
                day_row.update(plan_vs_actual_row_metrics(comp, config.target))
        elif not config.evaluation.use_ensemble:
            print(
                f"  [{opt_date.date()}] skip plan_vs_actual — "
                f"evaluation model {eval_winner!r} not configured"
            )

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
