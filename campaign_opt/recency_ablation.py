"""Ablation: exponential recency sample weights on the shipped baseline."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from campaign_opt.cv import cross_validate_model
from campaign_opt.features import train_holdout_split
from campaign_opt.hyperparam_cv import tune_hyperparams
from campaign_opt.modeling import FITTERS
from campaign_opt.schema import CampaignOptConfig

# Half-lives to sweep (days). ``None`` = uniform weights (production default).
RECENCY_ABLATION_HALF_LIVES: tuple[float | None, ...] = (None, 365.0, 180.0, 90.0, 45.0)


def _spec_name(half_life_days: float | None) -> str:
    if half_life_days is None:
        return "baseline"
    return f"half_life_{int(half_life_days)}"


def _config_with_recency_half_life(
    config: CampaignOptConfig,
    half_life_days: float | None,
) -> CampaignOptConfig:
    val = replace(config.model_policy.validation, recency_half_life_days=half_life_days)
    mp = replace(config.model_policy, validation=val)
    return replace(config, model_policy=mp)


def run_recency_ablation(
    df: pd.DataFrame,
    config: CampaignOptConfig,
    out_dir: Path,
    *,
    target: str | None = None,
    models: tuple[str, ...] = ("ridge", "xgboost"),
    holdout_days: int | None = None,
    half_lives: tuple[float | None, ...] | None = None,
    tune_models: bool = False,
) -> dict[str, Any]:
    """CV / holdout for each recency half-life on the current context feature spec."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    half_lives = half_lives or RECENCY_ABLATION_HALF_LIVES
    target = target or config.target
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not in modeling frame")

    holdout_days = (
        holdout_days if holdout_days is not None else config.model_policy.validation.holdout_days
    )
    train, holdout = train_holdout_split(df, holdout_days)
    n_folds = config.model_policy.validation.cv_folds

    rows: list[dict[str, Any]] = []
    for half_life in half_lives:
        spec_name = _spec_name(half_life)
        cfg = _config_with_recency_half_life(replace(deepcopy(config), target=target), half_life)
        from utils.campaign_features import get_context_feature_columns

        feature_cols = get_context_feature_columns(cfg.context_features)
        for model_name in models:
            fitter = FITTERS.get(model_name)
            if fitter is None:
                continue
            row: dict[str, Any] = {
                "spec": spec_name,
                "recency_half_life_days": half_life,
                "model": model_name,
                "target": target,
                "n_context_features": len(feature_cols),
            }
            try:
                hyperparams: dict[str, object] | None = None
                if tune_models and config.model_policy.validation.tune_hyperparams:
                    hyperparams, cv = tune_hyperparams(
                        model_name, fitter, train, cfg, feature_cols, n_folds=n_folds
                    )
                    row["best_hyperparams"] = hyperparams
                else:
                    cv = cross_validate_model(fitter, train, cfg, feature_cols, n_folds=n_folds)
                row["cv_rmse"] = cv["cv_rmse_levels"]
                row["cv_r2"] = cv["cv_r2_levels"]
                res = fitter(train, holdout, cfg, feature_cols, hyperparams=hyperparams)
                row["holdout_rmse"] = res.holdout_rmse
                row["holdout_r2"] = res.holdout_r2
                row["status"] = "ok"
            except Exception as exc:
                row["status"] = f"error: {exc}"
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "recency_ablation.csv", index=False)
    report = {
        "title": "Recency sample-weight ablation",
        "target": target,
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "cv_folds": n_folds,
        "half_lives": list(half_lives),
        "results": rows,
    }
    with open(out_dir / "recency_ablation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def print_recency_ablation_summary(report: dict[str, Any]) -> None:
    rows = [r for r in report.get("results", []) if r.get("status") == "ok"]
    if not rows:
        print("\n=== Recency sample-weight ablation ===")
        print("No successful runs.")
        return

    print(f"\n=== {report.get('title', 'Recency ablation')} ===")
    print(
        f"Target: {report.get('target')}  train rows: {report.get('train_rows')}  "
        f"holdout rows: {report.get('holdout_rows')}  CV folds: {report.get('cv_folds')}"
    )

    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        base = next((r for r in sub if r["spec"] == "baseline"), sub[0])
        print(f"\n--- {model} (baseline CV RMSE={base['cv_rmse']:.4f}, holdout R²={base['holdout_r2']:.4f}) ---")
        ranked = sorted(sub, key=lambda r: r["holdout_r2"], reverse=True)
        for r in ranked:
            d_cv = r["cv_rmse"] - base["cv_rmse"]
            d_ho = r["holdout_r2"] - base["holdout_r2"]
            hl = r.get("recency_half_life_days")
            hl_label = "uniform" if hl is None else f"{int(hl)}d"
            print(
                f"  {r['spec']:18s}  half_life={hl_label:7s}  "
                f"CV RMSE={r['cv_rmse']:.4f} ({d_cv:+.4f})  "
                f"holdout R²={r['holdout_r2']:.4f} ({d_ho:+.4f})"
            )
            if r.get("best_hyperparams"):
                print(f"    tuned: {r['best_hyperparams']}")
