"""Compare calendar feature sets with time-series CV and holdout metrics."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from campaign_opt.cv import cross_validate_model
from campaign_opt.features import train_holdout_split
from campaign_opt.modeling import FITTERS
from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import get_context_feature_columns

# Baseline matches typical campaign config (season, no month dummies).
_BASELINE_CALENDAR = [
    "day_of_week",
    "season",
    "is_weekend",
    "is_public_holiday",
    "days_to_next_course_start",
]

CALENDAR_ABLATION_SPECS: dict[str, list[str]] = {
    "baseline_season": list(_BASELINE_CALENDAR),
    "add_month": [*_BASELINE_CALENDAR, "month"],
    "add_month_cycle": [*_BASELINE_CALENDAR, "month_sin", "month_cos"],
    "month_replace_season": [
        "day_of_week",
        "month",
        "is_weekend",
        "is_public_holiday",
        "days_to_next_course_start",
    ],
    "month_cycle_replace_season": [
        "day_of_week",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_public_holiday",
        "days_to_next_course_start",
    ],
}


def _config_with_calendar(
    config: CampaignOptConfig,
    calendar_cols: list[str],
) -> CampaignOptConfig:
    ctx = deepcopy(config.context_features)
    ctx["calendar"] = calendar_cols
    return replace(config, context_features=ctx)


def run_calendar_ablation(
    df: pd.DataFrame,
    config: CampaignOptConfig,
    out_dir: Path,
    *,
    target: str | None = None,
    models: tuple[str, ...] = ("ridge", "xgboost"),
    holdout_days: int | None = None,
    specs: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """
    Side-by-side CV / holdout for each calendar spec (same target and models).

    Writes ``calendar_ablation.csv`` and ``calendar_ablation.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = specs or CALENDAR_ABLATION_SPECS
    target = target or config.target
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not in modeling frame")

    holdout_days = holdout_days if holdout_days is not None else config.model_policy.validation.holdout_days
    train, holdout = train_holdout_split(df, holdout_days)
    n_folds = config.model_policy.validation.cv_folds

    rows: list[dict[str, Any]] = []
    for spec_name, calendar_cols in specs.items():
        cfg = _config_with_calendar(replace(config, target=target), calendar_cols)
        feature_cols = get_context_feature_columns(cfg.context_features)
        for model_name in models:
            fitter = FITTERS.get(model_name)
            if fitter is None:
                continue
            row: dict[str, Any] = {
                "spec": spec_name,
                "model": model_name,
                "target": target,
                "calendar_cols": ",".join(calendar_cols),
                "n_calendar_features": len(calendar_cols),
            }
            try:
                cv = cross_validate_model(fitter, train, cfg, feature_cols, n_folds=n_folds)
                row["cv_rmse"] = cv["cv_rmse_levels"]
                row["cv_r2"] = cv["cv_r2_levels"]
                res = fitter(train, holdout, cfg, feature_cols)
                row["holdout_rmse"] = res.holdout_rmse
                row["holdout_r2"] = res.holdout_r2
                row["status"] = "ok"
            except Exception as exc:
                row["status"] = f"error: {exc}"
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "calendar_ablation.csv", index=False)
    report = {
        "target": target,
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "cv_folds": n_folds,
        "specs": specs,
        "results": rows,
    }
    with open(out_dir / "calendar_ablation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def print_calendar_ablation_summary(report: dict[str, Any]) -> None:
    rows = [r for r in report.get("results", []) if r.get("status") == "ok"]
    if not rows:
        print("\n=== Calendar ablation ===")
        print("  No successful runs.")
        return

    print(f"\n=== Calendar ablation (target={report.get('target')}) ===")
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        best_cv = min(sub, key=lambda r: r["cv_rmse"])
        best_ho = max(sub, key=lambda r: r["holdout_r2"])
        print(f"\n  {model}:")
        print(
            f"    best CV RMSE: {best_cv['spec']} "
            f"(rmse={best_cv['cv_rmse']:.3f}, r2={best_cv['cv_r2']:.3f})"
        )
        print(
            f"    best holdout R²: {best_ho['spec']} "
            f"(r2={best_ho['holdout_r2']:.3f}, rmse={best_ho['holdout_rmse']:.3f})"
        )
        print("    all specs (CV rmse / holdout r2):")
        for r in sorted(sub, key=lambda x: x["cv_rmse"]):
            print(
                f"      {r['spec']:28s}  cv_rmse={r['cv_rmse']:.3f}  "
                f"cv_r2={r['cv_r2']:.3f}  holdout_r2={r['holdout_r2']:.3f}"
            )
