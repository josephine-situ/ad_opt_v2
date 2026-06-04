"""Diagnose keyword efficiency vs lagged segment spend (regime confound check)."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campaign_opt.cv import cross_validate_model
from campaign_opt.efficiency_ablation import _config_with_efficiency_features
from campaign_opt.features import train_holdout_split
from campaign_opt.modeling import FITTERS
from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import get_context_feature_columns
from utils.keyword_efficiency_features import (
    efficiency_column_name,
    merge_keyword_efficiency_features,
)
from utils.lagged_segment_spend_features import (
    LAGGED_WINDOWS,
    add_lagged_segment_spend_features,
    all_lagged_segment_column_names,
    lagged_segment_column_name,
)

# Representative keyword-efficiency columns for correlation / ablation.
_KW_EFF_REP = [
    efficiency_column_name("last", "union", "mean", "cost"),
    efficiency_column_name("r7d", "union", "mean", "cost"),
    efficiency_column_name("last", "union", "mean", "budget"),
    efficiency_column_name("r7d", "union", "mean", "budget"),
]

_LAG_COST = [
    lagged_segment_column_name(w, "cost") for w in LAGGED_WINDOWS
]
_LAG_BUDGET = [
    lagged_segment_column_name(w, "budget") for w in LAGGED_WINDOWS
]
_LAG_SEG_EFF_COST = [
    lagged_segment_column_name(w, "eff_cost") for w in LAGGED_WINDOWS
]
_LAG_SEG_EFF_BUDGET = [
    lagged_segment_column_name(w, "eff_budget") for w in LAGGED_WINDOWS
]

SPEND_REGIME_ABLATION_SPECS: dict[str, list[str]] = {
    "baseline": [],
    # Lagged segment spend level (not keyword efficiency).
    "add_lagged_cost": list(_LAG_COST),
    "add_lagged_budget": list(_LAG_BUDGET),
    "add_lagged_cost_last": [lagged_segment_column_name("last", "cost")],
    "add_lagged_cost_r7d": [lagged_segment_column_name("r7d", "cost")],
    # Lagged segment conv_scaled / cost or / budget (segment efficiency, not kw-set).
    "add_lagged_seg_eff_cost": list(_LAG_SEG_EFF_COST),
    "add_lagged_seg_eff_budget": list(_LAG_SEG_EFF_BUDGET),
    # Keyword-set efficiency (from kw-day panel).
    "add_kw_eff_last_union_mean_cost": [
        efficiency_column_name("last", "union", "mean", "cost"),
    ],
    "add_kw_eff_r7d_union_mean_cost": [
        efficiency_column_name("r7d", "union", "mean", "cost"),
    ],
    "add_kw_eff_cost_full": _KW_EFF_REP
    + [efficiency_column_name(w, "union", "mean", "cost") for w in ("r14d", "r30d")],
}


def _config_with_extra_context(
    config: CampaignOptConfig,
    extra_cols: list[str],
    *,
    group: str = "lagged_spend",
) -> CampaignOptConfig:
    """Add columns to ``group``; keyword efficiency uses existing helper when group set."""
    if group == "keyword_efficiency":
        return _config_with_efficiency_features(config, extra_cols)
    ctx = deepcopy(config.context_features)
    if extra_cols:
        ctx[group] = list(dict.fromkeys(extra_cols))
    else:
        ctx.pop(group, None)
    return replace(config, context_features=ctx)


def _split_feature_groups(extra_cols: list[str]) -> tuple[list[str], list[str]]:
    kw = [c for c in extra_cols if c.startswith("hist_kw_eff_")]
    lag = [c for c in extra_cols if c.startswith("hist_seg_")]
    return kw, lag


def _config_for_spec(config: CampaignOptConfig, extra_cols: list[str]) -> CampaignOptConfig:
    kw, lag = _split_feature_groups(extra_cols)
    cfg = replace(config, context_features=deepcopy(config.context_features))
    ctx = deepcopy(cfg.context_features)
    ctx.pop("keyword_efficiency", None)
    ctx.pop("lagged_spend", None)
    if lag:
        ctx["lagged_spend"] = lag
    if kw:
        ctx["keyword_efficiency"] = kw
    return replace(cfg, context_features=ctx)


def build_spend_regime_frame(
    panel: pd.DataFrame,
    course: str,
) -> pd.DataFrame:
    """Modeling panel with keyword efficiency + lagged segment spend columns."""
    df = merge_keyword_efficiency_features(panel, course)
    if "cost" not in df.columns:
        raise ValueError("panel must include same-day 'cost' to build lagged segment features")
    return add_lagged_segment_spend_features(df)


def correlation_report(
    df: pd.DataFrame,
    *,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise correlations among representative efficiency vs lagged spend columns + target.

    Returns (pairwise long table, full corr submatrix).
    """
    kw_cols = [c for c in _KW_EFF_REP if c in df.columns]
    lag_cols = [c for c in all_lagged_segment_column_names() if c in df.columns]
    use_cols = list(dict.fromkeys(kw_cols + lag_cols + ([target] if target in df.columns else [])))
    sub = df[use_cols].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr()

    rows: list[dict[str, object]] = []
    for a in kw_cols:
        for b in lag_cols:
            if a in corr.columns and b in corr.index:
                rows.append(
                    {
                        "feature_a": a,
                        "feature_b": b,
                        "pearson_r": float(corr.loc[a, b]),
                        "kind": "kw_eff_vs_lagged_seg",
                    }
                )
    if target in corr.columns:
        for col in kw_cols + lag_cols:
            rows.append(
                {
                    "feature_a": col,
                    "feature_b": target,
                    "pearson_r": float(corr.loc[col, target]) if col in corr.index else np.nan,
                    "kind": "feature_vs_target",
                }
            )

    pairwise = pd.DataFrame(rows)
    return pairwise, corr


def run_spend_regime_ablation(
    df: pd.DataFrame,
    config: CampaignOptConfig,
    out_dir: Path,
    *,
    target: str | None = None,
    models: tuple[str, ...] = ("ridge", "xgboost"),
    holdout_days: int | None = None,
    specs: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """CV / holdout: lagged cost/budget vs keyword efficiency vs baseline."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = specs or SPEND_REGIME_ABLATION_SPECS
    target = target or config.target

    holdout_days = (
        holdout_days if holdout_days is not None else config.model_policy.validation.holdout_days
    )
    train, holdout = train_holdout_split(df, holdout_days)
    n_folds = config.model_policy.validation.cv_folds

    rows: list[dict[str, Any]] = []
    for spec_name, extra_cols in specs.items():
        cfg = _config_for_spec(replace(config, target=target), extra_cols)
        feature_cols = get_context_feature_columns(cfg.context_features)
        kw, lag = _split_feature_groups(extra_cols)
        for model_name in models:
            fitter = FITTERS.get(model_name)
            if fitter is None:
                continue
            row: dict[str, Any] = {
                "spec": spec_name,
                "model": model_name,
                "target": target,
                "n_kw_eff": len(kw),
                "n_lagged_spend": len(lag),
                "n_extra": len(extra_cols),
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
    table.to_csv(out_dir / "spend_regime_ablation.csv", index=False)
    report = {
        "title": "Spend regime diagnostic ablation",
        "target": target,
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "results": rows,
    }
    with open(out_dir / "spend_regime_ablation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def run_spend_regime_diagnostics(
    df: pd.DataFrame,
    config: CampaignOptConfig,
    out_dir: Path,
    *,
    target: str | None = None,
    models: tuple[str, ...] = ("ridge", "xgboost"),
    holdout_days: int | None = None,
) -> dict[str, Any]:
    """Write correlation tables + ablation CSV under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = target or config.target

    pairwise, corr = correlation_report(df, target=target)
    pairwise.to_csv(out_dir / "efficiency_vs_lagged_spend_corr.csv", index=False)
    corr.to_csv(out_dir / "feature_corr_matrix.csv")

    ablation = run_spend_regime_ablation(
        df,
        config,
        out_dir,
        target=target,
        models=models,
        holdout_days=holdout_days,
    )

    summary = {
        "target": target,
        "n_rows": len(df),
        "high_corr_pairs": pairwise[
            (pairwise["kind"] == "kw_eff_vs_lagged_seg")
            & (pairwise["pearson_r"].abs() > 0.7)
        ].to_dict(orient="records"),
        "ablation": ablation,
    }
    with open(out_dir / "spend_regime_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def print_spend_regime_summary(summary: dict[str, Any]) -> None:
    ablation = summary.get("ablation", {})
    rows = [r for r in ablation.get("results", []) if r.get("status") == "ok"]
    print(f"\n=== Spend regime diagnostics (target={summary.get('target')}) ===")

    high = summary.get("high_corr_pairs") or []
    if high:
        print("\n  |r| > 0.7 between kw efficiency and lagged segment spend:")
        for p in high[:12]:
            print(f"    {p['feature_a']}  vs  {p['feature_b']}: r={p['pearson_r']:.3f}")
    else:
        print("\n  No kw-eff vs lagged-seg pairs with |r| > 0.7 (see CSV for full table).")

    baseline_by_model: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["spec"] == "baseline":
            baseline_by_model[r["model"]] = r

    print("\n  Ablation (CV RMSE; lagged cost vs kw efficiency):")
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        base = baseline_by_model.get(model)
        print(f"\n    {model}:")
        if base:
            print(f"      baseline: {base['cv_rmse']:.3f}")
        for r in sorted(sub, key=lambda x: x["cv_rmse"]):
            d = ""
            if base:
                d = f"  d={r['cv_rmse'] - base['cv_rmse']:+.3f}"
            print(
                f"      {r['spec']:32s}  cv_rmse={r['cv_rmse']:.3f}{d}  "
                f"lag={r['n_lagged_spend']} kw={r['n_kw_eff']}"
            )
