"""Ablation for historical keyword efficiency context features."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from campaign_opt.cv import cross_validate_model
from campaign_opt.features import train_holdout_split
from campaign_opt.hyperparam_cv import tune_hyperparams
from campaign_opt.modeling import FITTERS
from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import get_context_feature_columns
from utils.keyword_efficiency_features import (
    EFFICIENCY_DENOMS,
    EFFICIENCY_POOLS,
    EFFICIENCY_STATS,
    EFFICIENCY_WINDOWS,
    all_efficiency_column_names,
    efficiency_column_name,
)


def _eff_cols(
    *,
    windows: tuple[str, ...] | None = None,
    pools: tuple[str, ...] | None = None,
    stats: tuple[str, ...] | None = None,
    denoms: tuple[str, ...] | None = None,
) -> list[str]:
    return all_efficiency_column_names(
        windows=windows or EFFICIENCY_WINDOWS,
        pools=pools or EFFICIENCY_POOLS,
        stats=stats or EFFICIENCY_STATS,
        denoms=denoms or EFFICIENCY_DENOMS,
    )


def _col(
    window: str,
    pool: str = "union",
    stat: str = "mean",
    denom: str = "cost",
) -> str:
    return efficiency_column_name(window, pool, stat, denom)


def _build_subset_ablation_specs() -> dict[str, list[str]]:
    """Focused grid: singles, volatility, and small combinations (union cost-denom)."""
    specs: dict[str, list[str]] = {"baseline": []}
    roll_windows = ("r7d", "r14d", "r30d")

    # --- Singles: time-mean efficiency (union, cost) ---
    for w in EFFICIENCY_WINDOWS:
        specs[f"single_{w}_union_mean_cost"] = [_col(w)]

    # --- Singles: temporal volatility (mean of per-kw time-std, union, cost) ---
    for w in roll_windows:
        specs[f"single_{w}_union_vol_cost"] = [_col(w, stat="vol")]

    # --- Singles: cross-keyword dispersion of time-means ---
    for w in EFFICIENCY_WINDOWS:
        specs[f"single_{w}_union_xkw_std_cost"] = [_col(w, stat="std")]

    # --- Volatility bundles ---
    specs["add_vol_union_cost_all_windows"] = [_col(w, stat="vol") for w in roll_windows]
    specs["add_vol_union_cost_r7d_r14d"] = [_col(w, stat="vol") for w in ("r7d", "r14d")]

    # --- Mean bundles (union cost) ---
    specs["add_mean_union_cost_roll_only"] = [_col(w) for w in roll_windows]
    specs["add_mean_union_cost_last_plus_roll"] = [_col(w) for w in EFFICIENCY_WINDOWS]

    # --- Mean + vol pairs (same window) ---
    for w in roll_windows:
        specs[f"pair_{w}_union_mean_vol_cost"] = [_col(w), _col(w, stat="vol")]

    # --- Best-practice small sets from prior ablation ---
    specs["lean_r7d_union_mean_cost"] = [_col("r7d")]
    specs["lean_r7d_last_union_mean_cost"] = [_col("r7d"), _col("last")]
    specs["lean_r7d_union_mean_vol_cost"] = [_col("r7d"), _col("r7d", stat="vol")]
    specs["lean_r7d_r14d_union_mean_cost"] = [_col("r7d"), _col("r14d")]
    specs["lean_r7d_mean_vol_last_mean_cost"] = [
        _col("r7d"),
        _col("r7d", stat="vol"),
        _col("last"),
    ]

    # --- Per match-type (r7d mean + vol, cost) ---
    for mt in ("broad", "phrase", "exact"):
        specs[f"single_r7d_{mt}_mean_cost"] = [_col("r7d", pool=mt)]
        specs[f"single_r7d_{mt}_vol_cost"] = [_col("r7d", pool=mt, stat="vol")]
    specs["add_r7d_per_mt_mean_cost"] = [_col("r7d", pool=mt) for mt in ("broad", "phrase", "exact")]
    specs["add_r7d_per_mt_mean_vol_cost"] = [
        c
        for mt in ("broad", "phrase", "exact")
        for c in (_col("r7d", pool=mt), _col("r7d", pool=mt, stat="vol"))
    ]

    # --- Stat / denom families (full window, union, cost) ---
    specs["add_xkw_std_union_cost_all_windows"] = [_col(w, stat="std") for w in EFFICIENCY_WINDOWS]
    specs["add_mean_vol_union_cost_all_roll"] = [
        c
        for w in roll_windows
        for c in (_col(w), _col(w, stat="vol"))
    ]

    # --- Forward-style greedy chains (growing from best single) ---
    specs["grow_from_r7d_mean"] = [
        _col("r7d"),
        _col("r7d", stat="vol"),
        _col("last"),
        _col("r14d"),
    ]
    specs["grow_from_r7d_vol"] = [
        _col("r7d", stat="vol"),
        _col("r7d"),
        _col("r14d", stat="vol"),
    ]

    # --- Pairwise means (choose 2 of 3 roll windows) ---
    for pair in combinations(roll_windows, 2):
        key = "pair_mean_" + "_".join(pair) + "_union_cost"
        specs[key] = [_col(w) for w in pair]

    return specs


EFFICIENCY_ABLATION_SPECS: dict[str, list[str]] = {
    "baseline": [],
    # Lookback window (all pools, stats, denoms).
    "add_eff_last": _eff_cols(windows=("last",)),
    "add_eff_r7d": _eff_cols(windows=("r7d",)),
    "add_eff_r14d": _eff_cols(windows=("r14d",)),
    "add_eff_r30d": _eff_cols(windows=("r30d",)),
    # Denominator.
    "add_eff_cost": _eff_cols(denoms=("cost",)),
    "add_eff_budget": _eff_cols(denoms=("budget",)),
    # Aggregation scope.
    "add_eff_union": _eff_cols(pools=("union",)),
    "add_eff_per_mt": _eff_cols(pools=("broad", "phrase", "exact")),
    # Statistic.
    "add_eff_mean_only": _eff_cols(stats=("mean",)),
    "add_eff_std_only": _eff_cols(stats=("std",)),
    "add_eff_vol_only": _eff_cols(stats=("vol",)),
    # Focused bundles (common modeling choices).
    "add_eff_last_union_mean_cost": [_col("last")],
    "add_eff_last_union_mean_budget": [_col("last", denom="budget")],
    "add_eff_r7d_union_mean_cost": [_col("r7d")],
    "add_eff_r7d_union_mean_budget": [_col("r7d", denom="budget")],
    "add_eff_r7d_union_vol_cost": [_col("r7d", stat="vol")],
    "add_eff_last_per_mt_mean_cost": _eff_cols(
        windows=("last",), pools=("broad", "phrase", "exact"), stats=("mean",), denoms=("cost",)
    ),
    "add_eff_roll_windows_union_mean_cost": [_col(w) for w in ("r7d", "r14d", "r30d")],
    "add_eff_r7d_union_mean_vol_cost": [_col("r7d"), _col("r7d", stat="vol")],
    "add_eff_all": _eff_cols(),
}

EFFICIENCY_SUBSET_ABLATION_SPECS: dict[str, list[str]] = _build_subset_ablation_specs()


def _config_with_efficiency_features(
    config: CampaignOptConfig,
    extra_cols: list[str],
) -> CampaignOptConfig:
    ctx = deepcopy(config.context_features)
    if extra_cols:
        ctx["keyword_efficiency"] = list(dict.fromkeys(extra_cols))
    else:
        ctx.pop("keyword_efficiency", None)
    return replace(config, context_features=ctx)


def run_efficiency_ablation(
    df: pd.DataFrame,
    config: CampaignOptConfig,
    out_dir: Path,
    *,
    target: str | None = None,
    models: tuple[str, ...] = ("ridge", "xgboost"),
    holdout_days: int | None = None,
    specs: dict[str, list[str]] | None = None,
    tune_models: bool = False,
) -> dict[str, Any]:
    """
    CV / holdout for each historical keyword-efficiency spec.

    ``df`` must include columns from ``merge_keyword_efficiency_features``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = specs or EFFICIENCY_ABLATION_SPECS
    target = target or config.target
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not in modeling frame")

    holdout_days = (
        holdout_days if holdout_days is not None else config.model_policy.validation.holdout_days
    )
    train, holdout = train_holdout_split(df, holdout_days)
    n_folds = config.model_policy.validation.cv_folds

    rows: list[dict[str, Any]] = []
    for spec_name, extra_cols in specs.items():
        cfg = _config_with_efficiency_features(replace(config, target=target), extra_cols)
        feature_cols = get_context_feature_columns(cfg.context_features)
        for model_name in models:
            fitter = FITTERS.get(model_name)
            if fitter is None:
                continue
            row: dict[str, Any] = {
                "spec": spec_name,
                "model": model_name,
                "target": target,
                "n_efficiency_features": len(extra_cols),
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
    table.to_csv(out_dir / "efficiency_ablation.csv", index=False)
    report = {
        "title": "Keyword efficiency feature ablation",
        "target": target,
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "cv_folds": n_folds,
        "specs": specs,
        "results": rows,
    }
    with open(out_dir / "efficiency_ablation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def rank_efficiency_ablation(
    report: dict[str, Any],
    *,
    model: str = "xgboost",
) -> pd.DataFrame:
    """Rank successful specs by CV RMSE for one model."""
    rows = [r for r in report.get("results", []) if r.get("status") == "ok" and r["model"] == model]
    if not rows:
        return pd.DataFrame()
    base = next((r for r in rows if r["spec"] == "baseline"), None)
    df = pd.DataFrame(rows)
    if base:
        df["d_cv_rmse"] = df["cv_rmse"] - base["cv_rmse"]
        df["d_cv_r2"] = df["cv_r2"] - base["cv_r2"]
    return df.sort_values("cv_rmse")


def print_efficiency_ablation_summary(report: dict[str, Any]) -> None:
    rows = [r for r in report.get("results", []) if r.get("status") == "ok"]
    if not rows:
        print("\n=== Keyword efficiency feature ablation ===")
        print("  No successful runs.")
        return

    baseline_by_model: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["spec"] == "baseline":
            baseline_by_model[r["model"]] = r

    print(f"\n=== {report.get('title')} (target={report.get('target')}) ===")
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        base = baseline_by_model.get(model)
        best_cv = min(sub, key=lambda r: r["cv_rmse"])
        ranked = rank_efficiency_ablation(report, model=model)
        print(f"\n  {model}:")
        if base:
            print(
                f"    baseline: cv_rmse={base['cv_rmse']:.3f} cv_r2={base['cv_r2']:.3f} "
                f"holdout_r2={base['holdout_r2']:.3f}"
            )
        print(
            f"    best CV RMSE: {best_cv['spec']} "
            f"(rmse={best_cv['cv_rmse']:.3f}, r2={best_cv['cv_r2']:.3f}, "
            f"n_eff={best_cv['n_efficiency_features']}, holdout_r2={best_cv['holdout_r2']:.3f})"
        )
        print("    top 12 by CV RMSE:")
        for _, r in ranked.head(12).iterrows():
            d_rmse = ""
            if base and "d_cv_rmse" in r:
                d_rmse = f"  d={r['d_cv_rmse']:+.3f}"
            print(
                f"      {r['spec']:40s}  cv_rmse={r['cv_rmse']:.3f}{d_rmse}  "
                f"ho_r2={r['holdout_r2']:.3f}  n={int(r['n_efficiency_features'])}"
            )
