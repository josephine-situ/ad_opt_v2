"""Ablation over all context feature groups (calendar, static, GKP, match-type)."""

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
from utils.campaign_features import (
    CALENDAR_BASELINE_COLS,
    CALENDAR_EXTENDED_COLS,
    GKP_SET_ALL_COLS,
    GKP_SET_MEAN_COLS,
    GKP_SET_P90_COLS,
    GKP_SET_STD_COLS,
    KEYWORD_SET_STATIC_BASELINE_COLS,
    SHIPPED_DEDUPED_CONTEXT,
    UNION_GKP_MEAN_COLS,
    MT_COHESION_FEATURE_COLS,
    MT_COUNT_FEATURE_COLS,
    MT_COURSE_SIM_FEATURE_COLS,
    MT_DISPERSION_FEATURE_COLS,
    MT_GKP_FEATURE_COLS,
    MT_GKP_MEAN_NO_BID,
    MT_SEMANTIC_FULL_FEATURE_COLS,
    MT_SHARE_FEATURE_COLS,
    get_context_feature_columns,
)

# Shipped default context (sys_think/default campaign_config.json).
SHIPPED_CONTEXT: dict[str, list[str]] = {
    "calendar": list(CALENDAR_BASELINE_COLS),
    "keyword_set_static": list(KEYWORD_SET_STATIC_BASELINE_COLS),
    "gkp_set": list(GKP_SET_MEAN_COLS),
    "match_type_set": list(MT_COUNT_FEATURE_COLS) + list(MT_DISPERSION_FEATURE_COLS),
}

MT_BASELINE_COLS = SHIPPED_CONTEXT["match_type_set"]
MT_ALL_SET_COLS = (
    list(MT_COUNT_FEATURE_COLS)
    + list(MT_SHARE_FEATURE_COLS)
    + list(MT_GKP_FEATURE_COLS)
    + list(MT_SEMANTIC_FULL_FEATURE_COLS)
)


def _ctx(
    *,
    calendar: list[str] | None = None,
    keyword_set_static: list[str] | None = None,
    gkp_set: list[str] | None = None,
    match_type_set: list[str] | None = None,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if calendar:
        out["calendar"] = list(calendar)
    if keyword_set_static:
        out["keyword_set_static"] = list(keyword_set_static)
    if gkp_set:
        out["gkp_set"] = list(gkp_set)
    if match_type_set:
        out["match_type_set"] = list(match_type_set)
    return out


def _minus_group(base: dict[str, list[str]], group: str) -> dict[str, list[str]]:
    return {k: v for k, v in base.items() if k != group}


# Current production config (shipped 20 minus correlated features; 10 context cols).
RECOMMENDED_CONTEXT: dict[str, list[str]] = deepcopy(SHIPPED_DEDUPED_CONTEXT)


FEATURE_ABLATION_SPECS: dict[str, dict[str, list[str]]] = {
    "minimal": {},
    "shipped_baseline": deepcopy(SHIPPED_CONTEXT),
    "recommended_config": deepcopy(RECOMMENDED_CONTEXT),
    # Leave-one-out from shipped baseline.
    "drop_calendar": _minus_group(SHIPPED_CONTEXT, "calendar"),
    "drop_keyword_set_static": _minus_group(SHIPPED_CONTEXT, "keyword_set_static"),
    "drop_gkp_set": _minus_group(SHIPPED_CONTEXT, "gkp_set"),
    "drop_match_type_set": _minus_group(SHIPPED_CONTEXT, "match_type_set"),
    # Additive from minimal.
    "add_calendar": _ctx(calendar=CALENDAR_BASELINE_COLS),
    "add_calendar_month_cycle": _ctx(calendar=CALENDAR_EXTENDED_COLS),
    "add_keyword_set_static": _ctx(keyword_set_static=KEYWORD_SET_STATIC_BASELINE_COLS),
    "add_gkp_mean": _ctx(gkp_set=GKP_SET_MEAN_COLS),
    "add_gkp_mean_std": _ctx(gkp_set=GKP_SET_MEAN_COLS + GKP_SET_STD_COLS),
    "add_gkp_all_stats": _ctx(gkp_set=GKP_SET_ALL_COLS),
    "add_mt_counts": _ctx(match_type_set=list(MT_COUNT_FEATURE_COLS)),
    "add_mt_counts_shares": _ctx(
        match_type_set=list(MT_COUNT_FEATURE_COLS) + list(MT_SHARE_FEATURE_COLS)
    ),
    "add_mt_gkp_mean": _ctx(
        match_type_set=[c for c in MT_GKP_FEATURE_COLS if c.endswith("_mean")]
    ),
    "add_mt_gkp_all_stats": _ctx(match_type_set=list(MT_GKP_FEATURE_COLS)),
    "add_mt_semantic_cohesion": _ctx(match_type_set=list(MT_COHESION_FEATURE_COLS)),
    "add_mt_semantic_dispersion": _ctx(match_type_set=list(MT_DISPERSION_FEATURE_COLS)),
    "add_mt_semantic_course_sim": _ctx(match_type_set=list(MT_COURSE_SIM_FEATURE_COLS)),
    "add_mt_semantic_full": _ctx(match_type_set=list(MT_SEMANTIC_FULL_FEATURE_COLS)),
    "add_all_mt_set": _ctx(match_type_set=MT_ALL_SET_COLS),
    # Shipped baseline + extensions.
    "baseline_plus_month_cycle": _ctx(
        calendar=CALENDAR_EXTENDED_COLS,
        keyword_set_static=SHIPPED_CONTEXT["keyword_set_static"],
        gkp_set=SHIPPED_CONTEXT["gkp_set"],
        match_type_set=SHIPPED_CONTEXT["match_type_set"],
    ),
    "baseline_gkp_mean_std": _ctx(
        calendar=SHIPPED_CONTEXT["calendar"],
        keyword_set_static=SHIPPED_CONTEXT["keyword_set_static"],
        gkp_set=GKP_SET_MEAN_COLS + GKP_SET_STD_COLS,
        match_type_set=SHIPPED_CONTEXT["match_type_set"],
    ),
    "baseline_gkp_all_stats": _ctx(
        calendar=SHIPPED_CONTEXT["calendar"],
        keyword_set_static=SHIPPED_CONTEXT["keyword_set_static"],
        gkp_set=GKP_SET_ALL_COLS,
        match_type_set=SHIPPED_CONTEXT["match_type_set"],
    ),
    "baseline_all_mt_set": _ctx(
        calendar=SHIPPED_CONTEXT["calendar"],
        keyword_set_static=SHIPPED_CONTEXT["keyword_set_static"],
        gkp_set=SHIPPED_CONTEXT["gkp_set"],
        match_type_set=MT_ALL_SET_COLS,
    ),
    "full_context": _ctx(
        calendar=CALENDAR_EXTENDED_COLS,
        keyword_set_static=KEYWORD_SET_STATIC_BASELINE_COLS,
        gkp_set=GKP_SET_ALL_COLS,
        match_type_set=MT_ALL_SET_COLS,
    ),
}


def _config_with_context(
    config: CampaignOptConfig,
    context_features: dict[str, list[str]],
) -> CampaignOptConfig:
    return replace(config, context_features=deepcopy(context_features))


def run_feature_ablation(
    df: pd.DataFrame,
    config: CampaignOptConfig,
    out_dir: Path,
    *,
    target: str | None = None,
    models: tuple[str, ...] = ("xgboost",),
    holdout_days: int | None = None,
    specs: dict[str, dict[str, list[str]]] | None = None,
    tune_models: bool = False,
) -> dict[str, Any]:
    """
    Side-by-side CV / holdout for each context-feature spec.

    ``df`` must include all columns referenced by any spec (calendar, set-level GKP, MT).
    Writes ``feature_ablation.csv`` and ``feature_ablation.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = specs or FEATURE_ABLATION_SPECS
    target = target or config.target
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not in modeling frame")

    holdout_days = (
        holdout_days if holdout_days is not None else config.model_policy.validation.holdout_days
    )
    train, holdout = train_holdout_split(df, holdout_days)
    n_folds = config.model_policy.validation.cv_folds

    rows: list[dict[str, Any]] = []
    for spec_name, context in specs.items():
        cfg = _config_with_context(replace(config, target=target), context)
        feature_cols = get_context_feature_columns(cfg.context_features)
        for model_name in models:
            fitter = FITTERS.get(model_name)
            if fitter is None:
                continue
            row: dict[str, Any] = {
                "spec": spec_name,
                "model": model_name,
                "target": target,
                "n_features": len(feature_cols),
                "feature_groups": ",".join(sorted(context.keys())),
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
    table.to_csv(out_dir / "feature_ablation.csv", index=False)
    report = {
        "title": "Context feature ablation",
        "target": target,
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "cv_folds": n_folds,
        "specs": specs,
        "results": rows,
    }
    with open(out_dir / "feature_ablation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def print_feature_ablation_summary(report: dict[str, Any]) -> None:
    rows = [r for r in report.get("results", []) if r.get("status") == "ok"]
    if not rows:
        print("\n=== Context feature ablation ===")
        print("  No successful runs.")
        return

    baseline_by_model: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["spec"] == "shipped_baseline":
            baseline_by_model[r["model"]] = r

    title = report.get("title", "Context feature ablation")
    print(f"\n=== {title} (target={report.get('target')}) ===")
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        base = baseline_by_model.get(model)
        best_cv = min(sub, key=lambda r: r["cv_rmse"])
        best_ho = max(sub, key=lambda r: r["holdout_r2"])
        print(f"\n  {model}:")
        if base:
            print(
                f"    shipped_baseline: cv_rmse={base['cv_rmse']:.3f} cv_r2={base['cv_r2']:.3f} "
                f"holdout_r2={base['holdout_r2']:.3f}"
            )
        print(
            f"    best CV RMSE: {best_cv['spec']} "
            f"(rmse={best_cv['cv_rmse']:.3f}, r2={best_cv['cv_r2']:.3f}, n_feat={best_cv['n_features']})"
        )
        print(
            f"    best holdout R²: {best_ho['spec']} "
            f"(r2={best_ho['holdout_r2']:.3f}, rmse={best_ho['holdout_rmse']:.3f})"
        )
        print("    all specs (cv_rmse, holdout_r2, delta vs shipped_baseline):")
        for r in sorted(sub, key=lambda x: x["cv_rmse"]):
            d_rmse = d_r2 = ""
            if base:
                d_rmse = f"  d_cv={r['cv_rmse'] - base['cv_rmse']:+.3f}"
                d_r2 = f"  d_ho_r2={r['holdout_r2'] - base['holdout_r2']:+.3f}"
            print(
                f"      {r['spec']:32s}  n={r['n_features']:3d}  "
                f"cv_rmse={r['cv_rmse']:.3f}{d_rmse}  "
                f"ho_r2={r['holdout_r2']:.3f}{d_r2}"
            )
