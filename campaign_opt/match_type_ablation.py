"""Ablation study for per-match-type keyword-set features (counts, GKP, cohesion)."""

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
    MT_COHESION_FEATURE_COLS,
    MT_COUNT_FEATURE_COLS,
    MT_COURSE_SIM_FEATURE_COLS,
    MT_DISPERSION_FEATURE_COLS,
    MT_GKP_FEATURE_COLS,
    MT_SEMANTIC_FULL_FEATURE_COLS,
    MT_SHARE_FEATURE_COLS,
    SEMANTIC_FEATURE_COLS,
    get_context_feature_columns,
)

# Union GKP columns in shipped campaign_config (baseline includes these).
_UNION_GKP_COLS = [
    "last_month_searches_mean",
    "competition_index_mean",
    "bid_low_mean",
]

MATCH_TYPE_ABLATION_SPECS: dict[str, list[str]] = {
    "baseline": [],
    "add_mt_counts": list(MT_COUNT_FEATURE_COLS),
    "add_mt_shares": list(MT_SHARE_FEATURE_COLS),
    "add_mt_counts_and_shares": list(MT_COUNT_FEATURE_COLS) + list(MT_SHARE_FEATURE_COLS),
    "add_per_mt_gkp": list(MT_GKP_FEATURE_COLS),
    "add_mt_counts_and_per_mt_gkp": list(MT_COUNT_FEATURE_COLS) + list(MT_GKP_FEATURE_COLS),
    "add_per_mt_cohesion": list(MT_COHESION_FEATURE_COLS),
    "add_all_mt_set": (
        list(MT_COUNT_FEATURE_COLS)
        + list(MT_SHARE_FEATURE_COLS)
        + list(MT_GKP_FEATURE_COLS)
        + list(MT_SEMANTIC_FULL_FEATURE_COLS)
    ),
    "replace_union_gkp_with_per_mt": list(MT_GKP_FEATURE_COLS),
}

# Embedding features per broad / phrase / exact list (see keyword_set_semantic_per_match_type).
SEMANTIC_MATCH_TYPE_ABLATION_SPECS: dict[str, list[str]] = {
    "baseline": [],
    "add_per_mt_cohesion": list(MT_COHESION_FEATURE_COLS),
    "add_per_mt_dispersion": list(MT_DISPERSION_FEATURE_COLS),
    "add_per_mt_course_sim": list(MT_COURSE_SIM_FEATURE_COLS),
    "add_per_mt_semantic_full": list(MT_SEMANTIC_FULL_FEATURE_COLS),
    "add_per_mt_semantic_on_union": list(MT_SEMANTIC_FULL_FEATURE_COLS),
    "replace_union_semantic_with_per_mt": list(MT_SEMANTIC_FULL_FEATURE_COLS),
}


def _config_with_match_type_features(
    config: CampaignOptConfig,
    extra_cols: list[str],
    *,
    replace_union_gkp: bool = False,
    replace_union_semantic: bool = False,
) -> CampaignOptConfig:
    ctx = deepcopy(config.context_features)
    if replace_union_gkp:
        gkp = [c for c in ctx.get("gkp_set", []) if c not in _UNION_GKP_COLS]
        ctx["gkp_set"] = gkp
    if replace_union_semantic:
        static = [c for c in ctx.get("keyword_set_static", []) if c not in SEMANTIC_FEATURE_COLS]
        ctx["keyword_set_static"] = static
    if extra_cols:
        mt_group = list(dict.fromkeys(extra_cols))
        ctx["match_type_set"] = mt_group
    else:
        ctx.pop("match_type_set", None)
    return replace(config, context_features=ctx)


def run_match_type_ablation(
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
    Side-by-side CV / holdout for each match-type feature spec.

    ``df`` must already include all ablation columns (via ``merge_match_type_set_features``).
    Writes ``match_type_ablation.csv`` and ``match_type_ablation.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = specs or MATCH_TYPE_ABLATION_SPECS
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
        replace_gkp = spec_name == "replace_union_gkp_with_per_mt"
        replace_sem = spec_name == "replace_union_semantic_with_per_mt"
        cfg = _config_with_match_type_features(
            replace(config, target=target),
            extra_cols,
            replace_union_gkp=replace_gkp,
            replace_union_semantic=replace_sem,
        )
        feature_cols = get_context_feature_columns(cfg.context_features)
        for model_name in models:
            fitter = FITTERS.get(model_name)
            if fitter is None:
                continue
            row: dict[str, Any] = {
                "spec": spec_name,
                "model": model_name,
                "target": target,
                "extra_cols": ",".join(extra_cols),
                "n_extra_features": len(extra_cols),
                "replace_union_gkp": replace_gkp,
                "replace_union_semantic": replace_sem,
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
    table.to_csv(out_dir / "match_type_ablation.csv", index=False)
    report = {
        "title": "Semantic match-type ablation"
        if specs is SEMANTIC_MATCH_TYPE_ABLATION_SPECS
        else "Match-type set feature ablation",
        "target": target,
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "cv_folds": n_folds,
        "specs": specs,
        "results": rows,
    }
    with open(out_dir / "match_type_ablation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def print_match_type_ablation_summary(report: dict[str, Any]) -> None:
    rows = [r for r in report.get("results", []) if r.get("status") == "ok"]
    if not rows:
        print("\n=== Match-type set feature ablation ===")
        print("  No successful runs.")
        return

    baseline_by_model: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["spec"] == "baseline":
            baseline_by_model[r["model"]] = r

    title = report.get("title", "Match-type set feature ablation")
    print(f"\n=== {title} (target={report.get('target')}) ===")
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        base = baseline_by_model.get(model)
        best_cv = min(sub, key=lambda r: r["cv_rmse"])
        best_ho = max(sub, key=lambda r: r["holdout_r2"])
        print(f"\n  {model}:")
        if base:
            print(
                f"    baseline: cv_rmse={base['cv_rmse']:.3f} cv_r2={base['cv_r2']:.3f} "
                f"holdout_r2={base['holdout_r2']:.3f}"
            )
        print(
            f"    best CV RMSE: {best_cv['spec']} "
            f"(rmse={best_cv['cv_rmse']:.3f}, r2={best_cv['cv_r2']:.3f})"
        )
        print(
            f"    best holdout R²: {best_ho['spec']} "
            f"(r2={best_ho['holdout_r2']:.3f}, rmse={best_ho['holdout_rmse']:.3f})"
        )
        print("    all specs (cv_rmse, holdout_r2, delta vs baseline):")
        for r in sorted(sub, key=lambda x: x["cv_rmse"]):
            d_rmse = d_r2 = ""
            if base:
                d_rmse = f"  d_cv={r['cv_rmse'] - base['cv_rmse']:+.3f}"
                d_r2 = f"  d_ho_r2={r['holdout_r2'] - base['holdout_r2']:+.3f}"
            print(
                f"      {r['spec']:32s}  cv_rmse={r['cv_rmse']:.3f}{d_rmse}  "
                f"ho_r2={r['holdout_r2']:.3f}{d_r2}"
            )
