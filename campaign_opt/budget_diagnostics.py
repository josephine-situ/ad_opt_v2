"""Budget-cap diagnostics and target comparison (``daily_budget`` only; not ``cost``)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campaign_opt.budget_identification import (
    fit_cell_fe_budget_ridge,
    identifiability_report,
    pooled_within_cell_budget_slopes,
)
from campaign_opt.features import train_holdout_split
from campaign_opt.coefficients import coeffs_from_linear_milp_design
from campaign_opt.linear_design import build_linear_milp_design_matrix
from campaign_opt.modeling import (
    FITTERS,
    model_feature_overview_lines,
)
from campaign_opt.schema import CampaignOptConfig
from campaign_opt.shap_effects import compute_mean_shap_effects, shap_available
from utils.campaign_features import get_context_feature_columns


def _ols_slope(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float | None:
    xv = pd.to_numeric(pd.Series(x), errors="coerce")
    yv = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = xv.notna() & yv.notna()
    xv = xv[mask].astype(float)
    yv = yv[mask].astype(float)
    if len(xv) < 2 or xv.std(ddof=0) == 0:
        return None
    return float(np.cov(xv, yv, ddof=0)[0, 1] / np.var(xv))


def bivariate_budget_slopes(
    df: pd.DataFrame,
    targets: list[str],
) -> pd.DataFrame:
    """Raw OLS slope of target ~ daily_budget by segment (no controls)."""
    rows: list[dict[str, Any]] = []
    for target in targets:
        if target not in df.columns or "daily_budget" not in df.columns:
            continue
        slope = _ols_slope(df["daily_budget"], df[target])
        rows.append(
            {
                "scope": "all",
                "segment": "",
                "target": target,
                "slope_budget": slope,
                "n": int(df[["daily_budget", target]].dropna().shape[0]),
            }
        )
        if "segment" not in df.columns:
            continue
        for segment, grp in df.groupby("segment", sort=False):
            rows.append(
                {
                    "scope": "segment",
                    "segment": str(segment),
                    "target": target,
                    "slope_budget": _ols_slope(grp["daily_budget"], grp[target]),
                    "n": int(grp[["daily_budget", target]].dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows)


def within_keyword_set_budget_slopes(
    df: pd.DataFrame,
    targets: list[str],
) -> pd.DataFrame:
    """
    OLS slope of target ~ daily_budget within each (segment, keyword_set_id) cell.

    Only cells with >=2 distinct budget levels are identifiable.
    """
    if "keyword_set_id" not in df.columns or "segment" not in df.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    grouped = df.groupby(["segment", "keyword_set_id"], sort=False)
    for (segment, set_id), grp in grouped:
        n_budget = int(grp["daily_budget"].nunique()) if "daily_budget" in grp.columns else 0
        base = {
            "segment": str(segment),
            "keyword_set_id": str(set_id),
            "n_rows": int(len(grp)),
            "n_budget_levels": n_budget,
            "identifiable": n_budget >= 2,
        }
        for target in targets:
            if target not in grp.columns:
                continue
            row = dict(base)
            row["target"] = target
            row["slope_budget"] = (
                _ols_slope(grp["daily_budget"], grp[target]) if n_budget >= 2 else None
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _unshrunk_ridge_budget_slopes(
    train: pd.DataFrame,
    config: CampaignOptConfig,
) -> dict[str, Any]:
    design = build_linear_milp_design_matrix(train, config)
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=1.0)
    model.fit(design.X.values, design.y)
    coeffs = coeffs_from_linear_milp_design(
        model,
        design,
        config,
        shrink_weight=0.0,
    )
    return {
        "global_budget_coef": float(model.coef_[design.x_columns.index("daily_budget")]),
        "segment_budget_slope": coeffs["segment_budget_slope"],
    }


def _try_fit_model(name: str, train: pd.DataFrame, holdout: pd.DataFrame, config: CampaignOptConfig):
    feature_cols = get_context_feature_columns(config.context_features)
    fitter = FITTERS.get(name)
    if fitter is None:
        return None
    try:
        return fitter(train, holdout, config, feature_cols)
    except Exception:
        return None


def compare_model_targets(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    config: CampaignOptConfig,
    targets: list[str],
    *,
    models: tuple[str, ...] = ("ridge", "xgboost"),
) -> dict[str, Any]:
    """Fit selected models per target; return holdout metrics and budget effects."""
    feature_cols = get_context_feature_columns(config.context_features)
    out: dict[str, Any] = {"targets": {}, "shap_available": shap_available()}

    for target in targets:
        if target not in train.columns:
            out["targets"][target] = {"error": f"column {target!r} missing"}
            continue

        cfg = replace(config, target=target)
        target_report: dict[str, Any] = {"models": {}}

        ridge_slopes = _unshrunk_ridge_budget_slopes(train, cfg)
        target_report["ridge_unshrunk"] = ridge_slopes

        for name in models:
            result = _try_fit_model(name, train, holdout, cfg)
            if result is None:
                target_report["models"][name] = {"status": "skipped"}
                continue

            shap_effects = compute_mean_shap_effects(
                result.pipeline, train, target, feature_cols
            )
            budget_shap = None
            if shap_effects:
                for key, val in shap_effects.items():
                    if key == "daily_budget" or key.endswith("daily_budget"):
                        budget_shap = float(val)
                        break

            target_report["models"][name] = {
                "holdout_r2": result.holdout_r2,
                "holdout_rmse": result.holdout_rmse,
                "budget_shap_mean": budget_shap,
                "overview": model_feature_overview_lines(
                    result, shap_effects=shap_effects
                ),
            }

        out["targets"][target] = target_report

    return out


def _maybe_plot(df: pd.DataFrame, out_dir: Path, targets: list[str]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = df.dropna(subset=["daily_budget"]).copy()
    sub["daily_budget"] = pd.to_numeric(sub["daily_budget"], errors="coerce")

    for target in targets:
        if target not in sub.columns:
            continue
        sub[target] = pd.to_numeric(sub[target], errors="coerce")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(sub["daily_budget"], sub[target], alpha=0.2, s=10)
        ax.set_xlabel("daily_budget")
        ax.set_ylabel(target)
        ax.set_title(f"{target} vs daily_budget (cap)")
        fig.tight_layout()
        p = out_dir / f"{target}_vs_daily_budget.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(str(p))

    return paths


def run_budget_diagnostics(
    df: pd.DataFrame,
    config: CampaignOptConfig,
    out_dir: Path,
    *,
    targets: list[str] | None = None,
    holdout_days: int | None = None,
    write_plots: bool = True,
    models: tuple[str, ...] = ("ridge", "xgboost"),
) -> dict[str, Any]:
    """Run full diagnostic bundle; write CSV/JSON under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = targets or [t for t in (config.target, "clicks", "all_conv") if t in df.columns]
    targets = list(dict.fromkeys(targets))

    holdout_days = holdout_days if holdout_days is not None else config.model_policy.validation.holdout_days
    train, holdout = train_holdout_split(df, holdout_days)

    report: dict[str, Any] = {
        "targets": targets,
        "decision_lever": "daily_budget",
        "n_rows": len(df),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "identification": identifiability_report(df),
    }

    bivariate = bivariate_budget_slopes(df, targets)
    within_set = within_keyword_set_budget_slopes(df, targets)
    bivariate.to_csv(out_dir / "bivariate_slopes.csv", index=False)
    within_set.to_csv(out_dir / "within_set_budget_slopes.csv", index=False)

    identifiable = within_set[within_set["identifiable"]] if len(within_set) else within_set
    report["within_set"] = {
        "n_cells": int(len(within_set)),
        "n_identifiable_cells": int(len(identifiable)),
        "n_identifiable_with_positive_budget_slope": int(
            (identifiable["slope_budget"] > 0).sum()
        )
        if len(identifiable) and "slope_budget" in identifiable.columns
        else 0,
    }

    report["model_comparison"] = compare_model_targets(
        train, holdout, config, targets, models=models
    )

    identified: dict[str, Any] = {}
    for target in targets:
        if target not in train.columns:
            continue
        cfg = replace(config, target=target)
        pooled = pooled_within_cell_budget_slopes(train, target)
        cell_fe = fit_cell_fe_budget_ridge(
            train, holdout, cfg, identifiable_only=True
        )
        identified[target] = {
            "pooled_within_cell_slopes": pooled.to_dict(orient="records"),
            "cell_fe_ridge": cell_fe,
        }
    report["identified_budget_models"] = identified
    if identified:
        pooled_all = []
        for target, block in identified.items():
            for row in block.get("pooled_within_cell_slopes", []):
                row = dict(row)
                row["target"] = target
                pooled_all.append(row)
        if pooled_all:
            pd.DataFrame(pooled_all).to_csv(
                out_dir / "pooled_within_cell_budget_slopes.csv", index=False
            )

    ident_cells = report["identification"].get("cells")
    if ident_cells:
        pd.DataFrame(ident_cells).to_csv(out_dir / "cell_identifiability.csv", index=False)

    if write_plots:
        report["plots"] = _maybe_plot(df, out_dir, targets)

    with open(out_dir / "budget_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def print_budget_diagnostics_summary(report: dict[str, Any]) -> None:
    """Human-readable summary to stdout."""
    print(f"\nDecision lever: {report.get('decision_lever', 'daily_budget')} (cost is not modeled)")

    id_block = report.get("identification", {})
    print("\n=== Budget identification (segment x keyword_set_id cells) ===")
    print(
        f"  cells={id_block.get('n_cells', 0)}  "
        f"identifiable (>={id_block.get('min_budget_levels', 2)} budget levels)="
        f"{id_block.get('n_identifiable_cells', 0)}  "
        f"share of rows in identifiable cells="
        f"{id_block.get('share_rows_identifiable', 0):.1%}"
    )
    ver = id_block.get("campaign_version") or {}
    if ver:
        print(
            f"  campaign_version: {ver.get('n_versions', 0)} versions, "
            f"{ver.get('versions_with_single_budget', 0)} with one budget level "
            f"(budget fixed within version)"
        )

    ws = report.get("within_set", {})
    print("\n=== Within (segment, keyword_set_id) budget variation ===")
    print(
        f"  cells={ws.get('n_cells', 0)}  identifiable (>=2 budget levels)="
        f"{ws.get('n_identifiable_cells', 0)}  positive raw slopes="
        f"{ws.get('n_identifiable_with_positive_budget_slope', 0)}"
    )

    print("\n=== Identified budget models (cell FE + calendar) ===")
    for target, block in report.get("identified_budget_models", {}).items():
        cell_fe = block.get("cell_fe_ridge", {})
        if cell_fe.get("status") != "ok":
            print(f"  {target}: cell-FE ridge {cell_fe.get('status', 'n/a')}")
            continue
        print(
            f"  {target}: budget_coef={cell_fe.get('budget_coef', 0):+.4g} "
            f"(train {cell_fe.get('train_rows_used')}/{cell_fe.get('train_rows_total')} rows, "
            f"{cell_fe.get('n_identifiable_cells')} identifiable cells)  "
            f"holdout R²={cell_fe.get('holdout_r2', 0):.4f}"
        )
        pooled = block.get("pooled_within_cell_slopes") or []
        if pooled:
            top = sorted(pooled, key=lambda r: abs(r.get("pooled_slope_budget", 0)), reverse=True)[:3]
            parts = ", ".join(
                f"{r['segment']}={r['pooled_slope_budget']:+.3g}" for r in top
            )
            print(f"    pooled within-cell slopes (top): {parts}")

    print("\n=== Model comparison (holdout R², budget effect) ===")
    for target, block in report.get("model_comparison", {}).get("targets", {}).items():
        if "error" in block:
            print(f"  {target}: {block['error']}")
            continue
        unshrunk = block.get("ridge_unshrunk", {})
        global_coef = unshrunk.get("global_budget_coef")
        print(f"\n  Target: {target}")
        if global_coef is not None:
            print(f"    ridge global daily_budget coef (unshrunk)={global_coef:+.4g}")
        seg_slopes = unshrunk.get("segment_budget_slope") or {}
        if seg_slopes:
            top = sorted(seg_slopes.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
            seg_str = ", ".join(f"{k}={v:+.3g}" for k, v in top)
            print(f"    ridge segment budget slopes (top): {seg_str}")
        for name, m in block.get("models", {}).items():
            if m.get("status") == "skipped":
                print(f"    {name}: skipped")
                continue
            shap_b = m.get("budget_shap_mean")
            shap_str = f"  budget_shap={shap_b:+.4g}" if shap_b is not None else ""
            print(
                f"    {name}: holdout R²={m.get('holdout_r2', 0):.4f}  "
                f"RMSE={m.get('holdout_rmse', 0):.4f}{shap_str}"
            )
