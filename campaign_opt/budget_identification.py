"""Identify budget effects controlling for keyword-set / version bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from campaign_opt.linear_design import (
    _encode_context_columns,
    calendar_context_columns,
)
from campaign_opt.schema import CampaignOptConfig


def cell_id(segment: object, keyword_set_id: object) -> str:
    return f"{segment} // {keyword_set_id}"


def add_cell_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_cell"] = [
        cell_id(seg, kid)
        for seg, kid in zip(out["segment"], out["keyword_set_id"], strict=True)
    ]
    return out


@dataclass
class IdentifiedBudgetDesign:
    X: pd.DataFrame
    y: np.ndarray
    sub: pd.DataFrame
    x_columns: list[str]
    design: str
    n_rows_total: int
    n_rows_used: int
    n_cells: int
    n_identifiable_cells: int


def identifiability_report(df: pd.DataFrame, *, min_budget_levels: int = 2) -> dict[str, Any]:
    """
    Summarize where ``daily_budget`` varies holding ``(segment, keyword_set_id)`` fixed.

    Within a ``campaign_version``, budget is constant by construction. Identification
    comes from the same keyword set appearing under multiple budget caps over time.
    """
    required = {"segment", "keyword_set_id", "daily_budget"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        return {"error": f"missing columns: {missing}"}

    sub = add_cell_id(df.dropna(subset=["segment", "keyword_set_id", "daily_budget"]))
    cell_stats = (
        sub.groupby("_cell", sort=False)
        .agg(
            segment=("segment", "first"),
            keyword_set_id=("keyword_set_id", "first"),
            n_rows=("daily_budget", "size"),
            n_budget_levels=("daily_budget", "nunique"),
            budgets=("daily_budget", lambda s: sorted({float(v) for v in s})),
        )
        .reset_index()
    )
    cell_stats["identifiable"] = cell_stats["n_budget_levels"] >= min_budget_levels

    identifiable = cell_stats[cell_stats["identifiable"]]
    rows_in_identifiable = int(
        sub[sub["_cell"].isin(identifiable["_cell"])].shape[0]
    ) if len(identifiable) else 0

    version_note = None
    if "campaign_version" in sub.columns:
        # Budget is (nearly) constant within version; flag 1:1 mappings.
        ver_budget = sub.groupby("campaign_version")["daily_budget"].nunique()
        version_note = {
            "n_versions": int(sub["campaign_version"].nunique()),
            "versions_with_single_budget": int((ver_budget == 1).sum()),
            "versions_with_multiple_budgets": int((ver_budget > 1).sum()),
        }

    return {
        "min_budget_levels": min_budget_levels,
        "n_rows": int(len(sub)),
        "n_cells": int(len(cell_stats)),
        "n_identifiable_cells": int(identifiable.shape[0]),
        "rows_in_identifiable_cells": rows_in_identifiable,
        "share_rows_identifiable": rows_in_identifiable / len(sub) if len(sub) else 0.0,
        "campaign_version": version_note,
        "cells": cell_stats.to_dict(orient="records"),
    }


def filter_identifiable_rows(
    df: pd.DataFrame,
    *,
    min_budget_levels: int = 2,
) -> pd.DataFrame:
    """Keep rows whose ``(segment, keyword_set_id)`` cell has >= ``min_budget_levels`` budgets."""
    sub = add_cell_id(df)
    levels = sub.groupby("_cell")["daily_budget"].transform("nunique")
    return sub[levels >= min_budget_levels].drop(columns=["_cell"]).copy()


def build_cell_fixed_effects_design(
    df: pd.DataFrame,
    config: CampaignOptConfig,
    *,
    min_budget_levels: int = 2,
    identifiable_only: bool = False,
    include_calendar: bool = True,
    columns: list[str] | None = None,
) -> IdentifiedBudgetDesign:
    """
    Design for identified budget effect:

      y ~ cell_fe(segment, keyword_set_id) + daily_budget [+ calendar]

    Static keyword-set features are omitted (collinear with cell FE).
    """
    target = config.target
    sub = df.dropna(subset=[target, "daily_budget", "segment", "keyword_set_id"]).copy()
    sub = add_cell_id(sub)
    n_total = len(sub)

    if identifiable_only:
        levels = sub.groupby("_cell")["daily_budget"].transform("nunique")
        sub = sub[levels >= min_budget_levels].copy()

    y = sub[target].astype(float).values
    cell_dummies = pd.get_dummies(sub["_cell"], prefix="cell", dtype=float)
    X_parts: list[pd.DataFrame] = [cell_dummies, sub[["daily_budget"]].astype(float)]

    if include_calendar:
        cal_cols = [c for c in calendar_context_columns(config) if c in sub.columns]
        cal_block = _encode_context_columns(sub, cal_cols)
        if not cal_block.empty:
            X_parts.append(cal_block)

    X = pd.concat(X_parts, axis=1).fillna(0.0)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)

    n_budget = sub.groupby("_cell")["daily_budget"].nunique()
    return IdentifiedBudgetDesign(
        X=X,
        y=y,
        sub=sub.drop(columns=["_cell"]),
        x_columns=list(X.columns),
        design="cell_fe_budget_calendar",
        n_rows_total=n_total,
        n_rows_used=len(sub),
        n_cells=int(cell_dummies.shape[1]),
        n_identifiable_cells=int((n_budget >= min_budget_levels).sum()),
    )


def build_version_fixed_effects_design(
    df: pd.DataFrame,
    config: CampaignOptConfig,
    *,
    include_calendar: bool = True,
    columns: list[str] | None = None,
) -> IdentifiedBudgetDesign | None:
    """
    y ~ campaign_version_fe + daily_budget [+ calendar]

  Note: often collinear because each version usually maps to one budget level.
    Prefer ``build_cell_fixed_effects_design`` for this panel.
    """
    if "campaign_version" not in df.columns:
        return None

    target = config.target
    sub = df.dropna(subset=[target, "daily_budget", "campaign_version"]).copy()
    n_total = len(sub)
    y = sub[target].astype(float).values
    ver_dummies = pd.get_dummies(sub["campaign_version"].astype(str), prefix="ver", dtype=float)
    X_parts: list[pd.DataFrame] = [ver_dummies, sub[["daily_budget"]].astype(float)]

    if include_calendar:
        cal_cols = [c for c in calendar_context_columns(config) if c in sub.columns]
        cal_block = _encode_context_columns(sub, cal_cols)
        if not cal_block.empty:
            X_parts.append(cal_block)

    X = pd.concat(X_parts, axis=1).fillna(0.0)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)

    return IdentifiedBudgetDesign(
        X=X,
        y=y,
        sub=sub,
        x_columns=list(X.columns),
        design="version_fe_budget_calendar",
        n_rows_total=n_total,
        n_rows_used=len(sub),
        n_cells=int(ver_dummies.shape[1]),
        n_identifiable_cells=int(sub.groupby("campaign_version")["daily_budget"].nunique().gt(1).sum()),
    )


def pooled_within_cell_budget_slopes(
    df: pd.DataFrame,
    target: str,
    *,
    min_budget_levels: int = 2,
) -> pd.DataFrame:
    """Per-segment, row-weighted average of cell-level OLS budget slopes."""
    from campaign_opt.budget_diagnostics import within_keyword_set_budget_slopes

    cells = within_keyword_set_budget_slopes(df, [target])
    if cells.empty:
        return pd.DataFrame()

    id_cells = cells[cells["identifiable"]].copy()
    id_cells = id_cells[id_cells["slope_budget"].notna()]
    if id_cells.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for segment, grp in id_cells.groupby("segment", sort=False):
        weights = grp["n_rows"].astype(float)
        slopes = grp["slope_budget"].astype(float)
        pooled = float(np.average(slopes, weights=weights))
        rows.append(
            {
                "segment": str(segment),
                "target": target,
                "pooled_slope_budget": pooled,
                "n_identifiable_cells": int(len(grp)),
                "n_rows": int(weights.sum()),
            }
        )
    return pd.DataFrame(rows)


def fit_cell_fe_budget_ridge(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    config: CampaignOptConfig,
    *,
    alpha: float = 1.0,
    min_budget_levels: int = 2,
    identifiable_only: bool = True,
) -> dict[str, Any]:
    """Ridge on cell-FE design; returns budget coef and holdout metrics."""
    train_design = build_cell_fixed_effects_design(
        train,
        config,
        min_budget_levels=min_budget_levels,
        identifiable_only=identifiable_only,
    )
    if train_design.n_rows_used < 10:
        return {
            "status": "insufficient_rows",
            "train_rows_used": train_design.n_rows_used,
            "train_rows_total": train_design.n_rows_total,
        }

    model = Ridge(alpha=alpha)
    model.fit(train_design.X.values, train_design.y)

    budget_idx = train_design.x_columns.index("daily_budget")
    budget_coef = float(model.coef_[budget_idx])

    holdout_design = build_cell_fixed_effects_design(
        holdout,
        config,
        min_budget_levels=1,
        identifiable_only=False,
        columns=train_design.x_columns,
    )
    pred = np.clip(model.predict(holdout_design.X.values), 0, None)
    y_true = holdout_design.y

    return {
        "status": "ok",
        "design": train_design.design,
        "identifiable_only": identifiable_only,
        "budget_coef": budget_coef,
        "train_rows_used": train_design.n_rows_used,
        "train_rows_total": train_design.n_rows_total,
        "n_cells": train_design.n_cells,
        "n_identifiable_cells": train_design.n_identifiable_cells,
        "holdout_r2": float(r2_score(y_true, pred)),
        "holdout_rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
    }
