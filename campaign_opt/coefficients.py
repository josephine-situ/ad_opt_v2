"""Export linear solver coefficients for Gurobi MILP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import get_context_feature_columns
from utils.shrinkage import segment_budget_level_counts, shrink_segment_slopes


def export_linear_solver_coeffs(
    train: pd.DataFrame,
    config: CampaignOptConfig,
    output_path: Path,
    *,
    shrink_weight: float = 0.5,
    min_budget_levels: int = 3,
) -> dict:
    """
    Fit interpretable segment-level linear model for MILP:
      y ~ segment + daily_budget*segment + keyword_set_id + calendar features
    """
    target = config.target
    if target not in train.columns:
        raise ValueError(f"Target column {target} missing from training frame")

    context_cols = [
        c
        for c in get_context_feature_columns(config.context_features)
        if c in train.columns and c not in ("num_unique_keywords",)
    ]
    cal_cols = [c for c in context_cols if c.startswith(("day_", "month", "is_", "days_"))]
    cal_numeric = {"is_weekend", "is_public_holiday", "days_to_next_course_start"}

    sub = train.dropna(subset=[target, "daily_budget", "segment", "keyword_set_id"]).copy()
    y = sub[target].astype(float).values

    seg_dummies = pd.get_dummies(sub["segment"], prefix="seg", dtype=float)
    set_dummies = pd.get_dummies(sub["keyword_set_id"].astype(str), prefix="set", dtype=float)
    X_parts = [seg_dummies, sub[["daily_budget"]].astype(float)]

    for col in cal_cols:
        if col not in sub.columns:
            continue
        if sub[col].dtype == object or str(sub[col].dtype).startswith("string"):
            X_parts.append(pd.get_dummies(sub[col].astype(str), prefix=col, dtype=float))
        else:
            X_parts.append(sub[[col]].apply(pd.to_numeric, errors="coerce").astype(float))

    X = pd.concat(X_parts, axis=1)
    for s_col in seg_dummies.columns:
        inter = seg_dummies[s_col] * sub["daily_budget"].astype(float)
        inter.name = f"budget_x_{s_col}"
        X[inter.name] = inter

    X = X.fillna(0.0)
    model = Ridge(alpha=1.0)
    model.fit(X.values, y)

    seg_slopes: dict[str, float] = {}
    seg_intercepts: dict[str, float] = {}
    global_slope = float(model.coef_[list(X.columns).index("daily_budget")])

    for seg in sub["segment"].unique():
        col = f"seg_{seg}"
        if col not in X.columns:
            continue
        idx = list(X.columns).index(col)
        inter_col = f"budget_x_{col}"
        slope = global_slope
        if inter_col in X.columns:
            slope += float(model.coef_[list(X.columns).index(inter_col)])
        seg_slopes[seg] = slope
        seg_intercepts[seg] = float(model.intercept_ + model.coef_[idx])

    counts = segment_budget_level_counts(sub)
    seg_slopes = shrink_segment_slopes(
        pd.Series(seg_slopes),
        global_slope=global_slope,
        min_levels=min_budget_levels,
        weight=shrink_weight,
    ).to_dict()

    set_effect: dict[str, float] = {}
    x_cols = list(X.columns)
    for col in set_dummies.columns:
        if col not in x_cols:
            continue
        set_id = col.replace("set_", "", 1)
        set_effect[set_id] = float(model.coef_[x_cols.index(col)])

    cal_effect = 0.0
    last = sub.iloc[-1]
    for col in cal_cols:
        if col not in last.index:
            continue
        val = last[col]
        dummy_cols = [c for c in X.columns if c.startswith(f"{col}_")]
        if dummy_cols:
            for dc in dummy_cols:
                level = dc[len(col) + 1 :]
                if str(val) == level:
                    cal_effect += float(model.coef_[list(X.columns).index(dc)])
        elif col in cal_numeric and col in X.columns:
            cal_effect += float(model.coef_[list(X.columns).index(col)] * float(val))

    coeffs = {
        "segment_intercept": seg_intercepts,
        "segment_budget_slope": seg_slopes,
        "keyword_set_effect": set_effect,
        "calendar_offset": cal_effect,
        "global_intercept": float(model.intercept_),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coeffs, f, indent=2)
    return coeffs
