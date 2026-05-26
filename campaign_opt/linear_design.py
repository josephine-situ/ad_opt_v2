"""Shared linear design matrix for tournament ridge and MILP coefficient export."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import (
    add_segment_match_type_indicators,
    get_context_feature_columns,
    parse_match_types,
)

SEGMENT_MATCH_COLS = ("has_broad", "has_phrase", "has_exact")
SEGMENT_STRUCTURAL_EXACT = frozenset({"daily_budget", *SEGMENT_MATCH_COLS})
SEGMENT_STRUCTURAL_PREFIXES = ("region_", "budget_x_region_", "budget_x_has_")


@dataclass
class LinearMilpDesign:
    X: pd.DataFrame
    y: np.ndarray
    sub: pd.DataFrame
    context_cols: list[str]
    static_cols: list[str]
    cal_cols: list[str]
    x_columns: list[str]


class LinearMilpRidgeModel:
    """Ridge on the MILP-linear design; used by tournament ridge and optional predict."""

    def __init__(self, model: Ridge, x_columns: list[str], config: CampaignOptConfig):
        self.model = model
        self.x_columns = x_columns
        self.config = config

    def predict_design_frame(self, df: pd.DataFrame) -> np.ndarray:
        design = build_linear_milp_design_matrix(df, self.config, columns=self.x_columns)
        return self.model.predict(design.X.values)


def static_context_columns(config: CampaignOptConfig) -> list[str]:
    cols: list[str] = []
    for group in ("keyword_set_static", "gkp_set"):
        cols.extend(config.context_features.get(group, []))
    return list(dict.fromkeys(cols))


def calendar_context_columns(config: CampaignOptConfig) -> list[str]:
    return list(config.context_features.get("calendar", []))


def split_context_columns_by_dtype(
    df: pd.DataFrame,
    cols: list[str],
) -> tuple[list[str], list[str]]:
    """Partition context columns the same way as the MILP-linear ridge design."""
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in cols:
        if col == "daily_budget" or col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def segment_match_indicators(segment: str) -> dict[str, float | str]:
    """Decompose a compound segment string into region + match-type flags."""
    region = segment.split(" / ", 1)[0].strip()
    match_part = segment.split(" / ", 1)[1].strip() if " / " in segment else ""
    types = parse_match_types(match_part)
    return {
        "region": region,
        "has_broad": float("Broad" in types),
        "has_phrase": float("Phrase" in types),
        "has_exact": float("Exact" in types),
    }


def segment_intercept_from_model(model: Ridge, x_columns: list[str], segment: str) -> float:
    """Intercept for one compound segment from region + match-type ridge coefficients."""
    ind = segment_match_indicators(segment)
    val = float(model.intercept_)
    region_col = f"region_{ind['region']}"
    if region_col in x_columns:
        val += float(model.coef_[x_columns.index(region_col)])
    for mt in SEGMENT_MATCH_COLS:
        if mt in x_columns:
            val += float(model.coef_[x_columns.index(mt)]) * float(ind[mt])
    return val


def segment_slope_from_model(model: Ridge, x_columns: list[str], segment: str) -> float:
    """Budget slope for one compound segment from region + match-type interactions."""
    ind = segment_match_indicators(segment)
    slope = float(model.coef_[x_columns.index("daily_budget")])
    region_col = f"budget_x_region_{ind['region']}"
    if region_col in x_columns:
        slope += float(model.coef_[x_columns.index(region_col)])
    for mt in SEGMENT_MATCH_COLS:
        inter_col = f"budget_x_{mt}"
        if inter_col in x_columns:
            slope += float(model.coef_[x_columns.index(inter_col)]) * float(ind[mt])
    return slope


def _encode_context_columns(sub: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    numeric_cols, categorical_cols = split_context_columns_by_dtype(sub, cols)
    parts: list[pd.DataFrame] = []
    for col in numeric_cols:
        parts.append(sub[[col]].apply(pd.to_numeric, errors="coerce").astype(float))
    for col in categorical_cols:
        parts.append(pd.get_dummies(sub[col].astype(str), prefix=col, dtype=float))
    if not parts:
        return pd.DataFrame(index=sub.index)
    return pd.concat(parts, axis=1)


def _build_segment_feature_block(sub: pd.DataFrame) -> pd.DataFrame:
    """Region one-hot + match-type indicators (aligned with tree-model segment features)."""
    region_dummies = pd.get_dummies(sub["region"].astype(str), prefix="region", dtype=float)
    match_block = sub[list(SEGMENT_MATCH_COLS)].astype(float)
    return pd.concat([region_dummies, match_block], axis=1)


def build_linear_milp_design_matrix(
    df: pd.DataFrame,
    config: CampaignOptConfig,
    *,
    columns: list[str] | None = None,
) -> LinearMilpDesign:
    """
    Design used by MILP linear backend and aligned tournament ridge:

      y ~ region + match_type_flags + daily_budget + budget×(region + match) + context_features

    Per-compound-segment intercepts/slopes for Gurobi are derived from these coefficients.
    """
    target = config.target
    context_cols = [
        c for c in get_context_feature_columns(config.context_features) if c in df.columns
    ]
    static_cols = [c for c in static_context_columns(config) if c in context_cols]
    cal_cols = [c for c in calendar_context_columns(config) if c in context_cols]

    sub = df.dropna(subset=[target, "daily_budget", "segment"]).copy()
    sub = add_segment_match_type_indicators(sub)
    y = sub[target].astype(float).values

    seg_block = _build_segment_feature_block(sub)
    context_block = _encode_context_columns(sub, context_cols)
    X_parts: list[pd.DataFrame] = [seg_block, sub[["daily_budget"]].astype(float)]
    if not context_block.empty:
        X_parts.append(context_block)

    X = pd.concat(X_parts, axis=1)
    budget = sub["daily_budget"].astype(float)
    for col in seg_block.columns:
        inter = seg_block[col] * budget
        inter.name = f"budget_x_{col}"
        X[inter.name] = inter

    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    # Missing context values (common on holdout) must not reach sklearn.
    X = X.fillna(0.0)

    x_columns = columns if columns is not None else list(X.columns)
    return LinearMilpDesign(
        X=X,
        y=y,
        sub=sub,
        context_cols=context_cols,
        static_cols=static_cols,
        cal_cols=cal_cols,
        x_columns=x_columns,
    )
