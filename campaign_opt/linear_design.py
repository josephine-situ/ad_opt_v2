"""Shared linear design matrix for tournament ridge and MILP coefficient export."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import (
    SEGMENT_BROAD_MATCH_COL,
    add_segment_match_type_indicators,
    get_context_feature_columns,
    is_broad_match_campaign,
)

SEGMENT_MATCH_COLS = (SEGMENT_BROAD_MATCH_COL,)
SEGMENT_STRUCTURAL_EXACT = frozenset({"daily_budget", *SEGMENT_MATCH_COLS})
SEGMENT_STRUCTURAL_PREFIXES = ("region_", "budget_x_region_", "budget_x_")
RIDGE_NO_SCALE_COLS = frozenset({"is_weekend", "is_public_holiday"})


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

    def __init__(
        self,
        model: Ridge,
        x_columns: list[str],
        config: CampaignOptConfig,
        *,
        scaler: StandardScaler | None = None,
        scale_cols: list[str] | None = None,
    ):
        self.model = model
        self.x_columns = x_columns
        self.config = config
        self.scaler = scaler
        self.scale_cols = list(scale_cols or [])

    def uses_scaled_fit(self) -> bool:
        return self.scaler is not None

    def _matrix_for_predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            return X.values
        X_scaled, _ = scale_milp_design_matrix(X, self.scale_cols, scaler=self.scaler)
        return X_scaled.values

    def predict_design_frame(self, df: pd.DataFrame) -> np.ndarray:
        design = build_linear_milp_design_matrix(df, self.config, columns=self.x_columns)
        return self.model.predict(self._matrix_for_predict(design.X))


def segment_budget_response_from_artifact(
    artifact: LinearMilpRidgeModel,
    template: pd.Series,
    segment: str,
    *,
    budget_lo: float = 0.0,
    budget_hi: float = 1.0,
) -> tuple[float, float]:
    """Predicted level at ``budget_lo`` and slope ``(level_hi - level_lo) / (budget_hi - budget_lo)``."""
    if budget_hi == budget_lo:
        raise ValueError("budget_hi must differ from budget_lo")
    rows = []
    for budget in (budget_lo, budget_hi):
        row = template.copy()
        row["segment"] = segment
        row["daily_budget"] = float(budget)
        rows.append(row)
    preds = artifact.predict_design_frame(pd.DataFrame(rows))
    level_lo = float(preds[0])
    slope = (float(preds[1]) - level_lo) / (budget_hi - budget_lo)
    return level_lo, slope


def template_row_for_segment(sub: pd.DataFrame, segment: str) -> pd.Series:
    """Representative panel row for a segment (for coefficient export / calendar offsets)."""
    seg_rows = sub[sub["segment"].astype(str) == str(segment)]
    if seg_rows.empty:
        raise ValueError(f"No rows for segment {segment!r}")
    return seg_rows.iloc[0]


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
    """Decompose a compound segment string into region + broad-only match flag."""
    region = segment.split(" / ", 1)[0].strip()
    match_part = segment.split(" / ", 1)[1].strip() if " / " in segment else ""
    return {
        "region": region,
        SEGMENT_BROAD_MATCH_COL: float(is_broad_match_campaign(match_part)),
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


def ridge_numeric_scale_column_names(
    sub: pd.DataFrame,
    context_cols: list[str],
    x_columns: list[str],
) -> list[str]:
    """Continuous MILP design columns to standardize before ridge (not dummies or interactions)."""
    numeric_ctx, _ = split_context_columns_by_dtype(sub, context_cols)
    return ["daily_budget"] + [
        c for c in numeric_ctx if c in x_columns and c not in RIDGE_NO_SCALE_COLS
    ]


def scale_milp_design_matrix(
    X: pd.DataFrame,
    scale_cols: list[str],
    *,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[pd.DataFrame, StandardScaler | None]:
    """
    Standardize continuous columns and rebuild ``budget_x_*`` from scaled budget.

    Region/match dummies and categorical context one-hots are left unchanged.
    """
    use_cols = [c for c in scale_cols if c in X.columns]
    if not use_cols:
        return X, scaler

    out = X.copy()
    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(out[use_cols])
    if scaler is None:
        raise ValueError("scaler is required when fit_scaler is False")
    out[use_cols] = scaler.transform(out[use_cols])
    budget_scaled = out["daily_budget"]
    for col in list(out.columns):
        if not col.startswith("budget_x_"):
            continue
        base_col = col[len("budget_x_") :]
        if base_col in out.columns:
            out[col] = out[base_col] * budget_scaled
    return out, scaler


def fit_linear_milp_ridge(
    design: LinearMilpDesign,
    config: CampaignOptConfig,
    *,
    alpha: float = 1.0,
) -> LinearMilpRidgeModel:
    """
    Fit ridge on the MILP design; standardize continuous columns when present.

    Coefficients are in scaled space; ``LinearMilpRidgeModel.predict_design_frame``
    applies the training scaler. MILP coefficient export uses prediction-based
    segment intercept/slope when scaling is active.
    """
    scale_cols = ridge_numeric_scale_column_names(
        design.sub, design.context_cols, design.x_columns
    )
    if not scale_cols:
        model = Ridge(alpha=alpha)
        model.fit(design.X.values, design.y)
        return LinearMilpRidgeModel(model, design.x_columns, config)

    X_scaled, scaler = scale_milp_design_matrix(
        design.X, scale_cols, fit_scaler=True
    )
    fitted = Ridge(alpha=alpha)
    fitted.fit(X_scaled.values, design.y)
    return LinearMilpRidgeModel(
        fitted,
        design.x_columns,
        config,
        scaler=scaler,
        scale_cols=scale_cols,
    )


def build_linear_milp_design_matrix(
    df: pd.DataFrame,
    config: CampaignOptConfig,
    *,
    columns: list[str] | None = None,
) -> LinearMilpDesign:
    """
    Design used by MILP linear backend and aligned tournament ridge:

      y ~ region + is_broad_match + daily_budget + budget×(region + is_broad_match) + context_features

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
