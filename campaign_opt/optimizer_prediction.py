"""Optimizer-facing predictions with observed-budget floor (not used for model fit / holdout R²)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from campaign_opt.decisions import observed_min_daily_budget
from campaign_opt.evaluation import EnsembleModel
from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import get_context_feature_columns


def _optimizer_feature_cols(model: EnsembleModel | Any, config: CampaignOptConfig) -> list[str]:
    if isinstance(model, EnsembleModel):
        return model.feature_cols
    cols = getattr(model, "feature_cols", None)
    if cols:
        return list(cols)
    return get_context_feature_columns(config.context_features)


def round_budgets_for_floor(budgets: np.ndarray) -> np.ndarray:
    """Round spend to cents before observed-min floor checks (matches displayed plan budgets)."""
    return np.round(np.asarray(budgets, dtype=float), 2)


def apply_observed_budget_floor(
    levels: np.ndarray,
    budgets: np.ndarray,
    segments: np.ndarray,
    min_by_segment: dict[str, float],
    *,
    budget_atol: float = 0.01,
) -> np.ndarray:
    """
    Zero levels when ``budget + budget_atol < min_by_segment[segment]``.

    Budgets are rounded to cents first so Gurobi values such as
    ``13.519999999999927`` match a displayed ``13.52`` plan row.

    Does not clip negative levels above the floor; the optimizer may spend 0 instead.
    ``budget_atol`` (default 1 cent) is an extra tolerance on top of cent rounding.
    """
    out = np.asarray(levels, dtype=float).copy()
    budgets = round_budgets_for_floor(budgets)
    segments = np.asarray(segments, dtype=str)
    atol = max(float(budget_atol), 0.0)
    for i in range(len(out)):
        seg = str(segments[i])
        bmin = float(min_by_segment.get(seg, min_by_segment.get(str(seg), 0.0)))
        if budgets[i] + atol < bmin:
            out[i] = 0.0
    return out


def floor_active_budget(
    budget: float,
    segment: str,
    min_by_segment: dict[str, float],
    *,
    budget_atol: float = 0.01,
) -> bool:
    """True when cent-rounded ``budget`` is at/above the segment's observed minimum."""
    bmin = float(min_by_segment.get(str(segment), min_by_segment.get(segment, 0.0)))
    if bmin <= 0.0:
        return True
    return float(round_budgets_for_floor(np.array([budget]))[0]) + max(float(budget_atol), 0.0) >= bmin


def predict_levels_optimizer(
    model: EnsembleModel | Any,
    rows: pd.DataFrame,
    panel: pd.DataFrame,
    config: CampaignOptConfig,
    *,
    floor_panel: pd.DataFrame | None = None,
) -> np.ndarray:
    """Level predictions with observed-budget floor when enabled in config."""
    if not config.evaluation.apply_observed_budget_floor:
        if isinstance(model, EnsembleModel):
            return model.predict_levels(rows)
        from campaign_opt.training_matrix import prep_xy

        target = config.target
        feature_cols = _optimizer_feature_cols(model, config)
        X, _ = prep_xy(rows, target, feature_cols)
        return np.asarray(model.predict(X), dtype=float)

    segments = rows["segment"].astype(str).unique().tolist()
    mins = observed_min_daily_budget(floor_panel if floor_panel is not None else panel, segments)
    if isinstance(model, EnsembleModel):
        raw = model.predict_levels(rows)
    else:
        from campaign_opt.training_matrix import prep_xy

        target = config.target
        feature_cols = _optimizer_feature_cols(model, config)
        X, _ = prep_xy(rows, target, feature_cols)
        raw = np.asarray(model.predict(X), dtype=float)
    return apply_observed_budget_floor(
        raw,
        rows["daily_budget"].to_numpy(),
        rows["segment"].to_numpy(),
        mins,
        budget_atol=float(config.evaluation.budget_floor_atol),
    )


def predict_incremental_optimizer(
    model: EnsembleModel | Any,
    decision_rows: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    panel: pd.DataFrame,
    config: CampaignOptConfig,
    *,
    floor_panel: pd.DataFrame | None = None,
) -> np.ndarray:
    """Gated f(decision) - f(baseline) per row."""
    floor_kw = {"floor_panel": floor_panel} if floor_panel is not None else {}
    f_dec = predict_levels_optimizer(model, decision_rows, panel, config, **floor_kw)
    f_zero = predict_levels_optimizer(model, baseline_rows, panel, config, **floor_kw)
    return f_dec - f_zero
