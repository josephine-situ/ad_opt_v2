"""Optimizer-facing predictions with observed-budget floor (not used for model fit / holdout R²)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from campaign_opt.decisions import observed_min_daily_budget
from campaign_opt.evaluation import EnsembleModel
from campaign_opt.schema import CampaignOptConfig


def apply_observed_budget_floor(
    levels: np.ndarray,
    budgets: np.ndarray,
    segments: np.ndarray,
    min_by_segment: dict[str, float],
) -> np.ndarray:
    """Zero levels where ``budget < min_by_segment[segment]``."""
    out = np.asarray(levels, dtype=float).copy()
    budgets = np.asarray(budgets, dtype=float)
    segments = np.asarray(segments, dtype=str)
    for i in range(len(out)):
        seg = str(segments[i])
        bmin = float(min_by_segment.get(seg, min_by_segment.get(str(seg), 0.0)))
        if budgets[i] < bmin:
            out[i] = 0.0
    return np.clip(out, 0, None)


def predict_levels_optimizer(
    model: EnsembleModel | Any,
    rows: pd.DataFrame,
    panel: pd.DataFrame,
    config: CampaignOptConfig,
) -> np.ndarray:
    """Level predictions with observed-budget floor when enabled in config."""
    if not config.evaluation.apply_observed_budget_floor:
        if isinstance(model, EnsembleModel):
            return model.predict_levels(rows)
        from campaign_opt.evaluation import _prep_xy

        target = config.target
        feature_cols = model.feature_cols if hasattr(model, "feature_cols") else []
        X, _ = _prep_xy(rows, target, feature_cols)
        return np.clip(np.asarray(model.predict(X), dtype=float), 0, None)

    segments = rows["segment"].astype(str).unique().tolist()
    mins = observed_min_daily_budget(panel, segments)
    if isinstance(model, EnsembleModel):
        raw = model.predict_levels(rows)
    else:
        from campaign_opt.evaluation import _prep_xy

        target = config.target
        feature_cols = getattr(model, "feature_cols", [])
        X, _ = _prep_xy(rows, target, feature_cols)
        raw = np.asarray(model.predict(X), dtype=float)
    return apply_observed_budget_floor(
        raw,
        rows["daily_budget"].to_numpy(),
        rows["segment"].to_numpy(),
        mins,
    )


def predict_incremental_optimizer(
    model: EnsembleModel | Any,
    decision_rows: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    panel: pd.DataFrame,
    config: CampaignOptConfig,
) -> np.ndarray:
    """Gated f(decision) - f(baseline) per row."""
    f_dec = predict_levels_optimizer(model, decision_rows, panel, config)
    f_zero = predict_levels_optimizer(model, baseline_rows, panel, config)
    return f_dec - f_zero
