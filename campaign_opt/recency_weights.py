"""Exponential recency sample weights for panel model fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from campaign_opt.schema import CampaignOptConfig


def recency_half_life_days(config: CampaignOptConfig) -> float | None:
    """Configured half-life in days; ``None`` or non-positive means uniform weights."""
    val = config.model_policy.validation.recency_half_life_days
    if val is None:
        return None
    half_life = float(val)
    return half_life if half_life > 0 else None


def recency_sample_weights(
    df: pd.DataFrame,
    *,
    half_life_days: float | None,
    date_col: str = "date",
    anchor: pd.Timestamp | None = None,
) -> np.ndarray | None:
    """
    Exponential weights: w = exp(-age_days / half_life), normalized to mean 1.

    ``anchor`` defaults to the max ``date_col`` in ``df`` (latest row in the train fold).
    """
    if half_life_days is None or half_life_days <= 0:
        return None
    if date_col not in df.columns or len(df) == 0:
        return None

    dates = pd.to_datetime(df[date_col])
    anchor_ts = pd.Timestamp(anchor) if anchor is not None else dates.max()
    age_days = (anchor_ts - dates).dt.total_seconds().to_numpy() / 86400.0
    w = np.exp(-age_days / float(half_life_days))
    mean_w = float(w.mean())
    if mean_w <= 0:
        return None
    return w / mean_w


def training_row_recency_weights(
    train: pd.DataFrame,
    config: CampaignOptConfig,
    *,
    y_col: str | None = None,
    date_col: str = "date",
) -> np.ndarray | None:
    """Weights aligned with tree/sklearn training rows (same dropna as ``_prep_xy``)."""
    from campaign_opt.modeling import _training_subframe

    half_life = recency_half_life_days(config)
    if half_life is None:
        return None
    target = config.target
    sub = _training_subframe(train, target, y_col=y_col)
    return recency_sample_weights(sub, half_life_days=half_life, date_col=date_col)
