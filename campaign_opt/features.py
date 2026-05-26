"""Feature matrix construction from config + campaign panel."""

from __future__ import annotations

import pandas as pd

from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import (
    build_modeling_frame,
    get_context_feature_columns,
)


def filter_modeling_lookback(
    df: pd.DataFrame,
    lookback_days: int | None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Keep rows with ``date_col`` in the last ``lookback_days`` through panel max date."""
    if not lookback_days or lookback_days <= 0:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    max_date = out[date_col].max()
    cutoff = max_date - pd.Timedelta(days=int(lookback_days))
    return out[out[date_col] >= cutoff].copy()


def prepare_modeling_data(config: CampaignOptConfig | str) -> pd.DataFrame:
    if isinstance(config, str):
        course = config
        target = "all_conv"
        context_features: dict[str, list[str]] = {}
        lookback_days = None
    else:
        course = config.course
        target = config.target
        context_features = config.context_features
        lookback_days = config.modeling_lookback_days

    df = build_modeling_frame(course, target_col=target)
    context_cols = get_context_feature_columns(context_features) if context_features else []
    if context_cols:
        for col in context_cols:
            if col not in df.columns:
                df[col] = pd.NA
    df = df.dropna(subset=["daily_budget", "segment"])
    return filter_modeling_lookback(df, lookback_days)


def train_holdout_split(
    df: pd.DataFrame,
    holdout_days: int,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split last ``holdout_days`` for static evaluation (fit_response_models)."""
    df = df.sort_values(date_col)
    cutoff = df[date_col].max() - pd.Timedelta(days=holdout_days)
    train = df[df[date_col] <= cutoff].copy()
    holdout = df[df[date_col] > cutoff].copy()
    return train, holdout


def train_before_date(
    df: pd.DataFrame,
    before: pd.Timestamp,
    date_col: str = "date",
) -> pd.DataFrame:
    """Training rows strictly before ``before`` (for walk-forward backtest)."""
    before = pd.Timestamp(before)
    return df[pd.to_datetime(df[date_col]) < before].copy()
