"""Time-series cross-validation on campaign-day panel."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd


def effective_min_train_days(
    n_dates: int,
    *,
    min_train_days: int = 0,
    min_train_fraction: float = 0.0,
) -> int:
    """Minimum training calendar days for the first CV fold."""
    floor_days = max(0, min_train_days)
    if min_train_fraction > 0:
        floor_days = max(floor_days, math.ceil(n_dates * min_train_fraction))
    return floor_days


def time_series_cv_folds(
    df: pd.DataFrame,
    n_folds: int,
    date_col: str = "date",
    *,
    min_train_days: int = 0,
    min_train_fraction: float = 0.5,
    min_val_days: int = 21,
    min_train_rows: int = 50,
    min_val_rows: int = 20,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Expanding-window CV: each fold trains on dates <= cutoff_i, validates on next chunk.

    Training on every fold uses at least ``min_train_fraction`` of unique panel dates
    (and at least ``min_train_days`` when set). Validation windows are at least
    ``min_val_days`` calendar days. ``n_folds`` may be reduced when the train panel
    is too short.
    """
    dates = np.array(sorted(df[date_col].unique()))
    n_dates = len(dates)
    min_train = effective_min_train_days(
        n_dates, min_train_days=min_train_days, min_train_fraction=min_train_fraction
    )
    if n_dates < min_train + min_val_days:
        return []

    usable = n_dates - min_train
    n_val = max(min_val_days, usable // max(n_folds, 1))
    n_folds_eff = min(n_folds, max(1, usable // n_val))

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(n_folds_eff):
        train_end_idx = min_train + i * n_val - 1
        val_end_idx = min(train_end_idx + n_val, n_dates - 1)
        if train_end_idx < min_train - 1 or val_end_idx <= train_end_idx:
            continue
        train_cutoff = dates[train_end_idx]
        val_end = dates[val_end_idx]
        train_fold = df[df[date_col] <= train_cutoff]
        val_fold = df[(df[date_col] > train_cutoff) & (df[date_col] <= val_end)]
        train_days = train_fold[date_col].nunique()
        val_days = val_fold[date_col].nunique()
        if (
            train_days >= min_train
            and val_days >= min_val_days
            and len(train_fold) >= min_train_rows
            and len(val_fold) >= min_val_rows
        ):
            folds.append((train_fold, val_fold))

    return folds


def _validation_kw(config) -> dict[str, int | float]:
    val = config.model_policy.validation
    return {
        "min_train_days": val.min_train_days,
        "min_train_fraction": val.min_train_fraction,
        "min_val_days": val.min_val_days,
        "min_train_rows": val.min_train_rows,
        "min_val_rows": val.min_val_rows,
    }


def cross_validate_model(
    fit_fn: Callable[..., Any],
    train: pd.DataFrame,
    config,
    feature_cols: list[str],
    *,
    n_folds: int = 5,
) -> dict[str, float]:
    """Run ``fit_fn`` on each CV fold; return mean level-scale metrics."""
    folds = time_series_cv_folds(train, n_folds, **_validation_kw(config))
    if not folds:
        # Fallback: single internal holdout (last 20% of dates)
        dates = sorted(train["date"].unique())
        cut = dates[int(len(dates) * 0.8)]
        folds = [
            (train[train["date"] <= cut], train[train["date"] > cut]),
        ]

    rmses: list[float] = []
    r2s: list[float] = []
    maes: list[float] = []

    for tr_fold, va_fold in folds:
        try:
            res = fit_fn(tr_fold, va_fold, config, feature_cols)
            rmses.append(res.holdout_rmse)
            r2s.append(res.holdout_r2)
            maes.append(res.holdout_mae)
        except Exception:
            continue

    if not rmses:
        return {"cv_rmse_levels": float("inf"), "cv_r2_levels": 0.0, "cv_mae_levels": float("inf")}

    return {
        "cv_rmse_levels": float(np.mean(rmses)),
        "cv_r2_levels": float(np.mean(r2s)),
        "cv_mae_levels": float(np.mean(maes)),
        "cv_n_folds": len(rmses),
    }
