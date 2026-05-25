"""Time-series cross-validation on campaign-day panel."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

def time_series_cv_folds(
    df: pd.DataFrame,
    n_folds: int,
    date_col: str = "date",
    min_train_days: int = 30,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Expanding-window CV: each fold trains on dates <= cutoff_i, validates on next chunk.

    Returns list of (train_fold, val_fold). Skips folds with insufficient training history.
    """
    dates = np.array(sorted(df[date_col].unique()))
    if len(dates) < min_train_days + n_folds:
        n_folds = max(1, len(dates) - min_train_days)

    # Validation chunks are consecutive date blocks after an expanding train window
    n_val = max(1, (len(dates) - min_train_days) // n_folds)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    for i in range(n_folds):
        train_end_idx = min_train_days + i * n_val - 1
        val_end_idx = min(train_end_idx + n_val, len(dates) - 1)
        if train_end_idx < min_train_days - 1 or val_end_idx <= train_end_idx:
            continue
        train_cutoff = dates[train_end_idx]
        val_end = dates[val_end_idx]
        train_fold = df[df[date_col] <= train_cutoff]
        val_fold = df[(df[date_col] > train_cutoff) & (df[date_col] <= val_end)]
        if len(train_fold) >= 10 and len(val_fold) >= 1:
            folds.append((train_fold, val_fold))

    return folds


def cross_validate_model(
    fit_fn: Callable[..., Any],
    train: pd.DataFrame,
    config,
    feature_cols: list[str],
    *,
    n_folds: int = 5,
) -> dict[str, float]:
    """Run ``fit_fn`` on each CV fold; return mean level-scale metrics."""
    folds = time_series_cv_folds(train, n_folds)
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
