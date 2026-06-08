"""Time-series cross-validation on campaign-day panel.

Expanding-window CV on the train calendar only (no campaign-version / phase splits).
See docs/cross_validation.md.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd


class CVFoldError(RuntimeError):
    """A CV fold failed during fit or scoring (no silent skip)."""


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
    is too short. All segments active in the val window are scored together.
    """
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    dates = np.array(sorted(work[date_col].unique()))
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
        train_fold = work[work[date_col] <= train_cutoff]
        val_fold = work[(work[date_col] > train_cutoff) & (work[date_col] <= val_end)]
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


def validation_cv_kwargs(config) -> dict[str, int | float]:
    """Extract ValidationConfig fields as kwargs for ``time_series_cv_folds``."""
    val = config.model_policy.validation
    return {
        "min_train_days": val.min_train_days,
        "min_train_fraction": val.min_train_fraction,
        "min_val_days": val.min_val_days,
        "min_train_rows": val.min_train_rows,
        "min_val_rows": val.min_val_rows,
    }


# Backward-compatible alias.
_validation_kw = validation_cv_kwargs


def _fold_context(train_fold: pd.DataFrame, val_fold: pd.DataFrame, date_col: str = "date") -> str:
    tr_d = pd.to_datetime(train_fold[date_col])
    va_d = pd.to_datetime(val_fold[date_col])
    return (
        f"train {tr_d.min().date()}..{tr_d.max().date()} "
        f"({tr_d.nunique()} days, {len(train_fold)} rows); "
        f"val {va_d.min().date()}..{va_d.max().date()} "
        f"({va_d.nunique()} days, {len(val_fold)} rows)"
    )


_cv_fold_log_cache: set[tuple] = set()


def print_cv_fold_dates(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    *,
    prefix: str = "CV",
    date_col: str = "date",
    dedupe_key: tuple | None = None,
) -> None:
    """Print train/val date ranges for each fold (once per ``dedupe_key`` when set)."""
    if dedupe_key is not None:
        if dedupe_key in _cv_fold_log_cache:
            return
        _cv_fold_log_cache.add(dedupe_key)
    if not folds:
        print(f"{prefix}: no folds")
        return
    print(f"{prefix}: {len(folds)} fold(s) (expanding-window)")
    for fold_idx, (tr_fold, va_fold) in enumerate(folds, start=1):
        print(f"  fold {fold_idx}/{len(folds)}: {_fold_context(tr_fold, va_fold, date_col)}")


def cross_validate_model(
    fit_fn: Callable[..., Any],
    train: pd.DataFrame,
    config,
    feature_cols: list[str],
    *,
    n_folds: int = 5,
    date_col: str = "date",
) -> dict[str, float]:
    """Run ``fit_fn`` on each CV fold; return mean level-scale metrics."""
    val = config.model_policy.validation
    n_folds_eff = int(getattr(val, "cv_folds", n_folds))
    folds = time_series_cv_folds(train, n_folds_eff, date_col=date_col, **validation_cv_kwargs(config))

    if not folds:
        warnings.warn(
            "No CV folds on train panel; using single internal holdout (last 20% of train dates).",
            stacklevel=2,
        )
        dates = sorted(pd.to_datetime(train[date_col]).unique())
        cut = dates[int(len(dates) * 0.8)]
        folds = [
            (train[train[date_col] <= cut], train[train[date_col] > cut]),
        ]

    tr_dates = pd.to_datetime(train[date_col])
    print_cv_fold_dates(
        folds,
        prefix="CV fold schedule",
        date_col=date_col,
        dedupe_key=(tr_dates.min(), tr_dates.max(), len(folds), n_folds_eff),
    )

    rmses: list[float] = []
    r2s: list[float] = []
    maes: list[float] = []

    for fold_idx, (tr_fold, va_fold) in enumerate(folds, start=1):
        try:
            res = fit_fn(tr_fold, va_fold, config, feature_cols)
        except Exception as exc:
            raise CVFoldError(
                f"CV fold {fold_idx}/{len(folds)} failed ({_fold_context(tr_fold, va_fold, date_col)}): {exc}"
            ) from exc
        rmses.append(res.holdout_rmse)
        r2s.append(res.holdout_r2)
        maes.append(res.holdout_mae)

    if not rmses:
        raise CVFoldError("CV produced no metrics after fold evaluation")

    return {
        "cv_rmse_levels": float(np.mean(rmses)),
        "cv_r2_levels": float(np.mean(r2s)),
        "cv_mae_levels": float(np.mean(maes)),
        "cv_n_folds": len(rmses),
    }
