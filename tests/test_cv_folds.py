"""Tests for time-series CV fold sizing."""

from __future__ import annotations

import pandas as pd

from campaign_opt.cv import effective_min_train_days, time_series_cv_folds


def _panel(n_days: int, rows_per_day: int = 4) -> pd.DataFrame:
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for s in range(rows_per_day):
            rows.append({"date": day, "segment": f"s{s}", "daily_budget": 100.0, "clicks": 1.0})
    return pd.DataFrame(rows)


def test_effective_min_train_days_uses_fraction():
    assert effective_min_train_days(200, min_train_fraction=0.5) == 100
    assert effective_min_train_days(200, min_train_days=120, min_train_fraction=0.5) == 120


def test_folds_respect_half_panel_train_fraction():
    df = _panel(200)
    folds = time_series_cv_folds(
        df, 3, min_train_fraction=0.5, min_val_days=21, min_train_rows=50, min_val_rows=20
    )
    assert len(folds) >= 1
    n_dates = df["date"].nunique()
    min_train = effective_min_train_days(n_dates, min_train_fraction=0.5)
    for tr, va in folds:
        assert tr["date"].nunique() >= min_train
        assert va["date"].nunique() >= 21
        assert len(tr) >= 50
        assert len(va) >= 20


def test_no_folds_when_panel_too_short_for_half_and_val():
    df = _panel(40)
    folds = time_series_cv_folds(df, 3, min_train_fraction=0.5, min_val_days=21)
    assert folds == []
