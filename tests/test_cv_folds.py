"""Tests for time-series CV fold sizing."""

from __future__ import annotations

import pandas as pd
import pytest

from campaign_opt.cv import (
    CVFoldError,
    add_calendar_period_id,
    add_run_period_id,
    calendar_period_ranges,
    cross_validate_model,
    effective_min_train_days,
    time_series_cv_folds,
    time_series_cv_folds_phase1_launch,
    time_series_cv_folds_phase2_daily,
)


def _panel(n_days: int, rows_per_day: int = 4, *, campaign_version: int | str = 1) -> pd.DataFrame:
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for s in range(rows_per_day):
            rows.append(
                {
                    "date": day,
                    "segment": f"s{s}",
                    "campaign_version": campaign_version,
                    "daily_budget": 100.0,
                    "clicks": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _panel_with_gap(gap_days: int = 30) -> pd.DataFrame:
    block1 = _panel(80, rows_per_day=4, campaign_version=1)
    block2 = _panel(80, rows_per_day=4, campaign_version=2)
    block2["date"] = block2["date"] + pd.Timedelta(days=80 + gap_days)
    return pd.concat([block1, block2], ignore_index=True)


def test_effective_min_train_days_uses_fraction():
    assert effective_min_train_days(200, min_train_fraction=0.5) == 100
    assert effective_min_train_days(200, min_train_days=120, min_train_fraction=0.5) == 120


def test_folds_respect_half_panel_train_fraction():
    df = _panel(200)
    folds = time_series_cv_folds(
        df,
        3,
        cv_profile="legacy_calendar",
        min_train_fraction=0.5,
        min_val_days=21,
        min_train_rows=50,
        min_val_rows=20,
        respect_campaign_periods=False,
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
    folds = time_series_cv_folds(
        df, 3, cv_profile="legacy_calendar", min_train_fraction=0.5, min_val_days=21
    )
    assert folds == []


def test_calendar_period_ranges_detects_gap():
    df = _panel_with_gap(gap_days=30)
    periods = calendar_period_ranges(df["date"], max_gap_days=7)
    assert len(periods) == 2


def test_phase2_daily_stride_caps_folds_with_multi_day_val():
    df = add_calendar_period_id(_panel(200), max_gap_days=7)
    folds = time_series_cv_folds_phase2_daily(
        df,
        15,
        min_train_fraction=0.25,
        phase2_val_days=7,
        min_train_rows=50,
        min_val_rows=4,
        fold_stride=5,
    )
    assert 1 <= len(folds) <= 15
    for _tr, va in folds:
        assert 1 <= va["date"].nunique() <= 7


def test_phase2_daily_val_is_short_horizon():
    df = add_run_period_id(_panel_with_gap(gap_days=30), max_gap_days=7)
    folds = time_series_cv_folds_phase2_daily(
        df,
        5,
        min_train_fraction=0.25,
        phase2_val_days=1,
        min_train_rows=50,
        min_val_rows=4,
        max_calendar_gap_days=7,
    )
    assert folds
    for _tr, va in folds:
        assert va["date"].nunique() == 1
        assert va["run_period_id"].nunique() == 1


def test_phase2_val_stays_within_campaign_version_run():
    early = _panel(60, rows_per_day=2, campaign_version=1)
    late = _panel(60, rows_per_day=2, campaign_version=2)
    late["date"] = late["date"] + pd.Timedelta(days=60)
    df = add_run_period_id(pd.concat([early, late], ignore_index=True), max_gap_days=7)
    folds = time_series_cv_folds_phase2_daily(
        df,
        20,
        min_train_fraction=0.2,
        phase2_val_days=7,
        min_train_rows=20,
        min_val_rows=4,
    )
    assert folds
    for _tr, va in folds:
        assert va["campaign_version"].nunique() == 1
        assert va["run_period_id"].nunique() == 1


def test_phase1_launch_train_before_period_start():
    df = _panel_with_gap(gap_days=30)
    folds = time_series_cv_folds_phase1_launch(
        df,
        3,
        min_train_fraction=0.25,
        phase1_launch_val_days=14,
        min_train_rows=50,
        min_val_rows=20,
        max_calendar_gap_days=7,
    )
    assert folds
    for tr, va in folds:
        assert tr["date"].max() < va["date"].min()
        assert va["date"].nunique() == 14


def test_phase1_launch_uses_pre_period_train_for_min_days():
    """Full-panel min_train_fraction must not block folds when pre-period history is shorter."""
    block1 = _panel(120, rows_per_day=4, campaign_version=1)
    block2 = _panel(120, rows_per_day=4, campaign_version=2)
    block2["date"] = block2["date"] + pd.Timedelta(days=120 + 30)
    df = add_run_period_id(pd.concat([block1, block2], ignore_index=True), max_gap_days=7)
    folds = time_series_cv_folds_phase1_launch(
        df,
        3,
        min_train_fraction=0.5,
        phase1_launch_val_days=14,
        min_train_rows=50,
        min_val_rows=20,
        max_calendar_gap_days=7,
    )
    assert folds, "period-2 launch fold should pass with 120 pre-period train days"


def test_phase2_skips_disallowed_match_types_and_excluded_regions():
    rows = []
    for d in range(60):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for seg, mt, ver in [
            ("USA / Broad", "Broad", 1),
            ("A / Phrase; Exact", "Phrase; Exact", 2),
            ("C / Broad", "Broad", 3),
        ]:
            rows.append(
                {
                    "date": day,
                    "segment": seg,
                    "region": seg.split(" / ")[0],
                    "match_types": mt,
                    "campaign_version": ver,
                    "daily_budget": 10.0,
                    "clicks": 1.0,
                }
            )
    df = add_run_period_id(pd.DataFrame(rows), max_gap_days=7)
    folds = time_series_cv_folds_phase2_daily(
        df,
        10,
        min_train_fraction=0.2,
        phase2_val_days=7,
        min_train_rows=30,
        min_val_rows=4,
        allowed_match_types=["Broad", "Phrase; Exact"],
        excluded_regions=["C"],
    )
    assert folds
    for _tr, va in folds:
        assert set(va["match_types"].unique()).issubset({"Broad", "Phrase; Exact"})
        assert "C" not in va["region"].unique()


def test_cross_validate_raises_on_fit_failure():
    from types import SimpleNamespace

    df = _panel(120)
    cfg = SimpleNamespace(
        model_policy=SimpleNamespace(
            validation=SimpleNamespace(
                min_train_days=0,
                min_train_fraction=0.5,
                min_val_days=21,
                min_train_rows=50,
                min_val_rows=20,
                cv_folds=3,
                cv_profile="phase2_daily",
                phase2_val_days=1,
                phase1_launch_val_days=14,
                report_phase1_cv=False,
                phase1_cv_for_selection=False,
                respect_campaign_periods=False,
                max_calendar_gap_days=7,
            )
        )
    )

    def bad_fit(tr, va, cfg, fc):
        raise ValueError("fit broke")

    with pytest.raises(CVFoldError, match="fit broke"):
        cross_validate_model(bad_fit, df, cfg, [], n_folds=3)
