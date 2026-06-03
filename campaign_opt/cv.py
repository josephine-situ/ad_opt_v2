"""Time-series cross-validation on campaign-day panel.

See docs/cross_validation.md for phase-1/phase-2 profiles, period splits, and config.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from campaign_opt.decisions import (
    parse_allowed_match_types,
    parse_excluded_regions,
    region_of_segment,
)

CVProfile = Literal["phase2_daily", "phase1_launch", "period_tail", "legacy_calendar"]

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


def calendar_period_ranges(
    dates: pd.Series | np.ndarray,
    *,
    max_gap_days: int = 7,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Contiguous active spans on the panel calendar.

    A gap of more than ``max_gap_days`` between consecutive active dates starts a new period
    (campaign off-air / no spend).
    """
    if max_gap_days < 1:
        raise ValueError("max_gap_days must be >= 1")
    ordered = pd.Series(pd.to_datetime(dates)).drop_duplicates().sort_values()
    if ordered.empty:
        return []

    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = pd.Timestamp(ordered.iloc[0])
    prev = start
    for d in ordered.iloc[1:]:
        d = pd.Timestamp(d)
        if (d - prev).days > max_gap_days:
            periods.append((start, prev))
            start = d
        prev = d
    periods.append((start, prev))
    return periods


def add_calendar_period_id(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    max_gap_days: int = 7,
    col: str = "calendar_period_id",
) -> pd.DataFrame:
    """Attach integer period id per row from :func:`calendar_period_ranges`."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    ranges = calendar_period_ranges(out[date_col], max_gap_days=max_gap_days)
    if not ranges:
        out[col] = np.nan
        return out

    period_by_date: dict[pd.Timestamp, int] = {}
    for pid, (start, end) in enumerate(ranges):
        mask = (out[date_col] >= start) & (out[date_col] <= end)
        for day in out.loc[mask, date_col].drop_duplicates():
            period_by_date[pd.Timestamp(day)] = pid
    out[col] = out[date_col].map(period_by_date)
    return out


def _version_periods_available(df: pd.DataFrame) -> bool:
    return "segment" in df.columns and "campaign_version" in df.columns


def add_run_period_id(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    segment_col: str = "segment",
    version_col: str = "campaign_version",
    max_gap_days: int = 7,
    col: str = "run_period_id",
) -> pd.DataFrame:
    """
    Attach ``run_period_id``: ``(segment, campaign_version)`` with gap splits inside each.

    Gaps ``> max_gap_days`` between active dates start a new run period (off-air within a version).
    """
    if not _version_periods_available(df):
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    run_ids = pd.Series(index=out.index, dtype="Int64")
    next_id = 0
    for (_seg, _ver), grp in out.groupby([segment_col, version_col], sort=False):
        dates = np.array(sorted(grp[date_col].unique()))
        for span_start, span_end in calendar_period_ranges(dates, max_gap_days=max_gap_days):
            mask = (
                (out[segment_col] == _seg)
                & (out[version_col] == _ver)
                & (out[date_col] >= span_start)
                & (out[date_col] <= span_end)
            )
            run_ids.loc[mask] = next_id
            next_id += 1
    out[col] = run_ids
    return out


def _iter_run_period_spans(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    segment_col: str = "segment",
    version_col: str = "campaign_version",
    run_period_col: str = "run_period_id",
    max_calendar_gap_days: int = 7,
) -> list[dict[str, Any]]:
    """Contiguous (segment, version, gap-split) spans with sorted active dates."""
    if run_period_col in df.columns and df[run_period_col].notna().any():
        spans: list[dict[str, Any]] = []
        for rid, grp in df.groupby(run_period_col, dropna=True):
            dates = np.array(sorted(pd.to_datetime(grp[date_col]).unique()))
            if len(dates) == 0:
                continue
            spans.append(
                {
                    "run_period_id": rid,
                    "segment": grp[segment_col].iloc[0],
                    "campaign_version": grp[version_col].iloc[0],
                    "dates": dates,
                    "start": pd.Timestamp(dates[0]),
                    "end": pd.Timestamp(dates[-1]),
                }
            )
        return sorted(spans, key=lambda s: (s["start"], str(s["segment"]), s["campaign_version"]))

    if not _version_periods_available(df):
        return []

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    spans: list[dict[str, Any]] = []
    next_id = 0
    for (_seg, _ver), grp in work.groupby([segment_col, version_col], sort=False):
        dates = np.array(sorted(grp[date_col].unique()))
        for span_start, span_end in calendar_period_ranges(
            dates, max_gap_days=max_calendar_gap_days
        ):
            span_dates = dates[(dates >= span_start) & (dates <= span_end)]
            if len(span_dates) == 0:
                continue
            spans.append(
                {
                    "run_period_id": next_id,
                    "segment": _seg,
                    "campaign_version": _ver,
                    "dates": span_dates,
                    "start": pd.Timestamp(span_dates[0]),
                    "end": pd.Timestamp(span_dates[-1]),
                }
            )
            next_id += 1
    return spans


def _cv_constraints_from_config(
    config: Any | None,
) -> tuple[list[str] | None, list[str]]:
    if config is None:
        return None, []
    constraints = getattr(config, "constraints", None) or {}
    return parse_allowed_match_types(constraints), parse_excluded_regions(constraints)


def _cv_validation_row_mask(
    df: pd.DataFrame,
    *,
    allowed_match_types: list[str] | None,
    excluded_regions: list[str] | None,
    segment_col: str = "segment",
) -> pd.Series:
    """Rows eligible for CV validation per ``constraints`` (match types + regions)."""
    mask = pd.Series(True, index=df.index)
    if allowed_match_types:
        if "match_types" not in df.columns:
            raise ValueError("CV match-type filter requires a match_types column")
        mask &= df["match_types"].isin(allowed_match_types)
    if excluded_regions:
        excluded = set(excluded_regions)
        if "region" in df.columns:
            mask &= ~df["region"].isin(excluded)
        elif segment_col in df.columns:
            mask &= ~df[segment_col].map(region_of_segment).isin(excluded)
    return mask


def _filter_cv_validation_rows(
    val_fold: pd.DataFrame,
    *,
    allowed_match_types: list[str] | None,
    excluded_regions: list[str] | None,
    segment_col: str = "segment",
) -> pd.DataFrame:
    if not allowed_match_types and not excluded_regions:
        return val_fold
    mask = _cv_validation_row_mask(
        val_fold,
        allowed_match_types=allowed_match_types,
        excluded_regions=excluded_regions,
        segment_col=segment_col,
    )
    return val_fold.loc[mask].copy()


def _span_allowed_for_cv(
    span: dict[str, Any],
    work: pd.DataFrame,
    *,
    segment_col: str,
    allowed_match_types: list[str] | None,
    excluded_regions: list[str] | None,
) -> bool:
    """Whether this run period's segment is in the optimization scope for CV val."""
    sub = work.loc[work[segment_col] == span["segment"]]
    if sub.empty:
        return False
    if allowed_match_types:
        if "match_types" not in sub.columns:
            return False
        if sub["match_types"].iloc[0] not in allowed_match_types:
            return False
    if excluded_regions:
        reg = (
            sub["region"].iloc[0]
            if "region" in sub.columns
            else region_of_segment(str(span["segment"]))
        )
        if reg in excluded_regions:
            return False
    return True


def _first_run_period_ids_per_segment(spans: list[dict[str, Any]]) -> set[Any]:
    seen: set[str] = set()
    first_ids: set[Any] = set()
    for span in sorted(spans, key=lambda s: (str(s["segment"]), s["start"])):
        seg = str(span["segment"])
        if seg in seen:
            continue
        seen.add(seg)
        first_ids.add(span["run_period_id"])
    return first_ids


def _fold_passes_sizing(
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    date_col: str,
    *,
    min_train_days: int,
    min_val_days: int,
    min_train_rows: int,
    min_val_rows: int,
) -> bool:
    return (
        train_fold[date_col].nunique() >= min_train_days
        and val_fold[date_col].nunique() >= min_val_days
        and len(train_fold) >= min_train_rows
        and len(val_fold) >= min_val_rows
    )


def _take_recent_folds(
    candidates: list[tuple],
    n_folds: int,
    *,
    stride: int = 1,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Keep up to ``n_folds`` distinct folds with the latest validation end dates.

    Each candidate is ``(train, val, val_end)`` or ``(train, val, val_end, dedupe_key)``.
    ``stride > 1`` subsamples the candidate timeline (every Nth val end from the recent
    tail) so phase-2 CV can average more diverse days without scoring every calendar day.
    """
    if not candidates or n_folds < 1:
        return []
    stride = max(1, int(stride))
    candidates.sort(key=lambda x: x[2])
    seen: set[Any] = set()
    deduped: list[tuple] = []
    for item in reversed(candidates):
        val_end = item[2]
        dedupe_key = item[3] if len(item) > 3 else val_end
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(item)
    deduped.reverse()
    tail = deduped[-(n_folds * stride) :]
    subsampled = tail[::stride][-n_folds:]
    return [(tr, va) for tr, va, *_ in subsampled]


def time_series_cv_folds_phase2_daily(
    df: pd.DataFrame,
    n_folds: int,
    date_col: str = "date",
    *,
    min_train_days: int = 0,
    min_train_fraction: float = 0.5,
    phase2_val_days: int = 1,
    min_train_rows: int = 50,
    min_val_rows: int = 20,
    max_calendar_gap_days: int = 7,
    fold_stride: int = 1,
    segment_col: str = "segment",
    run_period_col: str = "run_period_id",
    allowed_match_types: list[str] | None = None,
    excluded_regions: list[str] | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Phase 2 (daily budget): short forward windows inside one run period.

    Primary boundaries: ``(segment, campaign_version)`` with gap splits (``run_period_id``).
    Train on all rows with ``date`` strictly before validation start; validate on the next
    ``phase2_val_days`` within the same run period (fixed set/budget, in-run forecasting).
    Validation rows are limited to ``allowed_match_types`` / non-``excluded_regions``.

    Falls back to calendar-gap periods when ``campaign_version`` is absent.
    """
    if phase2_val_days < 1:
        raise ValueError("phase2_val_days must be >= 1")
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    all_dates = np.array(sorted(work[date_col].unique()))
    min_train = effective_min_train_days(
        len(all_dates), min_train_days=min_train_days, min_train_fraction=min_train_fraction
    )
    min_val_rows_eff = _scaled_min_val_rows(min_val_rows, phase2_val_days)

    candidates: list[tuple] = []
    use_version = _version_periods_available(work) or run_period_col in work.columns
    if use_version:
        if run_period_col not in work.columns:
            work = add_run_period_id(work, date_col=date_col, max_gap_days=max_calendar_gap_days)
        for span in _iter_run_period_spans(
            work,
            date_col=date_col,
            segment_col=segment_col,
            run_period_col=run_period_col,
            max_calendar_gap_days=max_calendar_gap_days,
        ):
            if not _span_allowed_for_cv(
                span,
                work,
                segment_col=segment_col,
                allowed_match_types=allowed_match_types,
                excluded_regions=excluded_regions,
            ):
                continue
            period_dates = span["dates"]
            rid = span["run_period_id"]
            if len(period_dates) < phase2_val_days:
                continue
            for val_start_idx in range(0, len(period_dates) - phase2_val_days + 1):
                val_start = period_dates[val_start_idx]
                val_end = period_dates[val_start_idx + phase2_val_days - 1]
                train_fold = work[work[date_col] < val_start]
                val_fold = work[
                    (work[run_period_col] == rid)
                    & (work[date_col] >= val_start)
                    & (work[date_col] <= val_end)
                ]
                val_fold = _filter_cv_validation_rows(
                    val_fold,
                    allowed_match_types=allowed_match_types,
                    excluded_regions=excluded_regions,
                    segment_col=segment_col,
                )
                if val_fold.empty:
                    continue
                if _fold_passes_sizing(
                    train_fold,
                    val_fold,
                    date_col,
                    min_train_days=min_train,
                    min_val_days=phase2_val_days,
                    min_train_rows=min_train_rows,
                    min_val_rows=min_val_rows_eff,
                ):
                    candidates.append(
                        (
                            train_fold,
                            val_fold,
                            pd.Timestamp(val_end),
                            (pd.Timestamp(val_end), rid),
                        )
                    )
    else:
        for p_start, p_end in calendar_period_ranges(
            all_dates, max_gap_days=max_calendar_gap_days
        ):
            period_dates = all_dates[(all_dates >= p_start) & (all_dates <= p_end)]
            if len(period_dates) < phase2_val_days:
                continue
            for val_start_idx in range(0, len(period_dates) - phase2_val_days + 1):
                val_start = period_dates[val_start_idx]
                val_end = period_dates[val_start_idx + phase2_val_days - 1]
                train_fold = work[work[date_col] < val_start]
                val_fold = work[
                    (work[date_col] >= val_start) & (work[date_col] <= val_end)
                ]
                val_fold = _filter_cv_validation_rows(
                    val_fold,
                    allowed_match_types=allowed_match_types,
                    excluded_regions=excluded_regions,
                    segment_col=segment_col,
                )
                if val_fold.empty:
                    continue
                if _fold_passes_sizing(
                    train_fold,
                    val_fold,
                    date_col,
                    min_train_days=min_train,
                    min_val_days=phase2_val_days,
                    min_train_rows=min_train_rows,
                    min_val_rows=min_val_rows_eff,
                ):
                    candidates.append((train_fold, val_fold, pd.Timestamp(val_end)))

    return _take_recent_folds(candidates, n_folds, stride=fold_stride)


def time_series_cv_folds_phase1_launch(
    df: pd.DataFrame,
    n_folds: int,
    date_col: str = "date",
    *,
    min_train_days: int = 0,
    min_train_fraction: float = 0.5,
    phase1_launch_val_days: int = 14,
    min_train_rows: int = 50,
    min_val_rows: int = 20,
    max_calendar_gap_days: int = 7,
    segment_col: str = "segment",
    run_period_col: str = "run_period_id",
    allowed_match_types: list[str] | None = None,
    excluded_regions: list[str] | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Phase 1 (run start): train on history before a run begins; validate on its opening window.

    Primary boundaries: new ``(segment, campaign_version)`` run periods (gap-split).
    Validation rows respect ``allowed_match_types`` / ``excluded_regions``.
    Falls back to calendar-gap periods when ``campaign_version`` is absent.
    """
    if phase1_launch_val_days < 1:
        raise ValueError("phase1_launch_val_days must be >= 1")
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    all_dates = np.array(sorted(work[date_col].unique()))
    min_val_rows_eff = _scaled_min_val_rows(min_val_rows, phase1_launch_val_days)

    candidates: list[tuple] = []
    use_version = _version_periods_available(work) or run_period_col in work.columns
    if use_version:
        if run_period_col not in work.columns:
            work = add_run_period_id(work, date_col=date_col, max_gap_days=max_calendar_gap_days)
        spans = _iter_run_period_spans(
            work,
            date_col=date_col,
            segment_col=segment_col,
            run_period_col=run_period_col,
            max_calendar_gap_days=max_calendar_gap_days,
        )
        skip_first = _first_run_period_ids_per_segment(spans)
        for span in spans:
            if span["run_period_id"] in skip_first:
                continue
            if not _span_allowed_for_cv(
                span,
                work,
                segment_col=segment_col,
                allowed_match_types=allowed_match_types,
                excluded_regions=excluded_regions,
            ):
                continue
            period_dates = span["dates"]
            if len(period_dates) < phase1_launch_val_days:
                continue
            v_start = span["start"]
            val_end = period_dates[phase1_launch_val_days - 1]
            rid = span["run_period_id"]
            train_fold = work[work[date_col] < v_start]
            val_fold = work[
                (work[run_period_col] == rid)
                & (work[date_col] >= v_start)
                & (work[date_col] <= val_end)
            ]
            val_fold = _filter_cv_validation_rows(
                val_fold,
                allowed_match_types=allowed_match_types,
                excluded_regions=excluded_regions,
                segment_col=segment_col,
            )
            if val_fold.empty:
                continue
            seg_pre = train_fold[train_fold[segment_col] == span["segment"]]
            n_pre_period_days = seg_pre[date_col].nunique()
            if n_pre_period_days < 1:
                continue
            min_train_fold = effective_min_train_days(
                n_pre_period_days,
                min_train_days=min_train_days,
                min_train_fraction=min_train_fraction,
            )
            if _fold_passes_sizing(
                train_fold,
                val_fold,
                date_col,
                min_train_days=min_train_fold,
                min_val_days=phase1_launch_val_days,
                min_train_rows=min_train_rows,
                min_val_rows=min_val_rows_eff,
            ):
                candidates.append(
                    (
                        train_fold,
                        val_fold,
                        pd.Timestamp(val_end),
                        (pd.Timestamp(val_end), rid),
                    )
                )
    else:
        for period_idx, (p_start, p_end) in enumerate(
            calendar_period_ranges(all_dates, max_gap_days=max_calendar_gap_days)
        ):
            if period_idx == 0:
                continue
            period_dates = all_dates[(all_dates >= p_start) & (all_dates <= p_end)]
            if len(period_dates) < phase1_launch_val_days:
                continue
            val_end = period_dates[phase1_launch_val_days - 1]
            train_fold = work[work[date_col] < p_start]
            val_fold = work[(work[date_col] >= p_start) & (work[date_col] <= val_end)]
            val_fold = _filter_cv_validation_rows(
                val_fold,
                allowed_match_types=allowed_match_types,
                excluded_regions=excluded_regions,
                segment_col=segment_col,
            )
            if val_fold.empty:
                continue
            n_pre_period_days = train_fold[date_col].nunique()
            if n_pre_period_days < 1:
                continue
            min_train_fold = effective_min_train_days(
                n_pre_period_days,
                min_train_days=min_train_days,
                min_train_fraction=min_train_fraction,
            )
            if _fold_passes_sizing(
                train_fold,
                val_fold,
                date_col,
                min_train_days=min_train_fold,
                min_val_days=phase1_launch_val_days,
                min_train_rows=min_train_rows,
                min_val_rows=min_val_rows_eff,
            ):
                candidates.append((train_fold, val_fold, pd.Timestamp(val_end)))

    return _take_recent_folds(candidates, n_folds)


def time_series_cv_folds(
    df: pd.DataFrame,
    n_folds: int,
    date_col: str = "date",
    *,
    cv_profile: CVProfile = "phase2_daily",
    min_train_days: int = 0,
    min_train_fraction: float = 0.5,
    min_val_days: int = 21,
    phase2_val_days: int = 7,
    phase1_launch_val_days: int = 14,
    min_train_rows: int = 50,
    min_val_rows: int = 20,
    respect_campaign_periods: bool = False,
    max_calendar_gap_days: int = 7,
    fold_stride: int = 1,
    config: Any | None = None,
    allowed_match_types: list[str] | None = None,
    excluded_regions: list[str] | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Dispatch fold construction by ``cv_profile``."""
    if config is not None and allowed_match_types is None and excluded_regions is None:
        allowed_match_types, excluded_regions = _cv_constraints_from_config(config)
    cv_kw = {
        "allowed_match_types": allowed_match_types,
        "excluded_regions": excluded_regions,
    }
    if cv_profile == "phase2_daily":
        n_phase2 = phase2_cv_fold_count(config) if config is not None else n_folds
        stride = fold_stride
        if config is not None:
            stride = getattr(config.model_policy.validation, "phase2_fold_stride", fold_stride)
        return time_series_cv_folds_phase2_daily(
            df,
            n_phase2,
            date_col=date_col,
            min_train_days=min_train_days,
            min_train_fraction=min_train_fraction,
            phase2_val_days=phase2_val_days,
            min_train_rows=min_train_rows,
            min_val_rows=min_val_rows,
            max_calendar_gap_days=max_calendar_gap_days,
            fold_stride=stride,
            **cv_kw,
        )
    if cv_profile == "phase1_launch":
        return time_series_cv_folds_phase1_launch(
            df,
            n_folds,
            date_col=date_col,
            min_train_days=min_train_days,
            min_train_fraction=min_train_fraction,
            phase1_launch_val_days=phase1_launch_val_days,
            min_train_rows=min_train_rows,
            min_val_rows=min_val_rows,
            max_calendar_gap_days=max_calendar_gap_days,
            **cv_kw,
        )
    if cv_profile == "period_tail" or (
        cv_profile == "legacy_calendar" and respect_campaign_periods
    ):
        return time_series_cv_folds_by_period(
            df,
            n_folds,
            date_col=date_col,
            min_train_days=min_train_days,
            min_train_fraction=min_train_fraction,
            min_val_days=min_val_days,
            min_train_rows=min_train_rows,
            min_val_rows=min_val_rows,
            max_calendar_gap_days=max_calendar_gap_days,
            **cv_kw,
        )

    # legacy_calendar: expanding windows across the full train calendar (may span gaps).
    dates = np.array(sorted(pd.to_datetime(df[date_col]).unique()))
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
        val_fold = _filter_cv_validation_rows(
            val_fold,
            allowed_match_types=allowed_match_types,
            excluded_regions=excluded_regions,
        )
        if val_fold.empty:
            continue
        if _fold_passes_sizing(
            train_fold,
            val_fold,
            date_col,
            min_train_days=min_train,
            min_val_days=min_val_days,
            min_train_rows=min_train_rows,
            min_val_rows=min_val_rows,
        ):
            folds.append((train_fold, val_fold))
    return folds


def time_series_cv_folds_by_period(
    df: pd.DataFrame,
    n_folds: int,
    date_col: str = "date",
    *,
    min_train_days: int = 0,
    min_train_fraction: float = 0.5,
    min_val_days: int = 21,
    min_train_rows: int = 50,
    min_val_rows: int = 20,
    max_calendar_gap_days: int = 7,
    allowed_match_types: list[str] | None = None,
    excluded_regions: list[str] | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Legacy period-tail windows (last ``min_val_days`` inside each active span)."""
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    all_dates = np.array(sorted(work[date_col].unique()))
    min_train = effective_min_train_days(
        len(all_dates), min_train_days=min_train_days, min_train_fraction=min_train_fraction
    )

    period_ranges = calendar_period_ranges(all_dates, max_gap_days=max_calendar_gap_days)
    candidates: list[tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]] = []

    for p_start, p_end in period_ranges:
        period_dates = all_dates[(all_dates >= p_start) & (all_dates <= p_end)]
        if len(period_dates) < min_val_days:
            continue

        usable = len(period_dates) - min_val_days
        n_val = max(min_val_days, usable // max(n_folds, 1))
        n_chunks = min(n_folds, max(1, usable // n_val)) if usable > 0 else 1

        for chunk in range(n_chunks):
            val_start_idx = len(period_dates) - min_val_days - chunk * n_val
            if val_start_idx < 0:
                continue
            val_start = period_dates[val_start_idx]
            val_end = period_dates[-1] if chunk == 0 else period_dates[
                min(val_start_idx + n_val - 1, len(period_dates) - 1)
            ]
            if val_end < val_start:
                val_end = period_dates[-1]

            train_fold = work[work[date_col] < val_start]
            val_fold = work[(work[date_col] >= val_start) & (work[date_col] <= val_end)]
            val_fold = _filter_cv_validation_rows(
                val_fold,
                allowed_match_types=allowed_match_types,
                excluded_regions=excluded_regions,
            )
            if val_fold.empty:
                continue
            if _fold_passes_sizing(
                train_fold,
                val_fold,
                date_col,
                min_train_days=min_train,
                min_val_days=min_val_days,
                min_train_rows=min_train_rows,
                min_val_rows=min_val_rows,
            ):
                candidates.append((train_fold, val_fold, pd.Timestamp(val_end)))

    return _take_recent_folds(candidates, n_folds)


def phase2_cv_fold_count(config) -> int:
    """Number of phase-2 CV folds (may exceed ``cv_folds`` for smoother metrics)."""
    val = config.model_policy.validation
    if getattr(val, "phase2_cv_folds", None) is not None:
        return int(val.phase2_cv_folds)
    return int(getattr(val, "cv_folds", 3))


def selection_cv_profile(config) -> CVProfile:
    """Profile used for tournament winner, hyperparameter tuning, and ensemble weights."""
    val = config.model_policy.validation
    profile = getattr(val, "cv_profile", "phase2_daily") or "phase2_daily"
    if profile in ("phase2_daily", "period_tail", "legacy_calendar"):
        return profile  # type: ignore[return-value]
    return "phase2_daily"


def _scaled_min_val_rows(min_val_rows: int, val_window_days: int, *, ref_val_days: int = 21) -> int:
    """Scale row floor when the validation window is shorter than ``ref_val_days``."""
    scale = val_window_days / max(ref_val_days, 1)
    return max(1, int(math.ceil(min_val_rows * scale)))


def _effective_min_val_rows(
    val,
    *,
    profile: CVProfile,
    phase2_val_days: int,
    phase1_launch_val_days: int = 14,
) -> int:
    """Scale row minimum when the validation window is shorter than ``min_val_days``."""
    if profile == "phase2_daily":
        return _scaled_min_val_rows(val.min_val_rows, phase2_val_days, ref_val_days=val.min_val_days)
    if profile == "phase1_launch":
        return _scaled_min_val_rows(
            val.min_val_rows, phase1_launch_val_days, ref_val_days=val.min_val_days
        )
    return val.min_val_rows


def _validation_kw(config, *, profile: CVProfile | None = None) -> dict[str, Any]:
    val = config.model_policy.validation
    prof = profile or selection_cv_profile(config)
    phase2_days = getattr(val, "phase2_val_days", 7)
    phase1_days = getattr(val, "phase1_launch_val_days", 14)
    allowed, excluded = _cv_constraints_from_config(config)
    return {
        "cv_profile": prof,
        "n_folds": phase2_cv_fold_count(config) if prof == "phase2_daily" else val.cv_folds,
        "min_train_days": val.min_train_days,
        "min_train_fraction": val.min_train_fraction,
        "min_val_days": val.min_val_days,
        "phase2_val_days": phase2_days,
        "phase1_launch_val_days": phase1_days,
        "min_train_rows": val.min_train_rows,
        "min_val_rows": _effective_min_val_rows(
            val,
            profile=prof,
            phase2_val_days=phase2_days,
            phase1_launch_val_days=phase1_days,
        ),
        "respect_campaign_periods": val.respect_campaign_periods,
        "max_calendar_gap_days": val.max_calendar_gap_days,
        "fold_stride": getattr(val, "phase2_fold_stride", 30),
        "allowed_match_types": allowed,
        "excluded_regions": excluded,
        "config": config,
    }


def _fold_context(train_fold: pd.DataFrame, val_fold: pd.DataFrame, date_col: str = "date") -> str:
    tr_d = pd.to_datetime(train_fold[date_col])
    va_d = pd.to_datetime(val_fold[date_col])
    ctx = (
        f"train {tr_d.min().date()}..{tr_d.max().date()} "
        f"({tr_d.nunique()} days, {len(train_fold)} rows); "
        f"val {va_d.min().date()}..{va_d.max().date()} "
        f"({va_d.nunique()} days, {len(val_fold)} rows)"
    )
    if "run_period_id" in val_fold.columns and val_fold["run_period_id"].notna().any():
        rid = int(val_fold["run_period_id"].iloc[0])
        ctx += f"; run_period_id={rid}"
        if "campaign_version" in val_fold.columns:
            ctx += f" v={val_fold['campaign_version'].iloc[0]}"
        if "segment" in val_fold.columns:
            ctx += f" seg={val_fold['segment'].iloc[0]}"
    elif "calendar_period_id" in val_fold.columns:
        ctx += f"; val period_id={int(val_fold['calendar_period_id'].iloc[0])}"
    return ctx


_cv_fold_log_cache: set[tuple] = set()


def print_cv_fold_dates(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    *,
    profile: str | None = None,
    prefix: str = "CV",
    date_col: str = "date",
    dedupe_key: tuple | None = None,
) -> None:
    """Print train/val date ranges for each fold (once per ``dedupe_key`` when set)."""
    if dedupe_key is not None:
        if dedupe_key in _cv_fold_log_cache:
            return
        _cv_fold_log_cache.add(dedupe_key)
    prof = f" profile={profile!r}" if profile else ""
    if not folds:
        print(f"{prefix}: no folds{prof}")
        return
    print(f"{prefix}: {len(folds)} fold(s){prof}")
    for fold_idx, (tr_fold, va_fold) in enumerate(folds, start=1):
        print(f"  fold {fold_idx}/{len(folds)}: {_fold_context(tr_fold, va_fold, date_col)}")


def _run_folds(
    fit_fn: Callable[..., Any],
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    config,
    feature_cols: list[str],
    date_col: str,
) -> dict[str, float]:
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


def cross_validate_model(
    fit_fn: Callable[..., Any],
    train: pd.DataFrame,
    config,
    feature_cols: list[str],
    *,
    n_folds: int = 5,
    date_col: str = "date",
    cv_profile: CVProfile | None = None,
) -> dict[str, float]:
    """
    Run ``fit_fn`` on each CV fold; return mean level-scale metrics.

    Default profile is ``selection_cv_profile(config)`` (phase 2 daily for production).
    Pass ``cv_profile='phase1_launch'`` for run-start reporting only.
    """
    profile = cv_profile or selection_cv_profile(config)
    kw = _validation_kw(config, profile=profile)
    n_folds_eff = int(kw.pop("n_folds", n_folds))
    kw.pop("config", None)
    folds = time_series_cv_folds(
        train, n_folds_eff, date_col=date_col, config=config, **kw
    )

    if not folds:
        warnings.warn(
            f"No CV folds for profile={profile!r}; using single internal holdout "
            "(last 20% of train dates).",
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
        profile=profile,
        prefix="CV fold schedule",
        date_col=date_col,
        dedupe_key=(
            profile,
            tr_dates.min(),
            tr_dates.max(),
            len(folds),
            pd.to_datetime(folds[0][1][date_col]).min() if folds else None,
            pd.to_datetime(folds[-1][1][date_col]).max() if folds else None,
        ),
    )

    out = _run_folds(fit_fn, folds, config, feature_cols, date_col)
    out["cv_profile"] = profile
    return out


def cross_validate_phase1_launch(
    fit_fn: Callable[..., Any],
    train: pd.DataFrame,
    config,
    feature_cols: list[str],
    *,
    n_folds: int = 5,
    date_col: str = "date",
) -> dict[str, float] | None:
    """Phase-1 launch-window CV metrics (reporting only unless configured otherwise)."""
    val = config.model_policy.validation
    if not getattr(val, "report_phase1_cv", True):
        return None
    metrics = cross_validate_model(
        fit_fn,
        train,
        config,
        feature_cols,
        n_folds=n_folds,
        date_col=date_col,
        cv_profile="phase1_launch",
    )
    return {
        "phase1_cv_rmse_levels": metrics["cv_rmse_levels"],
        "phase1_cv_r2_levels": metrics["cv_r2_levels"],
        "phase1_cv_mae_levels": metrics["cv_mae_levels"],
        "phase1_cv_n_folds": metrics["cv_n_folds"],
    }
