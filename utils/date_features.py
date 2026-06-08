"""Temporal feature extraction for campaign-day panels."""

from __future__ import annotations

import datetime
from typing import Iterable

import holidays
import numpy as np
import pandas as pd

def course_start_dates(course: str) -> list[str]:
    from utils.campaign_config import load_config

    cfg = load_config(course)
    return list(getattr(cfg, "start_dates", None) or [])


def region_to_country_code(region: object) -> str:
    if pd.isna(region):
        return "US"
    s = str(region).upper()
    if "CAN" in s or s == "CA":
        return "CA"
    if "USA" in s or s == "US":
        return "US"
    return "US"


def get_holiday_calendars(country_codes: Iterable[str], years: list[int] | None = None) -> dict:
    if years is None:
        years = [datetime.date.today().year]
    calendars: dict = {}
    for code in set(country_codes):
        try:
            calendars[code] = holidays.country_holidays(code, years=years)
        except Exception:
            calendars[code] = None
    return calendars


def season_from_month(month: int) -> str:
    """Meteorological season (Northern Hemisphere) from calendar month 1–12."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def calculate_days_to_next(d: pd.Timestamp, course_start_dts: list[str] | None = None) -> float:
    if course_start_dts is None:
        course_start_dts = []
    if not course_start_dts:
        return np.nan
    starts = pd.to_datetime(course_start_dts).sort_values()
    diffs = [int((cs - d).days) for cs in starts if (cs - d).days >= 0]
    return float(min(diffs)) if diffs else np.nan


def add_month_cycle_features(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Smooth annual cycle (2 harmonics) to reduce month-dummy overfit risk."""
    if isinstance(dates, pd.DatetimeIndex):
        month = dates.month.astype(float)
        index = dates
    else:
        month = dates.dt.month.astype(float)
        index = dates.index
    angle = 2.0 * np.pi * (month - 1.0) / 12.0
    return pd.DataFrame(
        {
            "month_sin": np.sin(angle),
            "month_cos": np.cos(angle),
        },
        index=index,
    )


def add_calendar_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    region_col: str = "region",
    course: str,
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["day_of_week"] = out[date_col].dt.day_name()
    out["is_weekend"] = (out[date_col].dt.weekday >= 5).astype(int)
    out["season"] = out[date_col].dt.month.map(season_from_month)
    out["month"] = out[date_col].dt.strftime("%b")
    cycle = add_month_cycle_features(out[date_col])
    out["month_sin"] = cycle["month_sin"]
    out["month_cos"] = cycle["month_cos"]

    years = sorted({int(y) for y in out[date_col].dt.year.unique()})
    country_codes = [region_to_country_code(r) for r in out[region_col].unique()]
    holiday_cals = get_holiday_calendars(country_codes, years=years)
    out["_country_code"] = out[region_col].map(region_to_country_code)

    def _is_hol(row: pd.Series) -> int:
        cal = holiday_cals.get(row["_country_code"])
        if cal is None:
            return 0
        d = row[date_col]
        try:
            date_obj = d.date()
        except AttributeError:
            date_obj = d
        return int(date_obj in cal)

    out["is_public_holiday"] = out.apply(_is_hol, axis=1)
    starts = course_start_dates(course)
    out["days_to_next_course_start"] = out[date_col].apply(
        lambda d: calculate_days_to_next(pd.Timestamp(d), starts)
    )
    return out.drop(columns=["_country_code"], errors="ignore")


def calendar_vector_for_date(
    planning_date: pd.Timestamp,
    region: str,
    course: str,
    *,
    calendar_cols: list[str] | None = None,
) -> dict[str, object]:
    row = pd.DataFrame(
        [{"date": planning_date, "region": region}],
    )
    enriched = add_calendar_features(row, course=course)
    default_cols = [
        "day_of_week",
        "season",
        "month",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_public_holiday",
        "days_to_next_course_start",
    ]
    use_cols = calendar_cols or default_cols
    return {c: enriched.iloc[0][c] for c in use_cols if c in enriched.columns}
