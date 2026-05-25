"""Temporal feature extraction for campaign-day panels."""

from __future__ import annotations

import datetime
from typing import Iterable

import holidays
import numpy as np
import pandas as pd

from config import COURSE_CONFIG


def course_start_dates(course: str) -> list[str]:
    cfg = COURSE_CONFIG.get(course, {})
    return list(cfg.get("start_dates") or [])


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
            calendars[code] = holidays.CountryHoliday(code, years=years)
        except Exception:
            calendars[code] = None
    return calendars


def calculate_days_to_next(d: pd.Timestamp, course_start_dts: list[str] | None = None) -> float:
    if course_start_dts is None:
        course_start_dts = []
    if not course_start_dts:
        return np.nan
    starts = pd.to_datetime(course_start_dts).sort_values()
    diffs = [int((cs - d).days) for cs in starts if (cs - d).days >= 0]
    return float(min(diffs)) if diffs else np.nan


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
    out["month"] = out[date_col].dt.strftime("%b")

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
) -> dict[str, object]:
    row = pd.DataFrame(
        [{"date": planning_date, "region": region}],
    )
    enriched = add_calendar_features(row, course=course)
    cols = [
        "day_of_week",
        "month",
        "is_weekend",
        "is_public_holiday",
        "days_to_next_course_start",
    ]
    return {c: enriched.iloc[0][c] for c in cols}
