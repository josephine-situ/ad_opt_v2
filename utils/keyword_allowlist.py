"""Enrollment-keyword allowlist for keyword-set construction and optimization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.campaign_features import MATCH_TYPE_LIST_COLS, resolve_positive_keyword_column
from utils.data_processing import (
    join_keyword_field,
    normalize_keyword,
    split_keyword_field,
)
from utils.paths import require_enrollment_allowlist

_KEYWORD_LIST_COLS = (
    "positive_keywords",
    "unique_keywords",
    *MATCH_TYPE_LIST_COLS.values(),
)


def _read_xlsx_first_sheet(path: Path) -> list[list[str]]:
    df = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")
    rows: list[list[str]] = []
    for _, row in df.iterrows():
        vals: list[str] = []
        for value in row.tolist():
            if value is None or (isinstance(value, float) and pd.isna(value)):
                vals.append("")
            else:
                vals.append(str(value))
        rows.append(vals)
    return rows


def load_enrollment_keyword_allowlist_ordered(course: str) -> list[str]:
    """
    Keywords from ``*Keywords*Enrollments*.xlsx`` in priority order.

    When a numeric enrollment column is present, sorts by enrollment descending;
    otherwise preserves spreadsheet row order.
    """
    path = require_enrollment_allowlist(course)
    rows = _read_xlsx_first_sheet(path)
    if not rows:
        return []

    entries: list[tuple[str, float]] = []
    for row in rows[1:]:
        if not row or not str(row[0]).strip():
            continue
        kw = normalize_keyword(row[0])
        if not kw:
            continue
        enroll = 0.0
        if len(row) > 1 and str(row[1]).strip():
            try:
                enroll = float(str(row[1]).replace(",", ""))
            except ValueError:
                enroll = 0.0
        entries.append((kw, enroll))

    if any(enroll > 0 for _, enroll in entries):
        entries.sort(key=lambda item: (-item[1], item[0]))

    ordered: list[str] = []
    seen: set[str] = set()
    for kw, _ in entries:
        if kw not in seen:
            seen.add(kw)
            ordered.append(kw)
    return ordered


def load_enrollment_keyword_allowlist(course: str) -> set[str]:
    """Keywords from the enrollment allowlist under ``sys_think/data/gkp/``."""
    return set(load_enrollment_keyword_allowlist_ordered(course))


def allowlist_keys_in_order(
    allowlist: set[str],
    allowlist_order: list[str] | None,
) -> list[str]:
    """Return allowlist keywords in priority order, with remaining keys appended alphabetically."""
    keys_in_order: list[str] = []
    seen: set[str] = set()
    for key in allowlist_order or sorted(allowlist):
        if key in allowlist and key not in seen:
            seen.add(key)
            keys_in_order.append(key)
    for key in sorted(allowlist):
        if key not in seen:
            seen.add(key)
            keys_in_order.append(key)
    return keys_in_order


def enrollment_allowlist_keywords(
    allowlist: set[str],
    kw_day: pd.DataFrame,
    segment_row: pd.Series,
    *,
    allowlist_order: list[str] | None = None,
) -> list[str]:
    """
    Full enrollment allowlist as a keyword list for one segment.

    Uses panel spelling when a keyword appears in the segment's region for that match type
    (from any campaign configuration in the region); otherwise the normalized allowlist text.
    """
    if not allowlist:
        return []

    region = segment_row.get("region")
    match_types = str(segment_row.get("match_types", ""))
    allowed_mt = {
        m.strip().title()
        for m in match_types.replace(";", " ").split()
        if m.strip()
    }

    canonical: dict[str, str] = {}
    if not kw_day.empty and pd.notna(region) and "keyword" in kw_day.columns:
        for mt in allowed_mt:
            sub = kw_day[(kw_day["region"] == region) & (kw_day["match_type"] == mt)]
            for kw in sub["keyword"].dropna().astype(str):
                key = normalize_keyword(kw)
                if key in allowlist:
                    canonical[key] = kw

    keys_in_order = allowlist_keys_in_order(allowlist, allowlist_order)

    for key in keys_in_order:
        canonical.setdefault(key, key)
    return [canonical[k] for k in keys_in_order]


def filter_keyword_list(keywords: list[str], allowlist: set[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for kw in keywords:
        key = normalize_keyword(kw)
        if key in allowlist and key not in seen:
            seen.add(key)
            out.append(kw or key)
    return out


def filter_keyword_sets_dataframe(
    keyword_sets: pd.DataFrame,
    allowlist: set[str],
    *,
    drop_empty: bool = True,
) -> pd.DataFrame:
    """Keep only allowlisted keywords in set list columns; optionally drop empty sets."""
    out = keyword_sets.copy()
    list_cols = [c for c in _KEYWORD_LIST_COLS if c in out.columns]

    for idx, row in out.iterrows():
        updated: dict[str, str] = {}
        for col in list_cols:
            filtered = filter_keyword_list(split_keyword_field(row.get(col)), allowlist)
            updated[col] = join_keyword_field(filtered)
        for col, value in updated.items():
            out.at[idx, col] = value

        if all(not str(updated.get(c, "")).strip() for c in list_cols):
            if "positive_keywords" in out.columns:
                out.at[idx, "positive_keywords"] = ""
            continue

        match_cols = [MATCH_TYPE_LIST_COLS[mt] for mt in MATCH_TYPE_LIST_COLS if MATCH_TYPE_LIST_COLS[mt] in out.columns]
        if match_cols:
            positive: set[str] = set()
            for col in match_cols:
                positive.update(split_keyword_field(out.at[idx, col]))
            joined = join_keyword_field(sorted(positive))
            if "positive_keywords" in out.columns:
                out.at[idx, "positive_keywords"] = joined
            if "unique_keywords" in out.columns:
                out.at[idx, "unique_keywords"] = joined

    out, _ = resolve_positive_keyword_column(out)
    positive_col = "positive_keywords"
    if drop_empty:
        keep = out[positive_col].fillna("").astype(str).str.strip().astype(bool)
        out = out[keep].copy()
    return out.reset_index(drop=True)


def apply_allowlist_to_keyword_sets(
    keyword_sets: pd.DataFrame,
    course: str,
    *,
    allowlist: set[str] | None = None,
) -> pd.DataFrame:
    if allowlist is None:
        allowlist = load_enrollment_keyword_allowlist(course)
    return filter_keyword_sets_dataframe(keyword_sets, allowlist)
