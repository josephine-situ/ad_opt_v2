"""Enrollment-keyword allowlist for keyword-set construction and optimization."""

from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from utils.campaign_features import MATCH_TYPE_LIST_COLS, resolve_positive_keyword_column

_XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_ENROLLMENT_FILE_GLOB = "*Keywords*Enrollments*.xlsx"
_KEYWORD_LIST_COLS = (
    "positive_keywords",
    "unique_keywords",
    *MATCH_TYPE_LIST_COLS.values(),
)


def clean_keyword_text(keyword: str) -> str:
    """Strip bracket/quote artifacts and collapse internal whitespace."""
    s = str(keyword).replace('"', "").replace("[", "").replace("]", "")
    return " ".join(s.split()).strip()


def normalize_keyword(keyword: str) -> str:
    return clean_keyword_text(keyword).lower()


def _read_xlsx_first_sheet(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(".//m:si", _XLSX_NS):
                shared.append("".join((t.text or "") for t in si.findall(".//m:t", _XLSX_NS)))
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        for row in sheet.findall(".//m:sheetData/m:row", _XLSX_NS):
            vals: list[str] = []
            for cell in row.findall("m:c", _XLSX_NS):
                ref = cell.get("t")
                value = cell.find("m:v", _XLSX_NS)
                if value is None:
                    vals.append("")
                elif ref == "s":
                    vals.append(shared[int(value.text)])
                else:
                    vals.append(value.text or "")
            rows.append(vals)
    return rows


def should_refresh_keyword_candidates(course: str, candidates_path: Path) -> bool:
    """True when candidates are missing or older than the enrollment allowlist file."""
    allow_path = enrollment_keyword_allowlist_path(course)
    if allow_path is None:
        return not candidates_path.exists()
    if not candidates_path.exists():
        return True
    return allow_path.stat().st_mtime > candidates_path.stat().st_mtime


def enrollment_keyword_allowlist_path(course: str) -> Path | None:
    gkp_dir = Path("data") / course / "gkp"
    if not gkp_dir.is_dir():
        return None
    matches = sorted(gkp_dir.glob(_ENROLLMENT_FILE_GLOB), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def load_enrollment_keyword_allowlist_ordered(course: str) -> list[str] | None:
    """
    Keywords from ``*Keywords*Enrollments*.xlsx`` in priority order.

    When a numeric enrollment column is present, sorts by enrollment descending;
    otherwise preserves spreadsheet row order. Returns ``None`` when no file exists.
    """
    path = enrollment_keyword_allowlist_path(course)
    if path is None:
        return None

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


def load_enrollment_keyword_allowlist(course: str) -> set[str] | None:
    """
    Keywords from ``*Keywords*Enrollments*.xlsx`` in ``data/<course>/gkp/``.

    Returns ``None`` when no allowlist file exists for the course.
    """
    ordered = load_enrollment_keyword_allowlist_ordered(course)
    if ordered is None:
        return None
    return set(ordered)


def _split_keyword_field(raw: object) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    return [clean_keyword_text(k) for k in re.split(r"[;\n]", str(raw)) if clean_keyword_text(k)]


def _join_keyword_field(keywords: list[str]) -> str:
    cleaned = [clean_keyword_text(k) for k in keywords if clean_keyword_text(k)]
    return "; ".join(sorted(dict.fromkeys(cleaned)))


def enrollment_allowlist_keywords(
    allowlist: set[str],
    kw_day: pd.DataFrame,
    segment_row: pd.Series,
    *,
    allowlist_order: list[str] | None = None,
) -> list[str]:
    """
    Full enrollment allowlist as a keyword list for one segment.

    Uses panel spelling when a keyword appears in the segment's region / match types;
    otherwise the normalized allowlist text.
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
    if not kw_day.empty and pd.notna(region):
        sub = kw_day[kw_day["region"] == region]
        if allowed_mt and "match_type" in sub.columns:
            sub = sub[sub["match_type"].isin(allowed_mt)]
        if not sub.empty and "keyword" in sub.columns:
            for kw in sub["keyword"].dropna().astype(str):
                key = normalize_keyword(kw)
                if key in allowlist:
                    canonical[key] = clean_keyword_text(kw)

    keys_in_order: list[str] = []
    seen_keys: set[str] = set()
    for key in allowlist_order or sorted(allowlist):
        if key in allowlist and key not in seen_keys:
            seen_keys.add(key)
            keys_in_order.append(key)
    for key in sorted(allowlist):
        if key not in seen_keys:
            seen_keys.add(key)
            keys_in_order.append(key)

    for key in keys_in_order:
        canonical.setdefault(key, key)
    return [canonical[k] for k in keys_in_order]


def filter_keyword_list(keywords: list[str], allowlist: set[str]) -> list[str]:
    if not allowlist:
        return keywords
    out: list[str] = []
    seen: set[str] = set()
    for kw in keywords:
        key = normalize_keyword(kw)
        if key in allowlist and key not in seen:
            seen.add(key)
            out.append(clean_keyword_text(kw) if clean_keyword_text(kw) else key)
    return out


def filter_keyword_sets_dataframe(
    keyword_sets: pd.DataFrame,
    allowlist: set[str],
    *,
    drop_empty: bool = True,
) -> pd.DataFrame:
    """Keep only allowlisted keywords in set list columns; optionally drop empty sets."""
    if not allowlist:
        return keyword_sets

    out = keyword_sets.copy()
    list_cols = [c for c in _KEYWORD_LIST_COLS if c in out.columns]

    for idx, row in out.iterrows():
        updated: dict[str, str] = {}
        for col in list_cols:
            filtered = filter_keyword_list(_split_keyword_field(row.get(col)), allowlist)
            updated[col] = _join_keyword_field(filtered)
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
                positive.update(_split_keyword_field(out.at[idx, col]))
            joined = _join_keyword_field(sorted(positive))
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
    if allowlist is None:
        return keyword_sets
    return filter_keyword_sets_dataframe(keyword_sets, allowlist)
