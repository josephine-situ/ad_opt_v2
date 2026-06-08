"""Build keywords_classified.csv from active campaign keywords only."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from utils.data_processing import _clean_keyword

EXISTING_ORIGIN = "existing"
MAX_KEYWORD_LEN = 80
MAX_KEYWORD_WORDS = 10


def clean_keyword_text(kw: object) -> str | None:
    """Match ad_opt compare_keywords cleaning (brackets, specials, quotes)."""
    if pd.isna(kw):
        return None

    text = str(kw).strip()
    text = text.replace("'", "").replace('"', "").replace("+", "")
    text = re.sub(r"\[.*?\]", "", text).strip()
    if not re.match(r"^[a-zA-Z0-9\s']*$", text):
        return None
    if not text or text.isspace():
        return None
    return text


def passes_length_filters(kw: str) -> bool:
    if len(kw) > MAX_KEYWORD_LEN:
        return False
    return len(kw.split()) <= MAX_KEYWORD_WORDS


def clean_and_filter_keyword(kw: object) -> str | None:
    cleaned = clean_keyword_text(kw)
    if cleaned is None:
        return None
    normalized = " ".join(cleaned.split()).lower()
    if not passes_length_filters(normalized):
        return None
    return normalized


def _keywords_from_kw_day_panel(path: Path, *, min_clicks: int) -> set[str]:
    df = pd.read_csv(path)
    if "keyword" not in df.columns:
        raise KeyError(f"{path} is missing a keyword column. Found: {list(df.columns)}")

    if min_clicks > 0 and "clicks" in df.columns:
        clicks = pd.to_numeric(df["clicks"], errors="coerce").fillna(0)
        df = df[clicks > 0]

    cleaned_series = _clean_keyword(df["keyword"].dropna())
    keywords: set[str] = set()
    for raw in cleaned_series.unique():
        cleaned = clean_and_filter_keyword(raw)
        if cleaned:
            keywords.add(cleaned)
    return keywords


def _keywords_from_keyword_sets(path: Path) -> set[str]:
    df = pd.read_csv(path)
    keywords: set[str] = set()
    list_columns = [
        col
        for col in (
            "positive_keywords",
            "unique_keywords",
            "broad_keywords",
            "phrase_keywords",
            "exact_keywords",
        )
        if col in df.columns
    ]
    for col in list_columns:
        for raw in df[col].dropna():
            for part in str(raw).split(";"):
                cleaned = clean_and_filter_keyword(part)
                if cleaned:
                    keywords.add(cleaned)
    return keywords


def collect_existing_keywords(
    course: str = "sys_think",
    *,
    min_clicks: int = 1,
    include_keyword_sets: bool = True,
) -> set[str]:
    """Unique normalized keywords from the kw-day-panel for a course."""
    from campaign_opt.paths import PROCESSED_DIR, REPORTS_DIR

    candidates: list[Path] = [
        PROCESSED_DIR / "kw-day-panel.csv",
        REPORTS_DIR / "kw-day-panel.csv",
    ]

    keywords: set[str] = set()
    for path in candidates:
        if not path.exists():
            continue
        found = _keywords_from_kw_day_panel(path, min_clicks=min_clicks)
        keywords |= found

    if include_keyword_sets:
        sets_path = PROCESSED_DIR / "campaign-keyword-sets.csv"
        if sets_path.exists():
            keywords |= _keywords_from_keyword_sets(sets_path)

    return keywords


def build_keywords_classified_dataframe(
    course: str,
    *,
    min_clicks: int = 1,
    include_keyword_sets: bool = True,
) -> pd.DataFrame:
    keywords = collect_existing_keywords(
        course,
        min_clicks=min_clicks,
        include_keyword_sets=include_keyword_sets,
    )
    rows = [{"Keyword": kw, "Origin": EXISTING_ORIGIN} for kw in sorted(keywords)]
    return pd.DataFrame(rows)


def write_keywords_classified(
    course: str,
    output_file: str | Path | None = None,
    *,
    min_clicks: int = 1,
    include_keyword_sets: bool = True,
) -> Path:
    from campaign_opt.paths import GKP_DIR

    out = Path(output_file or GKP_DIR / "keywords_classified.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = build_keywords_classified_dataframe(
        course,
        min_clicks=min_clicks,
        include_keyword_sets=include_keyword_sets,
    )
    frame.to_csv(out, index=False)
    return out
