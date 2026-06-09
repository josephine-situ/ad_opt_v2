from pathlib import Path
from typing import Any

import re

import pandas as pd

_ASCII_KEYWORD_RE = re.compile(r"^[a-zA-Z0-9\s']*$")


KW_DAY_PANEL_COLUMNS = [
    "date",
    "region",
    "keyword",
    "campaign",
    "match_type",
    "clicks",
    "cost",
    "all_conv",
    "currency_code",
    "first_page_cpc",
]


def _extract_region_from_campaign(campaign: Any) -> str | None:
    if pd.isna(campaign):
        return None

    parts = [part.strip() for part in str(campaign).split("-")]
    for part in parts:
        if ("USA" in part) or ("US" in part) or (part == "US"):
            return "USA"
        for region in ["A", "B", "C"]:
            if (
                part == region
                or part.startswith(f"{region} ")
                or part.startswith(f"{region}/")
                or part.startswith(f"{region}(")
            ):
                return region

    return None


def _clean_campaign(campaign: pd.Series) -> pd.Series:
    return campaign.astype("string").str.replace(r"\[.*?\]", "", regex=True).str.strip()


def clean_keyword_text(keyword: object, *, strict: bool = False) -> str | None:
    """Canonicalize one keyword token for storage and matching.

    Strips Ads syntax (quotes, brackets, ``+`` modifiers), collapses whitespace,
    and lowercases. When ``strict=True``, rejects non-ASCII-alphanumeric keywords
    and returns ``None``.
    """
    if keyword is None or (isinstance(keyword, float) and pd.isna(keyword)):
        return None if strict else ""

    text = str(keyword).strip()
    for char in ("'", '"', "+", "[", "]"):
        text = text.replace(char, "")
    text = " ".join(text.split()).strip().lower()

    if not text:
        return None if strict else ""
    if strict and not _ASCII_KEYWORD_RE.match(text):
        return None
    return text


def clean_keyword_series(keyword: pd.Series) -> pd.Series:
    """Vectorized :func:`clean_keyword_text` (keeps nulls as null)."""
    cleaned = keyword.map(clean_keyword_text)
    return cleaned.astype("string")


KEYWORD_SET_LIST_COLUMNS = (
    "positive_keywords",
    "unique_keywords",
    "broad_keywords",
    "phrase_keywords",
    "exact_keywords",
)


def split_keyword_field(raw: object) -> list[str]:
    """Split a semicolon/newline keyword list field (already-clean values only)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    return [part for part in (p.strip() for p in re.split(r"[;\n]", str(raw))) if part]


def join_keyword_field(keywords: list[str]) -> str:
    """Join keyword tokens into a canonical semicolon-separated field."""
    return "; ".join(sorted(dict.fromkeys(k for k in keywords if k)))


def clean_keyword_list_field(raw: object) -> str:
    """Clean a semicolon-separated keyword column (split → :func:`clean_keyword_text` each → join)."""
    cleaned = [text for part in split_keyword_field(raw) if (text := clean_keyword_text(part))]
    return join_keyword_field(cleaned)


def clean_keyword_sets_dataframe(
    keyword_sets: pd.DataFrame,
    list_cols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Clean semicolon-separated keyword columns once at load time.

    Downstream code assumes keyword tokens are already canonical (see
    :func:`clean_keyword_text`). Ingest boundaries: this helper, ``clean_keyword_series``
    (kw-day panel), allowlist xlsx load, and change-history parsing.
    """
    cols = [c for c in (list_cols or KEYWORD_SET_LIST_COLUMNS) if c in keyword_sets.columns]
    if not cols:
        return keyword_sets.copy()
    out = keyword_sets.copy()
    for col in cols:
        out[col] = out[col].map(clean_keyword_list_field)
    return out


def existing_panel_keywords(course: str, *, min_clicks: int = 1) -> list[str]:
    """Unique cleaned keywords from the course kw-day-panel."""
    from utils.paths import processed_dir, reports_dir

    panel_path: Path | None = None
    for path in (
        processed_dir(course) / "kw-day-panel.csv",
        reports_dir(course) / "kw-day-panel.csv",
    ):
        if path.is_file():
            panel_path = path
            break
    if panel_path is None:
        return []

    df = pd.read_csv(panel_path)
    if "keyword" not in df.columns:
        raise KeyError(f"{panel_path} is missing a keyword column. Found: {list(df.columns)}")

    if min_clicks > 0 and "clicks" in df.columns:
        clicks = pd.to_numeric(df["clicks"], errors="coerce").fillna(0)
        df = df.loc[clicks > 0]

    keywords = clean_keyword_series(df["keyword"].dropna()).dropna().unique()
    return sorted(str(kw) for kw in keywords if str(kw).strip())


def _clean_match_type(match_type: pd.Series) -> pd.Series:
    return match_type.astype("string").str.replace("_", " ", regex=False).str.title().str.strip()


def clean_kw_day_panel(input_file: str | Path, output_file: str | Path | None = None) -> pd.DataFrame:
    """Clean the API kw-day-panel and optionally write it to disk."""
    df = pd.read_csv(input_file)

    required = {"date", "keyword", "campaign", "match_type", "clicks", "cost"}
    missing = required - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"kw-day-panel input is missing required column(s): {missing_cols}")

    for column in ["all_conv", "currency_code", "first_page_cpc"]:
        if column not in df.columns:
            df[column] = 0.0 if column == "all_conv" else pd.NA

    required_columns = set(KW_DAY_PANEL_COLUMNS) - {"region"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"kw-day-panel is missing required column(s): {missing}")

    df = df.copy()
    df["campaign"] = _clean_campaign(df["campaign"])
    df["region"] = df["campaign"].apply(_extract_region_from_campaign)
    df["keyword"] = clean_keyword_series(df["keyword"])
    df["match_type"] = _clean_match_type(df["match_type"])
    df["date"] = pd.to_datetime(df["date"]).dt.date

    df["clicks"] = pd.to_numeric(df["clicks"], errors="coerce").fillna(0).astype("Int64")

    for column in ["cost", "all_conv", "first_page_cpc"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["all_conv"] = df["all_conv"].fillna(0.0)

    df = df[KW_DAY_PANEL_COLUMNS].sort_values(
        ["date", "region", "campaign", "keyword", "match_type"],
        na_position="last",
    )

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_path, index=False)

    return df


def _read_campaign_summary(campaign_summary_file: str | Path) -> pd.DataFrame:
    summary = pd.read_csv(campaign_summary_file)
    required_columns = {
        "campaign_version",
        "campaign",
        "start_date",
        "end_date",
        "daily_budget",
        "match_types",
    }
    missing_columns = required_columns - set(summary.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"campaign summary is missing required column(s): {missing}")

    summary = summary.copy()
    summary["campaign"] = _clean_campaign(summary["campaign"])
    summary["start_date"] = pd.to_datetime(summary["start_date"]).dt.date
    summary["end_date"] = pd.to_datetime(summary["end_date"], errors="coerce").dt.date
    summary["daily_budget"] = pd.to_numeric(summary["daily_budget"], errors="coerce")
    return summary


def _attach_campaign_versions(campaign_day: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    matched_groups = []
    for campaign, group in campaign_day.groupby("campaign", sort=False):
        campaign_summary = summary[summary["campaign"] == campaign]
        if campaign_summary.empty:
            continue

        group = group.copy()
        matched_group_parts = []
        for _, summary_row in campaign_summary.iterrows():
            mask = group["date"] >= summary_row["start_date"]
            if pd.notna(summary_row["end_date"]):
                mask &= group["date"] < summary_row["end_date"]
            matched = group[mask].copy()
            if matched.empty:
                continue
            matched["campaign_version"] = summary_row["campaign_version"]
            matched["daily_budget"] = summary_row["daily_budget"]
            matched["match_types"] = summary_row["match_types"]
            matched_group_parts.append(matched)

        if matched_group_parts:
            matched_groups.append(pd.concat(matched_group_parts, ignore_index=True))

    if not matched_groups:
        return pd.DataFrame(columns=[*campaign_day.columns, "campaign_version", "daily_budget"])

    return pd.concat(matched_groups, ignore_index=True)


def generate_campaign_day_panel(
    kw_day_panel_file: str | Path,
    campaign_summary_file: str | Path,
    campaign_day_output_file: str | Path | None = None,
    campaign_summary_output_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate active campaign-day aggregates keyed to campaign-summary versions."""
    kw_day = clean_kw_day_panel(kw_day_panel_file)
    campaign_summary = _read_campaign_summary(campaign_summary_file)

    kw_day["active_date_cost"] = kw_day.groupby(["date", "campaign"], dropna=False)[
        "cost"
    ].transform("sum")
    active_kw_day = kw_day[kw_day["active_date_cost"] > 0].copy()

    agg: dict[str, tuple[str, str]] = {
        "clicks": ("clicks", "sum"),
        "cost": ("cost", "sum"),
    }
    if "all_conv" in active_kw_day.columns:
        agg["all_conv"] = ("all_conv", "sum")

    campaign_day = (
        active_kw_day.groupby(["date", "campaign", "region"], dropna=False)
        .agg(**agg)
        .reset_index()
    )
    campaign_day = _attach_campaign_versions(campaign_day, campaign_summary)
    campaign_day["cost"] = campaign_day["cost"].round(2)
    if "all_conv" in campaign_day.columns:
        campaign_day["all_conv"] = campaign_day["all_conv"].fillna(0.0).round(4)

    output_columns = [
        "date",
        "campaign_version",
        "region",
        "daily_budget",
        "match_types",
        "clicks",
        "cost",
    ]
    if "all_conv" in campaign_day.columns:
        output_columns.append("all_conv")
    campaign_day = campaign_day[output_columns].sort_values(
        ["date", "region", "campaign_version"], na_position="last"
    )

    for df, output_file in [
        (campaign_day, campaign_day_output_file),
        (campaign_summary, campaign_summary_output_file),
    ]:
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(output_path, index=False)

    return campaign_day, campaign_summary
