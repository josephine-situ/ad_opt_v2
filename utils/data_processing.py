from pathlib import Path
from typing import Any

import pandas as pd


KW_DAY_PANEL_COLUMNS = [
    "date",
    "region",
    "keyword",
    "campaign",
    "match_type",
    "clicks",
    "cost",
    "conv_value",
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


def _clean_keyword(keyword: pd.Series) -> pd.Series:
    return (
        keyword.astype("string")
        .str.replace(r'["\[\]]', "", regex=True)
        .str.lower()
        .str.strip()
    )


def _clean_match_type(match_type: pd.Series) -> pd.Series:
    return match_type.astype("string").str.replace("_", " ", regex=False).str.title().str.strip()


def _join_sorted_unique(values: pd.Series) -> str:
    clean_values = values.dropna().astype(str)
    return "; ".join(sorted(clean_values.unique()))


def clean_kw_day_panel(input_file: str | Path, output_file: str | Path | None = None) -> pd.DataFrame:
    """Clean the keyword-day panel and optionally write it to disk."""
    df = pd.read_csv(input_file)

    raw_search_keyword_columns = {
        "Day",
        "Search keyword",
        "Search keyword match type",
        "Campaign",
        "Clicks",
        "Conv. value",
        "Currency code",
        "Cost",
        "First page CPC",
    }
    kw_day_panel_columns = {
        "date",
        "keyword",
        "campaign",
        "match_type",
        "clicks",
        "cost",
    }
    if raw_search_keyword_columns <= set(df.columns):
        df = df.rename(
            columns={
                "Day": "date",
                "Search keyword": "keyword",
                "Search keyword match type": "match_type",
                "Campaign": "campaign",
                "Clicks": "clicks",
                "Conv. value": "conv_value",
                "Currency code": "currency_code",
                "Cost": "cost",
                "First page CPC": "first_page_cpc",
            }
        )
    elif not kw_day_panel_columns <= set(df.columns):
        missing_columns = (raw_search_keyword_columns - set(df.columns)) or (
            kw_day_panel_columns - set(df.columns)
        )
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"kw-day-panel input is missing required column(s): {missing}")

    for column in ["conv_value", "currency_code", "first_page_cpc"]:
        if column not in df.columns:
            df[column] = pd.NA

    required_columns = set(KW_DAY_PANEL_COLUMNS) - {"region"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"kw-day-panel is missing required column(s): {missing}")

    df = df.copy()
    df["campaign"] = _clean_campaign(df["campaign"])
    df["region"] = df["campaign"].apply(_extract_region_from_campaign)
    df["keyword"] = _clean_keyword(df["keyword"])
    df["match_type"] = _clean_match_type(df["match_type"])
    df["date"] = pd.to_datetime(df["date"]).dt.date

    df["clicks"] = pd.to_numeric(df["clicks"], errors="coerce").fillna(0).astype("Int64")

    for column in ["cost", "conv_value", "first_page_cpc"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[KW_DAY_PANEL_COLUMNS].sort_values(
        ["date", "region", "campaign", "keyword", "match_type"],
        na_position="last",
    )

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_path, index=False)

    return df


def _read_clean_budget_history(budget_history_file: str | Path) -> pd.DataFrame:
    """Read campaign budget change history and normalize key fields."""
    budgets = pd.read_csv(budget_history_file)
    required_columns = {"date", "campaign", "daily budget"}
    missing_columns = required_columns - set(budgets.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"budget history is missing required column(s): {missing}")

    budgets = budgets.copy()
    budgets["date"] = pd.to_datetime(budgets["date"]).astype("datetime64[ns]")
    budgets["campaign"] = _clean_campaign(budgets["campaign"])
    budgets["daily_budget"] = pd.to_numeric(
        budgets["daily budget"].astype("string").str.replace(r"[$,]", "", regex=True),
        errors="coerce",
    )
    budgets = budgets.dropna(subset=["date", "campaign", "daily_budget"])
    budgets = budgets[["date", "campaign", "daily_budget"]].sort_values(["campaign", "date"])
    return budgets


def _add_daily_budgets(campaign_day: pd.DataFrame, budgets: pd.DataFrame) -> pd.DataFrame:
    """Attach the most recent budget change on or before each campaign-day."""
    if campaign_day.empty:
        campaign_day["daily_budget"] = pd.Series(dtype="float64")
        return campaign_day

    if budgets.empty:
        campaign_day["daily_budget"] = pd.NA
        return campaign_day

    campaign_day = campaign_day.copy()
    campaign_day["date"] = pd.to_datetime(campaign_day["date"]).astype("datetime64[ns]")

    budgeted_groups = []
    for campaign, group in campaign_day.groupby("campaign", sort=False):
        campaign_budgets = budgets[budgets["campaign"] == campaign]
        if campaign_budgets.empty:
            budgeted_group = group.copy()
            budgeted_group["daily_budget"] = pd.NA
        else:
            budgeted_group = pd.merge_asof(
                group.sort_values("date"),
                campaign_budgets.sort_values("date"),
                on="date",
                by="campaign",
                direction="backward",
            )
        budgeted_groups.append(budgeted_group)

    return pd.concat(budgeted_groups, ignore_index=True)


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

    campaign_day = (
        active_kw_day.groupby(["date", "campaign", "region"], dropna=False)
        .agg(
            clicks=("clicks", "sum"),
            cost=("cost", "sum"),
        )
        .reset_index()
    )
    campaign_day = _attach_campaign_versions(campaign_day, campaign_summary)
    campaign_day["cost"] = campaign_day["cost"].round(2)
    campaign_day = campaign_day[
        [
            "date",
            "campaign_version",
            "region",
            "daily_budget",
            "match_types",
            "clicks",
            "cost",
        ]
    ].sort_values(["date", "region", "campaign_version"], na_position="last")

    for df, output_file in [
        (campaign_day, campaign_day_output_file),
        (campaign_summary, campaign_summary_output_file),
    ]:
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(output_path, index=False)

    return campaign_day, campaign_summary
