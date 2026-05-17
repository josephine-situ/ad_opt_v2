from pathlib import Path
from typing import Any

import pandas as pd


KW_DAY_PANEL_COLUMNS = [
    "date",
    "region",
    "keyword",
    "campaign",
    "match_type",
    "impressions",
    "clicks",
    "cost",
    "conversions",
    "impression_share",
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

    required_columns = {
        "date",
        "keyword",
        "campaign",
        "match_type",
        "impressions",
        "clicks",
        "cost",
        "conversions",
        "impression_share",
    }
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

    for column in ["impressions", "clicks"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype("Int64")

    for column in ["cost", "conversions", "impression_share"]:
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


def generate_campaign_day_panel(
    kw_day_panel_file: str | Path,
    budget_history_file: str | Path,
    campaign_day_output_file: str | Path | None = None,
    campaign_summary_output_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate active campaign-day aggregates and a campaign-level summary."""
    kw_day = clean_kw_day_panel(kw_day_panel_file)
    budgets = _read_clean_budget_history(budget_history_file)

    kw_day["active_date_cost"] = kw_day.groupby(["date", "campaign"], dropna=False)[
        "cost"
    ].transform("sum")
    active_kw_day = kw_day[kw_day["active_date_cost"] > 0].copy()

    campaign_day = (
        active_kw_day.groupby(["date", "campaign", "region"], dropna=False)
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            cost=("cost", "sum"),
            conversions=("conversions", "sum"),
            match_types=("match_type", _join_sorted_unique),
            unique_keywords=("keyword", _join_sorted_unique),
            num_unique_keywords=("keyword", "nunique"),
        )
        .reset_index()
    )

    impression_share = (
        active_kw_day.assign(weighted_impression_share=lambda df: df["impression_share"] * df["impressions"])
        .groupby(["date", "campaign"], dropna=False)
        .agg(
            impression_share_weighted_sum=("weighted_impression_share", "sum"),
            impression_share_impressions=("impressions", "sum"),
            impression_share_mean=("impression_share", "mean"),
        )
        .reset_index()
    )
    impression_share["impression_share"] = impression_share[
        "impression_share_weighted_sum"
    ] / impression_share["impression_share_impressions"]
    impression_share.loc[
        impression_share["impression_share_impressions"] == 0, "impression_share"
    ] = impression_share["impression_share_mean"]
    campaign_day = campaign_day.merge(
        impression_share[["date", "campaign", "impression_share"]],
        on=["date", "campaign"],
        how="left",
    )

    campaign_day = _add_daily_budgets(campaign_day, budgets)
    campaign_day["date"] = pd.to_datetime(campaign_day["date"]).dt.date
    campaign_day = campaign_day[
        [
            "date",
            "campaign",
            "region",
            "daily_budget",
            "impressions",
            "clicks",
            "cost",
            "conversions",
            "impression_share",
            "match_types",
            "unique_keywords",
            "num_unique_keywords",
        ]
    ].sort_values(["date", "region", "campaign"], na_position="last")

    budgeted_active_kw_day = _add_daily_budgets(active_kw_day, budgets)
    budgeted_active_kw_day["date"] = pd.to_datetime(budgeted_active_kw_day["date"]).dt.date
    campaign_summary = (
        budgeted_active_kw_day.groupby(["campaign", "daily_budget", "region"], dropna=False)
        .agg(
            start_date=("date", "min"),
            match_types=("match_type", _join_sorted_unique),
            unique_keywords=("keyword", _join_sorted_unique),
            num_unique_keywords=("keyword", "nunique"),
        )
        .reset_index()
    )
    campaign_summary = campaign_summary[
        [
            "campaign",
            "start_date",
            "daily_budget",
            "region",
            "match_types",
            "unique_keywords",
            "num_unique_keywords",
        ]
    ].sort_values(["start_date", "region", "campaign"], na_position="last")

    for df, output_file in [
        (campaign_day, campaign_day_output_file),
        (campaign_summary, campaign_summary_output_file),
    ]:
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(output_path, index=False)

    return campaign_day, campaign_summary
