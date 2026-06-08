#!/usr/bin/env python3
"""Pull Google Ads data required by campaign_opt."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from dateutil.relativedelta import relativedelta
from google.ads.googleads.client import GoogleAdsClient

from campaign_opt.paths import GKP_DIR, REPORTS_DIR
from config import COURSE, COURSE_CONFIG
from utils.ads_reporting import generate_kw_day_panel_report, write_to_file
from utils.metrics import google_ads_metrics_client

CAMPAIGN_OPT = "campaign_opt"
KEYWORD_PLANNING = "keyword_planning"
VALID_DATASETS = {CAMPAIGN_OPT, KEYWORD_PLANNING}


def _gkp_month_header_sort_key(header: str) -> tuple[int, int]:
    month_str, year_str = header.replace("Searches: ", "", 1).rsplit(" ", 1)
    month_number = datetime.strptime(month_str, "%b").month
    return int(year_str), month_number


def validate_requested_datasets(datasets: Iterable[str]) -> set[str]:
    requested_datasets = {dataset.strip() for dataset in datasets if dataset.strip()}
    if not requested_datasets:
        print(
            f"Error: --datasets must include at least one of: {', '.join(sorted(VALID_DATASETS))}"
        )
        sys.exit(1)

    invalid_datasets = requested_datasets - VALID_DATASETS
    if invalid_datasets:
        print(f"Error: Invalid dataset(s): {', '.join(sorted(invalid_datasets))}")
        print(f"Valid choices are: {', '.join(sorted(VALID_DATASETS))}")
        sys.exit(1)

    return requested_datasets


def _resolve_date_range(
    start_date: str | None,
    end_date: str | None,
) -> tuple[str, str, str]:
    resolved_end = end_date or datetime.now().strftime("%Y-%m-%d")
    if start_date:
        resolved_start = start_date
        start_source = "--start-date"
    else:
        resolved_start = COURSE_CONFIG[COURSE]["min_date"]
        start_source = f"config min_date ({resolved_start})"
    return resolved_start, resolved_end, start_source


def pull_campaign_opt(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """Pull API reports used by the campaign_opt pipeline."""
    resolved_start, resolved_end, start_source = _resolve_date_range(start_date, end_date)

    print(f"Pulling campaign_opt datasets for {COURSE}...")
    print(f"Date range: {resolved_start} to {resolved_end} (start from {start_source})")
    print(f"Customer ID: {customer_id}")

    generate_kw_day_panel_report(
        google_ads_client,
        customer_id,
        COURSE,
        resolved_start,
        resolved_end,
    )

    print(f"Successfully generated campaign_opt reports for {COURSE}")


def generate_rows_from_gkp_response(response: Any) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    monthly_headers = set()

    for result in response.results:
        metrics = result.keyword_metrics
        low_bid = (
            Decimal(metrics.low_top_of_page_bid_micros) / 1_000_000
            if metrics.low_top_of_page_bid_micros
            else ""
        )
        high_bid = (
            Decimal(metrics.high_top_of_page_bid_micros) / 1_000_000
            if metrics.high_top_of_page_bid_micros
            else ""
        )

        row_parts = {
            "Keyword": result.text,
            "Avg. monthly searches": metrics.avg_monthly_searches
            if metrics.avg_monthly_searches
            else "",
            "Competition": metrics.competition.name.capitalize() if metrics.competition else "",
            "Competition (indexed value)": metrics.competition_index
            if metrics.competition_index
            else "",
            "Top of page bid (low range)": low_bid,
            "Top of page bid (high range)": high_bid,
        }

        for monthly_vol in metrics.monthly_search_volumes:
            header = f"Searches: {monthly_vol.month.name[:3].capitalize()} {monthly_vol.year}"
            monthly_headers.add(header)
            row_parts[header] = monthly_vol.monthly_searches if monthly_vol.monthly_searches else 0

        rows.append(row_parts)

    return rows, sorted(monthly_headers, key=_gkp_month_header_sort_key)


def pull_keyword_planning(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    keyword_planning_input_file: str,
) -> None:
    """Pull keyword planning data using generate_keyword_historical_metrics."""
    if not keyword_planning_input_file:
        keyword_planning_input_file = str(GKP_DIR / "keywords_classified.csv")

    keywords = []
    with open(keyword_planning_input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            origin = row.get("Origin", "").strip().lower()
            if origin and origin not in {"existing", "existing keywords"}:
                continue
            keyword = row.get("Keyword", "").strip()
            if keyword:
                keywords.append(keyword)

    print("Pulling keyword planning data...")
    print(f"Customer ID: {customer_id}")
    print(f"Keywords file: {keyword_planning_input_file}")

    if len(keywords) > 10_000:
        print("Error: Google Ads API supports up to 10,000 keywords per request.")
        sys.exit(1)

    keyword_plan_idea_service = google_ads_client.get_service("KeywordPlanIdeaService")
    request = google_ads_client.get_type("GenerateKeywordHistoricalMetricsRequest")
    request.customer_id = customer_id
    request.keywords = keywords
    request.keyword_plan_network = google_ads_client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH

    historical_metrics_options = google_ads_client.get_type("HistoricalMetricsOptions")
    current_date = datetime.now()
    start_date = datetime.strptime(
        COURSE_CONFIG[COURSE]["min_date"],
        "%Y-%m-%d",
    ) - relativedelta(months=6)
    end_date = current_date - relativedelta(months=1)

    month_of_year_enum = google_ads_client.enums.MonthOfYearEnum
    historical_metrics_options.year_month_range.start.year = start_date.year
    historical_metrics_options.year_month_range.start.month = getattr(
        month_of_year_enum,
        start_date.strftime("%B").upper(),
    )
    historical_metrics_options.year_month_range.end.year = end_date.year
    historical_metrics_options.year_month_range.end.month = getattr(
        month_of_year_enum,
        end_date.strftime("%B").upper(),
    )
    request.historical_metrics_options = historical_metrics_options

    print(f"Fetching historical metrics for {len(keywords)} keywords from Google Ads...")
    response = keyword_plan_idea_service.generate_keyword_historical_metrics(request=request)

    header_parts = [
        "Keyword",
        "Avg. monthly searches",
        "Competition",
        "Competition (indexed value)",
        "Top of page bid (low range)",
        "Top of page bid (high range)",
    ]
    rows, date_header_parts = generate_rows_from_gkp_response(response)
    header_parts.extend(date_header_parts)

    GKP_DIR.mkdir(parents=True, exist_ok=True)
    output_file = GKP_DIR / (
        f"Saved Keyword Stats {current_date.strftime('%Y-%m-%d')} "
        f"at {current_date.strftime('%H-%M-%S')}.csv"
    )

    write_to_file(header_parts, rows, output_file)
    google_ads_metrics_client.track_google_ads_operation_count(
        "generate_keyword_historical_metrics",
        1,
    )
    print(f"Keyword planning data written to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull Google Ads input data for campaign_opt.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated datasets: campaign_opt, keyword_planning (GKP).",
    )
    parser.add_argument(
        "--keyword-planning-input-file",
        type=str,
        default="",
        help="CSV file containing a Keyword column for Keyword Planner pulls.",
    )
    parser.add_argument(
        "--google-ads-yaml",
        type=str,
        required=True,
        help="Path to Google Ads YAML configuration file.",
    )
    parser.add_argument(
        "--customer-id",
        type=str,
        required=True,
        help="Google Ads customer ID.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="",
        help="YYYY-MM-DD start date for campaign_opt reports. Defaults to min_date in config.py.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="",
        help="YYYY-MM-DD end date for campaign_opt reports. Defaults to today.",
    )

    args = parser.parse_args()
    requested_datasets = validate_requested_datasets(args.datasets.split(","))
    google_ads_client = GoogleAdsClient.load_from_storage(args.google_ads_yaml)
    start_date = args.start_date.strip() or None
    end_date = args.end_date.strip() or None

    if CAMPAIGN_OPT in requested_datasets:
        pull_campaign_opt(
            google_ads_client,
            args.customer_id,
            start_date=start_date,
            end_date=end_date,
        )

    if KEYWORD_PLANNING in requested_datasets:
        pull_keyword_planning(
            google_ads_client,
            args.customer_id,
            args.keyword_planning_input_file,
        )

    print("All requested datasets pulled successfully")


if __name__ == "__main__":
    main()
