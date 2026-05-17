#!/usr/bin/env python3
"""Pull Google Ads reports and Keyword Planner data."""

import argparse
import csv
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from dateutil.relativedelta import relativedelta
from google.ads.googleads.client import GoogleAdsClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COURSE_CONFIG
from utils.ads_reporting import (
    ReportFunction,
    generate_age_clicks_and_conversion_report,
    generate_device_clicks_and_conversion_report,
    generate_hod_clicks_and_conversion_report,
    generate_kw_day_panel_report,
    generate_loc_clicks_and_conversion_report,
    generate_purchase_report,
    generate_search_keyword_report,
    generate_search_terms_report,
    write_to_file,
)
from utils.metrics import google_ads_metrics_client

ADS_REPORTS = "ads_reports"
KW_DAY_PANEL = "kw_day_panel"
KEYWORD_PLANNING = "keyword_planning"
VALID_DATASETS = {ADS_REPORTS, KW_DAY_PANEL, KEYWORD_PLANNING}


def _gkp_month_header_sort_key(header: str) -> tuple[int, int]:
    month_str, year_str = header.replace("Searches: ", "", 1).rsplit(" ", 1)
    month_number = datetime.strptime(month_str, "%b").month
    return int(year_str), month_number


def validate_requested_datasets(datasets: Iterable[str]) -> set[str]:
    requested_datasets = {dataset.strip() for dataset in datasets if dataset.strip()}
    if not requested_datasets:
        print(f"Error: --datasets must include at least one of: {', '.join(sorted(VALID_DATASETS))}")
        sys.exit(1)

    invalid_datasets = requested_datasets - VALID_DATASETS
    if invalid_datasets:
        print(f"Error: Invalid dataset(s): {', '.join(sorted(invalid_datasets))}")
        print(f"Valid choices are: {', '.join(sorted(VALID_DATASETS))}")
        sys.exit(1)

    return requested_datasets


def get_budget_history_start_date(output_course: str) -> str | None:
    budget_history_file = Path(f"data/{output_course}/change_history_budgets.csv")
    if not budget_history_file.exists():
        return None

    dates = []
    with open(budget_history_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_date = row.get("date", "").strip()
            if raw_date:
                dates.append(raw_date)

    return min(dates) if dates else None


def pull_ads_reports(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str | None = None,
    end_date: str | None = None,
    bid_adj_effectiveness_end_date: str | None = None,
) -> None:
    """Pull all Google Ads report CSVs for a course."""
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = COURSE_CONFIG[output_course]["min_date"]

    print(f"Pulling ads reports for course '{output_course}'...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Customer ID: {customer_id}")

    report_functions: list[ReportFunction] = [
        generate_search_keyword_report,
        generate_search_terms_report,
        generate_purchase_report,
        generate_hod_clicks_and_conversion_report,
        generate_age_clicks_and_conversion_report,
        generate_device_clicks_and_conversion_report,
        generate_loc_clicks_and_conversion_report,
    ]

    for report in report_functions:
        report(google_ads_client, customer_id, output_course, start_date, end_date)

    eff_end = bid_adj_effectiveness_end_date or (
        datetime.now() - timedelta(days=1)
    ).strftime("%Y-%m-%d")
    eff_start = (datetime.strptime(eff_end, "%Y-%m-%d") - timedelta(days=7)).strftime(
        "%Y-%m-%d"
    )

    print(f"Pulling 7d bid adjustment effectiveness reports ({eff_start} to {eff_end})...")
    for generator in [
        generate_hod_clicks_and_conversion_report,
        generate_age_clicks_and_conversion_report,
        generate_device_clicks_and_conversion_report,
        generate_loc_clicks_and_conversion_report,
    ]:
        generator(
            google_ads_client,
            customer_id,
            output_course,
            eff_start,
            eff_end,
            output_suffix="_7d",
        )

    print(f"Successfully generated all reports for {output_course}")


def pull_kw_day_panel(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """Pull only the keyword-day panel report for a course."""
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = get_budget_history_start_date(output_course) or COURSE_CONFIG[output_course]["min_date"]

    print(f"Pulling kw-day-panel for course '{output_course}'...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Customer ID: {customer_id}")

    generate_kw_day_panel_report(
        google_ads_client,
        customer_id,
        output_course,
        start_date,
        end_date,
    )


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
    output_course: str,
) -> None:
    """Pull keyword planning data using generate_keyword_historical_metrics."""
    if not keyword_planning_input_file:
        keyword_planning_input_file = f"data/{output_course}/gkp/keywords_classified.csv"

    keywords = []
    with open(keyword_planning_input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
        COURSE_CONFIG[output_course]["min_date"],
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

    output_dir = Path(f"data/{output_course}/gkp")
    output_file = output_dir / (
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
    parser = argparse.ArgumentParser(description="Pull Google Ads input data")
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated list of datasets to pull: ads_reports, kw_day_panel, keyword_planning",
    )
    parser.add_argument(
        "--keyword-planning-input-file",
        type=str,
        default="",
        help="CSV file containing a Keyword column for Keyword Planner pulls.",
    )
    parser.add_argument(
        "--output-course",
        type=str,
        choices=sorted(COURSE_CONFIG.keys()),
        required=True,
        help="Course key used to determine output locations and conversion action filters.",
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
        "--bid-adj-effectiveness-end-date",
        type=str,
        default="",
        help="YYYY-MM-DD end date for *_7d.csv bid-adjustment reports. Defaults to yesterday.",
    )

    args = parser.parse_args()
    requested_datasets = validate_requested_datasets(args.datasets.split(","))
    google_ads_client = GoogleAdsClient.load_from_storage(args.google_ads_yaml)

    if ADS_REPORTS in requested_datasets:
        pull_ads_reports(
            google_ads_client,
            args.customer_id,
            args.output_course,
            bid_adj_effectiveness_end_date=args.bid_adj_effectiveness_end_date.strip() or None,
        )

    if KW_DAY_PANEL in requested_datasets:
        pull_kw_day_panel(
            google_ads_client,
            args.customer_id,
            args.output_course,
        )

    if KEYWORD_PLANNING in requested_datasets:
        pull_keyword_planning(
            google_ads_client,
            args.customer_id,
            args.keyword_planning_input_file,
            args.output_course,
        )

    print("All requested datasets pulled successfully")


if __name__ == "__main__":
    main()
