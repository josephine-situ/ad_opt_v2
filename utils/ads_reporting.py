import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from google.ads.googleads.client import GoogleAdsClient

from config import COURSE_CONFIG
from utils.gaql_queries import KW_DAY_PANEL_REPORT_QUERY, KW_KEYWORD_ALL_CONV_QUERY
from utils.metrics import google_ads_metrics_client
from utils.report_row_generators import aggregate_kw_all_conv_totals, generate_kw_day_panel_rows


def write_to_file(
    header_parts: list[str],
    row_generator: Iterable[dict[str, Any]],
    output_file: Path,
    delimiter: str = "\t",
    restval: str = "0",
) -> None:
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header_parts, delimiter=delimiter, restval=restval)
        writer.writeheader()
        for row in row_generator:
            writer.writerow(row)


def generate_kw_day_panel_report(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str,
    end_date: str,
) -> None:
    """Keyword-day panel: clicks/cost plus filtered all_conv in one CSV."""
    output_path = Path(f"data/{output_course}/reports/kw-day-panel.csv")
    ads_service = google_ads_client.get_service("GoogleAdsService")

    clicks_query = KW_DAY_PANEL_REPORT_QUERY.format(start_date=start_date, end_date=end_date)
    clicks_stream = ads_service.search_stream(customer_id=customer_id, query=clicks_query)
    click_rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in generate_kw_day_panel_rows(clicks_stream):
        key = (row["date"], row["keyword"], row["campaign"], row["match_type"])
        click_rows[key] = row

    conversion_action_list = "', '".join(COURSE_CONFIG[output_course]["conversion_actions"])
    conv_query = KW_KEYWORD_ALL_CONV_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        conversion_action_list=conversion_action_list,
    )
    conv_stream = ads_service.search_stream(customer_id=customer_id, query=conv_query)
    conv_totals = aggregate_kw_all_conv_totals(conv_stream)

    for key, all_conv in conv_totals.items():
        if key in click_rows:
            click_rows[key]["all_conv"] = f"{all_conv:.2f}"
        else:
            date, keyword, campaign, match_type = key
            click_rows[key] = {
                "date": date,
                "keyword": keyword,
                "campaign": campaign,
                "match_type": match_type,
                "clicks": 0,
                "cost": "0.00",
                "currency_code": "",
                "first_page_cpc": "",
                "all_conv": f"{all_conv:.2f}",
            }

    for row in click_rows.values():
        row.setdefault("all_conv", "0.00")

    header_parts = [
        "date",
        "keyword",
        "campaign",
        "match_type",
        "clicks",
        "cost",
        "all_conv",
        "currency_code",
        "first_page_cpc",
    ]
    write_to_file(header_parts, click_rows.values(), output_path, delimiter=",")
    google_ads_metrics_client.track_google_ads_operation_count("search_stream", 2)
    print(f"Generated: {output_path}")
