import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from google.ads.googleads.client import GoogleAdsClient

from config import COURSE_CONFIG
from utils.gaql_queries import (
    AGE_CLICKS_REPORT_QUERY,
    AGE_CONVERSIONS_REPORT_QUERY,
    DEVICE_CLICKS_REPORT_QUERY,
    DEVICE_CONVERSIONS_REPORT_QUERY,
    HOD_CLICKS_REPORT_QUERY,
    HOD_CONVERSIONS_REPORT_QUERY,
    KW_DAY_PANEL_REPORT_QUERY,
    LOC_CLICKS_REPORT_QUERY,
    LOC_CONVERSIONS_REPORT_QUERY,
    PURCHASE_REPORT_QUERY,
    SEARCH_KEYWORD_REPORT_QUERY,
    SEARCH_TERM_REPORT_QUERY,
)
from utils.google_ads_api import get_location_resource_names_for_countries
from utils.metrics import google_ads_metrics_client
from utils.report_row_generators import (
    generate_age_clicks_rows,
    generate_age_conversions_rows,
    generate_device_clicks_rows,
    generate_device_conversions_rows,
    generate_hod_clicks_rows,
    generate_hod_conversions_rows,
    generate_kw_day_panel_rows,
    generate_loc_clicks_rows,
    generate_loc_conversions_rows,
    generate_purchase_report_rows,
    generate_search_keyword_rows,
    generate_search_terms_row,
)


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


class ReportFunction(Protocol):
    def __call__(
        self,
        google_ads_client: GoogleAdsClient,
        customer_id: str,
        output_course: str,
        start_date: str,
        end_date: str,
    ) -> None: ...


def generate_search_keyword_report(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str,
    end_date: str,
) -> None:
    output_path = Path(f"data/{output_course}/reports/Search keyword - raw input to models.csv")
    query = SEARCH_KEYWORD_REPORT_QUERY.format(start_date=start_date, end_date=end_date)
    stream = google_ads_client.get_service("GoogleAdsService").search_stream(
        customer_id=customer_id,
        query=query,
    )

    header_parts = [
        "Day",
        "Search keyword",
        "Search keyword match type",
        "Campaign",
        "Clicks",
        "Conv. value",
        "Currency code",
        "Cost",
        "First page CPC",
    ]
    write_to_file(header_parts, generate_search_keyword_rows(stream), output_path, delimiter=",")
    google_ads_metrics_client.track_google_ads_operation_count("search_stream", 1)
    print(f"Generated: {output_path}")


def generate_kw_day_panel_report(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str,
    end_date: str,
) -> None:
    output_path = Path(f"data/{output_course}/reports/kw-day-panel.csv")
    query = KW_DAY_PANEL_REPORT_QUERY.format(start_date=start_date, end_date=end_date)
    stream = google_ads_client.get_service("GoogleAdsService").search_stream(
        customer_id=customer_id,
        query=query,
    )

    header_parts = [
        "date",
        "keyword",
        "campaign",
        "match_type",
        "clicks",
        "cost",
        "conversions",
        "impression_share",
    ]
    write_to_file(header_parts, generate_kw_day_panel_rows(stream), output_path, delimiter=",")
    google_ads_metrics_client.track_google_ads_operation_count("search_stream", 1)
    print(f"Generated: {output_path}")


def generate_search_terms_report(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str,
    end_date: str,
) -> None:
    output_path = Path(f"data/{output_course}/reports/Search keyword - search terms.csv")
    query = SEARCH_TERM_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        conversion_action_list="', '".join(COURSE_CONFIG[output_course]["conversion_actions"]),
    )
    stream = google_ads_client.get_service("GoogleAdsService").search_stream(
        customer_id=customer_id,
        query=query,
    )
    header_parts = [
        "Search keyword",
        "Search keyword match type",
        "Search term",
        "Conversion action",
        "Conversions",
    ]
    write_to_file(header_parts, generate_search_terms_row(stream), output_path, delimiter=",")
    google_ads_metrics_client.track_google_ads_operation_count("search_stream", 1)
    print(f"Generated: {output_path}")


def generate_purchase_report(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str,
    end_date: str,
) -> None:
    output_path = Path(f"data/{output_course}/reports/Purchase report.csv")
    query = PURCHASE_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        purchase_action_list="', '".join(COURSE_CONFIG[output_course]["purchase_actions"]),
    )
    stream = google_ads_client.get_service("GoogleAdsService").search_stream(
        customer_id=customer_id,
        query=query,
    )
    write_to_file(
        ["Campaign", "Conversion action", "All conv."],
        generate_purchase_report_rows(stream),
        output_path,
        delimiter=",",
    )
    google_ads_metrics_client.track_google_ads_operation_count("search_stream", 1)
    print(f"Generated: {output_path}")


def generate_hod_clicks_and_conversion_report(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str,
    end_date: str,
    output_suffix: str = "",
) -> None:
    ads_service = google_ads_client.get_service("GoogleAdsService")
    output_path = Path(f"data/{output_course}/reports/bid_adj/hod_clicks{output_suffix}.csv")
    stream = ads_service.search_stream(
        customer_id=customer_id,
        query=HOD_CLICKS_REPORT_QUERY.format(start_date=start_date, end_date=end_date),
    )
    write_to_file(["Campaign", "Hour of the day", "Clicks"], generate_hod_clicks_rows(stream), output_path, delimiter=",")
    print(f"Generated: {output_path}")

    output_path_conv = Path(f"data/{output_course}/reports/bid_adj/hod_conv{output_suffix}.csv")
    stream_conv = ads_service.search_stream(
        customer_id=customer_id,
        query=HOD_CONVERSIONS_REPORT_QUERY.format(
            start_date=start_date,
            end_date=end_date,
            purchase_action_list="', '".join(COURSE_CONFIG[output_course]["purchase_actions"]),
        ),
    )
    write_to_file(
        ["Campaign", "Conversion action", "Hour of the day", "All conv."],
        generate_hod_conversions_rows(stream_conv),
        output_path_conv,
        delimiter=",",
    )
    google_ads_metrics_client.track_google_ads_operation_count("search_stream", 2)
    print(f"Generated: {output_path_conv}")


def generate_age_clicks_and_conversion_report(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str,
    end_date: str,
    output_suffix: str = "",
) -> None:
    ads_service = google_ads_client.get_service("GoogleAdsService")
    output_path = Path(f"data/{output_course}/reports/bid_adj/age_clicks{output_suffix}.csv")
    stream = ads_service.search_stream(
        customer_id=customer_id,
        query=AGE_CLICKS_REPORT_QUERY.format(start_date=start_date, end_date=end_date),
    )
    write_to_file(["Campaign", "Age", "Clicks"], generate_age_clicks_rows(stream), output_path, delimiter=",")
    print(f"Generated: {output_path}")

    output_path_conv = Path(f"data/{output_course}/reports/bid_adj/age_conv{output_suffix}.csv")
    stream_conv = ads_service.search_stream(
        customer_id=customer_id,
        query=AGE_CONVERSIONS_REPORT_QUERY.format(
            start_date=start_date,
            end_date=end_date,
            purchase_action_list="', '".join(COURSE_CONFIG[output_course]["purchase_actions"]),
        ),
    )
    write_to_file(
        ["Campaign", "Conversion action", "Age", "All conv."],
        generate_age_conversions_rows(stream_conv),
        output_path_conv,
        delimiter=",",
    )
    google_ads_metrics_client.track_google_ads_operation_count("search_stream", 2)
    print(f"Generated: {output_path_conv}")


def generate_device_clicks_and_conversion_report(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str,
    end_date: str,
    output_suffix: str = "",
) -> None:
    ads_service = google_ads_client.get_service("GoogleAdsService")
    output_path = Path(f"data/{output_course}/reports/bid_adj/device_clicks{output_suffix}.csv")
    stream = ads_service.search_stream(
        customer_id=customer_id,
        query=DEVICE_CLICKS_REPORT_QUERY.format(start_date=start_date, end_date=end_date),
    )
    write_to_file(["Campaign", "Device", "Clicks"], generate_device_clicks_rows(stream), output_path, delimiter=",")
    print(f"Generated: {output_path}")

    output_path_conv = Path(f"data/{output_course}/reports/bid_adj/device_conv{output_suffix}.csv")
    stream_conv = ads_service.search_stream(
        customer_id=customer_id,
        query=DEVICE_CONVERSIONS_REPORT_QUERY.format(
            start_date=start_date,
            end_date=end_date,
            purchase_action_list="', '".join(COURSE_CONFIG[output_course]["purchase_actions"]),
        ),
    )
    write_to_file(
        ["Campaign", "Conversion action", "Device", "All conv."],
        generate_device_conversions_rows(stream_conv),
        output_path_conv,
        delimiter=",",
    )
    google_ads_metrics_client.track_google_ads_operation_count("search_stream", 2)
    print(f"Generated: {output_path_conv}")


def generate_loc_clicks_and_conversion_report(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str,
    end_date: str,
    output_suffix: str = "",
) -> None:
    regions = COURSE_CONFIG[output_course]["regions"]
    all_locations = [location for locations in regions.values() for location in locations]
    location_resource_names = get_location_resource_names_for_countries(
        google_ads_client,
        all_locations,
    )
    country_criterion_ids = ", ".join(
        name.split("/")[-1] for name in location_resource_names.values()
    )

    ads_service = google_ads_client.get_service("GoogleAdsService")
    output_path = Path(f"data/{output_course}/reports/bid_adj/loc_clicks{output_suffix}.csv")
    stream = ads_service.search_stream(
        customer_id=customer_id,
        query=LOC_CLICKS_REPORT_QUERY.format(
            start_date=start_date,
            end_date=end_date,
            country_criterion_ids=country_criterion_ids,
        ),
    )
    write_to_file(
        ["Campaign", "Targeted location", "Clicks"],
        generate_loc_clicks_rows(stream, google_ads_client, customer_id),
        output_path,
        delimiter=",",
    )
    print(f"Generated: {output_path}")

    output_path_conv = Path(f"data/{output_course}/reports/bid_adj/loc_conv{output_suffix}.csv")
    stream_conv = ads_service.search_stream(
        customer_id=customer_id,
        query=LOC_CONVERSIONS_REPORT_QUERY.format(
            start_date=start_date,
            end_date=end_date,
            purchase_action_list="', '".join(COURSE_CONFIG[output_course]["purchase_actions"]),
            country_criterion_ids=country_criterion_ids,
        ),
    )
    write_to_file(
        ["Campaign", "Conversion action", "Targeted location", "All conv."],
        generate_loc_conversions_rows(stream_conv, google_ads_client, customer_id),
        output_path_conv,
        delimiter=",",
    )
    google_ads_metrics_client.track_google_ads_operation_count("search_stream", 2)
    print(f"Generated: {output_path_conv}")
