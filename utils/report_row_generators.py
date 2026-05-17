from collections import defaultdict
from decimal import Decimal
from typing import Any, Iterator

from google.ads.googleads.client import GoogleAdsClient

from utils.bid_adjustments import AGE_ENUM_TO_RANGE, DEVICE_ENUM_TO_NAME
from utils.google_ads_api import build_location_cache, get_from_location_cache


def generate_search_keyword_rows(stream: Any) -> Iterator[dict[str, Any]]:
    for batch in stream:
        for row in batch.results:
            first_page_bid = ""
            if row.ad_group_criterion.position_estimates.first_page_cpc_micros:
                first_page_bid = (
                    f"{Decimal(row.ad_group_criterion.position_estimates.first_page_cpc_micros) / 1_000_000:.2f}"
                )

            yield {
                "Day": row.segments.date,
                "Search keyword": row.ad_group_criterion.keyword.text,
                "Search keyword match type": row.ad_group_criterion.keyword.match_type.name.replace(
                    "_", " "
                ).title(),
                "Campaign": row.campaign.name,
                "Clicks": row.metrics.clicks,
                "Conv. value": f"{row.metrics.all_conversions_value:.2f}",
                "Currency code": row.customer.currency_code,
                "Cost": f"{Decimal(row.metrics.cost_micros) / 1_000_000:.2f}",
                "First page CPC": first_page_bid,
            }


def generate_kw_day_panel_rows(stream: Any) -> Iterator[dict[str, Any]]:
    for batch in stream:
        for row in batch.results:
            yield {
                "date": row.segments.date,
                "keyword": row.ad_group_criterion.keyword.text,
                "campaign": row.campaign.name,
                "match_type": row.ad_group_criterion.keyword.match_type.name.replace(
                    "_", " "
                ).title(),
                "impressions": row.metrics.impressions,
                "clicks": row.metrics.clicks,
                "cost": f"{Decimal(row.metrics.cost_micros) / 1_000_000:.2f}",
                "conversions": f"{row.metrics.conversions:.2f}",
                "impression_share": f"{row.metrics.search_impression_share:.4f}",
            }


def generate_purchase_report_rows(stream: Any) -> Iterator[dict[str, Any]]:
    for batch in stream:
        for row in batch.results:
            yield {
                "Campaign": row.campaign.name,
                "Conversion action": row.segments.conversion_action_name,
                "All conv.": f"{row.metrics.all_conversions:.2f}",
            }


def generate_hod_clicks_rows(stream: Any) -> Iterator[dict[str, Any]]:
    for batch in stream:
        for row in batch.results:
            yield {
                "Campaign": row.campaign.name,
                "Hour of the day": row.segments.hour,
                "Clicks": row.metrics.clicks,
            }


def generate_age_clicks_rows(stream: Any) -> Iterator[dict[str, Any]]:
    aggregated: defaultdict[tuple[str, str], int] = defaultdict(int)

    for batch in stream:
        for row in batch.results:
            age_type = row.ad_group_criterion.age_range.type_
            age_display = AGE_ENUM_TO_RANGE.get(age_type, "")
            aggregated[(row.campaign.name, age_display)] += row.metrics.clicks

    for (campaign, age), clicks in sorted(aggregated.items()):
        yield {"Campaign": campaign, "Age": age, "Clicks": clicks}


def generate_device_clicks_rows(stream: Any) -> Iterator[dict[str, Any]]:
    for batch in stream:
        for row in batch.results:
            device_display = DEVICE_ENUM_TO_NAME.get(row.segments.device, "")
            yield {
                "Campaign": row.campaign.name,
                "Device": device_display,
                "Clicks": row.metrics.clicks,
            }


def generate_loc_clicks_rows(
    stream: Any,
    google_ads_client: GoogleAdsClient,
    customer_id: str,
) -> Iterator[dict[str, Any]]:
    rows_data = []
    criterion_ids = set()

    for batch in stream:
        for row in batch.results:
            rows_data.append(row)
            if row.geographic_view.country_criterion_id:
                criterion_ids.add(row.geographic_view.country_criterion_id)

    build_location_cache(google_ads_client, customer_id, criterion_ids)

    for row in rows_data:
        yield {
            "Campaign": row.campaign.name,
            "Targeted location": get_from_location_cache(row.geographic_view.country_criterion_id),
            "Clicks": row.metrics.clicks,
        }


def generate_hod_conversions_rows(stream: Any) -> Iterator[dict[str, Any]]:
    for batch in stream:
        for row in batch.results:
            yield {
                "Campaign": row.campaign.name,
                "Conversion action": row.segments.conversion_action_name,
                "Hour of the day": row.segments.hour,
                "All conv.": f"{row.metrics.all_conversions:.2f}",
            }


def generate_age_conversions_rows(stream: Any) -> Iterator[dict[str, Any]]:
    aggregated: defaultdict[tuple[str, str, str], float] = defaultdict(float)

    for batch in stream:
        for row in batch.results:
            age_type = row.ad_group_criterion.age_range.type_
            age_display = AGE_ENUM_TO_RANGE.get(age_type, "")
            key = (row.campaign.name, row.segments.conversion_action_name, age_display)
            aggregated[key] += row.metrics.all_conversions

    for (campaign, conversion_action, age), conversions in sorted(aggregated.items()):
        yield {
            "Campaign": campaign,
            "Conversion action": conversion_action,
            "Age": age,
            "All conv.": f"{conversions:.2f}",
        }


def generate_device_conversions_rows(stream: Any) -> Iterator[dict[str, Any]]:
    for batch in stream:
        for row in batch.results:
            device_display = DEVICE_ENUM_TO_NAME.get(row.segments.device, "")
            yield {
                "Campaign": row.campaign.name,
                "Conversion action": row.segments.conversion_action_name,
                "Device": device_display,
                "All conv.": f"{row.metrics.all_conversions:.2f}",
            }


def generate_loc_conversions_rows(
    stream: Any,
    google_ads_client: GoogleAdsClient,
    customer_id: str,
) -> Iterator[dict[str, Any]]:
    rows_data = []
    criterion_ids = set()

    for batch in stream:
        for row in batch.results:
            rows_data.append(row)
            if row.geographic_view.country_criterion_id:
                criterion_ids.add(row.geographic_view.country_criterion_id)

    build_location_cache(google_ads_client, customer_id, criterion_ids)

    for row in rows_data:
        yield {
            "Campaign": row.campaign.name,
            "Conversion action": row.segments.conversion_action_name,
            "Targeted location": get_from_location_cache(row.geographic_view.country_criterion_id),
            "All conv.": f"{row.metrics.all_conversions:.2f}",
        }


def generate_search_terms_row(stream: Any) -> Iterator[dict[str, Any]]:
    for batch in stream:
        for row in batch.results:
            yield {
                "Search keyword": row.segments.keyword.info.text,
                "Search keyword match type": row.segments.keyword.info.match_type.name.replace(
                    "_", " "
                ).title(),
                "Search term": row.search_term_view.search_term,
                "Conversion action": row.segments.conversion_action_name,
                "Conversions": f"{row.metrics.all_conversions:.2f}",
            }
