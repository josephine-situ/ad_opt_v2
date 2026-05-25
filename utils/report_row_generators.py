from collections import defaultdict
from decimal import Decimal
from typing import Any, Iterator


def _format_match_type(match_type_enum: Any) -> str:
    return match_type_enum.name.replace("_", " ").title()


def aggregate_kw_all_conv_totals(stream: Any) -> dict[tuple[str, str, str, str], float]:
    """Sum filtered all_conversions by keyword-day key."""
    totals: dict[tuple[str, str, str, str], float] = defaultdict(float)
    for batch in stream:
        for row in batch.results:
            key = (
                row.segments.date,
                row.ad_group_criterion.keyword.text,
                row.campaign.name,
                _format_match_type(row.ad_group_criterion.keyword.match_type),
            )
            totals[key] += row.metrics.all_conversions
    return totals


def generate_kw_day_panel_rows(stream: Any) -> Iterator[dict[str, Any]]:
    for batch in stream:
        for row in batch.results:
            first_page_bid = ""
            if row.ad_group_criterion.position_estimates.first_page_cpc_micros:
                first_page_bid = (
                    f"{Decimal(row.ad_group_criterion.position_estimates.first_page_cpc_micros) / 1_000_000:.2f}"
                )

            yield {
                "date": row.segments.date,
                "keyword": row.ad_group_criterion.keyword.text,
                "campaign": row.campaign.name,
                "match_type": _format_match_type(row.ad_group_criterion.keyword.match_type),
                "clicks": row.metrics.clicks,
                "cost": f"{Decimal(row.metrics.cost_micros) / 1_000_000:.2f}",
                "currency_code": row.customer.currency_code,
                "first_page_cpc": first_page_bid,
            }
