#!/usr/bin/env python3
"""Query recent Google Ads change_event rows for auto-applied recommendations."""

import argparse
import csv
import sys
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

from google.ads.googleads.client import GoogleAdsClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.gaql_queries import AUTO_APPLIED_RECOMMENDATIONS_QUERY
from utils.metrics import google_ads_metrics_client

MAX_LOOKBACK_DAYS = 30


def query_change_events(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    lookback_days: int,
    end_datetime: datetime | None = None,
) -> list[dict[str, Any]]:
    """Query the change_event table for auto-applied recommendation changes."""
    if lookback_days > MAX_LOOKBACK_DAYS:
        print(
            f"Warning: --lookback-days ({lookback_days}) exceeds the API limit of "
            f"{MAX_LOOKBACK_DAYS} days. Clamping to {MAX_LOOKBACK_DAYS}.",
            file=sys.stderr,
        )
        lookback_days = MAX_LOOKBACK_DAYS

    if end_datetime is None:
        end_datetime = datetime.now(UTC)

    start_datetime = end_datetime - timedelta(days=lookback_days)
    query = AUTO_APPLIED_RECOMMENDATIONS_QUERY.format(
        start_datetime=start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        end_datetime=end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
    )

    stream = google_ads_client.get_service("GoogleAdsService").search_stream(
        customer_id=customer_id,
        query=query,
    )

    rows = []
    for batch in stream:
        for row in batch.results:
            event = row.change_event
            rows.append(
                {
                    "resource_name": event.resource_name,
                    "change_date_time": event.change_date_time,
                    "change_resource_name": event.change_resource_name,
                    "change_resource_type": event.change_resource_type.name,
                    "resource_change_operation": event.resource_change_operation.name,
                    "changed_fields": ",".join(event.changed_fields.paths),
                    "client_type": event.client_type.name,
                    "user_email": event.user_email,
                    "campaign": event.campaign,
                    "ad_group": event.ad_group,
                }
            )

    google_ads_metrics_client.track_google_ads_operation_count("search_stream", 1)
    return rows


def print_results(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "resource_name",
        "change_date_time",
        "change_resource_name",
        "change_resource_type",
        "resource_change_operation",
        "changed_fields",
        "client_type",
        "user_email",
        "campaign",
        "ad_group",
    ]

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    print(buf.getvalue(), end="")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query Google Ads auto-applied recommendation changes."
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
        "--lookback-days",
        type=int,
        default=7,
        help=f"Number of days to look back (default: 7, max: {MAX_LOOKBACK_DAYS}).",
    )

    args = parser.parse_args()
    if args.lookback_days <= 0:
        print("Error: --lookback-days must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    google_ads_client = GoogleAdsClient.load_from_storage(args.google_ads_yaml)
    rows = query_change_events(
        google_ads_client=google_ads_client,
        customer_id=args.customer_id,
        lookback_days=args.lookback_days,
    )

    print_results(rows)
    print(f"Queried {args.lookback_days} day(s) of change events ({len(rows)} total).")


if __name__ == "__main__":
    main()
