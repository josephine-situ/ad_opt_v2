from collections.abc import Iterable
from typing import Any

from google.ads.googleads.client import GoogleAdsClient

from utils.gaql_queries import BUILD_LOCATION_CACHE_QUERY
from utils.metrics import google_ads_metrics_client

LOCATION_CACHE: dict[int, str] = {}
GEO_TARGET_BATCH_SIZE = 25


def get_location_resource_names_for_countries(
    google_ads_client: GoogleAdsClient,
    unique_countries: Iterable[str],
) -> dict[str, str]:
    """Resolve country/region names to geo target constant resource names."""
    unique_countries = list(set(unique_countries))
    if not unique_countries:
        return {}

    geo_target_service = google_ads_client.get_service("GeoTargetConstantService")
    location_map: dict[str, str] = {}

    for i in range(0, len(unique_countries), GEO_TARGET_BATCH_SIZE):
        batch = unique_countries[i : i + GEO_TARGET_BATCH_SIZE]
        request = google_ads_client.get_type("SuggestGeoTargetConstantsRequest")
        request.location_names.names.extend(batch)
        request.locale = "en"

        suggestions = geo_target_service.suggest_geo_target_constants(request=request)
        for suggestion in suggestions.geo_target_constant_suggestions:
            search_term = suggestion.search_term
            if search_term not in location_map:
                location_map[search_term] = suggestion.geo_target_constant.resource_name

    google_ads_metrics_client.track_google_ads_operation_count("suggest_geo_target_constants", 1)
    missing = set(unique_countries) - set(location_map.keys())
    if missing:
        raise ValueError(f"Locations not found for countries: {missing}")

    return location_map


def build_location_cache(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    country_criterion_ids: Iterable[int],
) -> None:
    """Build a human-readable location cache for country criterion IDs."""
    if not country_criterion_ids:
        return

    valid_ids = [
        criterion_id
        for criterion_id in set(country_criterion_ids)
        if criterion_id is not None and criterion_id not in LOCATION_CACHE
    ]
    if not valid_ids:
        return

    try:
        ads_service = google_ads_client.get_service("GoogleAdsService")
        ids_str = ", ".join(str(criterion_id) for criterion_id in valid_ids)
        query = BUILD_LOCATION_CACHE_QUERY.format(ids_str=ids_str)
        response = ads_service.search(customer_id=customer_id, query=query)
        google_ads_metrics_client.track_google_ads_operation_count("search", 1)

        for row in response:
            LOCATION_CACHE[row.geo_target_constant.id] = row.geo_target_constant.canonical_name

        for criterion_id in valid_ids:
            if criterion_id not in LOCATION_CACHE:
                print(f"Warning: Could not find location name for criterion {criterion_id}")
                LOCATION_CACHE[criterion_id] = f"Location {criterion_id}"
    except Exception as exc:
        print(f"Warning: Error fetching location names: {exc}")
        for criterion_id in valid_ids:
            LOCATION_CACHE.setdefault(criterion_id, f"Location {criterion_id}")


def get_from_location_cache(criterion_id: int) -> Any:
    return LOCATION_CACHE[criterion_id]
