"""Utility functions for reading state from the Google Ads API."""

from collections.abc import Iterable

from google.ads.googleads.client import GoogleAdsClient

from utils.metrics import google_ads_metrics_client

GEO_TARGET_BATCH_SIZE = 25  # API limit per SuggestGeoTargetConstantsRequest


def get_location_resource_names_for_countries(
    google_ads_client: GoogleAdsClient, unique_countries: Iterable[str]
) -> dict[str, str]:
    """
    Get geo target constant resource names for a list of country/region names.
    Returns a dict mapping each input name to its geo target constant resource name.
    Sends names in batches of up to 25 (API limit) and raises ValueError for any unresolved names.
    """
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

        # When multiple suggestions share the same search_term, keep only the first (best) match.
        for suggestion in suggestions.geo_target_constant_suggestions:
            search_term = suggestion.search_term
            if search_term not in location_map:
                location_map[search_term] = suggestion.geo_target_constant.resource_name

    google_ads_metrics_client.track_google_ads_operation_count("suggest_geo_target_constants", 1)
    missing = set(unique_countries) - set(location_map.keys())
    if missing:
        raise ValueError(f"Locations not found for countries: {missing}")

    return location_map
