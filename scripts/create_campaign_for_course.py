#!/usr/bin/env python3
"""
Script to create Google Ads campaigns and ad groups for a given course.
Creates campaigns following the structure: one campaign per (Course, Region, Match Type) tuple.
Each campaign contains exactly one ad group.
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from typing import Optional

from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.v23.services.types.campaign_budget_service import CampaignBudgetOperation
from google.ads.googleads.v23.services.types.campaign_service import CampaignOperation
from google.ads.googleads.v23.services.types.ad_group_service import AdGroupOperation
from google.ads.googleads.v23.services.types.campaign_criterion_service import (
    CampaignCriterionOperation,
)
from google.ads.googleads.v23.services.types.ad_group_criterion_service import (
    AdGroupCriterionOperation,
)

from utils.campaign_config import load_config_dict
from utils.gaql_queries import SELECT_EXISTING_KEYWORDS_BY_AD_GROUP_RESOURCE, SELECT_AD_GROUPS_BY_CAMPAIGN_NAME
from utils.bid_adjustments import AGE_RANGE_MAP
from utils.google_ads_api import get_location_resource_names_for_countries
from utils.metrics import google_ads_metrics_client
from utils.name_generation import construct_campaign_name_for_args, construct_ad_group_name_for_args, construct_budget_name_for_args, get_match_types_for_label
from utils.paths import prod_dir, processed_dir

BATCH_SIZE = 5000  # Google Ads API limit
MATCH_TYPE_MAP = {"Exact match": "EXACT", "Phrase match": "PHRASE", "Broad match": "BROAD"}


# We may need to change this in the future, but for now this encapsulates the related resources we create for a campaign
# We'll need to probably add the notion of run, but for now, this will scaffold out campaigns acceptably.
@dataclass
class CampaignSpec:
    """Specification for a campaign and its associated resources."""

    campaign_name: str
    ad_group_name: str
    budget_name: str
    default_budget: int
    region_label: str
    match_type: str
    countries: list[str]
    budget_resource_name: Optional[str] = None
    campaign_resource_name: Optional[str] = None
    ad_group_resource_name: Optional[str] = None

    def __hash__(self) -> int:
        return hash(self.campaign_name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CampaignSpec):
            return NotImplemented
        return self.campaign_name == other.campaign_name


def find_spec_by_name(specs: list[CampaignSpec], name: str, field: str) -> Optional[CampaignSpec]:
    """
    Find a CampaignSpec by searching for a name in the specified field.

    Args:
        specs: List of CampaignSpec objects to search
        name: The name to search for
        field: The field to search in ('budget_name', 'campaign_name', or 'ad_group_name')

    Returns:
        The matching CampaignSpec or None if not found
    """
    for spec in specs:
        if getattr(spec, field) == name:
            return spec
    return None


def _load_fixed_keyword_sets(course: str) -> dict[str, str]:
    path = prod_dir(course) / "two_stage_plan" / "fixed_keyword_sets.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_keyword_set(course: str, filename: str) -> dict[str, list[str]]:
    """Load a keyword set CSV and return a dict mapping uppercase match type to keyword list."""
    path = processed_dir(course) / "keyword-sets-display" / f"{filename}.csv"
    if not path.exists():
        print(f"Warning: Keyword set file not found: {path}")
        return {}
    keywords_by_match_type: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col, value in row.items():
                if value and value.strip():
                    match_type = col.strip().upper()
                    keywords_by_match_type.setdefault(match_type, []).append(value.strip())
    return keywords_by_match_type


def get_keywords_to_create(
    course: str, campaign_specs: list[CampaignSpec]
) -> dict[CampaignSpec, dict[str, list[str]]]:
    """
    Get keywords to create for a list of CampaignSpecs.

    Looks up each spec in the course's fixed_keyword_sets.json (keyed by
    "{region} / {match_type_label}") to find the corresponding keyword set file,
    then reads keywords per match type from that file.

    A campaign spec's match_type field may contain one or more semicolon-separated
    match type labels (e.g. "Phrase; Exact"), and keywords are returned for each.

    Returns a dict keyed by CampaignSpec, where each value is a dict mapping individual
    match type strings (e.g. "EXACT") to the list of keyword texts for that spec.
    """
    fixed_keyword_sets = _load_fixed_keyword_sets(course)
    if not fixed_keyword_sets:
        print(f"Warning: No fixed_keyword_sets.json found for course '{course}'. No keywords will be created.")
        return {}

    result: dict[CampaignSpec, dict[str, list[str]]] = {}
    for spec in campaign_specs:
        lookup_key = f"{spec.region_label} / {spec.match_type}"
        filename = fixed_keyword_sets.get(lookup_key)
        if not filename:
            print(f"Warning: No keyword set found for '{lookup_key}' in fixed_keyword_sets.json. Skipping '{spec.campaign_name}'.")
            result[spec] = {}
            continue

        all_keywords = _load_keyword_set(course, filename)
        match_types = get_match_types_for_label(spec.match_type)
        result[spec] = {mt: all_keywords[mt] for mt in match_types if mt in all_keywords}

    print("Found keywords to create for the following campaigns:")
    for spec, mt_keywords in result.items():
        for match_type, kw_list in mt_keywords.items():
            print(f"  - {spec.campaign_name} | {match_type}: {len(kw_list)} keywords")

    return result


def create_campaign_budget_operation(
    google_ads_client: GoogleAdsClient, budget_name: str, daily_budget_micros: int
) -> CampaignBudgetOperation:
    """Create a campaign budget operation."""
    operation = google_ads_client.get_type("CampaignBudgetOperation")
    campaign_budget = operation.create
    campaign_budget.name = budget_name
    campaign_budget.amount_micros = daily_budget_micros
    campaign_budget.delivery_method = google_ads_client.enums.BudgetDeliveryMethodEnum.STANDARD
    # By default, budgets are created as "shared", which doesn't appear to be how the existing ones are set up.
    # We'll match that pattern for the time being, but it's easy enough to change in the future.
    # See https://developers.google.com/google-ads/api/reference/rpc/v23/CampaignBudget#explicitly_shared for more info
    campaign_budget.explicitly_shared = False
    return operation


def create_campaign_operation(
    google_ads_client: GoogleAdsClient, campaign_name: str, budget_resource_name: str
) -> CampaignOperation:
    """Create a Search campaign operation."""
    operation = google_ads_client.get_type("CampaignOperation")
    campaign = operation.create
    campaign.name = campaign_name
    campaign.advertising_channel_type = google_ads_client.enums.AdvertisingChannelTypeEnum.SEARCH
    campaign.status = google_ads_client.enums.CampaignStatusEnum.PAUSED
    campaign.campaign_budget = budget_resource_name

    # Set manual CPC bidding strategy
    campaign.manual_cpc.enhanced_cpc_enabled = False

    # Set network settings
    # TODO: Not quite sure if this what we want, but it's good enough for now.
    campaign.network_settings.target_google_search = True
    campaign.network_settings.target_search_network = True
    campaign.network_settings.target_content_network = False
    campaign.network_settings.target_partner_search_network = False
    campaign.contains_eu_political_advertising = (
        google_ads_client.enums.EuPoliticalAdvertisingStatusEnum.DOES_NOT_CONTAIN_EU_POLITICAL_ADVERTISING
    )
    return operation


def create_ad_group_operation(
    google_ads_client: GoogleAdsClient, ad_group_name: str, campaign_resource_name: str
) -> AdGroupOperation:
    """Create an ad group operation."""
    operation = google_ads_client.get_type("AdGroupOperation")
    ad_group = operation.create
    ad_group.name = ad_group_name
    ad_group.campaign = campaign_resource_name
    ad_group.status = google_ads_client.enums.AdGroupStatusEnum.ENABLED
    ad_group.type_ = google_ads_client.enums.AdGroupTypeEnum.SEARCH_STANDARD
    return operation


def create_ad_group_keyword_operations(
    google_ads_client: GoogleAdsClient,
    ad_group_resource_name: str,
    keywords: list[str],
    match_type: str,
) -> list[AdGroupCriterionOperation]:
    """
    Create an operation to add a keyword criterion to an ad group.

    Args:
        google_ads_client: The Google Ads client instance.
        ad_group_resource_name: The resource name of the ad group to add the keyword to.
        keywords: List of keyword text values (e.g. "machine learning course").
        match_type: The keyword match type — one of "exact", "phrase", or "broad".

    Returns:
        An AdGroupCriterionOperation ready to be submitted via AdGroupCriterionService.
    """
    operations = []
    for keyword_text in keywords:
        operation = google_ads_client.get_type("AdGroupCriterionOperation")
        criterion = operation.create
        criterion.ad_group = ad_group_resource_name
        criterion.status = google_ads_client.enums.AdGroupCriterionStatusEnum.PAUSED
        criterion.keyword.text = keyword_text
        # TODO: Redo this match type setting. We should be more defensive
        criterion.keyword.match_type = match_type.upper()
        operations.append(operation)
    return operations


def create_age_range_criteria_operations(
    google_ads_client: GoogleAdsClient, ad_group_resource_name: str
) -> list[AdGroupCriterionOperation]:
    """Create age range criterion operations for all age groups in AGE_RANGE_MAP."""
    operations = []
    for age_range_type in AGE_RANGE_MAP.values():
        operation = google_ads_client.get_type("AdGroupCriterionOperation")
        criterion = operation.create
        criterion.ad_group = ad_group_resource_name
        criterion.status = google_ads_client.enums.AdGroupCriterionStatusEnum.ENABLED
        criterion.age_range.type_ = age_range_type
        operations.append(operation)
    return operations


def create_ad_schedule_operations(
    google_ads_client: GoogleAdsClient, campaign_resource_name: str
) -> list[CampaignCriterionOperation]:
    """
    Create ad schedule operations for a campaign.
    Creates 6 four-hour windows starting at 00:00 (covering the full day).
    """
    operations = []

    # Create 6 four-hour windows: 0-4, 4-8, 8-12, 12-16, 16-20, 20-24
    for window in range(6):
        start_hour = window * 4
        end_hour = (window + 1) * 4

        # Create ad schedule for each day of the week
        days_of_week = [
            google_ads_client.enums.DayOfWeekEnum.MONDAY,
            google_ads_client.enums.DayOfWeekEnum.TUESDAY,
            google_ads_client.enums.DayOfWeekEnum.WEDNESDAY,
            google_ads_client.enums.DayOfWeekEnum.THURSDAY,
            google_ads_client.enums.DayOfWeekEnum.FRIDAY,
            google_ads_client.enums.DayOfWeekEnum.SATURDAY,
            google_ads_client.enums.DayOfWeekEnum.SUNDAY,
        ]

        for day_of_week in days_of_week:
            operation = google_ads_client.get_type("CampaignCriterionOperation")
            criterion = operation.create
            criterion.campaign = campaign_resource_name
            criterion.status = google_ads_client.enums.CampaignCriterionStatusEnum.ENABLED

            # Set ad schedule
            ad_schedule = criterion.ad_schedule
            ad_schedule.start_hour = start_hour
            ad_schedule.start_minute = google_ads_client.enums.MinuteOfHourEnum.ZERO
            ad_schedule.end_hour = end_hour
            ad_schedule.end_minute = google_ads_client.enums.MinuteOfHourEnum.ZERO
            ad_schedule.day_of_week = day_of_week

            operations.append(operation)

    return operations


def create_location_operations(
    google_ads_client: GoogleAdsClient,
    campaign_resource_name: str,
    countries: list[str],
    location_map: dict[str, str],
) -> list[CampaignCriterionOperation]:
    """
    Create location targeting operations for a campaign using pre-fetched geo target resource names.
    """
    operations = []

    for country in countries:
        operation = google_ads_client.get_type("CampaignCriterionOperation")
        criterion = operation.create
        criterion.campaign = campaign_resource_name
        criterion.status = google_ads_client.enums.CampaignCriterionStatusEnum.ENABLED
        criterion.location.geo_target_constant = location_map[country]

        operations.append(operation)

    return operations

def create_remaining_keyword_criteria(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    course: str,
    campaign_specs: list[CampaignSpec],
    keywords_limit: int
) -> None:
    """
    Create keyword criteria for existing ad groups based on the provided campaign specifications.
    This is used in --only-keywords mode, where we assume campaigns and ad groups have already been created in a previous run with --skip-keywords.
    We look up ad groups by campaign name (which we control and is reliable) to get their resource
    names, query all already-existing keyword criteria for those ad groups by resource name, and
    only create keywords that are not yet present.
    """
    google_ads_service = google_ads_client.get_service("GoogleAdsService")
    ad_group_criterion_service = google_ads_client.get_service("AdGroupCriterionService")
    spec_to_keywords = get_keywords_to_create(course, campaign_specs)

    # Query 1: get ad group resource names via campaign name to avoid ambiguity from non-unique ad group names.
    print(f"\nFetching ad group resource names for {len(campaign_specs)} campaigns...")
    campaign_names_list = "', '".join(spec.campaign_name for spec in campaign_specs)
    ag_stream = google_ads_service.search_stream(
        customer_id=customer_id,
        query=SELECT_AD_GROUPS_BY_CAMPAIGN_NAME.format(campaign_names_list=campaign_names_list),
    )
    google_ads_metrics_client.track_google_ads_operation_count('search_stream', 1)
    ad_group_count = 0
    for batch in ag_stream:
        for row in batch.results:
            spec = find_spec_by_name(campaign_specs, row.campaign.name, "campaign_name")
            spec.ad_group_resource_name = row.ad_group.resource_name
            ad_group_count += 1

    print(f"Found {ad_group_count} of {len(campaign_specs)} ad groups.")

    # Query 2: fetch existing keyword criteria by ad group resource name to avoid duplicate creation.
    print(f"Fetching existing keyword criteria for {ad_group_count} ad groups...")
    existing_keywords: set[tuple[str, str, str]] = set()
    resource_names = [spec.ad_group_resource_name for spec in campaign_specs if spec.ad_group_resource_name]
    if resource_names:
        ad_group_resource_names_list = "', '".join(resource_names)
        kw_stream = google_ads_service.search_stream(
            customer_id=customer_id,
            query=SELECT_EXISTING_KEYWORDS_BY_AD_GROUP_RESOURCE.format(
                ad_group_resource_names_list=ad_group_resource_names_list
            ),
        )
        google_ads_metrics_client.track_google_ads_operation_count('search_stream', 1)
        for batch in kw_stream:
            for row in batch.results:
                existing_keywords.add((
                    row.ad_group.resource_name,
                    row.ad_group_criterion.keyword.text.lower(),
                    row.ad_group_criterion.keyword.match_type.name,
                ))
    print(f"Found {len(existing_keywords)} existing keyword criteria.")

    # Create only keywords not already present in each ad group
    remaining_keyword_operations = keywords_limit
    exhausted_limit = False
    for spec in campaign_specs:
        ad_group_resource_name = spec.ad_group_resource_name
        if not ad_group_resource_name:
            continue

        keywords_by_match_type = spec_to_keywords.get(spec, {})
        if not keywords_by_match_type:
            print(
                f"Warning: No keywords found for campaign '{spec.campaign_name}' with region '{spec.region_label}' and match type '{spec.match_type}'."
            )
            continue

        for match_type, all_keywords in keywords_by_match_type.items():
            new_keywords = [
                kw for kw in all_keywords
                if (ad_group_resource_name, kw.lower(), match_type) not in existing_keywords
            ]

            skipped = len(all_keywords) - len(new_keywords)
            if skipped:
                print(f"Skipping {skipped} already-existing keywords for ad group '{spec.ad_group_name}' ({match_type}).")
            if not new_keywords:
                print(f"All keywords already exist for ad group '{spec.ad_group_name}' ({match_type}). Nothing to create.")
                continue

            operations = create_ad_group_keyword_operations(
                google_ads_client, ad_group_resource_name, new_keywords, match_type
            )

            # If we have a keyword limit set
            if remaining_keyword_operations and len(operations) > remaining_keyword_operations:
                print(f"Keyword operations for ad group '{spec.ad_group_name}' exceed remaining limit of {remaining_keyword_operations}. Only creating a portion of keywords for this ad group.")  # noqa: E501
                operations = operations[:remaining_keyword_operations]
                remaining_keyword_operations = 0
                exhausted_limit = True
            elif remaining_keyword_operations:
                remaining_keyword_operations -= len(operations)

            try:
                for i in range(0, len(operations), BATCH_SIZE):
                    batch = operations[i : i + BATCH_SIZE]
                    request = google_ads_client.get_type("MutateAdGroupCriteriaRequest")
                    request.customer_id = customer_id
                    request.operations = batch
                    response = ad_group_criterion_service.mutate_ad_group_criteria(request=request)
                    google_ads_metrics_client.track_google_ads_operation_count('mutate_ad_group_criteria', len(batch))
                    print(f"✓ Created {len(response.results)} keyword criteria for ad group '{spec.ad_group_name}' ({match_type}) (batch {i // BATCH_SIZE + 1})")
            except Exception as e:
                print(f"✗ Error creating keyword criteria for ad group '{spec.ad_group_name}' ({match_type}): {e}")

            if exhausted_limit:
                print(f"Keyword limit reached, skipping additional keywords.")
                return

def create_campaigns_for_course(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    course: str,
    execute: bool,
    skip_keywords: bool = False,
    only_keywords: bool = False,
    keywords_limit: int = 0,
) -> list[CampaignSpec]:
    """Create all campaigns and ad groups for a given course."""
    course_config = load_config_dict(course)
    if not course_config.get("regions"):
        print(f"Error: Course '{course}' has no 'regions' defined in its course.yaml")
        sys.exit(1)

    regions = course_config.get("regions", {})
    match_types = course_config.get("match_types", [])
    default_budget = course_config.get("default_daily_budget_micros", 1_000_000)

    # Collect all unique countries across all regions, deduplicate in case of manual errors in config
    all_countries = set()
    for countries in regions.values():
        all_countries.update(countries)

    # Fetch all location resource names via GeoTargetConstantService
    print(f"\n{'='*60}")
    print(f"Fetching geo target constants for {len(all_countries)} countries...")
    print(f"{'='*60}")
    location_map = get_location_resource_names_for_countries(google_ads_client, all_countries)
    print(f"✓ Retrieved {len(location_map)} geo target constants")

    # Prepare all campaign specifications
    campaign_specs = []
    for region_label, countries in regions.items():
        for match_type in match_types:
            campaign_name = construct_campaign_name_for_args(course, match_type, region_label)
            ad_group_name = construct_ad_group_name_for_args(course, match_type, region_label)
            budget_name = construct_budget_name_for_args(course, match_type, region_label)

            spec = CampaignSpec(
                campaign_name=campaign_name,
                ad_group_name=ad_group_name,
                budget_name=budget_name,
                default_budget=default_budget,
                region_label=region_label,
                countries=countries,
                match_type=match_type,
            )
            campaign_specs.append(spec)

            print(f"Planned: {campaign_name}")

    if not execute:
        print(f"\n[DRY RUN] Would create {len(campaign_specs)} campaigns with ad groups:")
        for spec in campaign_specs:
            print(
                f"  - Campaign: {spec.campaign_name} | Ad Group: {spec.ad_group_name} | "
                f"Budget: {spec.budget_name} ({spec.default_budget} micros) | "
                f"Region: {spec.region_label} | Countries: {', '.join(spec.countries)}"
            )
        return []

    if only_keywords:
        print("\n[KEYWORD-ONLY MODE] Only creating keyword criteria for existing ad groups.")
        print("Assuming campaigns and ad groups have already been created in a previous run with --skip-keywords.")
        print("Will attempt to find ad groups by name and add keywords to them, but will not create any new campaigns or ad groups.")
        create_remaining_keyword_criteria(google_ads_client, customer_id, course, campaign_specs, keywords_limit)
        return []

    # Batch create all budgets
    print(f"\n{'='*60}")
    print(f"Creating {len(campaign_specs)} campaign budgets...")
    print(f"{'='*60}")

    campaign_budget_service = google_ads_client.get_service("CampaignBudgetService")
    budget_operations = [
        create_campaign_budget_operation(google_ads_client, spec.budget_name, spec.default_budget)
        for spec in campaign_specs
    ]

    try:
        # Create request with response_content_type to get full resource data back
        request = google_ads_client.get_type("MutateCampaignBudgetsRequest")
        request.customer_id = customer_id
        request.operations = budget_operations
        # MUTABLE_RESOURCE is required for things like result.campaign_budget.name to be populated in the response.
        request.response_content_type = (
            google_ads_client.enums.ResponseContentTypeEnum.MUTABLE_RESOURCE
        )

        budget_response = campaign_budget_service.mutate_campaign_budgets(request=request)
        google_ads_metrics_client.track_google_ads_operation_count('mutate_campaign_budgets', len(budget_operations))
        print(f"✓ Created {len(budget_response.results)} budgets")

        # Map each result back to the corresponding spec using the budget name from response
        for result in budget_response.results:
            budget_name = result.campaign_budget.name
            spec = find_spec_by_name(campaign_specs, budget_name, "budget_name")
            if spec:
                spec.budget_resource_name = result.resource_name
    except Exception as e:
        print(f"✗ Error creating budgets: {e}")
        sys.exit(1)

    # Batch create all campaigns
    print(f"\n{'='*60}")
    print(f"Creating {len(campaign_specs)} campaigns...")
    print(f"{'='*60}")

    campaign_service = google_ads_client.get_service("CampaignService")
    campaign_operations = [
        create_campaign_operation(google_ads_client, spec.campaign_name, spec.budget_resource_name)
        for spec in campaign_specs
    ]

    try:
        request = google_ads_client.get_type("MutateCampaignsRequest")
        request.customer_id = customer_id
        request.operations = campaign_operations
        request.response_content_type = (
            google_ads_client.enums.ResponseContentTypeEnum.MUTABLE_RESOURCE
        )

        campaign_response = campaign_service.mutate_campaigns(request=request)
        google_ads_metrics_client.track_google_ads_operation_count('mutate_campaigns', len(campaign_operations))
        print(f"✓ Created {len(campaign_response.results)} campaigns")

        for result in campaign_response.results:
            campaign_name = result.campaign.name
            spec = find_spec_by_name(campaign_specs, campaign_name, "campaign_name")
            if spec:
                spec.campaign_resource_name = result.resource_name
    except Exception as e:
        print(f"✗ Error creating campaigns: {e}")
        sys.exit(1)

    # Batch create all ad groups
    print(f"\n{'='*60}")
    print(f"Creating {len(campaign_specs)} ad groups...")
    print(f"{'='*60}")

    ad_group_service = google_ads_client.get_service("AdGroupService")
    ad_group_operations = [
        create_ad_group_operation(
            google_ads_client, spec.ad_group_name, spec.campaign_resource_name
        )
        for spec in campaign_specs
    ]

    try:
        request = google_ads_client.get_type("MutateAdGroupsRequest")
        request.customer_id = customer_id
        request.operations = ad_group_operations
        request.response_content_type = (
            google_ads_client.enums.ResponseContentTypeEnum.MUTABLE_RESOURCE
        )

        ad_group_response = ad_group_service.mutate_ad_groups(request=request)
        google_ads_metrics_client.track_google_ads_operation_count('mutate_ad_groups', len(ad_group_operations))
        print(f"✓ Created {len(ad_group_response.results)} ad groups")

        for result in ad_group_response.results:
            ad_group_name = result.ad_group.name
            spec = find_spec_by_name(campaign_specs, ad_group_name, "ad_group_name")
            if spec:
                spec.ad_group_resource_name = result.resource_name
    except Exception as e:
        print(f"✗ Error creating ad groups: {e}")
        sys.exit(1)


    ad_group_criterion_service = google_ads_client.get_service("AdGroupCriterionService")
    if skip_keywords:
        print("\nSkipping keyword criteria population (--skip-keywords flag set).")
    else:
        # TODO: We may need to pull this out to allow for partial execution of keywords if we dont get standard api access
        # As this works now, we attempt to create all keywords for all campaigns in one batch.
        # Some courses have a dataset too large for this with API Basic Access quotas
        spec_to_keywords = get_keywords_to_create(course, campaign_specs)
        keyword_operations = []
        for spec in campaign_specs:
            ad_group_resource_name = spec.ad_group_resource_name
            keywords_by_match_type = spec_to_keywords.get(spec, {})
            if not keywords_by_match_type:
                print(
                    f"Warning: No keywords found for campaign '{spec.campaign_name}' with region '{spec.region_label}' and match type '{spec.match_type}'."
                )
            else:
                for match_type, keywords in keywords_by_match_type.items():
                    keyword_operations.extend(
                        create_ad_group_keyword_operations(
                            google_ads_client, ad_group_resource_name, keywords, match_type
                        )
                    )

        total_created = 0
        try:
            for i in range(0, len(keyword_operations), BATCH_SIZE):
                batch = keyword_operations[i : i + BATCH_SIZE]
                request = google_ads_client.get_type("MutateAdGroupCriteriaRequest")
                request.customer_id = customer_id
                request.operations = batch
                response = ad_group_criterion_service.mutate_ad_group_criteria(request=request)
                google_ads_metrics_client.track_google_ads_operation_count('mutate_ad_group_criteria', len(batch))
                total_created += len(response.results)
                print(f"✓ Created {len(response.results)} keyword criteria (batch {i // BATCH_SIZE + 1})")
        except Exception as e:
            print(f"✗ Error creating keyword criteria: {e}")
            sys.exit(1)
        print(f"✓ Created {total_created} keyword criteria in total")

    # Batch create age range criteria for all ad groups
    print(f"\n{'='*60}")
    print(f"Creating age range criteria for {len(campaign_specs)} ad groups...")
    print(f"{'='*60}")

    all_age_range_operations = []
    for spec in campaign_specs:
        all_age_range_operations.extend(
            create_age_range_criteria_operations(google_ads_client, spec.ad_group_resource_name)
        )

    try:
        request = google_ads_client.get_type("MutateAdGroupCriteriaRequest")
        request.customer_id = customer_id
        request.operations = all_age_range_operations

        age_range_response = ad_group_criterion_service.mutate_ad_group_criteria(request=request)
        google_ads_metrics_client.track_google_ads_operation_count('mutate_ad_group_criteria', len(all_age_range_operations))
        print(f"✓ Created {len(age_range_response.results)} age range criteria")
    except Exception as e:
        print(f"✗ Error creating age range criteria: {e}")
        sys.exit(1)

    # Batch create ad schedules for all campaigns
    print(f"\n{'='*60}")
    print(f"Creating ad schedules for {len(campaign_specs)} campaigns...")
    print(f"{'='*60}")

    # Ad schedules and location targeting could be batched together, but we'll keep them seperate for now
    # It's not much less efficient, and this lets us have more granular logging and error handling for each type of criterion if we need it.
    campaign_criterion_service = google_ads_client.get_service("CampaignCriterionService")
    all_ad_schedule_operations = []

    for spec in campaign_specs:
        ad_schedule_ops = create_ad_schedule_operations(
            google_ads_client, spec.campaign_resource_name
        )
        all_ad_schedule_operations.extend(ad_schedule_ops)

    try:
        request = google_ads_client.get_type("MutateCampaignCriteriaRequest")
        request.customer_id = customer_id
        request.operations = all_ad_schedule_operations

        ad_schedule_response = campaign_criterion_service.mutate_campaign_criteria(request=request)
        google_ads_metrics_client.track_google_ads_operation_count('mutate_campaign_criteria',
                                                                   len(all_ad_schedule_operations))
        print(f"✓ Created {len(ad_schedule_response.results)} ad schedule criteria")
    except Exception as e:
        print(f"✗ Error creating ad schedules: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Creating location targeting for {len(campaign_specs)} campaigns...")
    print(f"{'='*60}")

    all_location_operations = []

    for spec in campaign_specs:
        location_ops = create_location_operations(
            google_ads_client, spec.campaign_resource_name, spec.countries, location_map
        )
        all_location_operations.extend(location_ops)

    try:
        request = google_ads_client.get_type("MutateCampaignCriteriaRequest")
        request.customer_id = customer_id
        request.operations = all_location_operations

        location_response = campaign_criterion_service.mutate_campaign_criteria(request=request)
        google_ads_metrics_client.track_google_ads_operation_count('mutate_campaign_criteria',
                                                                   len(all_location_operations))
        print(f"✓ Created {len(location_response.results)} location targeting criteria")
    except Exception as e:
        print(f"✗ Error creating location targeting: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(
        f"Summary: Successfully created {len(campaign_specs)} campaigns with ad schedules, location targeting, and age range criteria"
    )
    print(f"{'='*60}")

    return campaign_specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Google Ads campaigns and ad groups for a given course. Be advised that nothing in this script is transactional and it doesn't attempt to avoid collisions. Use with caution."
    )
    parser.add_argument(
        "--course",
        type=str,
        choices=["gen_ai", "ml", "sys_eng", "sys_think"],
        required=True,
        help="The course to create campaigns for",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Execute the campaign creation (without this flag, runs in dry-run mode)",
    )
    parser.add_argument(
        "--google-ads-yaml",
        type=str,
        required=True,
        help="Path to Google Ads YAML configuration file",
    )
    parser.add_argument(
        "--customer-id",
        type=str,
        required=True,
        help="Google Ads customer ID",
    )

    parser.add_argument(
        "--skip-keywords",
        action="store_true",
        default=False,
        help="Skip populating keyword criteria for ad groups",
    )
    parser.add_argument(
        "--only-keywords",
        action="store_true",
        default=False,
        help="Only create keyword criteria for existing ad groups. Assumes this script has been run w/ --skip-keywords previously",
    )
    parser.add_argument(
        "--keywords-limit",
        type=int,
        default=0,
        help="When specified with --only-keywords, limits the number of operations to perform.",
    )

    args = parser.parse_args()

    yaml_path = args.google_ads_yaml
    customer_id = args.customer_id

    # Initialize Google Ads client
    google_ads_client = GoogleAdsClient.load_from_storage(yaml_path)

    # Create campaigns
    create_campaigns_for_course(google_ads_client, customer_id, args.course, args.execute, args.skip_keywords, args.only_keywords, args.keywords_limit)


if __name__ == "__main__":
    main()
