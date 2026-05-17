SEARCH_KEYWORD_REPORT_QUERY = """
    SELECT
        segments.date,
        ad_group_criterion.keyword.text,
        ad_group_criterion.keyword.match_type,
        campaign.name,
        metrics.clicks,
        metrics.all_conversions_value,
        customer.currency_code,
        metrics.cost_micros,
        ad_group_criterion.position_estimates.first_page_cpc_micros
    FROM keyword_view
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    AND ad_group_criterion.keyword.match_type IN ('EXACT', 'PHRASE', 'BROAD')
    AND metrics.clicks > 0
    ORDER BY segments.date
"""

KW_DAY_PANEL_REPORT_QUERY = """
    SELECT
        segments.date,
        ad_group_criterion.keyword.text,
        campaign.name,
        ad_group_criterion.keyword.match_type,
        metrics.clicks,
        metrics.cost_micros,
        metrics.conversions,
        metrics.search_impression_share
    FROM keyword_view
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    AND ad_group_criterion.keyword.match_type IN ('EXACT', 'PHRASE', 'BROAD')
    AND metrics.clicks > 0
    ORDER BY segments.date, campaign.name, ad_group_criterion.keyword.text
"""

PURCHASE_REPORT_QUERY = """
    SELECT
        campaign.name,
        segments.conversion_action_name,
        metrics.all_conversions
    FROM campaign
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND metrics.all_conversions > 0
    AND campaign.advertising_channel_type = 'SEARCH'
    AND segments.conversion_action_name IN ('{purchase_action_list}')
    ORDER BY campaign.name, segments.conversion_action_name
"""

HOD_CLICKS_REPORT_QUERY = """
    SELECT
        campaign.name,
        segments.hour,
        metrics.clicks
    FROM campaign
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    ORDER BY campaign.name, segments.hour
"""

AGE_CLICKS_REPORT_QUERY = """
    SELECT
        campaign.name,
        ad_group_criterion.age_range.type,
        metrics.clicks
    FROM age_range_view
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    ORDER BY campaign.name, ad_group_criterion.age_range.type
"""

DEVICE_CLICKS_REPORT_QUERY = """
    SELECT
        campaign.name,
        segments.device,
        metrics.clicks
    FROM campaign
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    ORDER BY campaign.name, segments.device
"""

LOC_CLICKS_REPORT_QUERY = """
    SELECT
        campaign.name,
        geographic_view.location_type,
        geographic_view.country_criterion_id,
        metrics.clicks,
        campaign.advertising_channel_type
    FROM geographic_view
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    AND geographic_view.country_criterion_id IN ({country_criterion_ids})
    ORDER BY campaign.name, geographic_view.location_type
"""

HOD_CONVERSIONS_REPORT_QUERY = """
    SELECT
        campaign.name,
        segments.conversion_action_name,
        segments.hour,
        metrics.all_conversions
    FROM campaign
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    AND metrics.all_conversions > 0
    AND segments.conversion_action_name IN ('{purchase_action_list}')
    ORDER BY campaign.name, segments.conversion_action_name, segments.hour
"""

AGE_CONVERSIONS_REPORT_QUERY = """
    SELECT
        campaign.name,
        segments.conversion_action_name,
        ad_group_criterion.age_range.type,
        metrics.all_conversions
    FROM age_range_view
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    AND metrics.all_conversions > 0
    AND segments.conversion_action_name IN ('{purchase_action_list}')
    ORDER BY campaign.name, segments.conversion_action_name, ad_group_criterion.age_range.type
"""

DEVICE_CONVERSIONS_REPORT_QUERY = """
    SELECT
        campaign.name,
        segments.conversion_action_name,
        segments.device,
        metrics.all_conversions
    FROM campaign
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    AND metrics.all_conversions > 0
    AND segments.conversion_action_name IN ('{purchase_action_list}')
    ORDER BY campaign.name, segments.conversion_action_name, segments.device
"""

LOC_CONVERSIONS_REPORT_QUERY = """
    SELECT
        campaign.name,
        segments.conversion_action_name,
        geographic_view.location_type,
        geographic_view.country_criterion_id,
        metrics.all_conversions,
        campaign.advertising_channel_type
    FROM geographic_view
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    AND metrics.all_conversions > 0
    AND segments.conversion_action_name IN ('{purchase_action_list}')
    AND geographic_view.country_criterion_id IN ({country_criterion_ids})
    ORDER BY campaign.name, segments.conversion_action_name, geographic_view.location_type
"""

SEARCH_TERM_REPORT_QUERY = """
    SELECT
        search_term_view.search_term,
        segments.keyword.info.text,
        segments.keyword.info.match_type,
        segments.conversion_action_name,
        metrics.all_conversions
    FROM search_term_view
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND metrics.all_conversions > 0
    AND segments.conversion_action_name IN ('{conversion_action_list}')
    AND segments.keyword.info.match_type IN ('EXACT', 'PHRASE', 'BROAD')
    ORDER BY segments.keyword.info.text
"""

BUILD_LOCATION_CACHE_QUERY = """
    SELECT
        geo_target_constant.id,
        geo_target_constant.canonical_name
    FROM geo_target_constant
    WHERE geo_target_constant.id IN ({ids_str})
"""

AUTO_APPLIED_RECOMMENDATIONS_QUERY = """
    SELECT
        change_event.resource_name,
        change_event.change_date_time,
        change_event.change_resource_name,
        change_event.change_resource_type,
        change_event.resource_change_operation,
        change_event.changed_fields,
        change_event.client_type,
        change_event.user_email,
        change_event.campaign,
        change_event.ad_group
    FROM change_event
    WHERE change_event.change_date_time >= '{start_datetime}'
    AND change_event.change_date_time <= '{end_datetime}'
    AND change_event.user_email = 'Recommendations Auto-Apply'
    ORDER BY change_event.change_date_time DESC
    LIMIT 10000
"""
