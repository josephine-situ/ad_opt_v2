_SEARCH_CAMPAIGN_FILTERS = """
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.name NOT LIKE '%Experiment%'
    AND campaign.advertising_channel_type = 'SEARCH'"""

KW_DAY_PANEL_REPORT_QUERY = f"""
    SELECT
        segments.date,
        ad_group_criterion.keyword.text,
        campaign.name,
        ad_group_criterion.keyword.match_type,
        metrics.clicks,
        metrics.cost_micros,
        customer.currency_code,
        ad_group_criterion.position_estimates.first_page_cpc_micros
    FROM keyword_view
    WHERE segments.date BETWEEN '{{start_date}}' AND '{{end_date}}'
    {_SEARCH_CAMPAIGN_FILTERS}
    AND ad_group_criterion.keyword.match_type IN ('EXACT', 'PHRASE', 'BROAD')
    AND metrics.clicks > 0
    ORDER BY segments.date, campaign.name, ad_group_criterion.keyword.text
"""

KW_KEYWORD_ALL_CONV_QUERY = f"""
    SELECT
        segments.date,
        ad_group_criterion.keyword.text,
        campaign.name,
        ad_group_criterion.keyword.match_type,
        segments.conversion_action_name,
        metrics.all_conversions
    FROM keyword_view
    WHERE segments.date BETWEEN '{{start_date}}' AND '{{end_date}}'
    {_SEARCH_CAMPAIGN_FILTERS}
    AND ad_group_criterion.keyword.match_type IN ('EXACT', 'PHRASE', 'BROAD')
    AND segments.conversion_action_name IN ('{{conversion_action_list}}')
    AND metrics.all_conversions > 0
    ORDER BY segments.date, campaign.name, ad_group_criterion.keyword.text
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

SELECT_AD_GROUPS_BY_CAMPAIGN_NAME = """
        SELECT
            campaign.name,
            ad_group.resource_name
        FROM ad_group
        WHERE campaign.name IN ('{campaign_names_list}')
        AND campaign.status != 'REMOVED'
        AND campaign.advertising_channel_type = 'SEARCH'
    """

SELECT_EXISTING_KEYWORDS_BY_AD_GROUP_RESOURCE = """
        SELECT
            ad_group.resource_name,
            ad_group_criterion.keyword.text,
            ad_group_criterion.keyword.match_type
        FROM ad_group_criterion
        WHERE ad_group_criterion.ad_group IN ('{ad_group_resource_names_list}')
        AND ad_group_criterion.type = 'KEYWORD'
        AND campaign.advertising_channel_type = 'SEARCH'
    """

