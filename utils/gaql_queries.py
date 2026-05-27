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
