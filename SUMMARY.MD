# System Thinking Ads Data Summary

This repo currently has two main processed datasets for the System Thinking ads analysis:

- `data/sys_think/processed/campaign-day-panel.csv`: campaign-day level panel for modeling clicks and spend against daily campaign budget.
- `data/sys_think/processed/kw-day-panel.csv`: keyword-day level panel for understanding keyword performance, match-type behavior, and keyword overlap across regions.

The exploratory notebook for this analysis is `eda_clicks_budget_keywords.ipynb`.

## Data Overview

### `campaign-day-panel.csv`

This file is the primary dataset for budget allocation modeling. Each row is one campaign version observed on one day.

Key columns:

- `date`: reporting date.
- `campaign_version`: version identifier that links to `campaign-summary.csv`.
- `region`: campaign region bucket (`A`, `B`, `C`, or `USA`).
- `daily_budget`: campaign daily budget in USD.
- `match_types`: campaign-level match-type configuration, such as `Broad`, `Exact; Phrase`, or `Broad; Exact; Phrase`.
- `clicks`: daily clicks.
- `cost`: daily cost.

Current coverage:

- Rows: 2,222 campaign-days.
- Campaign versions: 47 date-bounded campaign configurations.
- Date range: 2024-06-26 to 2026-05-13.
- Regions: `A`, `B`, `C`, `USA`.
- Match configurations: `Broad`, `Broad; Exact`, `Broad; Exact; Phrase`, `Broad; Phrase`, `Exact`, `Exact; Phrase`, `Phrase`.
- Total clicks: 82,119.
- Total cost: $198,601.35.
- Daily budget range: $6.18 to $296.47.

A `campaign_version` is a date-bounded campaign configuration from `campaign-summary.csv`. For modeling purposes, it is determined by the tuple of `region`, `daily_budget`, `match_types`, and `keyword_set_id`, where `keyword_set_id` captures the positive and negative keyword configuration in effect during that window. A new version is created when one of those configuration elements changes, such as a daily budget change, match-type/keyword-set change, campaign pause/reactivation, keyword removals/additions, or the initial raw keyword coverage window. When `campaign-day-panel.csv` is generated, each active campaign-day is matched to the summary row whose campaign/date window contains that day, then inherits that row's `campaign_version`, `daily_budget`, `match_types`, and keyword-set metadata.

### `kw-day-panel.csv`

This file is the keyword-level dataset. It is useful for selecting keywords, checking keyword concentration, and comparing overlap across regions and match types.

Key columns:

- `date`: reporting date.
- `region`: campaign region bucket.
- `keyword`: keyword text.
- `campaign`: source campaign name.
- `match_type`: individual keyword match type.
- `clicks`: daily keyword clicks.
- `cost`: daily keyword cost.
- `conv_value`: conversion value reported for the keyword-day.
- `currency_code`: reporting currency.
- `first_page_cpc`: first-page CPC estimate when available.

Current coverage:

- Rows: 64,528 keyword-days.
- Date range: 2022-07-01 to 2026-05-13.
- The keyword panel starts earlier than the campaign-day panel, so analyses that compare the two should align date windows.

## Initial Plan

1. Select keywords to use for each campaign.
2. Model clicks as a function of budget.
3. Allocate budget across campaigns.

## Initial EDA Findings

### Keyword Count vs Click Performance

The number of positive keywords in a campaign configuration does not appear to explain much of the observed click performance in this initial cut.

![Keyword count vs clicks](./figures/keyword-count-vs-clicks.png)

Across campaign versions:

- Correlation between `num_unique_keywords` and `clicks_per_day`: 0.051.
- Correlation between `num_unique_keywords` and `clicks_per_day` normalized by budget: 0.078.

This suggests keyword count alone is not a strong driver of clicks. Keyword quality, match type, region, budget level, and campaign timing likely matter more than raw keyword inventory size.

### Budget vs Clicks

Increased daily budget generally corresponds to more clicks, but the level and slope differ by region and match-type configuration.

![Aggregated clicks vs daily budget](./figures/clicks-vs-budget-aggregated.png)

Notable patterns:

- Region B has strong click volume at relatively low budgets, especially for mixed match configurations such as `Broad; Phrase` and `Broad; Exact; Phrase`.
- USA campaigns operate at much higher budgets, but the clicks response is not uniformly higher at every budget level.
- `Exact; Phrase` appears across several regions and budget levels, making it a useful starting point for modeling a budget response curve.
- Match-type configuration should be included in the click model; pooling all campaigns together would hide important differences.

### Top Keyword Overlap

The notebook also compares top-20 keywords across `(region, match_type)` groups. Early results suggest top keywords overlap more across regions within the same match type than across match types within a single region.

This is useful for keyword selection: region-specific campaigns may be able to share some high-performing keywords, but match-type-specific lists should still be handled separately.

## Next Steps

1. Define the candidate keyword set for each campaign.
   - Start from top-click and top-value keywords in `kw-day-panel.csv`.
   - Compare top keywords across regions and match types.
   - Remove or flag keywords with low relevance, poor cost efficiency, or inconsistent performance.

2. Build a click response model.
   - Use `campaign-day-panel.csv` as the main modeling table.
   - Model `clicks` as a function of `daily_budget`, `region`, `match_types`, and keyword-set attributes.
   - Consider nonlinear budget effects because click gains may flatten at higher budgets.
   - Keep region and match-type interactions in scope because the budget response differs by segment.

3. Estimate marginal clicks per dollar.
   - Convert the click response model into expected incremental clicks for a small budget increase.
   - Compare marginal returns across campaigns instead of comparing average clicks alone.

4. Allocate budget across campaigns.
   - Start with a constrained optimization that respects total budget, campaign minimums, and campaign maximums.
   - Allocate dollars toward the campaigns with the highest estimated marginal clicks per dollar.
   - Re-run allocation under multiple assumptions to check sensitivity.

5. Validate against holdout periods.
   - Backtest model predictions on later dates or campaign versions.
   - Compare predicted clicks, actual clicks, and spend efficiency.
   - Check whether recommendations are stable enough to use operationally.
