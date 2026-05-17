# ad_opt_v2

Small repo for pulling Google Ads data used by the ad optimization workflow.

This repo was split from `ad_opt` and intentionally keeps only the Google Ads API read/pull layer:

- ads performance reports
- keyword planning historical metrics
- recent change events

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Keep Google Ads credentials outside the repo. The `.gitignore` excludes common `google-ads*.yaml` and `google-ads*.json` credential filenames.

## Pull Ads Reports

```powershell
python scripts/pull_input_data.py `
  --datasets ads_reports `
  --output-course sys_think `
  --google-ads-yaml ..\google-ads-prod.yaml `
  --customer-id 1234567890
```

Outputs are written under `data/<course>/reports`.

The raw Search keyword export is written to
`data/<course>/reports/Search keyword - raw input to models.csv`. The processed
keyword-day panel is derived from that full export, not from a separate keyword-day
report.

## Parsing Procedure

The processed analysis files are generated in three steps.

First, parse the saved Google Ads change-history HTML. This writes compact budget
history, campaign-version rows, and deduplicated keyword sets:

```powershell
uv run python scripts/parse_change_history_html.py `
  "data/sys_think/Change history - MIT xPRO - System Thinking - Google Ads.html" `
  -o data/sys_think/change_history_budgets.csv `
  --campaign-summary-output data/sys_think/processed/campaign-summary.csv `
  --keyword-sets-output data/sys_think/processed/campaign-keyword-sets.csv `
  --search-keyword-report "data/sys_think/reports/Search keyword - raw input to models.csv"
```

`campaign-summary.csv` is raw-report driven: intervals come from Search keyword
coverage, positive keywords and match-type counts come from the raw Search keyword
export, and budget/negative-keyword metadata comes from change history. If a
campaign was renamed from an earlier run, the parser can fill missing budgets from
the earlier campaign's budget changes. Rows that still have no daily budget are
filtered out.

`campaign-keyword-sets.csv` stores the full positive and negative keyword lists.
`campaign-summary.csv` references those rows by `keyword_set_id`. Budget-only
splits share the union of observed keywords across the whole budget-only run, so
a pure budget change does not create a different keyword set.

Next, clean the full raw Search keyword export into the processed keyword-day
panel:

```powershell
uv run python scripts/process_input_data.py --output-course sys_think
```

This writes `data/sys_think/processed/kw-day-panel.csv` with date, region,
keyword, campaign, match type, clicks, cost, conversion value, currency, and first
page CPC. It does not include impressions.

Finally, build the campaign-day panel from the processed keyword-day panel and
campaign summary:

```powershell
uv run python scripts/generate_campaign_day_panel.py --output-course sys_think
```

This writes `data/sys_think/processed/campaign-day-panel.csv` with date,
campaign version, region, daily budget, match types, clicks, and cost.

## Change History Signals

`data/sys_think/processed/campaign-summary.csv` uses the saved Google Ads change
history HTML for campaign budget changes, negative keywords, campaign status, and
campaign rename lineage. Positive keyword inventory and match-type counts come
from the raw Search keyword report because that is more reliable for observed
active keywords than inferring positives from change history.

The same HTML contains other campaign changes that are useful for interpretation
but are not yet represented as structured columns in the summary:

- Landing page URL changes: about 27 change actions, with 113 individual final
  URL detail rows in the saved System Thinking HTML. The clearest observed
  change is a switch from the `xpro.mit.edu` course URL to
  `https://learn-xpro.mit.edu/system-thinking`, which changes the post-click
  destination associated with the same campaign/keyword setup.
- Ad copy changes: about 105 responsive-search-ad creation/change actions across
  9 dates and 14 campaign names. The HTML includes created/changed responsive
  search ads, headline text, description text, pinned headline positions, and
  display paths such as `certificate`.
- Asset and extension changes: about 189 campaign asset or extension actions
  across 9 dates. The observed asset types include callouts, sitelinks, images,
  business logos, and promotion assets.
- Geo exclusions: the parsed history shows 2 exclusion detail rows for 1 unique
  excluded country: `Pakistan`, changed from included to excluded on 2026-04-09
  for `Course - System Thinking - Run 19 - Search - Prospecting - B`.

If campaign summaries are used to explain performance shifts, these signals
should be added either as compact flags/counts in `campaign-summary.csv` or as
separate history files joined by campaign and date. The highest-value additions
are likely `landing_page_url`, `num_rsa_created`, `num_rsa_changed`,
`num_asset_changes`, and `geo_exclusions`.

## Process Pulled Reports

```powershell
python scripts/process_input_data.py `
  --output-course sys_think
```

By default, this cleans
`data/<course>/reports/Search keyword - raw input to models.csv`, infers `region`
from the campaign name, and writes `data/<course>/processed/kw-day-panel.csv`.

## Pull Keyword Planner Metrics

```powershell
python scripts/pull_input_data.py `
  --datasets keyword_planning `
  --output-course sys_think `
  --keyword-planning-input-file data\sys_think\gkp\keywords_classified.csv `
  --google-ads-yaml ..\google-ads-prod.yaml `
  --customer-id 1234567890
```

Outputs are written under `data/<course>/gkp`.

## Query Auto-Applied Recommendation Changes

```powershell
python scripts/query_change_events.py `
  --google-ads-yaml ..\google-ads-prod.yaml `
  --customer-id 1234567890 `
  --lookback-days 7
```
