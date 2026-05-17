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

This also writes `data/<course>/reports/kw-day-panel.csv`, a keyword-day panel with
date, keyword, campaign, match type, impressions, clicks, cost, conversions, and
search impression share.

To pull only the keyword-day panel:

```powershell
python scripts/pull_input_data.py `
  --datasets kw_day_panel `
  --output-course sys_think `
  --google-ads-yaml ..\google-ads-prod.yaml `
  --customer-id 1234567890
```

When `data/<course>/change_history_budgets.csv` exists, the keyword-day panel
starts at the earliest date in that file.

## Change History Signals

`data/sys_think/processed/campaign-summary.csv` currently uses the saved Google
Ads change history HTML to summarize campaign budget changes, negative keywords,
campaign status, and campaign rename lineage. Positive keyword inventory and
match-type counts come from the raw Search keyword report matched by exact
campaign name and date window (`start_date <= Day < end_date`), because that is
more reliable for observed active keywords than inferring positives from change
history. Deduplicated positive/negative keyword configurations are written to
`data/sys_think/processed/campaign-keyword-sets.csv` and referenced from
`campaign-summary.csv` by `keyword_set_id`. Per-campaign raw-vs-change-history
keyword comparisons are written to
`data/sys_think/processed/campaign-keyword-checks.csv`.

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

By default, this cleans `data/<course>/reports/kw-day-panel.csv`, infers `region`
from the campaign name, and writes `data/<course>/processed/kw-day-panel-clean.csv`.

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
