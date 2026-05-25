# ad_opt_v2

Small repo for pulling Google Ads data used by the ad optimization workflow.

This repo was split from `ad_opt` and intentionally keeps only the Google Ads API read/pull layer:

- ads performance reports
- keyword planning historical metrics
- recent change events

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```powershell
uv sync
```

Run scripts with `uv run python ...` or activate `.venv`:

```powershell
.\.venv\Scripts\Activate.ps1
```

Keep Google Ads credentials outside the repo. The `.gitignore` excludes common `google-ads*.yaml` and `google-ads*.json` credential filenames.

## Pull Ads Reports

```powershell
uv run python scripts/pull_input_data.py `
  --datasets campaign_opt `
  --output-course sys_think `
  --google-ads-yaml ..\google-ads-prod.yaml `
  --customer-id 1234567890
```

Pulls **campaign_opt** data from the Google Ads API:

- `reports/kw-day-panel.csv` — keyword-day clicks, cost, and filtered `all_conv` (via `keyword_view` GAQL; excludes Experiment campaigns)

Add `keyword_planning` to `--datasets` when GKP set features are enabled in config.

Outputs are written under `data/<course>/reports/`.

## Parsing Procedure

Run in order: **pull → clean → parse change history → build campaign-day panel**.

### 1. Pull Google Ads reports

```powershell
uv run python scripts/pull_input_data.py `
  --datasets campaign_opt `
  --output-course sys_think `
  --google-ads-yaml ..\google-ads-prod.yaml `
  --customer-id 1234567890
```

Start date defaults to `min_date` in `config.py` (`2024-06-01` for `sys_think`).

→ `data/<course>/reports/kw-day-panel.csv`

### 2. Clean the kw-day-panel

```powershell
uv run python scripts/process_input_data.py --output-course sys_think
```

→ `data/<course>/processed/kw-day-panel.csv`

### 3. Parse change history

Save the change-history HTML under `data/<course>/`, then:

```powershell
uv run python scripts/parse_change_history_html.py --output-course sys_think
```

Uses the cleaned kw-day-panel for live keyword inventory. Writes `campaign-summary.csv` and `campaign-keyword-sets.csv`.

### 4. Build campaign-day panel

```powershell
uv run python scripts/generate_campaign_day_panel.py --output-course sys_think
```

→ `data/<course>/processed/campaign-day-panel.csv` — clicks, cost, and `all_conv` per campaign version

```powershell
uv run python scripts/generate_campaign_day_panel.py --output-course sys_think
```

Joins the cleaned kw-day-panel with campaign summary. Writes:

- `data/sys_think/processed/campaign-day-panel.csv` — date, campaign version, region, daily budget, match types, clicks, cost, and `all_conv`

## Change History Signals

`data/sys_think/processed/campaign-summary.csv` uses the saved Google Ads change
history HTML for campaign budget changes, negative keywords, campaign status, and
campaign rename lineage. Positive keyword inventory and match-type counts come
Positive keyword inventory and match-type counts come from the cleaned kw-day-panel.

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

Step 1 of the parsing procedure above — cleans the raw API export only:

```powershell
uv run python scripts/process_input_data.py `
  --output-course sys_think
```

Reads `data/<course>/reports/kw-day-panel.csv`, infers `region` from campaign name,
and writes `data/<course>/processed/kw-day-panel.csv`.

## Build keywords_classified (existing campaign keywords only)

```powershell
uv run python scripts/build_keywords_classified.py --course sys_think
```

Writes `data/<course>/gkp/keywords_classified.csv` with `Origin=existing` only
(keywords from kw-day-panel, not search terms or Semrush
candidates). Re-run after pulling new keyword performance data.

## Pull Keyword Planner Metrics

```powershell
uv run python scripts/pull_input_data.py `
  --datasets keyword_planning `
  --output-course sys_think `
  --keyword-planning-input-file data\sys_think\gkp\keywords_classified.csv `
  --google-ads-yaml ..\google-ads-prod.yaml `
  --customer-id 1234567890
```

Outputs are written under `data/<course>/gkp`.

## Campaign budget + keyword-set optimization

Config-driven pipeline (Python 3.11+). Install core + solver + ML extras:

```powershell
uv sync --extra optimization --extra ml
```

Default experiment config: `opt_results/sys_think/campaign/default/campaign_config.json`

```powershell
# Full pipeline (panel → GKP set features → candidates → model tournament → Gurobi MILP)
uv run python scripts/run_campaign_pipeline.py --course sys_think

# Individual steps
uv run python scripts/build_gkp_set_features.py --course sys_think
uv run python scripts/build_keyword_candidates.py --course sys_think
uv run python scripts/fit_response_models.py --course sys_think
uv run python scripts/optimize_campaign.py --course sys_think --budget 400

# Walk-forward daily backtest (optimize each day in range; like ad_opt backtest_daily)
uv run python scripts/backtest_campaign.py --course sys_think --start 2025-10-01 --end 2025-12-31

# Production monitor (compare latest plan vs actuals)
uv run python scripts/monitor_campaign_production.py --course sys_think --lag 1
```

Notebook EDA (`eda_clicks_budget_keywords.ipynb`) is optional; add `--extra notebook` to `uv sync`. Production scripts do not import the notebook.

See [`campaign_opt/README.md`](campaign_opt/README.md) for package layout, CV, and MILP backends.

## Query Auto-Applied Recommendation Changes

```powershell
uv run python scripts/query_change_events.py `
  --google-ads-yaml ..\google-ads-prod.yaml `
  --customer-id 1234567890 `
  --lookback-days 7
```
