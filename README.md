# ad_opt_v2 — Multi-course campaign optimization

Two-stage Google Ads optimization: fix keyword sets once per window (stage 1), then walk-forward daily budget allocation (stage 2). Includes data prep, model fitting, production planning, and walk-forward backtest.

Each course is a self-contained bundle under `<course>/` with its own `data/` and `opt_results/`. All CLIs accept `--course` (default: `sys_think`).

## Repository layout

```
<course>/                        # e.g. sys_think/, ml/
  data/                          # inputs + processed panels
    Change history*.html         # required: budget/keyword history
    gkp/
      *Keywords*Enrollments*.xlsx   # required: enrollment allowlist
      Saved Keyword Stats*.csv      # cached GKP search-volume stats
    processed/                   # campaign-day panel, candidates, features
  opt_results/
    campaign/default/
      campaign_config.json       # experiment config
      backtest/<start>_<end>/    # backtest summaries + plans/
campaign_opt/                    # library + CLI (campaign_opt/cli/)
utils/                           # data-processing helpers
config.py                        # COURSE_CONFIG per course
tests/
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```powershell
uv sync --extra optimization --extra ml
```

Keep Google Ads credentials outside the repo (`.gitignore` excludes `google-ads*.yaml`).

**Required before candidate build:** `<course>/data/gkp/*Keywords*Enrollments*.xlsx`

## Adding a new course

1. Add an entry to `COURSE_CONFIG` and `REGION_CONFIG` in `config.py` (see the `ml` stub as a template).
2. Create the bundle directories:
   ```powershell
   mkdir <course>\data\gkp
   mkdir <course>\opt_results\campaign\default
   ```
3. Copy and edit `campaign_config.json` from `sys_think/opt_results/campaign/default/`, setting `"course": "<course>"`.
4. Place change-history HTML, enrollment allowlist xlsx, and run data prep with `--course <course>`.

All pipeline commands accept `--course` (default `sys_think`).

## Overall inputs and outputs

| Inputs | Outputs |
|--------|---------|
| Change-history HTML | `<course>/data/processed/campaign-summary.csv`, `campaign-keyword-sets.csv` |
| Google Ads API pulls | `<course>/data/reports/kw-day-panel.csv` → cleaned `processed/kw-day-panel.csv` |
| Enrollment allowlist xlsx | Filtered `segment-keyword-candidates.csv` (synthetic `*_allowlist_*` sets) |
| Cached GKP stats CSV | `keyword-set-features.csv` (`last_month_searches_mean`, etc.) |
| `campaign_config.json` + fitted models | `model_manifest.json`, `holdout_metrics.json` |
| Two-stage production run | `two_stage_plan/fixed_keyword_sets.json`, `stage2_budgets/YYYYMMDD/campaign_plan.csv` |
| Backtest window | `backtest/<start>_<end>/daily_backtest_summary.csv`, `backtest_summary.csv` |

## Quick start (sys_think)

```powershell
# 1. Data prep (omit --skip-pull to pull from Google Ads API)
uv run prepare-data --google-ads-yaml ..\google-ads-prod.yaml --customer-id 1234567890

# 2. Features + model fit
uv run python -m campaign_opt.cli.build_keyword_candidates --verify
uv run python -m campaign_opt.cli.build_gkp_set_features
uv run fit-models

# 3. Two-stage production plan
uv run run-pipeline --window-start 2026-05-12 --window-end 2026-05-25 --planning-date 2026-05-12

# 4. Walk-forward backtest + analysis
uv run backtest --start 2026-05-12 --end 2026-05-25 --use-actual-budget --analyze
uv run analyze-backtest --start 2026-05-12 --end 2026-05-25
```

For another course, add `--course <name>` to any command above.

## Per-component I/O

| Component | Inputs | Outputs |
|-----------|--------|---------|
| `prepare-data` | allowlist xlsx, credentials, change-history HTML | all `processed/` panels |
| `cli.pull_input_data` | Google Ads YAML, customer ID | `reports/kw-day-panel.csv`; optional KWP stats (keywords from panel) |
| `cli.process_input_data` | raw kw-day panel | `processed/kw-day-panel.csv` |
| `cli.parse_change_history_html` | change-history HTML | `campaign-summary.csv`, `campaign-keyword-sets.csv` |
| `cli.generate_campaign_day_panel` | processed artifacts | `campaign-day-panel.csv` |
| `cli.build_keyword_candidates` | panel + **required** allowlist | `segment-keyword-candidates.csv`, `campaign-keyword-sets-extended.csv` |
| `cli.build_gkp_set_features` | cached GKP stats + keyword sets | `keyword-set-features.csv` |
| `fit-models` | modeling panel + config | `model_manifest.json`, `holdout_metrics.json` |
| **Stage 1** `select_keyword_sets_for_window` | train (`date < window_start`), candidates, manifest | `fixed_keyword_sets.json`, `keyword_set_plan.csv` |
| **Stage 2** `optimize_budgets_for_day` | train (`date < t`), fixed sets, manifest | `campaign_plan.csv`, `optimizer_xgboost.joblib` |
| `backtest` | same + date window | `daily_backtest_summary.csv`, `plans/YYYYMMDD/plan_vs_actual.csv` |
| `analyze-backtest` | `plan_vs_actual` files | `backtest_summary.csv`, `evaluation_results.csv` |

**Production vs backtest:** `run-pipeline` / `plan_two_stage_campaign` runs stage 2 for **one** `--planning-date`. `backtest` runs stage 2 for **every day** in `[start, end]`.

## Config

Default: `sys_think/opt_results/campaign/default/campaign_config.json`

See [`campaign_opt/README.md`](campaign_opt/README.md) for modeling notes and config fields.

## Tests

```powershell
uv run pytest tests/ -q
```

Golden parity fixtures: `tests/fixtures/backtest_golden/2026-05-12_2026-05-25/`
