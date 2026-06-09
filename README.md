# ad_opt_v2 — Search campaign optimization

Two-stage Google Ads optimization: fix keyword sets once per window (stage 1), then walk-forward daily budget allocation (stage 2).

Each course is a bundle under `<course>/` with `data/`, `prod/`, and `backtests/`. Shared optimization defaults live in `config/default.yaml`; per-course overrides in `<course>/course.yaml`.

## Repository layout

```
config/
  default.yaml                 # shared optimization defaults (all courses)
<course>/                      # e.g. sys_think/
  course.yaml                  # optional per-course overrides
  data/                        # inputs + processed panels
  prod/                        # model fit + production pipeline outputs
  experiments/                 # debug dumps + ablation diagnostics (not prod)
  backtests/<start>_<end>/     # walk-forward backtest windows
scripts/                       # CLI entry points
utils/                         # library code
tests/
```

## Setup

```powershell
uv sync
```

Keep Google Ads credentials outside the repo (`.gitignore` excludes `google-ads*.yaml`).

**Required before candidate build:** `<course>/data/gkp/*Keywords*Enrollments*.xlsx`

## Expected Google Ads account structure

The pipeline expects a hierarchical Google Ads account structure with a single **Manager Account (MCC)** linked to multiple **course accounts**, one per course. Authentication uses a service account with API access to the Manager Account, which automatically provides access to all linked course accounts.

### Account hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                   Manager Account (MCC)                      │
│          API Key (Service Account) provides access           │
│                   to all linked accounts                     │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─────────────┬─────────────┬─────────────┬──────────────
             │             │             │             │
             ▼             ▼             ▼             ▼
      ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
      │ Generative │ │  Machine   │ │  Systems   │ │  Systems   │
      │  AI Course │ │  Learning  │ │ Engineering│ │  Thinking  │
      │  Account   │ │  Course    │ │   Course   │ │   Course   │
      └────────────┘ └────────────┘ └────────────┘ └────────────┘
```

### Campaign and ad group structure

Each course account contains **one campaign per (Course Name, Region, Match Type) tuple**. Match types are configured as **`Broad`** or **`Phrase; Exact`** (see `allowed_match_types` in `config/default.yaml`). Each campaign contains **exactly one ad group** matching the same tuple.

**Example for System Thinking:**

```
System Thinking Course Account
│
├── Campaign: Course - System Thinking - USA - Broad
│   └── Ad Group: Course - System Thinking - USA - Broad
│       ├── Keyword: systems thinking course (Broad Match)
│       └── Keyword: strategic thinking workshop (Broad Match)
│
├── Campaign: Course - System Thinking - USA - Phrase; Exact
│   └── Ad Group: Course - System Thinking - USA - Phrase; Exact
│       ├── Keyword: [systems thinking course] (Exact Match)
│       ├── Keyword: [mit systems thinking] (Exact Match)
│       ├── Keyword: "systems thinking" (Phrase Match)
│       └── Keyword: "mit xpro systems thinking" (Phrase Match)
│
├── Campaign: Course - System Thinking - A - Broad
│   └── Ad Group: Course - System Thinking - A - Broad
│       └── [Broad match keywords for Region A]
│
├── Campaign: Course - System Thinking - A - Phrase; Exact
│   └── Ad Group: Course - System Thinking - A - Phrase; Exact
│       └── [Phrase and Exact match keywords for Region A]
│
└── ... (additional Region × Match Type combinations)
```

**Key principles:**

- **One campaign = one (Course, Region, Match Type) tuple** — maps to an optimizer segment such as `USA / Broad` or `USA / Phrase; Exact`
- **One ad group per campaign** — simplifies keyword-set pushes
- **Match type grouping** — `Broad` campaigns contain only broad keywords; `Phrase; Exact` campaigns contain both phrase and exact keywords in the same ad group
- **Campaign-level daily budgets** — set from stage-2 `campaign_plan.csv` per segment
- **Keyword sets** — add, pause, or remove keywords from the ad group when stage 1 selects a new `keyword_set_id`

This structure ensures stage-1 keyword sets and stage-2 budgets map cleanly to Google Ads entities without ambiguity.

### Experiment campaigns and naming

The past experiment campaigns used a **different bidding strategy** (manual CPC) than production. Mark them with **`Experiment` in the campaign name** (e.g. `… - Search - Experiment - …`). The pipeline excludes them automatically:

- **API pulls** — `campaign.name NOT LIKE '%Experiment%'` in `utils/gaql_queries.py`
- **Change-history parsing** — `is_search_campaign()` skips names containing `Experiment`

Avoid putting `Experiment` in production campaign names. Region `C` is also excluded from optimization (`excluded_regions` in `config/default.yaml`).

## Change history

`campaign-summary.csv` (budget intervals, campaign versions) and `campaign-keyword-sets.csv` (historical keyword sets) drive the campaign-day panel. Bootstrap and updates can come from any of:

1. **Saved change-history HTML** (current default for bootstrap and keyword changes) — save the Google Ads change history page to `<course>/data/`, then:

   ```powershell
   uv run python -m scripts.parse_change_history_html --course sys_think
   ```

2. **Persisted daily budget changes** (lighter option for stage-2 pushes) — when you apply budgets from `campaign_plan.csv`, append new `daily_budget` rows to `campaign-summary.csv` (new `campaign_version` intervals with the pushed amounts). No HTML re-parse needed for budget-only updates. Keyword adds/removes still need option 1 or 3.

3. **Change History API** (not yet wired) — replace manual HTML saves going forward.

`prepare-data` requires `campaign-summary.csv` before it can build `campaign-day-panel.csv`. It also writes `segment-conv-per-click-rates.csv` (course-wide `sum(all_conv)/sum(clicks)` per region × match type) used as fixed scaling for the `conv_scaled_clicks` target. Commit updated summary/keyword-set CSVs to git after keyword changes or budget appends.

## Config

`load_config(course)` merges:

1. `config/default.yaml` — target, features, model policy, constraints, evaluation
2. `<course>/course.yaml` — optional overrides (e.g. `min_date`, `conversion_actions`, `start_dates`, `daily_budget_cap`)

Adding a course:

```powershell
mkdir my_course\data\gkp
# optional: my_course\course.yaml with course-specific fields
```

Courses are discovered automatically (top-level dirs with a `data/` folder).

## Production schedule

| Cadence | Step | What it does |
|---------|------|--------------|
| **Daily** | Pull Google Ads data | `prepare-data` — API pull → clean kw-day panel → campaign-day panel |
| **Daily** | Stage 2 budgets | `run-pipeline --skip-stage1` — refit model walk-forward, optimize tomorrow's segment budgets |
| **Daily** | Plan-vs-actual monitoring | Automatic in `run-pipeline` (before model refit) — scores yesterday's saved plan vs realized `conv_scaled_clicks` |
| **Monthly or ad-hoc** | Stage 1 keyword sets | `run-pipeline --window-start … --window-end …` — pick one keyword set per segment for the window, then stage 2 for the planning date |
| **When Ads config changes** | Re-parse change history | `parse_change_history_html` — after keyword adds/removes, pauses, or renames in the UI |
| **After daily budget push** | Append budget to summary | Update `campaign-summary.csv` with pushed amounts (see Change history §2) — or re-parse HTML |
| **When keyword pool changes** | Rebuild candidates | `build_keyword_candidates` + `build_gkp_set_features` (included in full `run-pipeline`) |

### Daily run

```powershell
uv run prepare-data --google-ads-yaml ..\google-ads-prod.yaml --customer-id 1234567890
uv run run-pipeline --skip-stage1 --skip-candidates --skip-gkp
```

### Monthly / ad-hoc stage 1

```powershell
uv run run-pipeline --window-start 2026-06-15 --window-end 2026-07-15
```

`run-pipeline` fits `model_policy.optimizer_winner` only (no tournament, no evaluation-model refit) and (by default) rebuilds keyword candidates and GKP set features. Use `fit-models` without `--optimizer-only` when you want the full tournament and evaluation scorer (e.g. before backtest). Use `--skip-candidates` / `--skip-gkp` on daily runs if those inputs have not changed. Use `--skip-monitoring` to skip plan-vs-actual scoring.

### Production monitoring

Each `run-pipeline` call (after `prepare-data`) scores unscored days in the last 7 days that have a saved stage-2 plan. Monitoring compares **saved plan predictions** (`milp_pred` from `campaign_plan.csv`) to **realized outcomes** from the modeling panel (Google Ads data). It does not load or refit an evaluation model. Outputs under `<course>/prod/monitoring/`:

| File | Purpose |
|------|---------|
| `daily_metrics.csv` | One row per scored day: RMSE, nRMSE, bias %, pred/observed totals |
| `rolling_summary.json` | 7-day and 30-day rolling mean bias and nRMSE |
| `plan_vs_actual/YYYYMMDD/plan_vs_actual.csv` | Per-segment detail: plan preds vs panel actuals |

Example log line:

```
[monitoring] 2026-06-08: observed=142.3, pred=158.1, bias=+11.1%, RMSE=18.4, nRMSE=0.13, segments=6
[monitoring] 7d rolling bias=+6.2%, nRMSE=0.11
```

When `GRAFANA_URL`, `GRAFANA_USERNAME`, and `GRAFANA_TOKEN` are set, metrics are also pushed under the `campaign_opt_monitoring` prefix.

## Push outputs to Google Ads

Pipeline push is **manual today** (no upload script yet). After each run, apply:

| What to set in Google Ads | Source | When |
|---------------------------|--------|------|
| **Keywords in each campaign ad group** | `<course>/data/processed/keyword-sets-display/<keyword_set_id>.csv` — columns `Broad`, `Phrase`, `Exact` | After **stage 1** when `fixed_keyword_sets.json` assigns new sets (compare to live ad groups; add, pause, or remove to match) |
| **Campaign daily budgets** | `<course>/prod/two_stage_plan/stage2_budgets/YYYYMMDD/campaign_plan.csv` — `daily_budget` per `segment` | **Every daily stage-2 run** — one budget per segment (Region × Match Type) → corresponding campaign |

Map each `segment` row (e.g. `USA / Phrase; Exact`) to the campaign for that region and match type. Keyword-set display files are keyed by `keyword_set_id` from `fixed_keyword_sets.json` or `stage1_keyword_sets/campaign_plan.csv` (multi-day window plan; same filename as stage 2).

## What to keep between runs

Almost everything is **regenerated** each run. `pull_input_data` rewrites `data/reports/kw-day-panel.csv` from `min_date` through today on every `prepare-data` call, so reports are a working cache, not something you need to back up separately.

### Track in git (the durable record)

| Artifact | Why |
|----------|-----|
| `data/processed/campaign-summary.csv` | Campaign versions and budget intervals; update via change-history parse or budget append |
| `data/processed/campaign-keyword-sets.csv` | Historical keyword sets for candidate pool |
| `prod/two_stage_plan/fixed_keyword_sets.json` | Active keyword-set assignment between stage-1 runs |
| `data/gkp/*Keywords*Enrollments*.xlsx` | Allowlist input |
| `data/Change history*.html` | Bootstrap archive for keyword changes (optional once summary CSVs are current) |
| `config/`, `<course>/course.yaml` | Pipeline configuration |
| `prod/monitoring/daily_metrics.csv` | Append-only plan-vs-actual scores (optional; not gitignored) |

With those in git plus API credentials, a fresh clone can run `prepare-data` → `run-pipeline`. Monitoring history is optional for reruns but needed for rolling bias/nRMSE trends.

### Production monitoring (keep on disk)

Plan-vs-actual scoring (see **Production monitoring** above) appends to a cumulative history and reads saved stage-2 plans. Do not delete these between daily runs:

| Artifact | Keep? | Why |
|----------|-------|-----|
| `prod/monitoring/daily_metrics.csv` | **Yes** | Append-only score history; 7d/30d rolling stats read from here |
| `prod/monitoring/plan_vs_actual/YYYYMMDD/` | **Yes** | Per-segment detail for each scored day |
| `prod/monitoring/rolling_summary.json` | Optional | Regenerated from `daily_metrics.csv` on every `run-pipeline` |
| `prod/two_stage_plan/stage2_budgets/YYYYMMDD/` | **Yes, ~lookback window** | Monitoring scores a day only if that date's `campaign_plan.csv` still exists; default lookback is 7 days (`monitoring.lookback_days` in `config/default.yaml`). Also the budget push source for that planning date. |

`prod/monitoring/` is not gitignored — commit `daily_metrics.csv` (and detail dirs if you want audits in git). Other `prod/two_stage_plan/` outputs are gitignored except `fixed_keyword_sets.json`; retain recent `stage2_budgets/` dirs locally for monitoring and budget pushes.

### Regenerated each run (no backup needed)

| Artifact | How |
|----------|-----|
| `data/reports/kw-day-panel.csv` | Full API re-pull (`prepare-data`) from `min_date` to today |
| `data/processed/kw-day-panel.csv`, `campaign-day-panel.csv`, `segment-conv-per-click-rates.csv` | `process_input_data` + `generate_campaign_day_panel` |
| Candidate / feature CSVs, keyword-set display | `run-pipeline` (or individual build scripts) |
| `prod/*.joblib`, model manifests, today's stage-2 plan | `run-pipeline` |

### Ephemeral (gitignored working files)

- `prod/two_stage_plan/` except `fixed_keyword_sets.json` — stage-1 outputs, `stage2_budgets/` (keep on disk per monitoring section above; prune dirs once scored and past lookback)
- `experiments/`, backtest MILP/plan dirs, `logs/`

Processed panels are git-tracked today but are optional in git if you always run `prepare-data` first; the summary CSVs are what you cannot reconstruct from the API alone.

## Backtest

Data prep:

```powershell
uv run prepare-data --google-ads-yaml ..\google-ads-prod.yaml --customer-id 1234567890
```

Build modeling artifacts (skip if you already ran production), then walk-forward over the window:

```powershell
uv run python -m scripts.build_keyword_candidates --verify
uv run python -m scripts.build_gkp_set_features
uv run fit-models
uv run backtest --start 2026-05-12 --end 2026-05-25 --use-actual-budget --analyze
uv run analyze-backtest --start 2026-05-12 --end 2026-05-25
```

`--analyze` on `backtest` runs the same summary as `analyze-backtest`; use the latter to re-summarize an existing window without re-running the backtest.

Use `--course <name>` on any command. Use `--config path.yaml` for a one-off override file.

## Output paths

| Kind | Location |
|------|----------|
| Processed panels | `<course>/data/processed/` |
| Keyword set display (for push) | `<course>/data/processed/keyword-sets-display/` |
| Model fit | `<course>/prod/model_manifest.json`, `holdout_metrics.json` |
| Production plan | `<course>/prod/two_stage_plan/` |
| Experiments / debug | `<course>/experiments/features/`, `experiments/diagnostics/` |
| Backtest window | `<course>/backtests/<start>_<end>/` |

## Tests

```powershell
uv run pytest tests/ -q
```

Golden parity: `tests/fixtures/backtest_golden/2026-05-12_2026-05-25/` vs `sys_think/backtests/2026-05-12_2026-05-25/`
