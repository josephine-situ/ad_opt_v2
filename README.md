# ad_opt_v2 — Multi-course campaign optimization

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
  backtests/<start>_<end>/     # walk-forward backtest windows
scripts/                       # CLI entry points
utils/                         # library code
tests/
```

## Setup

```powershell
uv sync --extra optimization --extra ml
```

Keep Google Ads credentials outside the repo (`.gitignore` excludes `google-ads*.yaml`).

**Required before candidate build:** `<course>/data/gkp/*Keywords*Enrollments*.xlsx`

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

## Quick start (sys_think)

```powershell
uv run prepare-data --google-ads-yaml ..\google-ads-prod.yaml --customer-id 1234567890
uv run python -m scripts.build_keyword_candidates --verify
uv run python -m scripts.build_gkp_set_features
uv run fit-models
uv run run-pipeline --window-start 2026-05-12 --window-end 2026-05-25 --planning-date 2026-05-12
uv run backtest --start 2026-05-12 --end 2026-05-25 --use-actual-budget --analyze
uv run analyze-backtest --start 2026-05-12 --end 2026-05-25
```

Use `--course <name>` on any command. Use `--config path.yaml` for a one-off override file.

## Output paths

| Kind | Location |
|------|----------|
| Processed panels | `<course>/data/processed/` |
| Model fit | `<course>/prod/model_manifest.json`, `holdout_metrics.json` |
| Production plan | `<course>/prod/two_stage_plan/` |
| Backtest window | `<course>/backtests/<start>_<end>/` |

## Tests

```powershell
uv run pytest tests/ -q
```

Golden parity: `tests/fixtures/backtest_golden/2026-05-12_2026-05-25/` vs `sys_think/backtests/2026-05-12_2026-05-25/`
