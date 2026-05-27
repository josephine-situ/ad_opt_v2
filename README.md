# ad_opt_v2

Google Ads data pull, campaign-level budget/keyword-set optimization, and walk-forward backtests for the System Thinking course.

## Repo layout

| Path | Purpose |
|------|---------|
| `eda_clicks_budget_keywords.ipynb` | Exploratory analysis (clicks, budget, keyword sets) |
| `data/<course>/` | Change-history HTML, processed panels, GKP inputs |
| `campaign_opt/` | Optimization library (features, models, MILP, backtest) |
| `scripts/` | CLI entry points for pull → panel → pipeline → backtest |
| `opt_results/<course>/campaign/<exp>/` | `campaign_config.json` + backtest summaries |
| `presentation/` | Beamer slides (`SUMMARY_presentation.tex`, `BACKTEST_presentation.tex`) |
| `tests/` | Unit tests for the optimization package |

Notebook figures go under `figures/` (gitignored; created when you run the notebook).

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```powershell
uv sync --extra optimization --extra ml
```

Optional notebook EDA: `uv sync --extra notebook`.

Keep Google Ads credentials outside the repo (`.gitignore` excludes `google-ads*.yaml` / `google-ads*.json`).

## Data pipeline

Run in order: **pull → clean → parse change history → campaign-day panel**.

```powershell
uv run python scripts/pull_input_data.py --datasets campaign_opt --output-course sys_think --google-ads-yaml ..\google-ads-prod.yaml --customer-id 1234567890
uv run python scripts/process_input_data.py --output-course sys_think
uv run python scripts/parse_change_history_html.py --output-course sys_think
uv run python scripts/generate_campaign_day_panel.py --output-course sys_think
```

`build_keywords_classified.py` and `pull_input_data.py --datasets keyword_planning` are needed when GKP set features are enabled in config. Details: [`campaign_opt/README.md`](campaign_opt/README.md).

## Optimization and backtest

Default config: `opt_results/sys_think/campaign/default/campaign_config.json`

```powershell
uv run python scripts/run_campaign_pipeline.py --course sys_think
uv run python scripts/backtest_campaign.py --course sys_think --start 2026-05-12 --end 2026-05-25 --analyze
uv run python scripts/analyze_backtest_results.py --course sys_think --start 2026-05-12 --end 2026-05-25
```

Cluster array jobs: `submit_backtest.sh` (one day per task; last task runs analysis). Rerun gaps: `submit_backtest_missing.sh`.

Per-day plan folders and fitted `.joblib` files under `opt_results/.../backtest/` are regeneratable and not tracked in git.

## Presentations

```powershell
cd presentation
latexmk BACKTEST_presentation.tex
latexmk SUMMARY_presentation.tex
```

Build artifacts go to `presentation/build/`.
