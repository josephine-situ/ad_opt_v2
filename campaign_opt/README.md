# campaign_opt

Config-driven **two-stage** campaign optimization for System Thinking.

## Pipeline

1. **Data prep** — `uv run prepare-data` (see root [README.md](../README.md))
2. **Keyword candidates** — requires enrollment allowlist xlsx under `sys_think/data/gkp/`
3. **GKP features** — `cli.build_gkp_set_features` (uses cached `Saved Keyword Stats*.csv`)
4. **Model fit** — `uv run fit-models` → `model_manifest.json`
5. **Two-stage plan** — `uv run run-pipeline --window-start … --window-end …`
6. **Backtest** — `uv run backtest --start … --end …`
7. **Analyze** — `uv run analyze-backtest` or `--analyze` on backtest

## Modeling notes

- Default target: `conv_scaled_clicks` (see `campaign_config.json`)
- Optimizer: `optimizer_winner: xgboost` with `tree_embed` MILP backend
- Evaluation scorer may differ from MILP optimizer (fit on full panel during backtest)
- Holdout R² on segment-day targets is expected to be modest (~0.3–0.7 depending on target)

## Two-stage optimization

**Stage 1** — one multi-day MILP over `[window_start, window_end]`; picks one `keyword_set_id` per segment (train: `date < window_start`).

**Stage 2** — single-day MILP per planning date with fixed sets (train: walk-forward `date < t`, CV-tuned).

Implementation: [`two_stage_plan.py`](two_stage_plan.py), [`backtest_two_stage.py`](backtest_two_stage.py).

## Config reference

Key `campaign_config.json` fields:

- `target` — optimization objective (`conv_scaled_clicks`, `clicks`, `all_conv`)
- `context_features` — calendar, keyword-set semantic/GKP columns
- `constraints.regional_order` — e.g. USA ≥ A ≥ B spend
- `model_policy.optimizer_winner` — MILP embed model (shipped: `xgboost`)
- `model_policy.validation` — time-series CV, `min_train_rows`
- `evaluation` — plan-vs-actual scoring (`use_ensemble`, `apply_observed_budget_floor`, etc.)

Cross-validation design: [`docs/cross_validation.md`](docs/cross_validation.md)
