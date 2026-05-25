# campaign_opt

Config-driven **campaign-level** optimization: daily budget per `(region, match_types)` segment plus discrete **keyword set** choice.

Follow the steps below in order. Prerequisites and package reference are at the end.

---

## 1. Install

```powershell
pip install -e ".[optimization,ml]"
```

Gurobi license required for optimization. Keep Google Ads credentials outside the repo (see root [`README.md`](../README.md)).

---

## 2. Config

Default experiment: [`opt_results/sys_think/campaign/default/campaign_config.json`](../opt_results/sys_think/campaign/default/campaign_config.json)

Key fields:

- `target` — `all_conv` or `clicks` (optimization objective)
- `context_features` — calendar, keyword-set semantic/GKP columns
- `constraints.regional_order` — e.g. USA ≥ A ≥ B spend
- `model_policy.validation` — `time_series_cv` with `cv_folds`, or `time_holdout` for last-N-day reporting only
- `evaluation` — `use_ensemble`, `baseline_budget`, `weight_by_cv_rmse` (plan vs actual scoring)

Copy or edit this file per course/experiment under `opt_results/<course>/campaign/<exp_name>/`.

---

## 3. Prepare input data

All paths below use `<course>` (e.g. `sys_think`). Processed files must exist before `run_campaign_pipeline.py` or the individual campaign scripts.

### 3a. Pull Google Ads reports

```powershell
uv run python scripts/pull_input_data.py `
  --datasets ads_reports `
  --output-course sys_think `
  --google-ads-yaml ..\google-ads-prod.yaml `
  --customer-id 1234567890
```

Writes under `data/<course>/reports/`, including  
`Search keyword - raw input to models.csv` (source for the keyword-day panel).

### 3b. Parse change history → campaign summary & keyword sets

Save the Google Ads **Change history** HTML under `data/<course>/`, then:

```powershell
uv run python scripts/parse_change_history_html.py `
  "data/sys_think/Change history - MIT xPRO - System Thinking - Google Ads.html" `
  -o data/sys_think/change_history_budgets.csv `
  --campaign-summary-output data/sys_think/processed/campaign-summary.csv `
  --keyword-sets-output data/sys_think/processed/campaign-keyword-sets.csv `
  --search-keyword-report "data/sys_think/reports/Search keyword - raw input to models.csv"
```

### 3c. Keyword-day panel

```powershell
uv run python scripts/process_input_data.py --output-course sys_think
```

→ `data/<course>/processed/kw-day-panel.csv`

### 3d. Campaign-day panel

```powershell
uv run python scripts/generate_campaign_day_panel.py --output-course sys_think
```

→ `data/<course>/processed/campaign-day-panel.csv` (main modeling input for `campaign_opt`)

### 3e. (Optional) Keyword Planner / GKP metrics

If `context_features.gkp_set` is enabled in config:

```powershell
uv run python scripts/pull_input_data.py `
  --datasets keyword_planning `
  --output-course sys_think `
  --keyword-planning-input-file data\sys_think\gkp\keywords_classified.csv `
  --google-ads-yaml ..\google-ads-prod.yaml `
  --customer-id 1234567890
```

→ `data/<course>/gkp/`

More detail on pulls and parsing: root [`README.md`](../README.md).

---

## 4. Build GKP set features & keyword candidates

```powershell
uv run python scripts/build_gkp_set_features.py --course sys_think
uv run python scripts/build_keyword_candidates.py --course sys_think
```

Produces set-level GKP aggregates and `data/<course>/processed/segment-keyword-candidates.csv` used by the MILP.

---

## 5. Fit response models

```powershell
uv run python scripts/fit_response_models.py --course sys_think
```

Tournament (ridge, power, RF, XGB) with **level-scale** metrics; writes `model_manifest.json`, `winner_model.joblib`, `holdout_metrics.json` under `opt_results/<course>/campaign/<exp>/`.

**Model selection rules:**

1. All candidates scored on **levels** (comparable RMSE / R²).
2. With `validation.scheme: time_series_cv`, winner picked by **mean CV RMSE** on train.
3. Holdout metrics logged in `holdout_metrics.json`.
4. `optimizer_backend: auto` maps winner → `linear` | `piecewise_linear` | `tree_embed`.

---

## 6. Optimize (Gurobi MILP)

```powershell
uv run python scripts/optimize_campaign.py --course sys_think --budget 400
```

Or run steps 4–6 in one command (regenerates panel if missing):

```powershell
uv run python scripts/run_campaign_pipeline.py --course sys_think
```

Use `--skip-gkp` / `--skip-candidates` if those artifacts are already built.

---

## 7. Walk-forward backtest (optional)

```powershell
uv run python scripts/backtest_campaign.py --course sys_think --start 2025-10-01 --end 2025-12-31
```

One MILP per day in range (train on `date < t`, optimize, score vs actual). See [Outputs](#outputs).

---

## 8. Production monitor & evaluation ensemble (optional)

Plan vs actual uses a **separate ensemble** (all `model_policy` candidates on available train data), not the MILP winner:

| Metric | Definition |
|--------|------------|
| **f(0)** | Baseline: `evaluation.baseline_budget` (default 0) + modal keyword set per segment on train |
| **pred_lift** | f(plan) − f(0) |
| **actual_model_lift** | f(actual budget & set) − f(0), same ensemble |

Primary comparison: `pred_lift` vs `actual_model_lift`. Observed totals are reference only.

```powershell
# Fit ensemble on full history (saved for monitor)
uv run python scripts/fit_evaluation_ensemble.py --course sys_think

# Compare latest plan to actuals (uses saved ensemble or refits on pre-eval train)
uv run python scripts/monitor_campaign_production.py --course sys_think --lag 1
```

---

## MILP structure (shared core)

For each segment `s`:

- Continuous `x[s]` = daily budget (bounded by history)
- Binary `y[s,k]` = select keyword set `k`; ∑_k y[s,k] = 1

Backends only differ in how per-segment predicted `target` is expressed in Gurobi (`linear`, `piecewise_linear`, `tree_embed` via [`backends/milp_core.py`](backends/milp_core.py)).

---

## Outputs

Under `opt_results/<course>/campaign/<exp_name>/`:

- `model_manifest.json`, `winner_model.joblib`, `holdout_metrics.json`
- `campaign_plan.csv`, `linear_coeffs.json`, `solver_status.json`
- `ensemble_model.joblib` (after `fit_evaluation_ensemble.py` or backtest with `use_ensemble`)
- Backtest: `backtest/<start>_<end>/plans/YYYYMMDD/`, `plan_vs_actual.csv`, `daily_backtest_summary.csv`

---

## Package layout

| Module | Purpose |
|--------|---------|
| [`schema.py`](schema.py) | Load `campaign_config.json` |
| [`features.py`](features.py) | Build modeling frame from `campaign-day-panel` |
| [`evaluation.py`](evaluation.py) | Ensemble fit + incremental `f(decision)−f(0)` comparison |
| [`cv.py`](cv.py) | Expanding-window time-series cross-validation |
| [`modeling.py`](modeling.py) | Model tournament with level-scale metrics |
| [`coefficients.py`](coefficients.py) | Export linear coeffs for MILP objective |
| [`optimize.py`](optimize.py) | Dispatch solver backend from manifest |
| [`backtest.py`](backtest.py) | Walk-forward daily backtest loop |
| [`backends/milp_core.py`](backends/milp_core.py) | Shared Gurobi MILP |
| [`backends/linear.py`](backends/linear.py) | Linear objective backend |
| [`backends/piecewise_linear.py`](backends/piecewise_linear.py) | Piecewise budget curve backend |
| [`backends/tree_embed.py`](backends/tree_embed.py) | Tree → PW surrogate for Gurobi |
