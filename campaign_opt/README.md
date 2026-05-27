# campaign_opt

Config-driven **campaign-level** optimization: daily budget per `(region, match_types)` segment plus discrete **keyword set** choice.

Run steps **1–7** in order to produce a campaign plan (tournament → evaluation ensemble → MILP). Steps **8–9** are optional validation and production monitoring.

```mermaid
flowchart TD
  S1[1 Install] --> S2[2 Config]
  S2 --> S3[3 Prepare input data]
  S3 --> S4[4 GKP + keyword candidates]
  S4 --> S5[5 Fit response models]
  S5 --> S5b[6 Fit evaluation ensemble]
  S5b --> S6[7 Optimize MILP]
  S6 --> S7[8 Backtest optional]
  S6 --> S8[9 Monitor optional]
```



More on API pulls and HTML parsing: root `[README.md](../README.md)`.

---

## Modeling considerations

Raw `**all_conv**` is a poor default optimization target because of how the panel is built and what we can identify from history. The shipped `default` experiment uses `**conv_scaled_clicks**` (clicks weighted by segment conversion-per-click); `**clicks**` remains a supported alternative.

**Decision lever.** Response models use `**daily_budget`** (the configured cap from change history). `**cost**` is observed spend, not a controllable input, and is excluded from models and budget diagnostics.

**Limited budget variation.** Within a `campaign_version`, budget is fixed. Budget only changes when the campaign is reconfigured (new version). That gives few within-cell budget levels — mostly when the same `(segment, keyword_set_id)` appears at multiple budgets across versions. Full models that pool all rows often show a **negative budget coefficient on conversions** because budget moves coincide with keyword-set and strategy changes (Lead Gen → Run 19 prospecting), not because higher caps reduce conversions.

**Conversions are unstable over time.** `all_conv` levels and mix shifted substantially over the past two years (campaign type, match types, tracking). Time-series holdout and CV R² stay low for conversion models, and OOS forecasts are dominated by regime change rather than budget or set features.

**Clicks are more stable and identifiable.** On identifiable `(segment, keyword_set_id)` cells, within-set budget slopes for clicks are typically **positive**. Clicks respond more directly to auction volume at a given cap, so they are a better proxy for the budget lever even when conversion efficiency moves.

**Implication for config.** Prefer `target: conv_scaled_clicks` or `clicks` over `all_conv` for optimization and MILP objectives. Keep `all_conv` in `secondary_metrics` for reporting and diagnostics. Run `diagnose_budget_response.py` before trusting budget signs in fitted coefficients. Holdout R² on segment-day targets (~0.3–0.7 for tree/ensemble models, depending on target) is expected to stay well below in-sample EDA benchmarks; prioritize correct budget direction and relative ranking over chasing high R².

---

## 1. Install

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```powershell
uv sync --extra optimization --extra ml
```

The `ml` extra includes XGBoost and SHAP (signed directional effects for tree models in fit logs / `model_manifest.json`).

Run scripts with `uv run python ...` or activate the project venv:

```powershell
.\.venv\Scripts\Activate.ps1
```

Optional notebook EDA: `uv sync --extra notebook`.

Gurobi license required for optimization. Keep Google Ads credentials outside the repo (see root `[README.md](../README.md)`).

---

## 2. Config

Default experiment: `[opt_results/sys_think/campaign/default/campaign_config.json](../opt_results/sys_think/campaign/default/campaign_config.json)`

Key fields:

- `target` — optimization objective (see [Modeling considerations](#modeling-considerations)). Shipped `default` config: `conv_scaled_clicks`. Also supported: `clicks`, `all_conv` (clicks scaled by historical conversion-per-click per `(region, match_types)` with global fallback for zero-click segments). Use `secondary_metrics` to track conversions alongside the target.
- `context_features` — calendar, keyword-set semantic/GKP columns
- `constraints.regional_order` — e.g. USA ≥ A ≥ B spend
- `constraints.budget_tiebreak_penalty` — optional (default `1e-8`); subtract `penalty × Σ daily_budget` from the MILP objective so equal predicted-target solutions prefer lower total spend
- `model_policy.validation` — `time_series_cv` with `cv_folds`, `min_train_fraction` (e.g. `0.5` = each fold trains on at least half of train-panel days), `min_val_days`, or `time_holdout` for last-N-day reporting only
- `evaluation` — `use_ensemble`, `baseline_budget`, `weight_by_cv_rmse`, `objective` (`levels` or `incremental`), `apply_observed_budget_floor` (predict **0** when `daily_budget` &lt; segment min observed cap in training panel; same rule in MILP and plan-vs-actual; does not change holdout R²), `budget_floor_atol` (default `0.01`), optional `max_level_ub` (McCormick cap for tree backends), `milp_external_level_tol` (default `0.01`, MILP vs gated sklearn level check)

Copy or edit this file per course/experiment under `opt_results/<course>/campaign/<exp_name>/`.

---

## 3. Prepare input data

All paths below use `<course>` (e.g. `sys_think`). Performance and conversion data come from the **Google Ads API**; campaign budgets and negative keywords still come from saved **change-history HTML** (until a change_event API path exists).

### 3a. Pull Google Ads reports (API)

```powershell
uv run python scripts/pull_input_data.py `
  --datasets campaign_opt `
  --output-course sys_think `
  --google-ads-yaml ..\google-ads-prod.yaml `
  --customer-id 1234567890
```

Uses `min_date` from `config.py` (override with `--start-date` if needed).

→ `data/<course>/reports/kw-day-panel.csv` — clicks, cost, filtered `all_conv`

If `context_features.gkp_set` is enabled:

```powershell
uv run python scripts/build_keywords_classified.py --course sys_think
uv run python scripts/pull_input_data.py `
  --datasets campaign_opt,keyword_planning `
  --output-course sys_think `
  --keyword-planning-input-file data\sys_think\gkp\keywords_classified.csv `
  --google-ads-yaml ..\google-ads-prod.yaml `
  --customer-id 1234567890
```

### 3b. Clean keyword-day panel

```powershell
uv run python scripts/process_input_data.py --output-course sys_think
```

→ `data/<course>/processed/kw-day-panel.csv`

### 3c. Parse change history (HTML)

Save the Change history HTML under `data/<course>/`, then:

```powershell
uv run python scripts/parse_change_history_html.py --output-course sys_think
```

Uses the cleaned kw-day-panel to fill keyword inventory gaps. Writes `campaign-summary.csv` and `campaign-keyword-sets.csv` (budgets are on each summary row).

### 3d. Campaign-day panel

```powershell
uv run python scripts/generate_campaign_day_panel.py --output-course sys_think
```

→ `data/<course>/processed/campaign-day-panel.csv` — clicks, cost, and `all_conv` per campaign version (main modeling input)

---

## 4. Build GKP set features & keyword candidates

Required when `context_features.gkp_set` is set in config; otherwise use `--skip-gkp` in the pipeline.

```powershell
uv run python scripts/build_keyword_candidates.py --course sys_think --top-n-values 10,20,40
uv run python scripts/build_gkp_set_features.py --course sys_think
```

Produces set-level GKP aggregates and `data/<course>/processed/segment-keyword-candidates.csv` used by the MILP.

**Keyword candidates (`build_keyword_candidates.py`)**

- **Segment** = `(region, match_types)` where `match_types` is the campaign-level configuration from change history (`Broad`, `Phrase; Exact`, `Exact`, or `Broad; Phrase; Exact`). The MILP picks one keyword set per segment, not separate lists per match type.
- **Enrollment allowlist** — when `data/<course>/gkp/*Keywords*Enrollments*.xlsx` exists, `load_enrollment_keyword_allowlist()` restricts **all** historical and synthetic sets to approved keywords; sets with no allowlisted keywords are dropped from `segment-keyword-candidates.csv`.
- **Historical** candidates: every `keyword_set_id` observed in that segment (allowlist-filtered).
- **`synthetic_top_conv` / `synthetic_top_conv_n{N}`** — rank `kw-day-panel.csv` by **`all_conv`** (not clicks): `volume_col="all_conv"`, `require_positive_volume=True`, within segment region and `allowed_match_types`. The pool is the union of top‑`N` converters and conversion‑efficiency keywords (`all_conv` per dollar). If fewer than `N` converters exist, pad from the enrollment allowlist (by enrollment count in the GKP file, else sheet order). With `--top-n-values 10,20,40`, emit separate sets `synthetic_top_conv_n10`, `_n20`, `_n40` per segment.
- **`synthetic_allowlist` / `synthetic_allowlist_n{N}`** — first `N` allowlist keywords by enrollment priority (full list when only one `top_n` is used). Only emitted when the enrollments spreadsheet exists.
- **`synthetic_semantic` / `synthetic_dispersion` / `synthetic_composite`** (each with `_n10` / `_n20` / `_n40` when multiple caps are set) — built from the same top‑conv performance pool; semantic = course‑anchor similarity ranking; dispersion = greedy max spread; composite = greedy max `z(course_sim) + z(dispersion)`.
- Default shipped caps: pass **`--top-n-values 10,20,40`** (see command above). Without it, a single cap uses `--top-n` (default 30).

Disable variants with `--no-top-conv-synthetic`, `--no-allowlist-synthetic`, `--no-semantic-synthetic`, `--no-dispersion-synthetic`, or `--no-composite-synthetic`.

Keywords **do not** need to be identical across Broad / Phrase / Exact within a campaign. Historical sets store separate `broad_keywords`, `phrase_keywords`, and `exact_keywords` columns; synthetic top‑conv sets assign match-type columns using **dominant `all_conv`** per keyword in the panel (`match_type_rank_col="all_conv"`). Union-level semantic/GKP features use `positive_keywords`; **match-type structure** features (counts, Jaccard overlap, per-type course similarity, cross-type embedding similarity) are computed from the split lists in `build_gkp_set_features.py` and exposed as `keyword_set_match_type` in the campaign config.

Outputs:


| File                                 | Contents                                            |
| ------------------------------------ | --------------------------------------------------- |
| `segment-keyword-candidates.csv`     | segment → `keyword_set_id`, `source`                |
| `campaign-keyword-sets-extended.csv` | keyword lists (+ match-type columns for synthetics) |


Run **candidates first**, then **GKP/set features**, so `campaign-keyword-sets-extended.csv` exists before `build_keyword_set_feature_table()` merges synthetic sets into `keyword-set-features.csv`.

**Budget diagnostics** (before trusting budget coefficients):

```powershell
uv run python scripts/diagnose_budget_response.py --course sys_think
```

Compares `all_conv` vs `clicks` (ridge + XGB) and within-`(segment, keyword_set_id)` **daily_budget** slopes. Outputs under `opt_results/<course>/campaign/<exp>/diagnostics/budget/`. Cost is not used in response models or these diagnostics.

**Calendar ablation** (season vs month vs sin/cos month):

```powershell
uv run python scripts/diagnose_budget_response.py --course sys_think --calendar-ablation --ablation-target clicks
```

Writes `diagnostics/budget/calendar_ablation/calendar_ablation.csv` with CV RMSE and holdout R² per spec.

---

## 5. Fit response models

```powershell
uv run python scripts/fit_response_models.py --course sys_think
```

Tournament (ridge, power, RF, XGB) with **level-scale** metrics; writes `model_manifest.json`, `winner_model.joblib`, `holdout_metrics.json`, and `**linear_coeffs.json`** under `opt_results/<course>/campaign/<exp>/`.

**Ridge uses the same design as the linear MILP** (`region + match_type + budget×(region + match) + context_features` via `[linear_design.py](linear_design.py)`). Keyword-set selection in the MILP uses `**static_context_lift`** — per-set scores derived from static context-feature coefficients (semantic, GKP), not `keyword_set_id` dummies. Saved debug matrices land in `features/`:


| File                                   | Purpose                            |
| -------------------------------------- | ---------------------------------- |
| `modeling_frame_train/holdout.csv`     | Wide modeling frame                |
| `linear_milp_design_train/holdout.csv` | Aligned ridge / MILP design matrix |
| `linear_milp_design_columns.json`      | Column order for holdout reindex   |
| `context_design_train/holdout.csv`     | Tree-model context OHE matrix      |
| `artifact_manifest.json`               | Paths index                        |


**Model selection rules:**

1. All candidates scored on **levels** (comparable RMSE / R²).
2. With `validation.scheme: time_series_cv`, winner picked by **mean CV RMSE** on train.
3. Holdout metrics logged in `holdout_metrics.json`.
4. Tournament winner is the lowest CV RMSE (with `time_series_cv`) or lowest holdout RMSE otherwise; `optimizer_backend: auto` uses that model's backend (`linear` | `piecewise_linear` | `tree_embed`).

---

## 6. Fit evaluation ensemble (plan vs actual scoring)

```powershell
uv run python scripts/fit_evaluation_ensemble.py --course sys_think
```

Fits the 5-member ensemble on the **full modeling panel** with CV-RMSE weights from `holdout_metrics.json` (same as backtest). Writes `ensemble_model.joblib` and `ensemble_meta.json`.

## 7. Optimize (Gurobi MILP)

```powershell
uv run python scripts/optimize_campaign.py --course sys_think --budget 400
```

Walk-forward train (`date < planning_date`), **3-fold time-series CV** hyperparameter tuning, then MILP with `optimizer_winner` (default `ensemble_ridge_xgb` / `tree_embed`). Optional `--planning-date YYYY-MM-DD` (default: latest panel date).

`run_campaign_pipeline.py` runs steps **4–5** and **7** only (candidates, GKP features if enabled, tournament, single-day optimize). Run step **6** (`fit_evaluation_ensemble.py`) separately before production monitor or if you want a saved `ensemble_model.joblib` without running a backtest.

Or run steps **4–7** manually after step 3 is complete (regenerates panel if missing):

```powershell
uv run python scripts/run_campaign_pipeline.py --course sys_think
```

Use `--skip-gkp` / `--skip-candidates` when GKP set features are disabled or already built.

---

## 8. Walk-forward backtest (optional)

### Daily mode (default)

```powershell
uv run python scripts/backtest_campaign.py --course sys_think --start 2025-10-01 --end 2025-12-31 --use-actual-budget
```

One MILP per day in range: fit the **optimizer** on walk-forward train (`date < t`) with time-series CV hyperparameter search, embed in Gurobi, then optimize budget + keyword set. Requires `fit_response_models.py` artifacts (`model_manifest.json`, `holdout_metrics.json` when `evaluation.weight_by_cv_rmse` is true). The **evaluation** scorer is fit **once on the full modeling panel** during the backtest (also available via `fit_evaluation_ensemble.py` as `ensemble_model.joblib`).

Optional flags: `--strategy daily` (explicit); `--use-actual-budget` sets each day's MILP cap to the panel sum of segment `daily_budget` (see `presentation/BACKTEST_presentation.tex` May 2026 example).

### Two-stage mode (fixed keyword sets + daily budgets)

Operational backtest that separates **keyword-set choice** (slow, once per window) from **budget allocation** (fast, walk-forward each day). Implemented in [`backtest_two_stage.py`](backtest_two_stage.py); uses the same `ensemble_ridge_xgb` / `tree_embed` backend as daily mode (not a separate linear-only path).

```powershell
uv run python scripts/backtest_campaign.py --course sys_think --start 2025-10-01 --end 2025-12-31 --strategy two_stage
# Optional: panel caps per day (like daily May example)
uv run python scripts/backtest_campaign.py --course sys_think --start 2026-05-12 --end 2026-05-25 --strategy two_stage --use-actual-budget --analyze
# Re-run stage 2 only after stage 1 completed:
uv run python scripts/backtest_campaign.py --course sys_think --start 2026-05-12 --end 2026-05-25 --strategy two_stage --skip-stage1 --analyze
```

**Stage 1 — keyword sets for the full window**

1. Train the optimizer on all modeling rows with `date < start` (requires ≥ `min_train_rows`, default 50).
2. Run one **multi-day** MILP over every calendar day in `[start, end]` (`planning_dates` = full window; `tune_optimizer=True`). Each day can have its own budget variables; keyword-set binaries are shared across days so each segment picks **one** list for the whole window.
3. Write `fixed_keyword_sets.json` (segment → `keyword_set_id`) and `stage1_keyword_sets/keyword_set_plan.csv`. Stage‑1 budget cap: `--budget` total, or first-day panel sum when `--use-actual-budget`.

**Stage 2 — daily budgets with fixed sets**

For each day `t` in `[start, end]`:

1. Train on `date < t` (walk-forward).
2. Single-day MILP with `fixed_keyword_sets` from stage 1 (`tune_optimizer=False`); only `daily_budget` per segment is free.
3. Score plan vs panel with the evaluation ensemble → `plans/YYYYMMDD/plan_vs_actual.csv` and `campaign_plan.csv`.

**Flags**

| Flag | Effect |
|------|--------|
| `--use-actual-budget` | Each day's MILP cap = panel sum of segment `daily_budget` (stage 1 uses first-day cap). |
| `--skip-stage1` | Load `fixed_keyword_sets.json` from a prior run; run stage 2 only. |
| `--budget` | Fixed daily cap when not using actual budgets (course default from `COURSE_CONFIG`). |

Config block (optional; CLI `--strategy` overrides):

```json
"backtest": {
  "strategy": "daily",
  "keyword_set_horizon": "period",
  "budget_cadence": "W-MON"
}
```

(`budget_cadence` is reserved for future weekly aggregation; the current two-stage implementation optimizes **daily** budgets in stage 2.)

**Outputs** under `backtest/<start>_<end>/`:

- `fixed_keyword_sets.json`, `stage1_keyword_sets/`
- `daily_backtest_summary.csv`, `daily_backtest_summary.json` (per-day plan vs actual metrics)
- `plans/YYYYMMDD/plan_vs_actual.csv`, `campaign_plan.csv` (gitignored under `plans/`)
- After `--analyze`: same summary tables as daily mode (`evaluation_results.csv`, `backtest_summary.csv`, …) when `plan_vs_actual` files exist

### Summarize performance

After a backtest window finishes, compile daily evaluation rows and write summary tables (CSV + LaTeX):

```powershell
uv run python scripts/analyze_backtest_results.py --course sys_think --start 2025-10-06 --end 2025-10-12
```

Or pass `--analyze` to `backtest_campaign.py` after a local full-window run. Outputs:

- `evaluation_results.csv` — daily totals (`pred_lift`, `actual_model_lift`, observed, budgets)
- `backtest_summary.csv` — Model / Actual rows with conversions, budget, conv/$, and improvement % 
- `regional_breakdown.csv` — opt vs actual spend/lift shares by region
- `backtest_summary.tex` — LaTeX performance table

On a cluster, run one `backtest_campaign.py` job per day with `--day YYYY-MM-DD`, then `analyze_backtest_results.py` after the window completes.

---

## 9. Production monitor (optional)

Plan vs actual uses the **evaluation ensemble** (5 tournament members, CV-RMSE weights from `holdout_metrics.json`), not the MILP optimizer model:


| Metric                | Definition                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------- |
| **f(0)**              | `evaluation.baseline_budget` (default 0) + **same keyword set** as the row being scored (plan set for Model, campaign set for Actual) |
| **pred_lift**         | `max(pred_lift_raw, 0)` on plan rows                                                        |
| **pred_lift_raw**     | `f(plan) − f(0)` (signed); equals gated `f(plan)` when `baseline_budget=0` and floor zeros `f(0)` |
| **f_plan_level**      | Gated level `f(plan budget, plan set)`                                                      |
| **f_zero**            | Gated level at `baseline_budget` with plan/market keyword set                               |
| **actual_model_lift** | `max(actual_model_lift_raw, 0)` on market rows                                              |
| **actual_model_lift_raw** | `f(panel budget, panel set) − f(0 with panel set)`; gated `f(panel)` when `f(0)=0`     |


Primary comparison: `pred_lift` vs `actual_model_lift`. Observed totals are reference only.

```powershell
# Fit ensemble on full history (saved for monitor)
uv run python scripts/fit_evaluation_ensemble.py --course sys_think

# Compare latest plan to actuals (loads ensemble_model.joblib or fits via fit_evaluation_model)
uv run python scripts/monitor_campaign_production.py --course sys_think --lag 1
# Force refit after new panel data:
uv run python scripts/monitor_campaign_production.py --course sys_think --lag 1 --refit-ensemble
```

---

## MILP structure (shared core)

For each segment `s`:

- Continuous `x[s]` = daily budget (bounded by historical min/max caps on `daily_budget`)
- Binary `y[s,k]` = select keyword set `k`; ∑_k y[s,k] = 1

Backends differ in how per-segment predicted `target` is expressed in Gurobi (`linear`, `piecewise_linear`, exact `tree_embed` via `[backends/tree_embedding.py](backends/tree_embedding.py)`).

**Ungated prediction.** Let `f̃_s(x)` be the embedded model level at budget `x` (ridge + XGB trees, or linear/piecewise form) for the chosen keyword set and planning-day calendar.

**Observed-budget floor** (`evaluation.apply_observed_budget_floor`, default `true` in shipped config):

```
x_s^min = min{ daily_budget : segment s in training panel }
F_s(x)  = 0                         if x < x_s^min  (within evaluation.budget_floor_atol)
        = f̃_s(x)                    otherwise
```

- **MILP:** McCormick “gating” constraints in [`backends/milp_core.py`](backends/milp_core.py) / [`prediction_gating.py`](backends/prediction_gating.py) force the solver's `pred_vars[s]` to follow `F_s(x_s)` (tree leaf paths are not re-cut; gating wraps the segment level).
- **Post-solve / evaluation:** [`optimizer_prediction.py`](optimizer_prediction.py) applies the same rule in numpy for `external_model_pred` and `plan_vs_actual` scoring.
- **Holdout tournament fit** stays on ungated sklearn predictions (floor does not change reported CV / holdout R²).

**Objective** (`evaluation.objective`):

- `levels` (shipped `default` config): maximize `Σ_s F_s(x_s)` minus budget tie-break.
- `incremental`: maximize `Σ_s [F_s(x_s) − F_s(baseline)]` at `evaluation.baseline_budget` with the same keyword set (baseline levels also gated when the floor is on).

Subtract `constraints.budget_tiebreak_penalty` (code default `1e-8`; shipped config uses `1e-4`) × Σ daily budgets to break ties toward lower spend.

**Plan vs actual (evaluation ensemble, not the MILP objective):** code computes `pred_lift_raw = f(plan) − f(0)` at `evaluation.baseline_budget` (default `0`) with the **same** keyword-set features for plan and counterfactual rows; `pred_lift = max(pred_lift_raw, 0)`. With `apply_observed_budget_floor: true`, `f(0)` is **0** (budget `0` is below each segment’s min observed cap), so backtest headline totals are **gated levels** `f(plan)` / `f(panel)`, not a meaningful nonzero baseline subtraction. Columns `f_plan_level` and `f_zero` in `plan_vs_actual.csv` expose the level and baseline explicitly.

For `ensemble_ridge_xgb`, each candidate baseline `f_k(baseline)` is `EnsembleModel.predict_levels` on the same `build_segment_decision_rows` features used in backtest scoring (not a separate analytic formula).

---

## Outputs

Under `opt_results/<course>/campaign/<exp_name>/`:

- `model_manifest.json`, `winner_model.joblib`, `holdout_metrics.json`
- `features/` — saved modeling frames and design matrices (see §5)
- `campaign_plan.csv`, `linear_coeffs.json`, `solver_status.json`

`campaign_plan.csv` columns:


| Column                | Meaning                                                                                                                                                                                  |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `milp_pred`           | Level prediction from the solver embedding at the optimum: `f(plan)` (Gurobi value of `pred_vars[segment]`; what the MILP maximizes, up to the budget tie-break)                       |
| `pred_over_base`      | Incremental lift `f(plan) - f(0)` in solver space; `f(0)` uses the **same chosen keyword set** with `daily_budget = 0`                                                                   |
| `external_model_pred` | After solve: same **objective-side** gated level as the MILP (raw sklearn level, then 0 if `daily_budget` &lt; min observed cap on `panel`); warns if `|milp_pred - external_model_pred| > evaluation.milp_external_level_tol` (default `0.01`) |
| `n_planning_days`     | Number of calendar days in the MILP objective (1 for single-day optimize / stage‑2 two-stage; full window length for stage‑1 multi-day keyword-set solve)                              |

`milp_pred` and `external_model_pred` are **level** predictions. Use `pred_over_base` for incremental lift aligned with backtest `pred_lift` (same keyword set, zero budget baseline). Linear/piecewise runs leave `external_model_pred` empty.


- `ensemble_model.joblib` (after `fit_evaluation_ensemble.py` or backtest with `use_ensemble`)
- Backtest (daily): `backtest/<start>_<end>/plans/YYYYMMDD/`, `plan_vs_actual.csv`, `daily_backtest_summary.csv`
- Backtest (two-stage): `fixed_keyword_sets.json`, `stage1_keyword_sets/`, `daily_backtest_summary.csv`, `plans/YYYYMMDD/`
- Backtest analysis: `evaluation_results.csv`, `backtest_summary.csv`, `regional_breakdown.csv`, `backtest_summary.tex`

---

## Package layout


| Module                                                         | Purpose                                                  |
| -------------------------------------------------------------- | -------------------------------------------------------- |
| `[schema.py](schema.py)`                                       | Load `campaign_config.json`                              |
| `[features.py](features.py)`                                   | Build modeling frame from `campaign-day-panel`           |
| `[evaluation.py](evaluation.py)`                               | Ensemble fit + incremental `f(decision)−f(0)` comparison |
| `[cv.py](cv.py)`                                               | Expanding-window time-series cross-validation            |
| `[modeling.py](modeling.py)`                                   | Model tournament with level-scale metrics                |
| `[coefficients.py](coefficients.py)`                           | Export linear coeffs for MILP objective                  |
| `[linear_design.py](linear_design.py)`                         | Shared MILP-linear design + aligned ridge                |
| `[feature_artifacts.py](feature_artifacts.py)`                 | Persist modeling / design matrices at fit time           |
| `[optimize.py](optimize.py)`                                   | Dispatch solver backend from manifest                    |
| `[backtest_analysis.py](backtest_analysis.py)`                 | Compile plan vs actual + summary / LaTeX tables          |
| `[backtest.py](backtest.py)`                                   | Walk-forward daily backtest loop                         |
| `[backtest_two_stage.py](backtest_two_stage.py)`               | Stage‑1 fixed keyword sets + stage‑2 daily budget backtest |
| `[backends/milp_core.py](backends/milp_core.py)`               | Shared Gurobi MILP                                       |
| `[backends/linear.py](backends/linear.py)`                     | Linear objective backend                                 |
| `[backends/piecewise_linear.py](backends/piecewise_linear.py)` | Piecewise budget curve backend                           |
| `[backends/tree_embed.py](backends/tree_embed.py)`             | Exact RF/XGB tree embedding in Gurobi                    |
| `[backends/tree_embedding.py](backends/tree_embedding.py)`     | Leaf-level Big-M constraints (from ad_opt)               |


