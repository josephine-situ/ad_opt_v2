# Historical keyword efficiency experiments (not shipped)

**Status:** Explored June 2026; **not added** to production `campaign_config.json`. Production remains the **shipped deduped 10** context features (see [feature_selection_and_modeling.md](feature_selection_and_modeling.md)).

Implementation remains in the repo for re-runs and MILP experiments: `utils/keyword_efficiency_features.py`, `utils/lagged_segment_spend_features.py`, `campaign_opt/efficiency_ablation.py`, `campaign_opt/spend_regime_diagnostics.py`.

---

## What we tried

Causal **keyword-set** context from `kw-day-panel.csv`:

1. Per keyword-day: `conv_scaled_clicks / cost` or `/ segment daily_budget` (fixed segment conv/click rates).
2. Per keyword, strictly before panel `date`: **last** observed day or **rolling mean** over calendar 7 / 14 / 30d.
3. Per keyword, optional **temporal volatility**: std of daily efficiency in the window (`vol`; undefined for `last`).
4. Aggregate across keywords in the set: **mean** or **std** across keywords (std = cross-keyword dispersion of time-means).
5. Pools: **union** of all lists, or **broad / phrase / exact** lists separately.

Column pattern: `hist_kw_eff_{window}_{pool}_{stat}_{denom}`  
Example: `hist_kw_eff_r7d_broad_mean_cost`.

We also ran **spend-regime diagnostics**: lagged segment `cost` / `daily_budget` / segment-level `conv_scaled/cost` (`hist_seg_*`) to test whether efficiency gains were just historic spend level. See `scripts/diagnose_efficiency_vs_lagged_spend.py`.

---

## Decision

| Question | Answer |
|----------|--------|
| Add to production config? | **No** — keep original deduped 10-feat baseline. |
| Best candidate if revisited? | **`add_r7d_per_mt_mean_cost`** (3 cols), cost-denom, 7d time-mean, per match type. |
| Add lagged segment `cost`? | **No** — hurts or barely helps XGB CV; not the same signal as efficiency. |
| Add budget-denom efficiency? | **No** — weaker than cost-denom on CV. |
| Add volatility (`vol`)? | **No** — no CV gain vs mean-only; often worse. |
| Add union-only single col? | Possible but weaker than per-MT on tuned XGB. |

**Why not ship despite CV lift:** Tuned XGB CV improved strongly with per-MT r7d efficiency, but **holdout R² fell** (0.248 baseline → 0.096 with features). Same CV–holdout gap concern as other aggressive context expansions. Ridge holdout remains poor either way.

---

## Key results (sys_think / default, `conv_scaled_clicks`)

### Untuned ablation (`diagnostics/efficiency_ablation/subset/`)

| Spec | XGB CV RMSE | Δ vs baseline (0.390) | XGB holdout R² |
|------|------------:|----------------------:|---------------:|
| baseline | 0.390 | — | 0.223 |
| `add_eff_cost` (32 cols) | 0.353 | −0.037 | 0.211 |
| `add_r7d_per_mt_mean_cost` | 0.349 | −0.041 | 0.170 |
| `single_r7d_union_mean_cost` | 0.383 | −0.007 | 0.161 |
| `single_r7d_union_vol_cost` | 0.397 | +0.007 | 0.122 |

Volatility and union-only bundles did not beat the per-MT mean trio on CV.

### Tuned: baseline vs `add_r7d_per_mt_mean_cost`

Command:

```powershell
uv run python scripts/run_efficiency_ablation.py --course sys_think --only-spec add_r7d_per_mt_mean_cost --tune
```

Output: `diagnostics/efficiency_ablation_tuned/add_r7d_per_mt_mean_cost/`

| Spec | Model | CV RMSE | CV R² | Holdout R² | Tuned params |
|------|-------|--------:|------:|-----------:|--------------|
| baseline | xgboost | 0.362 | 0.423 | **0.248** | `n_estimators=20`, `max_depth=3`, `lr=0.1` |
| + 3 eff cols | xgboost | **0.309** | **0.607** | 0.096 | `n_estimators=10`, `max_depth=3`, `lr=0.3` |
| baseline | ridge | 0.430 | 0.173 | −0.946 | `alpha=100` |
| + 3 eff cols | ridge | 0.423 | 0.210 | −0.555 | `alpha=100` |

**Columns in the tuned candidate:**

- `hist_kw_eff_r7d_broad_mean_cost`
- `hist_kw_eff_r7d_phrase_mean_cost`
- `hist_kw_eff_r7d_exact_mean_cost`

### Spend-regime diagnostic (`diagnostics/spend_regime/`)

- Keyword efficiency (**cost-denom**) is **not** the same as lagged segment **cost** (correlations modestly negative, ~−0.3).
- Keyword efficiency correlates ~0.7 with **lagged segment** `conv_scaled/cost` (`hist_seg_eff_*`) — overlapping information at segment level.
- Lagged raw `hist_seg_cost_*` did **not** reproduce XGB CV gains; often hurt.

---

## Clarifications (common confusion)

| Name suffix | Meaning |
|-------------|---------|
| `*_cost` on `hist_kw_eff_*` | Efficiency uses **keyword cost** in the denominator — **not** raw cost as a feature. |
| `*_budget` | Efficiency uses **segment daily_budget** in the denominator (weak in ablations). |
| `*_vol_*` | Mean across keywords of **within-keyword** std of daily efficiency over the window. |
| `*_std_*` (not `vol`) | Std **across keywords** of per-keyword time-mean efficiencies. |

Ablation **without** `--tune` uses ridge `alpha=1`, XGB defaults (`n_estimators=10`, `max_depth=3`, `lr=0.1`). **With** `--tune`, each spec gets its own CV grid search (same grids as tournament).

---

## How to re-run

```powershell
# Subset grid (~39 specs), untuned
uv run python scripts/run_efficiency_ablation.py --course sys_think --spec-set subset

# Single spec + tuned (vs baseline)
uv run python scripts/run_efficiency_ablation.py --course sys_think --only-spec add_r7d_per_mt_mean_cost --tune

# Efficiency vs lagged spend
uv run python scripts/diagnose_efficiency_vs_lagged_spend.py --course sys_think
```

---

## If revisiting later

1. Re-check **holdout** and **backtest** after any regime change — not CV alone.
2. Prefer **3-column per-MT r7d mean cost** over full 32/64-column bundles.
3. Do **not** add same-day or lagged segment **cost** as a context feature for budget optimization.
4. Consider nested validation or holdout-aligned tuning if CV and holdout continue to diverge.
