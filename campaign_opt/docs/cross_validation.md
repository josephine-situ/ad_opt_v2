# Cross-validation design

This document describes how `campaign_opt` builds time-series CV folds and which config knobs control behavior. Implementation lives in [`cv.py`](../cv.py); tournament wiring is in [`modeling.py`](../modeling.py).

## Holdout vs CV

| Split | Data | Purpose |
|-------|------|---------|
| **Train** | All dates except last `holdout_days` | CV + hyperparameter tuning |
| **Holdout** | Last `holdout_days` (default **75**) | Fixed recent-window report in `holdout_metrics.json`; not used to slide CV folds |

With `validation.scheme: time_series_cv`, the **tournament winner** is the candidate with lowest **mean CV RMSE** on the train panel. Holdout RMSE/R² are logged for comparison and for `selection_metric` when scheme is not time-series CV.

## Expanding-window CV (default)

CV respects **time only**: folds are built on the train calendar with an expanding training window. Validation chunks are successive forward windows on the same timeline. The splitter does **not** split by campaign version, calendar gaps, or launch vs daily-budget phases—all segments active in each val window are scored together.

Algorithm (`time_series_cv_folds`):

1. Sort unique panel dates.
2. Require at least `min_train_fraction` (default **0.5**) of dates in train for every fold, and at least `min_val_days` (default **21**) in validation.
3. Divide the usable post-train span into up to `cv_folds` (default **3**) validation chunks.
4. Fold *i*: train on dates `<= cutoff_i`, validate on the next chunk (non-overlapping forward windows).

```
Panel dates:  |---- train grows ----|-- val1 --|-- val2 --|-- val3 --|
Holdout:                                              |--- holdout ---|
```

Row floors: `min_train_rows` / `min_val_rows` (defaults **50** / **20**). If no fold passes, `cross_validate_model` warns and falls back to a single internal 80/20 date split on train.

## Config reference

Example block (see `opt_results/<course>/campaign/default/campaign_config.json`):

```json
"validation": {
  "scheme": "time_series_cv",
  "holdout_days": 75,
  "cv_folds": 3,
  "min_train_fraction": 0.5,
  "min_train_days": 0,
  "min_val_days": 21,
  "min_train_rows": 50,
  "min_val_rows": 20,
  "tune_hyperparams": true,
  "refit_on_full_data": true
}
```

| Field | Default | Role |
|-------|---------|------|
| `scheme` | `time_series_cv` | Use CV RMSE for winner when `time_series_cv` |
| `holdout_days` | 75 | Recent holdout strip |
| `cv_folds` | 3 | Maximum expanding-window folds |
| `min_train_fraction` | 0.5 | Minimum fraction of train-panel dates in train |
| `min_train_days` | 0 | Optional absolute train-day floor |
| `min_val_days` | 21 | Minimum validation calendar days per fold |
| `min_train_rows` / `min_val_rows` | 50 / 20 | Row floors per fold |

**More folds / shorter val:** Increase `cv_folds` or decrease `min_val_days`. If the log shows fewer folds than requested, relax `min_train_fraction` (e.g. `0.25`) so the panel still has enough history.

## Errors and logging

- Failed fits on a fold raise **`CVFoldError`** (no silent skip).
- Tournament start prints fold count, e.g. `CV (expanding-window): 3 folds on 487 train days (...)`.
- First `cross_validate_model` call prints each fold’s train/val date ranges (`CV fold schedule: …`), once per train panel.
- Hyperparameter search (`hyperparam_cv.py`) uses the same `_validation_kw()` as the tournament.
- Default grids live in [`train_specs.py`](../train_specs.py) (ridge `alpha` 10–100; XGB/RF `max_depth` 2–3, `n_estimators` ≤ 20). Rationale and feature-selection context: [feature_selection_and_modeling.md](feature_selection_and_modeling.md).

## Interpreting metrics

- **CV RMSE (levels)** is the primary selection metric for `time_series_cv`.
- **CV R²** on a sparse daily panel is often negative even when the model is useful; treat it as a secondary diagnostic. Trust CV RMSE, holdout RMSE, and backtest lift.
- **Holdout** is one fixed recent window; **CV** averages several forward windows on the train slice.

## Code map

| Symbol | File | Role |
|--------|------|------|
| `time_series_cv_folds` | `cv.py` | Expanding-window fold generation |
| `effective_min_train_days` | `cv.py` | Train-day floor from fraction / absolute min |
| `_validation_kw` | `cv.py` | Config → fold kwargs |
| `cross_validate_model` | `cv.py` | Fit/score loop over folds |
| `run_tournament` | `modeling.py` | Logs fold preview; CV per candidate |

## Tests

`tests/test_cv_folds.py` covers train fractions, empty panels, and `CVFoldError` on fit failure.
