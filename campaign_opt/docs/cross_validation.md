# Cross-validation design

This document describes how `campaign_opt` builds time-series CV folds, how that maps to production, and which config knobs control behavior. Implementation lives in [`cv.py`](../cv.py); tournament wiring is in [`modeling.py`](../modeling.py).

## Production workflow (two phases)

Campaign optimization is split into two decisions that happen on different horizons:

| Phase | Decision | Typical horizon | CV profile |
|-------|----------|-----------------|------------|
| **Phase 1** | Which keyword set to launch for a run | First ~2 weeks after a new period / copy change | `phase1_launch` (reporting only by default) |
| **Phase 2** | Daily budget within a fixed set for the rest of the run | Next few days, repeatedly | `phase2_daily` (default for selection) |

**Phase 2** matches day-to-day planning: keyword sets are fixed for the run; the model only forecasts short forward windows *inside* the current active calendar period.

**Phase 1** matches run-start forecasting: train on all history *before* a period begins; validate on the opening days of that period (new launch / post-gap behavior).

Phase-1 metrics are written as `phase1_cv_*` in `holdout_metrics.json` when `report_phase1_cv: true`. They do **not** pick the tournament winner unless `phase1_cv_for_selection: true`.

## Run periods (`campaign_version` + gaps)

When the modeling frame has `segment` and `campaign_version` (from `campaign-summary.csv`), `prepare_modeling_data` attaches **`run_period_id`**:

- **Primary boundary:** each `(segment, campaign_version)` — fixed daily budget and keyword set.
- **Secondary boundary:** within that version, a gap longer than `max_calendar_gap_days` (default **7**) without panel rows starts a new run period (off-air).

Phase-1 and phase-2 CV use `run_period_id` for validation windows. Training for phase 2 still uses all rows with `date < val_start` (pooled history); validation rows are restricted to a single `run_period_id`.

**Optimization scope:** CV validation rows (and version-based run-period candidates) respect `constraints.allowed_match_types` and `constraints.excluded_regions` from `campaign_config.json` — same filters as keyword-candidate build and MILP. Training rows remain the full pooled history unless you filter the modeling frame separately.

Legacy **`calendar_period_id`** (gap-only on the full calendar) remains on the frame for `period_tail` / debugging but is not used by default phase-1/phase-2 profiles.

If `campaign_version` is missing (synthetic tests), CV falls back to calendar-gap periods.

```
Segment A, v3  |---- in-run val (phase2) ----|
               ^ run_period_id (gap may split v3 into two ids)

Phase 1: train date < run_start; val = first N days of that run_period only
```

## Holdout vs CV

| Split | Data | Purpose |
|-------|------|---------|
| **Train** | All dates except last `holdout_days` | CV + hyperparameter tuning |
| **Holdout** | Last `holdout_days` (default **75**) | Fixed recent-window report in `holdout_metrics.json`; not used to slide CV folds |

With `validation.scheme: time_series_cv`, the **tournament winner** is the candidate with lowest **mean CV RMSE** on the train panel. Holdout RMSE/R² are logged for comparison and for `selection_metric` when scheme is not time-series CV.

## CV profiles

Set `validation.cv_profile` in `campaign_config.json`. Dispatch is `time_series_cv_folds()` in `cv.py`.

### `phase2_daily` (default)

**Question CV answers:** “Given everything before day *t*, how well do we predict the next few days *in the same run*?”

For each `run_period_id`:

1. Enumerate candidate validation starts on that run’s contiguous active dates.
2. **Train:** all rows with `date < val_start`.
3. **Val:** rows with the same `run_period_id` and `date` in `[val_start, val_start + phase2_val_days - 1]`.

**Noise vs cost:** One-day validation is very noisy on this panel. Defaults use a **7-day** validation window and **~15 folds** averaged for selection.

Many candidates exist on long periods; we do not score every day:

1. Sort candidates by validation end date.
2. Subsample with `phase2_fold_stride` (default **30**: keep every 30th candidate val start from the recent tail).
3. Cap at `phase2_cv_folds` (default **15**), preferring the most recent windows.

```
Candidates:  |--7d--|--7d--|--7d--|--7d--|--7d--| ... (one per slide day)
Stride 30:   x                                              (fewer folds)
Reported:    mean metric over up to 15 retained folds
```

Fold sizing also requires:

- `min_train_fraction` / `min_train_days` on **training** calendar days
- `min_train_rows` / `min_val_rows` on row counts (val row minimum is scaled down for short `phase2_val_days`)

### `phase1_launch`

**Question CV answers:** “Before this period started, could we forecast its opening fortnight?”

For each `run_period_id` after that segment’s first run:

- **Train:** all rows with `date < run_start` (pooled across segments)
- **Val:** first `phase1_launch_val_days` (default **14**) of that run only (`run_period_id`)

`min_train_fraction` applies to **pre-run days for that segment**, not the full panel.

Used for diagnostics (`phase1_cv_rmse_levels`, etc.), not default tournament selection.

### Legacy profiles

| Profile | Behavior |
|---------|----------|
| `period_tail` | Last `min_val_days` inside each period |
| `legacy_calendar` | Expanding windows on full calendar; may span gaps unless `respect_campaign_periods` |

Prefer `phase2_daily` / `phase1_launch` for new configs.

## Config reference

Example block (see `opt_results/<course>/campaign/default/campaign_config.json`):

```json
"validation": {
  "scheme": "time_series_cv",
  "holdout_days": 75,
  "cv_folds": 3,
  "cv_profile": "phase2_daily",
  "phase2_val_days": 7,
  "phase2_cv_folds": 15,
  "phase2_fold_stride": 30,
  "phase1_launch_val_days": 14,
  "report_phase1_cv": true,
  "phase1_cv_for_selection": false,
  "min_train_fraction": 0.5,
  "min_train_days": 0,
  "min_val_days": 21,
  "min_train_rows": 50,
  "min_val_rows": 20,
  "respect_campaign_periods": true,
  "max_calendar_gap_days": 7
}
```

| Field | Default | Role |
|-------|---------|------|
| `scheme` | `time_series_cv` | Use CV for winner when `time_series_cv` |
| `holdout_days` | 75 | Recent holdout strip |
| `cv_folds` | 3 | Legacy cap; phase-2 uses `phase2_cv_folds` when set |
| `cv_profile` | `phase2_daily` | Fold constructor |
| `phase2_val_days` | 7 | Length of each phase-2 validation window |
| `phase2_cv_folds` | 15 | Max phase-2 folds retained after stride |
| `phase2_fold_stride` | 30 | Subsample every Nth candidate val start |
| `phase1_launch_val_days` | 14 | Phase-1 validation length |
| `report_phase1_cv` | true | Attach `phase1_cv_*` to tournament output |
| `phase1_cv_for_selection` | false | If true, phase-1 could drive selection (unusual) |
| `min_train_fraction` | 0.5 | Minimum fraction of panel dates in train |
| `min_train_rows` / `min_val_rows` | 50 / 20 | Row floors per fold |
| `max_calendar_gap_days` | 7 | Gap threshold for new period |

**Tuning more folds:** Increase `phase2_cv_folds` or decrease `phase2_fold_stride`. If the tournament log shows fewer folds than requested, relax `min_train_fraction` (e.g. `0.25`) so recent windows still have enough train history.

## Errors and logging

- Failed fits on a fold raise **`CVFoldError`** (no silent `continue`).
- Tournament start prints fold count, e.g. `CV (phase2_daily): 15 folds on 487 train days (requested=15, phase2_val_days=7, ...)`.
- First `cross_validate_model` call prints each fold’s train/val date ranges (`CV fold schedule: …`), including `run_period_id` / `campaign_version` / `segment` when present. Phase-1 uses the same logger with `profile='phase1_launch'` (once per train panel).
- Hyperparameter search (`hyperparam_cv.py`) uses the same `_validation_kw()` / profile as the tournament.

## Interpreting metrics

- **CV RMSE (levels)** is the primary selection metric for `time_series_cv`.
- **CV R²** on a sparse daily panel is often negative even when the model is useful; treat it as a secondary diagnostic. Trust CV RMSE, holdout RMSE, and backtest lift.
- **Holdout** is one fixed recent window; **CV** averages many short forward windows on the train slice—better aligned with repeated phase-2 planning.

## Code map

| Symbol | File | Role |
|--------|------|------|
| `time_series_cv_folds_phase2_daily` | `cv.py` | Phase-2 candidate generation + stride cap |
| `time_series_cv_folds_phase1_launch` | `cv.py` | Phase-1 period-start folds |
| `add_run_period_id` | `cv.py` | `(segment, campaign_version)` + gap → `run_period_id` |
| `calendar_period_ranges` | `cv.py` | Gap-only spans (legacy fallback) |
| `_validation_kw` | `cv.py` | Config → fold kwargs (incl. scaled `min_val_rows`) |
| `cross_validate_model` | `cv.py` | Fit/score loop over folds |
| `run_tournament` | `modeling.py` | Logs fold preview; CV per candidate |
| `cross_validate_phase1_launch` | `cv.py` | Optional phase-1 report metrics |

## Tests

`tests/test_cv_folds.py` covers train fractions, gap splitting, phase-2 horizon, phase-1 train-before-period, and `CVFoldError` on fit failure.
