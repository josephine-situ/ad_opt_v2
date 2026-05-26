"""Time-series CV hyperparameter search for tournament candidates."""

from __future__ import annotations

from itertools import product
from typing import Any, Callable

import pandas as pd

from campaign_opt.cv import cross_validate_model
from campaign_opt.train_specs import DEFAULT_HYPERPARAM_GRIDS


def iter_param_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    return [dict(zip(keys, combo)) for combo in product(*param_grid.values())]


def tune_hyperparams(
    model_name: str,
    fitter: Callable[..., Any],
    train: pd.DataFrame,
    config,
    feature_cols: list[str],
    *,
    param_grid: dict[str, list[Any]] | None = None,
    n_folds: int = 5,
) -> tuple[dict[str, Any], dict[str, float]]:
    """
    Grid-search hyperparameters with expanding-window CV on ``train``.

    Returns best params and mean level-scale CV metrics at those params.
    """
    grid = param_grid if param_grid is not None else DEFAULT_HYPERPARAM_GRIDS.get(model_name, {})
    combos = iter_param_grid(grid)

    best_params: dict[str, Any] = {}
    best_cv = {"cv_rmse_levels": float("inf"), "cv_r2_levels": 0.0, "cv_mae_levels": float("inf")}

    for params in combos:
        def fit_with_params(
            tr: pd.DataFrame,
            va: pd.DataFrame,
            cfg,
            fc: list[str],
            *,
            _params: dict[str, Any] = params,
        ):
            return fitter(tr, va, cfg, fc, hyperparams=_params)

        cv_metrics = cross_validate_model(fit_with_params, train, config, feature_cols, n_folds=n_folds)
        if cv_metrics["cv_rmse_levels"] < best_cv["cv_rmse_levels"]:
            best_params = params
            best_cv = cv_metrics

    return best_params, best_cv
