"""Per-candidate training specs for fit-on-train and ensemble prediction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# Tree models: shallow trees, n_estimators capped at 20 (no max_depth=4).
_AD_OPT_XGB_GRID = {
    "n_estimators": [5, 10, 20],
    "max_depth": [2, 3],
    "learning_rate": [0.1, 0.3],
}

_RIDGE_ALPHA_GRID = {"alpha": [10.0, 100.0]}

DEFAULT_HYPERPARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "ridge": dict(_RIDGE_ALPHA_GRID),
    "random_forest": {
        "n_estimators": _AD_OPT_XGB_GRID["n_estimators"],
        "max_depth": _AD_OPT_XGB_GRID["max_depth"],
        "min_samples_leaf": [10, 20],
    },
    "xgboost": dict(_AD_OPT_XGB_GRID),
}


@dataclass(frozen=True)
class TrainSpec:
    name: str
    backend: str
    estimator: Any
    budget_col: str = "daily_budget"
    transform: Any | None = None
    inverse_pred: Any | None = None
    fit_y_col: str | None = None


def build_estimator(name: str, hyperparams: dict[str, Any] | None = None) -> Any:
    """Build a sklearn estimator for ``name`` using optional tuned hyperparameters."""
    hp = hyperparams or {}
    if name == "ridge":
        return Ridge(alpha=float(hp.get("alpha", 1.0)))
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(hp.get("n_estimators", 10)),
            max_depth=int(hp.get("max_depth", 3)),
            min_samples_leaf=int(hp.get("min_samples_leaf", 20)),
            random_state=42,
            n_jobs=-1,
        )
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(hp.get("n_estimators", 10)),
            max_depth=int(hp.get("max_depth", 3)),
            learning_rate=float(hp.get("learning_rate", 0.1)),
            subsample=float(hp.get("subsample", 1.0)),
            colsample_bytree=float(hp.get("colsample_bytree", 1.0)),
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model name: {name}")


def get_train_specs() -> dict[str, TrainSpec]:
    specs: dict[str, TrainSpec] = {
        "ridge": TrainSpec("ridge", "linear", build_estimator("ridge")),
        "random_forest": TrainSpec(
            "random_forest",
            "tree_embed",
            build_estimator("random_forest"),
        ),
    }
    try:
        build_estimator("xgboost")
        specs["xgboost"] = TrainSpec("xgboost", "tree_embed", build_estimator("xgboost"))
    except ImportError:
        pass
    return specs


def get_train_spec(name: str, hyperparams: dict[str, Any] | None = None) -> TrainSpec | None:
    base = get_train_specs().get(name)
    if base is None:
        return None
    if not hyperparams:
        return base
    return replace(base, estimator=build_estimator(name, hyperparams))
