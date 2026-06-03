"""Per-candidate training specs for fit-on-train and ensemble prediction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
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
    "power_log": dict(_RIDGE_ALPHA_GRID),
    "power_level": dict(_RIDGE_ALPHA_GRID),
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
    transform: Callable[[pd.DataFrame, str], pd.DataFrame] | None = None
    inverse_pred: Callable[[np.ndarray], np.ndarray] | None = None
    fit_y_col: str | None = None


def power_transform(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.dropna(subset=["daily_budget"]).copy()
    if target in out.columns:
        out["y_log"] = np.log1p(out[target].astype(float))
    else:
        out["y_log"] = 0.0
    out["log_budget"] = np.log(out["daily_budget"].astype(float).clip(lower=0.01))
    return out


def power_level_transform(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.dropna(subset=["daily_budget"]).copy()
    out["daily_budget"] = np.log(out["daily_budget"].astype(float).clip(lower=0.01))
    return out


def build_estimator(name: str, hyperparams: dict[str, Any] | None = None) -> Any:
    """Build a sklearn estimator for ``name`` using optional tuned hyperparameters."""
    hp = hyperparams or {}
    if name in ("ridge", "power_log", "power_level"):
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
        "power_log": TrainSpec(
            "power_log",
            "piecewise_linear",
            build_estimator("power_log"),
            budget_col="log_budget",
            transform=power_transform,
            inverse_pred=np.expm1,
            fit_y_col="y_log",
        ),
        "power_level": TrainSpec(
            "power_level",
            "piecewise_linear",
            build_estimator("power_level"),
            transform=power_level_transform,
        ),
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
