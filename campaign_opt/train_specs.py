"""Per-candidate training specs for fit-on-train and ensemble prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


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
    out["log_budget"] = np.log(out["daily_budget"].astype(float).clip(lower=0.01))
    return out.rename(columns={"log_budget": "daily_budget"})


def get_train_specs() -> dict[str, TrainSpec]:
    specs: dict[str, TrainSpec] = {
        "ridge": TrainSpec("ridge", "linear", Ridge(alpha=1.0)),
        "power_log": TrainSpec(
            "power_log",
            "piecewise_linear",
            Ridge(alpha=1.0),
            budget_col="log_budget",
            transform=power_transform,
            inverse_pred=np.expm1,
            fit_y_col="y_log",
        ),
        "power_level": TrainSpec(
            "power_level",
            "piecewise_linear",
            Ridge(alpha=1.0),
            transform=power_level_transform,
        ),
        "random_forest": TrainSpec(
            "random_forest",
            "tree_embed",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }
    try:
        from xgboost import XGBRegressor

        specs["xgboost"] = TrainSpec(
            "xgboost",
            "tree_embed",
            XGBRegressor(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.08,
                min_child_weight=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            ),
        )
    except ImportError:
        pass
    return specs
