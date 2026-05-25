"""Model tournament with time-series CV and level-scale holdout metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from campaign_opt.cv import cross_validate_model
from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import get_context_feature_columns
from utils.shrinkage import segment_budget_level_counts


@dataclass
class ModelResult:
    name: str
    pipeline: Any
    holdout_rmse: float
    holdout_r2: float
    holdout_mae: float
    log_r2_diagnostic: float | None = None
    backend: str = "linear"
    cv_rmse: float | None = None
    cv_r2: float | None = None
    cv_mae: float | None = None
    extra: dict[str, Any] | None = None


def _level_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 0, None)
    return {
        "holdout_rmse_levels": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "holdout_r2_levels": float(r2_score(y_true, y_pred)),
        "holdout_mae_levels": float(mean_absolute_error(y_true, y_pred)),
    }


def _prep_xy(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    segment_col: str = "segment",
    *,
    y_col: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    y_name = y_col or target
    use_cols = [segment_col, "daily_budget", *feature_cols]
    sub = df.dropna(subset=[y_name, "daily_budget"]).copy()
    return sub[use_cols], sub[y_name].astype(float).values


def _build_preprocessor(feature_cols: list[str], segment_col: str = "segment") -> ColumnTransformer:
    cat_cols = [c for c in feature_cols if c != "daily_budget"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["daily_budget"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [segment_col] + cat_cols),
        ]
    )


def _fit_and_evaluate(
    name: str,
    backend: str,
    estimator,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    config: CampaignOptConfig,
    feature_cols: list[str],
    *,
    budget_col: str = "daily_budget",
    transform_train: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    transform_holdout: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    inverse_pred: Callable[[np.ndarray], np.ndarray] | None = None,
    log_r2_diagnostic: Callable[[pd.DataFrame, np.ndarray], float] | None = None,
    fit_y_col: str | None = None,
) -> ModelResult:
    """Shared fit path for sklearn pipelines (reduces duplication across model types)."""
    target = config.target
    tr = transform_train(train) if transform_train else train
    ho = transform_holdout(holdout) if transform_holdout else holdout

    X_train, y_train = _prep_xy(tr, target, feature_cols, y_col=fit_y_col)
    X_hold, y_eval = _prep_xy(ho, target, feature_cols)  # always evaluate on level-scale target
    if budget_col != "daily_budget":
        X_train = X_train.rename(columns={budget_col: "daily_budget"})
        X_hold = X_hold.rename(columns={budget_col: "daily_budget"})

    pipe = Pipeline([("prep", _build_preprocessor(feature_cols)), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_hold)
    if inverse_pred is not None:
        pred = inverse_pred(pred)
    m = _level_metrics(y_eval, pred)

    log_diag = log_r2_diagnostic(ho, pipe.predict(X_hold)) if log_r2_diagnostic else None
    return ModelResult(
        name=name,
        pipeline=pipe,
        backend=backend,
        log_r2_diagnostic=log_diag,
        holdout_rmse=m["holdout_rmse_levels"],
        holdout_r2=m["holdout_r2_levels"],
        holdout_mae=m["holdout_mae_levels"],
    )


def fit_ridge(train, holdout, config, feature_cols) -> ModelResult:
    return _fit_and_evaluate("ridge", "linear", Ridge(alpha=1.0), train, holdout, config, feature_cols)


def fit_random_forest(train, holdout, config, feature_cols) -> ModelResult:
    return _fit_and_evaluate(
        "random_forest",
        "tree_embed",
        RandomForestRegressor(
            n_estimators=100, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=-1
        ),
        train,
        holdout,
        config,
        feature_cols,
    )


def fit_xgboost(train, holdout, config, feature_cols) -> ModelResult:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("Install xgboost: pip install -e '.[ml]'") from exc
    return _fit_and_evaluate(
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
        train,
        holdout,
        config,
        feature_cols,
    )


def _power_transform(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.dropna(subset=[target, "daily_budget"]).copy()
    out["y_log"] = np.log1p(out[target].astype(float))
    out["log_budget"] = np.log(out["daily_budget"].astype(float).clip(lower=0.01))
    return out


def fit_power_log(train, holdout, config, feature_cols) -> ModelResult:
    def _inv(p):
        return np.expm1(p)

    def _log_diag(ho, pred_log):
        if "y_log" not in ho.columns:
            return None
        return float(r2_score(ho["y_log"], pred_log))

    return _fit_and_evaluate(
        "power_log",
        "piecewise_linear",
        Ridge(alpha=1.0),
        train,
        holdout,
        config,
        feature_cols,
        budget_col="log_budget",
        transform_train=lambda d: _power_transform(d, config.target),
        transform_holdout=lambda d: _power_transform(d, config.target),
        inverse_pred=_inv,
        log_r2_diagnostic=lambda ho, p: _log_diag(ho, p),
        fit_y_col="y_log",
    )


def fit_power_level(train, holdout, config, feature_cols) -> ModelResult:
    def _to_log_budget(df):
        out = df.dropna(subset=[config.target, "daily_budget"]).copy()
        out["log_budget"] = np.log(out["daily_budget"].astype(float).clip(lower=0.01))
        return out.rename(columns={"log_budget": "daily_budget"})

    return _fit_and_evaluate(
        "power_level",
        "piecewise_linear",
        Ridge(alpha=1.0),
        train,
        holdout,
        config,
        feature_cols,
        transform_train=_to_log_budget,
        transform_holdout=_to_log_budget,
    )


FITTERS: dict[str, Callable[..., ModelResult]] = {
    "ridge": fit_ridge,
    "power_log": fit_power_log,
    "power_level": fit_power_level,
    "random_forest": fit_random_forest,
    "xgboost": fit_xgboost,
}


def resolve_backend(winner: ModelResult, config: CampaignOptConfig, ridge_rmse: float) -> str:
    policy = config.model_policy
    if policy.optimizer_backend != "auto":
        return policy.optimizer_backend
    cand_rmse = winner.cv_rmse if winner.cv_rmse is not None else winner.holdout_rmse
    gain = (ridge_rmse - cand_rmse) / max(ridge_rmse, 1e-9)
    if winner.name == "ridge":
        return "linear"
    if winner.backend == "tree_embed" and gain < policy.min_holdout_gain_vs_ridge:
        return "linear"
    if winner.backend == "piecewise_linear" and gain < policy.min_holdout_gain_vs_ridge:
        return "linear"
    return winner.backend


def _selection_score(res: ModelResult, config: CampaignOptConfig) -> float:
    """Lower is better for RMSE; higher is better for R2."""
    metric = config.model_policy.selection_metric
    use_cv = config.model_policy.validation.scheme == "time_series_cv"
    if "r2" in metric:
        val = (res.cv_r2 if use_cv and res.cv_r2 is not None else res.holdout_r2) or 0.0
        return -val
    val = res.cv_rmse if use_cv and res.cv_rmse is not None else res.holdout_rmse
    return val


def run_tournament(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    config: CampaignOptConfig,
    *,
    run_cv: bool | None = None,
) -> tuple[ModelResult, dict[str, dict[str, float]], dict[str, Any]]:
    feature_cols = get_context_feature_columns(config.context_features)
    results: list[ModelResult] = []
    metrics_table: dict[str, dict[str, float]] = {}

    if run_cv is None:
        run_cv = config.model_policy.validation.scheme in ("time_series_cv", "cv")
    n_folds = config.model_policy.validation.cv_folds

    for name in config.model_policy.candidates:
        fitter = FITTERS.get(name)
        if fitter is None:
            continue
        try:
            # Final model fit on full train; metrics on holdout
            res = fitter(train, holdout, config, feature_cols)
            if run_cv:
                cv_metrics = cross_validate_model(fitter, train, config, feature_cols, n_folds=n_folds)
                res.cv_rmse = cv_metrics["cv_rmse_levels"]
                res.cv_r2 = cv_metrics["cv_r2_levels"]
                res.cv_mae = cv_metrics["cv_mae_levels"]
            results.append(res)
            metrics_table[name] = {
                "holdout_rmse_levels": res.holdout_rmse,
                "holdout_r2_levels": res.holdout_r2,
                "holdout_mae_levels": res.holdout_mae,
            }
            if res.cv_rmse is not None:
                metrics_table[name]["cv_rmse_levels"] = res.cv_rmse
                metrics_table[name]["cv_r2_levels"] = res.cv_r2
            if res.log_r2_diagnostic is not None:
                metrics_table[name]["log_r2_diagnostic"] = res.log_r2_diagnostic
            cv_str = f" CV_RMSE={res.cv_rmse:.4f}" if res.cv_rmse is not None else ""
            print(f"  {name}: holdout RMSE={res.holdout_rmse:.4f}{cv_str}")
        except Exception as exc:
            print(f"  {name}: skipped ({exc})")

    if not results:
        raise RuntimeError("No models succeeded in tournament")

    ridge_res = next((r for r in results if r.name == "ridge"), results[0])
    winner = min(results, key=lambda r: _selection_score(r, config))

    backend = resolve_backend(winner, config, ridge_res.holdout_rmse)
    if backend == "linear" and winner.name != "ridge":
        winner = ridge_res

    manifest: dict[str, Any] = {
        "winner": winner.name,
        "backend": backend,
        "target": config.target,
        "secondary_metrics": config.secondary_metrics,
        "holdout_metrics": metrics_table,
        "segment_budget_levels": segment_budget_level_counts(train).to_dict(),
        "feature_cols": feature_cols,
        "cv_folds": n_folds if run_cv else 0,
        "fallback_note": (
            "solver uses ridge linear coefficients"
            if winner.name != "ridge" and backend == "linear"
            else None
        ),
    }
    return winner, metrics_table, manifest


def save_manifest(manifest: dict[str, Any], model: ModelResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    joblib.dump(model.pipeline, path.parent / "winner_model.joblib")
