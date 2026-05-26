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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from campaign_opt.cv import cross_validate_model, effective_min_train_days, time_series_cv_folds
from campaign_opt.coefficients import export_linear_solver_coeffs, extract_context_feature_coefs
from campaign_opt.hyperparam_cv import tune_hyperparams
from campaign_opt.linear_design import (
    SEGMENT_MATCH_COLS,
    LinearMilpRidgeModel,
    build_linear_milp_design_matrix,
    split_context_columns_by_dtype,
)
from campaign_opt.schema import CampaignOptConfig
from campaign_opt.shap_effects import compute_mean_shap_effects, format_top_shap_effects
from campaign_opt.train_specs import build_estimator, get_train_spec
from utils.campaign_features import (
    TREE_SEGMENT_FEATURE_COLS,
    add_segment_match_type_indicators,
    get_context_feature_columns,
)
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
    best_hyperparams: dict[str, Any] | None = None
    extra: dict[str, Any] | None = None


def _level_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 0, None)
    return {
        "holdout_rmse_levels": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "holdout_r2_levels": float(r2_score(y_true, y_pred)),
        "holdout_mae_levels": float(mean_absolute_error(y_true, y_pred)),
    }


def eval_pipeline_holdout(
    pipeline: Any,
    holdout: pd.DataFrame,
    config: CampaignOptConfig,
    feature_cols: list[str],
) -> dict[str, float] | None:
    """Level-scale RMSE / R² / MAE on holdout rows (walk-forward backtest diagnostic)."""
    if holdout.empty:
        return None
    target = config.target
    sub = holdout.dropna(subset=[target, "daily_budget", "region"])
    if sub.empty:
        return None
    X, y = _prep_xy(sub, target, feature_cols)
    if len(X) == 0:
        return None
    pred = np.clip(pipeline.predict(X), 0, None)
    m = _level_metrics(y, pred)
    return {
        "holdout_r2": m["holdout_r2_levels"],
        "holdout_rmse": m["holdout_rmse_levels"],
        "holdout_mae": m["holdout_mae_levels"],
        "n_holdout": float(len(X)),
    }


def _ensure_tree_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    if all(col in df.columns for col in TREE_SEGMENT_FEATURE_COLS):
        return df
    return add_segment_match_type_indicators(df)


def _prep_xy(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    *,
    y_col: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    y_name = y_col or target
    sub = _ensure_tree_segment_features(
        df.dropna(subset=[y_name, "daily_budget", "region"]).copy()
    )
    numeric_ctx, _ = split_context_columns_by_dtype(sub, feature_cols)
    for col in numeric_ctx:
        sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)
    use_cols = [*TREE_SEGMENT_FEATURE_COLS, "daily_budget", *feature_cols]
    return sub[use_cols], sub[y_name].astype(float).values


def _build_preprocessor(feature_cols: list[str], sample: pd.DataFrame) -> ColumnTransformer:
    ctx_numeric, ctx_cat = split_context_columns_by_dtype(sample, feature_cols)
    transformers: list[tuple[str, Any, list[str]]] = [
        ("num", StandardScaler(), ["daily_budget"]),
        (
            "region",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["region"],
        ),
        ("match", "passthrough", ["has_broad", "has_phrase", "has_exact"]),
    ]
    if ctx_numeric:
        transformers.append(("ctx_num", "passthrough", ctx_numeric))
    if ctx_cat:
        transformers.append(
            (
                "ctx_cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ctx_cat,
            )
        )
    return ColumnTransformer(transformers)


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
    hyperparams: dict[str, Any] | None = None,
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

    pipe = Pipeline([("prep", _build_preprocessor(feature_cols, tr)), ("model", estimator)])
    pipe.fit(X_train, y_train)
    if len(X_hold) == 0:
        # Train-only refit (e.g. optimizer_winner); metrics are undefined.
        return ModelResult(
            name=name,
            pipeline=pipe,
            backend=backend,
            log_r2_diagnostic=None,
            holdout_rmse=float("nan"),
            holdout_r2=float("nan"),
            holdout_mae=float("nan"),
            best_hyperparams=hyperparams,
        )

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
        best_hyperparams=hyperparams,
    )


def fit_ridge(train, holdout, config, feature_cols, *, hyperparams: dict[str, Any] | None = None) -> ModelResult:
    del feature_cols  # ridge uses MILP-linear design (region + match + budget interactions + context)
    alpha = float((hyperparams or {}).get("alpha", 1.0))
    train_design = build_linear_milp_design_matrix(train, config)
    model = Ridge(alpha=alpha)
    model.fit(train_design.X.values, train_design.y)

    holdout_design = build_linear_milp_design_matrix(
        holdout, config, columns=train_design.x_columns
    )
    if len(holdout_design.X) == 0:
        m = {
            "holdout_rmse_levels": float("nan"),
            "holdout_r2_levels": float("nan"),
            "holdout_mae_levels": float("nan"),
        }
    else:
        pred = np.clip(model.predict(holdout_design.X.values), 0, None)
        m = _level_metrics(holdout_design.y, pred)

    artifact = LinearMilpRidgeModel(model, train_design.x_columns, config)
    return ModelResult(
        name="ridge",
        pipeline=artifact,
        backend="linear",
        holdout_rmse=m["holdout_rmse_levels"],
        holdout_r2=m["holdout_r2_levels"],
        holdout_mae=m["holdout_mae_levels"],
        best_hyperparams=hyperparams,
        extra={
            "milp_model": model,
            "milp_design": train_design,
        },
    )


def fit_ridge_full(
    train: pd.DataFrame,
    config: CampaignOptConfig,
    *,
    hyperparams: dict[str, Any] | None = None,
) -> LinearMilpRidgeModel:
    """Fit aligned ridge on all training rows (ensemble / production)."""
    alpha = float((hyperparams or {}).get("alpha", 1.0))
    design = build_linear_milp_design_matrix(train, config)
    model = Ridge(alpha=alpha)
    model.fit(design.X.values, design.y)
    return LinearMilpRidgeModel(model, design.x_columns, config)


def fit_random_forest(
    train, holdout, config, feature_cols, *, hyperparams: dict[str, Any] | None = None
) -> ModelResult:
    return _fit_and_evaluate(
        "random_forest",
        "tree_embed",
        build_estimator("random_forest", hyperparams),
        train,
        holdout,
        config,
        feature_cols,
        hyperparams=hyperparams,
    )


def _power_transform(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.dropna(subset=[target, "daily_budget"]).copy()
    out["y_log"] = np.log1p(out[target].astype(float))
    out["log_budget"] = np.log(out["daily_budget"].astype(float).clip(lower=0.01))
    return out


def hyperparams_from_manifest(manifest: dict[str, Any], model_name: str) -> dict[str, Any] | None:
    """Best hyperparameters for a tournament candidate from a saved manifest."""
    top = manifest.get("best_hyperparams") or {}
    if model_name in top:
        return top[model_name]
    holdout = manifest.get("holdout_metrics") or {}
    entry = holdout.get(model_name) or {}
    return entry.get("best_hyperparams")


def refit_optimizer_model(
    model_name: str,
    train: pd.DataFrame,
    config: CampaignOptConfig,
    manifest: dict[str, Any],
) -> Any:
    """Refit a named tournament candidate on ``train`` for production optimization."""
    fitter = FITTERS.get(model_name)
    if fitter is None:
        raise ValueError(f"Unknown optimizer_winner: {model_name!r}")
    feature_cols = manifest.get("feature_cols") or get_context_feature_columns(
        config.context_features
    )
    hyperparams = hyperparams_from_manifest(manifest, model_name)
    empty_holdout = train.iloc[0:0]
    res = fitter(train, empty_holdout, config, feature_cols, hyperparams=hyperparams)
    return res.pipeline


def refit_winner_on_data(
    winner: ModelResult,
    df: pd.DataFrame,
    config: CampaignOptConfig,
    feature_cols: list[str],
) -> ModelResult:
    """
    Refit the tournament winner on all available rows for production / optimization.

    Evaluation metrics on ``winner`` (CV / holdout from the train split) are preserved.
    """
    spec = get_train_spec(winner.name, winner.best_hyperparams)
    if spec is None:
        raise ValueError(f"Unknown winner model: {winner.name}")

    if winner.name == "ridge":
        alpha = float((winner.best_hyperparams or {}).get("alpha", 1.0))
        design = build_linear_milp_design_matrix(df, config)
        model = Ridge(alpha=alpha)
        model.fit(design.X.values, design.y)
        pipeline = LinearMilpRidgeModel(model, design.x_columns, config)
        extra = {"milp_model": model, "milp_design": design}
    else:
        target = config.target
        tr = spec.transform(df, target) if spec.transform else df
        sub = tr.dropna(subset=[target, "daily_budget", "segment"])
        X, y = _prep_xy(sub, target, feature_cols, y_col=spec.fit_y_col)
        if spec.budget_col != "daily_budget":
            X = X.rename(columns={spec.budget_col: "daily_budget"})
        pipeline = Pipeline(
            [("prep", _build_preprocessor(feature_cols, sub)), ("model", spec.estimator)]
        )
        pipeline.fit(X, y)
        extra = winner.extra

    return ModelResult(
        name=winner.name,
        pipeline=pipeline,
        backend=winner.backend,
        holdout_rmse=winner.holdout_rmse,
        holdout_r2=winner.holdout_r2,
        holdout_mae=winner.holdout_mae,
        log_r2_diagnostic=winner.log_r2_diagnostic,
        cv_rmse=winner.cv_rmse,
        cv_r2=winner.cv_r2,
        cv_mae=winner.cv_mae,
        best_hyperparams=winner.best_hyperparams,
        extra=extra,
    )


def fit_power_log(
    train, holdout, config, feature_cols, *, hyperparams: dict[str, Any] | None = None
) -> ModelResult:
    def _inv(p):
        return np.expm1(p)

    def _log_diag(ho, pred_log):
        if "y_log" not in ho.columns:
            return None
        return float(r2_score(ho["y_log"], pred_log))

    return _fit_and_evaluate(
        "power_log",
        "piecewise_linear",
        build_estimator("power_log", hyperparams),
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
        hyperparams=hyperparams,
    )


def fit_power_level(
    train, holdout, config, feature_cols, *, hyperparams: dict[str, Any] | None = None
) -> ModelResult:
    def _to_log_budget(df):
        out = df.dropna(subset=[config.target, "daily_budget"]).copy()
        out["daily_budget"] = np.log(out["daily_budget"].astype(float).clip(lower=0.01))
        return out

    return _fit_and_evaluate(
        "power_level",
        "piecewise_linear",
        build_estimator("power_level", hyperparams),
        train,
        holdout,
        config,
        feature_cols,
        transform_train=_to_log_budget,
        transform_holdout=_to_log_budget,
        hyperparams=hyperparams,
    )


def fit_xgboost(
    train, holdout, config, feature_cols, *, hyperparams: dict[str, Any] | None = None
) -> ModelResult:
    try:
        build_estimator("xgboost", hyperparams)
    except ImportError as exc:
        raise ImportError("Install xgboost: pip install -e '.[ml]'") from exc
    return _fit_and_evaluate(
        "xgboost",
        "tree_embed",
        build_estimator("xgboost", hyperparams),
        train,
        holdout,
        config,
        feature_cols,
        hyperparams=hyperparams,
    )


FITTERS: dict[str, Callable[..., ModelResult]] = {
    "ridge": fit_ridge,
    "power_log": fit_power_log,
    "power_level": fit_power_level,
    "random_forest": fit_random_forest,
    "xgboost": fit_xgboost,
}


def _short_name(name: str, *, max_len: int = 36) -> str:
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def _format_top_pairs(pairs: list[tuple[str, float]], *, limit: int) -> str:
    top = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:limit]
    return ", ".join(f"{_short_name(k)}={v:+.3g}" for k, v in top)


def _ridge_milp_overview(artifact: LinearMilpRidgeModel, *, top_n: int) -> list[str]:
    model = artifact.model
    cols = artifact.x_columns
    coef = model.coef_
    lines: list[str] = []

    ctx = extract_context_feature_coefs(model, cols)
    if ctx:
        lines.append(f"    context: {_format_top_pairs(list(ctx.items()), limit=top_n)}")

    global_slope = float(coef[cols.index("daily_budget")]) if "daily_budget" in cols else 0.0
    region_slopes = [
        (col[len("budget_x_region_") :], global_slope + float(coef[cols.index(col)]))
        for col in cols
        if col.startswith("budget_x_region_")
    ]
    if region_slopes:
        lines.append(f"    budget slope by region: {_format_top_pairs(region_slopes, limit=top_n)}")
    match_slopes = [
        (col[len("budget_x_") :], float(coef[cols.index(col)]))
        for col in cols
        if col.startswith("budget_x_has_")
    ]
    if match_slopes:
        lines.append(f"    budget slope by match: {_format_top_pairs(match_slopes, limit=top_n)}")
    elif "daily_budget" in cols:
        lines.append(f"    budget slope: (shared)={global_slope:+.3g}")

    region_intercepts = [
        (col[len("region_") :], float(coef[cols.index(col)]))
        for col in cols
        if col.startswith("region_")
    ]
    if region_intercepts:
        lines.append(f"    region: {_format_top_pairs(region_intercepts, limit=top_n)}")
    match_intercepts = [
        (col, float(coef[cols.index(col)]))
        for col in SEGMENT_MATCH_COLS
        if col in cols
    ]
    if match_intercepts:
        lines.append(f"    match type: {_format_top_pairs(match_intercepts, limit=top_n)}")
    return lines


def _pipeline_overview(pipe: Pipeline, *, top_n: int) -> list[str]:
    prep = pipe.named_steps["prep"]
    est = pipe.named_steps["model"]
    try:
        names = list(prep.get_feature_names_out())
    except Exception:
        return []

    if hasattr(est, "coef_"):
        vals = np.asarray(est.coef_).ravel()
        label = "coef"
    elif hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_).ravel()
        label = "importance"
    else:
        return []

    pairs = [(n, float(v)) for n, v in zip(names, vals)]
    cleaned = [(n.split("__", 1)[-1] if "__" in n else n, v) for n, v in pairs]
    return [f"    top {label}: {_format_top_pairs(cleaned, limit=top_n)}"]


def pipeline_feature_overview_lines(pipeline: Any, *, top_n: int = 6) -> list[str]:
    """Compact coefficient / importance summary for a fitted pipeline."""
    if isinstance(pipeline, LinearMilpRidgeModel):
        return _ridge_milp_overview(pipeline, top_n=top_n)
    if isinstance(pipeline, Pipeline):
        return _pipeline_overview(pipeline, top_n=top_n)
    return []


def model_feature_overview_lines(
    res: ModelResult,
    *,
    top_n: int = 6,
    shap_data: pd.DataFrame | None = None,
    target: str | None = None,
    feature_cols: list[str] | None = None,
    shap_effects: dict[str, float] | None = None,
) -> list[str]:
    lines = pipeline_feature_overview_lines(res.pipeline, top_n=top_n)
    if shap_effects is None and shap_data is not None and target and feature_cols:
        shap_effects = compute_mean_shap_effects(
            res.pipeline, shap_data, target, feature_cols
        )
    if shap_effects:
        lines.append(f"    top shap (mean): {format_top_shap_effects(shap_effects, top_n=top_n)}")
    return lines


def resolve_backend(winner: ModelResult, config: CampaignOptConfig) -> str:
    policy = config.model_policy
    if policy.optimizer_backend != "auto":
        return policy.optimizer_backend
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
    export_dir: Path | None = None,
) -> tuple[ModelResult, dict[str, dict[str, float]], dict[str, Any]]:
    feature_cols = get_context_feature_columns(config.context_features)
    results: list[ModelResult] = []
    metrics_table: dict[str, dict[str, float]] = {}

    if run_cv is None:
        run_cv = config.model_policy.validation.scheme in ("time_series_cv", "cv")
    val_cfg = config.model_policy.validation
    n_folds = val_cfg.cv_folds
    tune = val_cfg.tune_hyperparams
    best_hyperparams_all: dict[str, dict[str, Any]] = {}

    if run_cv or tune:
        cv_folds = time_series_cv_folds(
            train,
            n_folds,
            min_train_days=val_cfg.min_train_days,
            min_train_fraction=val_cfg.min_train_fraction,
            min_val_days=val_cfg.min_val_days,
            min_train_rows=val_cfg.min_train_rows,
            min_val_rows=val_cfg.min_val_rows,
        )
        n_train_dates = train["date"].nunique()
        min_train_eff = effective_min_train_days(
            n_train_dates,
            min_train_days=val_cfg.min_train_days,
            min_train_fraction=val_cfg.min_train_fraction,
        )
        print(
            f"CV: {len(cv_folds)} folds on {n_train_dates} train days "
            f"(requested={n_folds}, min_train>={min_train_eff} days "
            f"[{val_cfg.min_train_fraction:.0%} of panel], min_val_days={val_cfg.min_val_days})"
        )

    for name in config.model_policy.candidates:
        fitter = FITTERS.get(name)
        if fitter is None:
            continue
        try:
            hyperparams: dict[str, Any] | None = None
            if tune:
                hyperparams, cv_metrics = tune_hyperparams(
                    name, fitter, train, config, feature_cols, n_folds=n_folds
                )
                best_hyperparams_all[name] = hyperparams
                res = fitter(train, holdout, config, feature_cols, hyperparams=hyperparams)
                res.cv_rmse = cv_metrics["cv_rmse_levels"]
                res.cv_r2 = cv_metrics["cv_r2_levels"]
                res.cv_mae = cv_metrics["cv_mae_levels"]
                res.best_hyperparams = hyperparams or None
            else:
                res = fitter(train, holdout, config, feature_cols)
                if run_cv:
                    cv_metrics = cross_validate_model(
                        fitter, train, config, feature_cols, n_folds=n_folds
                    )
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
            if res.best_hyperparams:
                metrics_table[name]["best_hyperparams"] = res.best_hyperparams
            if res.log_r2_diagnostic is not None:
                metrics_table[name]["log_r2_diagnostic"] = res.log_r2_diagnostic
            cv_str = ""
            if res.cv_rmse is not None:
                cv_str = f" CV_RMSE={res.cv_rmse:.4f}"
                if res.cv_r2 is not None:
                    cv_str += f" CV_R^2={res.cv_r2:.4f}"
            hp_str = f" params={res.best_hyperparams}" if res.best_hyperparams else ""
            print(
                f"  {name}: holdout RMSE={res.holdout_rmse:.4f} R^2={res.holdout_r2:.4f}"
                f"{cv_str}{hp_str}"
            )
            shap_effects = compute_mean_shap_effects(
                res.pipeline, train, config.target, feature_cols
            )
            for line in model_feature_overview_lines(res, shap_effects=shap_effects):
                print(line)
            if shap_effects:
                metrics_table[name]["shap_mean_effects"] = shap_effects
        except Exception as exc:
            print(f"  {name}: skipped ({exc})")

    if not results:
        raise RuntimeError("No models succeeded in tournament")

    ridge_res = next((r for r in results if r.name == "ridge"), None)
    winner = min(results, key=lambda r: _selection_score(r, config))
    backend = resolve_backend(winner, config)

    refit_full = config.model_policy.validation.refit_on_full_data
    full = (
        pd.concat([train, holdout], ignore_index=True).sort_values("date")
        if len(holdout)
        else train
    )
    if refit_full and len(holdout):
        winner = refit_winner_on_data(winner, full, config, feature_cols)
        print(f"  Refit winner ({winner.name}) on full data ({len(full)} rows) for optimization")
        if ridge_res is not None and ridge_res.name != winner.name:
            ridge_res = refit_winner_on_data(ridge_res, full, config, feature_cols)

    coeffs_source = winner if winner.name == "ridge" else ridge_res
    if export_dir is not None and coeffs_source is not None and coeffs_source.extra:
        export_linear_solver_coeffs(
            full if refit_full and len(holdout) else train,
            config,
            Path(export_dir) / "linear_coeffs.json",
            prefit_model=coeffs_source.extra["milp_model"],
            prefit_design=coeffs_source.extra["milp_design"],
        )

    winner_shap = compute_mean_shap_effects(
        winner.pipeline,
        full if refit_full and len(holdout) else train,
        config.target,
        feature_cols,
    )
    if winner_shap and winner.name in metrics_table:
        metrics_table[winner.name]["shap_mean_effects"] = winner_shap

    manifest: dict[str, Any] = {
        "winner": winner.name,
        "backend": backend,
        "target": config.target,
        "secondary_metrics": config.secondary_metrics,
        "holdout_metrics": metrics_table,
        "shap_mean_effects": winner_shap,
        "segment_budget_levels": segment_budget_level_counts(
            full if refit_full and len(holdout) else train
        ).to_dict(),
        "feature_cols": feature_cols,
        "linear_design": "region + match_type + budget×(region + match) + context_features",
        "cv_folds": n_folds if (run_cv or tune) else 0,
        "tune_hyperparams": tune,
        "refit_on_full_data": refit_full,
        "production_rows": len(full),
        "best_hyperparams": best_hyperparams_all,
    }
    return winner, metrics_table, manifest


def save_manifest(manifest: dict[str, Any], model: ModelResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    joblib.dump(model.pipeline, path.parent / "winner_model.joblib")
