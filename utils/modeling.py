"""Model tournament with time-series CV and level-scale holdout metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from utils.cv import (
    cross_validate_model,
    effective_min_train_days,
    time_series_cv_folds,
    validation_cv_kwargs,
)
from utils.coefficients import export_linear_solver_coeffs, extract_context_feature_coefs
from utils.cv import tune_hyperparams
from utils.linear_design import (
    SEGMENT_MATCH_COLS,
    LinearMilpRidgeModel,
    build_linear_milp_design_matrix,
    fit_linear_milp_ridge,
)
from utils.recency_weights import (
    recency_half_life_days,
    recency_sample_weights,
    training_row_recency_weights,
)
from utils.campaign_config import CampaignOptConfig
from utils.shap_effects import compute_mean_shap_effects, format_top_shap_effects
from utils.train_specs import build_estimator, get_train_spec
from utils.training_matrix import (
    build_preprocessor,
    ensure_tree_segment_features,
    prep_xy,
    training_subframe,
)
from utils.campaign_features import (
    SEGMENT_BROAD_MATCH_COL,
    get_context_feature_columns,
)
# Meta-candidates: weighted average of base learners (not MILP backends themselves).
ENSEMBLE_CANDIDATE_NAMES = frozenset({"ensemble", "ensemble_ridge_xgb"})
BASE_TOURNAMENT_CANDIDATES = [
    "ridge",
    "random_forest",
    "xgboost",
]
MEAN_BASELINE_CANDIDATE = "mean_baseline"


def is_mean_baseline_candidate(name: str) -> bool:
    return name == MEAN_BASELINE_CANDIDATE


@dataclass
class TrainingMeanBaseline:
    """Constant level predictor: training-set mean of the optimization target."""

    mean_level: float

    def predict(self, X: Any) -> np.ndarray:
        n = int(getattr(X, "shape", [len(X)])[0])
        return np.full(n, self.mean_level, dtype=float)
ENSEMBLE_MEMBER_GROUPS: dict[str, list[str]] = {
    "ensemble": list(BASE_TOURNAMENT_CANDIDATES),
    "ensemble_ridge_xgb": ["ridge", "xgboost"],
}


def is_ensemble_candidate(name: str) -> bool:
    return name in ENSEMBLE_CANDIDATE_NAMES


def base_tournament_candidates(candidates: list[str]) -> list[str]:
    """Drop ensemble meta-candidates; used for walk-forward evaluation ensembles."""
    return [n for n in candidates if not is_ensemble_candidate(n)]


def _cv_rmse_member_weights(
    metrics_table: dict[str, dict[str, float]],
    member_names: list[str],
) -> dict[str, float]:
    inv: dict[str, float] = {}
    for name in member_names:
        m = metrics_table.get(name) or {}
        rmse = m.get("cv_rmse_levels") or m.get("holdout_rmse_levels") or float("inf")
        inv[name] = 1.0 / max(rmse, 1e-9)
    total = sum(inv.values()) or 1.0
    return {k: v / total for k, v in inv.items()}


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


# Segment-day holdout R² below this is treated as a weak fit (see campaign_opt/README.md).
POOR_R2_THRESHOLD = 0.3


def warn_if_poor_r2(
    r2: float,
    *,
    scope: str,
    label: str | None = None,
    threshold: float = POOR_R2_THRESHOLD,
) -> None:
    """Print a user-visible warning when level-scale R² is below ``threshold``."""
    if not np.isfinite(r2) or r2 >= threshold:
        return
    who = f" ({label})" if label else ""
    print(
        f"[Warn] Poor {scope} R²={r2:.4f} (< {threshold}){who}; "
        "model fit is weak — treat predictions with caution."
    )


def tournament_winner_name(manifest: dict) -> str | None:
    """Tournament-selected model from ``model_manifest.json``."""
    winner = manifest.get("winner")
    return str(winner) if winner else None


def configured_evaluation_model_name(config: CampaignOptConfig) -> str:
    """Model used for plan-vs-actual scoring."""
    if config.evaluation.use_ensemble:
        return "ensemble"
    from utils.optimize import require_optimizer_winner

    return require_optimizer_winner(config)


def warn_if_not_tournament_winner(
    configured_model: str,
    manifest: dict,
    *,
    role: str,
) -> None:
    """Warn when optimizer or evaluation uses a model other than the CV tournament winner."""
    tw = tournament_winner_name(manifest)
    if not tw or configured_model == tw:
        return
    print(
        f"[Warn] {role} uses {configured_model!r} but tournament winner is {tw!r}; "
        "consider switching to the tournament winner."
    )


def eval_pipeline_holdout(
    pipeline: Any,
    holdout: pd.DataFrame,
    config: CampaignOptConfig,
    feature_cols: list[str],
    *,
    label: str | None = None,
) -> dict[str, float] | None:
    """Level-scale RMSE / R² / MAE on holdout rows (walk-forward backtest diagnostic)."""
    if holdout.empty:
        return None
    target = config.target
    sub = holdout.dropna(subset=[target, "daily_budget", "region"])
    if sub.empty:
        return None
    from utils.evaluation import EnsembleModel

    if isinstance(pipeline, EnsembleModel):
        if "segment" not in sub.columns:
            return None
        pred = pipeline.predict_levels(sub)
        y = sub[target].astype(float).values
    else:
        X, y = _prep_xy(sub, target, feature_cols)
        if len(X) == 0:
            return None
        pred = np.clip(pipeline.predict(X), 0, None)
    m = _level_metrics(y, pred)
    warn_if_poor_r2(m["holdout_r2_levels"], scope="holdout", label=label)
    return {
        "holdout_r2": m["holdout_r2_levels"],
        "holdout_rmse": m["holdout_rmse_levels"],
        "holdout_mae": m["holdout_mae_levels"],
        "n_holdout": float(len(sub)),
    }


# Backward-compatible aliases — canonical definitions live in training_matrix.py.
_ensure_tree_segment_features = ensure_tree_segment_features
_training_subframe = training_subframe
_prep_xy = prep_xy
_build_preprocessor = build_preprocessor


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
    fit_kw: dict[str, Any] = {}
    sample_weight = training_row_recency_weights(
        tr, config, y_col=fit_y_col, date_col="date"
    )
    if sample_weight is not None:
        fit_kw["model__sample_weight"] = sample_weight
    pipe.fit(X_train, y_train, **fit_kw)
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


def fit_mean_baseline(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    config: CampaignOptConfig,
    feature_cols: list[str],
    *,
    hyperparams: dict[str, Any] | None = None,
) -> ModelResult:
    """Predict the training-set mean target level (reference for tournament comparison)."""
    del feature_cols, hyperparams
    target = config.target
    y_train = train.dropna(subset=[target])[target].astype(float)
    if y_train.empty:
        raise ValueError(f"No training rows with target {target!r} for mean baseline")
    mean_level = float(y_train.mean())
    model = TrainingMeanBaseline(mean_level)

    if holdout.empty:
        m = {
            "holdout_rmse_levels": float("nan"),
            "holdout_r2_levels": float("nan"),
            "holdout_mae_levels": float("nan"),
        }
    else:
        ho = holdout.dropna(subset=[target])
        y = ho[target].astype(float).values
        pred = np.clip(model.predict(np.zeros((len(ho), 1))), 0, None)
        m = _level_metrics(y, pred)

    return ModelResult(
        name=MEAN_BASELINE_CANDIDATE,
        pipeline=model,
        backend="baseline",
        holdout_rmse=m["holdout_rmse_levels"],
        holdout_r2=m["holdout_r2_levels"],
        holdout_mae=m["holdout_mae_levels"],
        extra={"train_mean": mean_level},
    )


def fit_ridge(train, holdout, config, feature_cols, *, hyperparams: dict[str, Any] | None = None) -> ModelResult:
    del feature_cols  # ridge uses MILP-linear design (region + match + budget interactions + context)
    alpha = float((hyperparams or {}).get("alpha", 1.0))
    train_design = build_linear_milp_design_matrix(train, config)
    half_life = recency_half_life_days(config)
    sample_weight = recency_sample_weights(train_design.sub, half_life_days=half_life)
    artifact = fit_linear_milp_ridge(
        train_design, config, alpha=alpha, sample_weight=sample_weight
    )

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
        pred = np.clip(artifact.predict_design_frame(holdout), 0, None)
        m = _level_metrics(holdout_design.y, pred)

    return ModelResult(
        name="ridge",
        pipeline=artifact,
        backend="linear",
        holdout_rmse=m["holdout_rmse_levels"],
        holdout_r2=m["holdout_r2_levels"],
        holdout_mae=m["holdout_mae_levels"],
        best_hyperparams=hyperparams,
        extra={
            "milp_model": artifact.model,
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
    half_life = recency_half_life_days(config)
    sample_weight = recency_sample_weights(design.sub, half_life_days=half_life)
    return fit_linear_milp_ridge(
        design, config, alpha=alpha, sample_weight=sample_weight
    )


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
    *,
    tune: bool | None = None,
) -> Any:
    """
    Refit a named tournament candidate on ``train`` for MILP embedding.

    When ``tune`` is true (walk-forward backtest), hyperparameters are chosen by
    time-series CV on ``train``. When false (production optimize), hyperparameters
    must be present in ``manifest``.
    """
    val = config.model_policy.validation
    do_tune = val.tune_hyperparams if tune is None else tune
    feature_cols = manifest.get("feature_cols")
    if not feature_cols:
        raise ValueError("manifest missing feature_cols; run fit_response_models.py")
    n_folds = val.cv_folds
    empty_holdout = train.iloc[0:0]

    if model_name == "ensemble_ridge_xgb":
        member_names = ENSEMBLE_MEMBER_GROUPS["ensemble_ridge_xgb"]
        if do_tune:
            member_hp = _ensure_member_hyperparams(
                member_names,
                train,
                config,
                feature_cols,
                {},
                tune=True,
                n_folds=n_folds,
            )
        else:
            member_hp = {}
            for m in member_names:
                hp = hyperparams_from_manifest(manifest, m)
                if not hp:
                    raise ValueError(
                        f"manifest missing best_hyperparams for ensemble member {m!r}"
                    )
                member_hp[m] = hp
        return fit_ensemble_tournament(
            "ensemble_ridge_xgb",
            member_names,
            train,
            empty_holdout,
            config,
            feature_cols,
            member_hyperparams=member_hp,
            member_weights=None,
        ).pipeline
    if is_ensemble_candidate(model_name):
        raise ValueError(
            f"optimizer_winner {model_name!r} is not supported for MILP optimization; "
            "use 'ensemble_ridge_xgb' or a base model such as 'xgboost'."
        )
    fitter = FITTERS.get(model_name)
    if fitter is None:
        raise ValueError(f"Unknown optimizer_winner: {model_name!r}")
    if do_tune:
        hyperparams, _ = tune_hyperparams(
            model_name, fitter, train, config, feature_cols, n_folds=n_folds
        )
    else:
        hyperparams = hyperparams_from_manifest(manifest, model_name)
        if hyperparams is None:
            raise ValueError(f"manifest missing best_hyperparams for {model_name!r}")
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
    if is_mean_baseline_candidate(winner.name):
        res = fit_mean_baseline(df, df.iloc[0:0], config, feature_cols)
        return ModelResult(
            name=winner.name,
            pipeline=res.pipeline,
            backend=winner.backend,
            holdout_rmse=winner.holdout_rmse,
            holdout_r2=winner.holdout_r2,
            holdout_mae=winner.holdout_mae,
            cv_rmse=winner.cv_rmse,
            cv_r2=winner.cv_r2,
            cv_mae=winner.cv_mae,
            extra=res.extra,
        )
    if is_ensemble_candidate(winner.name):
        member_names = ENSEMBLE_MEMBER_GROUPS[winner.name]
        member_hp = winner.best_hyperparams if isinstance(winner.best_hyperparams, dict) else {}
        if member_hp and not any(isinstance(v, dict) for v in member_hp.values()):
            member_hp = {}
        empty = df.iloc[0:0]
        refit = fit_ensemble_tournament(
            winner.name,
            member_names,
            df,
            empty,
            config,
            feature_cols,
            member_hyperparams=member_hp,
            member_weights=None,
        )
        return ModelResult(
            name=winner.name,
            pipeline=refit.pipeline,
            backend=winner.backend,
            holdout_rmse=winner.holdout_rmse,
            holdout_r2=winner.holdout_r2,
            holdout_mae=winner.holdout_mae,
            log_r2_diagnostic=winner.log_r2_diagnostic,
            cv_rmse=winner.cv_rmse,
            cv_r2=winner.cv_r2,
            cv_mae=winner.cv_mae,
            best_hyperparams=winner.best_hyperparams,
            extra=refit.extra,
        )

    spec = get_train_spec(winner.name, winner.best_hyperparams)
    if spec is None:
        raise ValueError(f"Unknown winner model: {winner.name}")

    if winner.name == "ridge":
        alpha = float((winner.best_hyperparams or {}).get("alpha", 1.0))
        design = build_linear_milp_design_matrix(df, config)
        half_life = recency_half_life_days(config)
        sample_weight = recency_sample_weights(design.sub, half_life_days=half_life)
        pipeline = fit_linear_milp_ridge(
            design, config, alpha=alpha, sample_weight=sample_weight
        )
        extra = {"milp_model": pipeline.model, "milp_design": design}
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
        fit_kw: dict[str, Any] = {}
        sample_weight = training_row_recency_weights(
            tr, config, y_col=spec.fit_y_col, date_col="date"
        )
        if sample_weight is not None:
            fit_kw["model__sample_weight"] = sample_weight
        pipeline.fit(X, y, **fit_kw)
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


def fit_ensemble_tournament(
    name: str,
    member_names: list[str],
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    config: CampaignOptConfig,
    feature_cols: list[str],
    *,
    member_hyperparams: dict[str, dict[str, Any]] | None = None,
    member_weights: dict[str, float] | None = None,
    hyperparams: dict[str, Any] | None = None,
) -> ModelResult:
    """Fit member models on train and score level predictions on holdout (equal or CV weights)."""
    from utils.evaluation import EnsembleModel, fit_member_on_train

    del hyperparams  # per-member params live in member_hyperparams
    members = []
    mhp = member_hyperparams or {}
    for mname in member_names:
        hp = mhp.get(mname)
        spec = get_train_spec(mname, hp)
        if spec is None:
            continue
        member = fit_member_on_train(spec, train, config, feature_cols, hyperparams=hp)
        w = 1.0 if not member_weights else member_weights.get(mname, 0.0)
        if member_weights is not None and w <= 0:
            continue
        member.weight = w
        members.append(member)

    if not members:
        raise RuntimeError(f"No ensemble members fitted for {name}")

    ensemble = EnsembleModel(
        members=members,
        feature_cols=feature_cols,
        target=config.target,
        baseline_budget=float(config.evaluation.baseline_budget),
    )

    target = config.target
    sub = holdout.dropna(subset=[target, "daily_budget", "segment"])
    if len(sub) == 0:
        m = {
            "holdout_rmse_levels": float("nan"),
            "holdout_r2_levels": float("nan"),
            "holdout_mae_levels": float("nan"),
        }
    else:
        pred = ensemble.predict_levels(sub)
        m = _level_metrics(sub[target].astype(float).values, pred)

    return ModelResult(
        name=name,
        pipeline=ensemble,
        backend="tree_embed",
        holdout_rmse=m["holdout_rmse_levels"],
        holdout_r2=m["holdout_r2_levels"],
        holdout_mae=m["holdout_mae_levels"],
        best_hyperparams=mhp or None,
        extra={"member_names": member_names, "ensemble": ensemble},
    )


def fit_ensemble(
    train,
    holdout,
    config,
    feature_cols,
    *,
    hyperparams: dict[str, Any] | None = None,
    member_hyperparams: dict[str, dict[str, Any]] | None = None,
    member_weights: dict[str, float] | None = None,
) -> ModelResult:
    mhp = member_hyperparams or (
        hyperparams if hyperparams and any(isinstance(v, dict) for v in hyperparams.values()) else None
    )
    return fit_ensemble_tournament(
        "ensemble",
        ENSEMBLE_MEMBER_GROUPS["ensemble"],
        train,
        holdout,
        config,
        feature_cols,
        member_hyperparams=mhp,
        member_weights=member_weights,
    )


def fit_ensemble_ridge_xgb(
    train,
    holdout,
    config,
    feature_cols,
    *,
    hyperparams: dict[str, Any] | None = None,
    member_hyperparams: dict[str, dict[str, Any]] | None = None,
    member_weights: dict[str, float] | None = None,
) -> ModelResult:
    mhp = member_hyperparams or (
        hyperparams if hyperparams and any(isinstance(v, dict) for v in hyperparams.values()) else None
    )
    return fit_ensemble_tournament(
        "ensemble_ridge_xgb",
        ENSEMBLE_MEMBER_GROUPS["ensemble_ridge_xgb"],
        train,
        holdout,
        config,
        feature_cols,
        member_hyperparams=mhp,
        member_weights=member_weights,
    )


FITTERS: dict[str, Callable[..., ModelResult]] = {
    "ridge": fit_ridge,
    "random_forest": fit_random_forest,
    "xgboost": fit_xgboost,
    "ensemble": fit_ensemble,
    "ensemble_ridge_xgb": fit_ensemble_ridge_xgb,
}


def _ensure_member_hyperparams(
    member_names: list[str],
    train: pd.DataFrame,
    config: CampaignOptConfig,
    feature_cols: list[str],
    best_hyperparams_all: dict[str, Any],
    *,
    tune: bool,
    n_folds: int,
) -> dict[str, dict[str, Any]]:
    member_hp: dict[str, dict[str, Any]] = {}
    for mname in member_names:
        existing = best_hyperparams_all.get(mname)
        if isinstance(existing, dict) and existing:
            member_hp[mname] = existing
            continue
        base_fitter = FITTERS.get(mname)
        if base_fitter is None:
            continue
        if tune:
            hp, _ = tune_hyperparams(mname, base_fitter, train, config, feature_cols, n_folds=n_folds)
            member_hp[mname] = hp
            best_hyperparams_all[mname] = hp
        else:
            member_hp[mname] = {}
    return member_hp


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
    if SEGMENT_BROAD_MATCH_COL in cols:
        inter_col = f"budget_x_{SEGMENT_BROAD_MATCH_COL}"
        if inter_col in cols:
            lines.append(
                f"    budget slope (broad vs phrase/exact): "
                f"{SEGMENT_BROAD_MATCH_COL}={float(coef[cols.index(inter_col)]):+.3g}"
            )
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
    if isinstance(pipeline, TrainingMeanBaseline):
        return [f"    train mean level: {pipeline.mean_level:.4g}"]
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
    if (
        shap_effects is None
        and shap_data is not None
        and target
        and feature_cols
        and not is_mean_baseline_candidate(res.name)
    ):
        shap_effects = compute_mean_shap_effects(
            res.pipeline, shap_data, target, feature_cols
        )
    if shap_effects:
        lines.append(f"    top shap (mean): {format_top_shap_effects(shap_effects, top_n=top_n)}")
    return lines


def report_model_fit_diagnostics(
    res: ModelResult,
    train: pd.DataFrame,
    config: CampaignOptConfig,
    feature_cols: list[str],
    *,
    shap_effects: dict[str, float] | None = None,
) -> dict[str, float] | None:
    """Print tournament-style holdout metrics, warnings, and feature/SHAP overview."""
    cv_str = ""
    if res.cv_rmse is not None:
        cv_str = f" CV_RMSE={res.cv_rmse:.4f}"
        if res.cv_r2 is not None:
            cv_str += f" CV_R^2={res.cv_r2:.4f}"
    hp_str = f" params={res.best_hyperparams}" if res.best_hyperparams else ""
    if np.isfinite(res.holdout_rmse):
        print(
            f"  {res.name}: holdout RMSE={res.holdout_rmse:.4f} R^2={res.holdout_r2:.4f}"
            f"{cv_str}{hp_str}"
        )
    else:
        print(f"  {res.name}: refit on full panel (no holdout metrics){hp_str}")
    warn_if_poor_r2(res.holdout_r2, scope="holdout", label=res.name)
    if res.cv_r2 is not None:
        warn_if_poor_r2(res.cv_r2, scope="CV", label=res.name)
    if shap_effects is None and not is_mean_baseline_candidate(res.name):
        shap_effects = compute_mean_shap_effects(
            res.pipeline, train, config.target, feature_cols
        )
    for line in model_feature_overview_lines(res, shap_effects=shap_effects):
        print(line)
    return shap_effects


def resolve_backend(winner: ModelResult) -> str:
    return winner.backend


def print_tournament_metric_summary(
    metrics_table: dict[str, dict[str, float]],
    *,
    winner_name: str | None = None,
) -> None:
    """Print holdout / CV $R^2$ for ensemble candidates and the tournament winner."""
    print("\n--- Tournament metric summary ---")
    summary_rows = [
        ("ensemble ridge+xgb", "ensemble_ridge_xgb"),
        ("mean baseline", MEAN_BASELINE_CANDIDATE),
    ]
    if "ensemble" in metrics_table:
        summary_rows.insert(0, ("ensemble", "ensemble"))
    for label, key in summary_rows:
        row = metrics_table.get(key) or {}
        cv_r2 = row.get("cv_r2_levels")
        ho_r2 = row.get("holdout_r2_levels")
        cv_rmse = row.get("cv_rmse_levels")
        ho_rmse = row.get("holdout_rmse_levels")
        cv_r2_s = f"{cv_r2:.4f}" if cv_r2 is not None else "n/a"
        ho_r2_s = f"{ho_r2:.4f}" if ho_r2 is not None else "n/a"
        cv_rmse_s = f"{cv_rmse:.4f}" if cv_rmse is not None else "n/a"
        ho_rmse_s = f"{ho_rmse:.4f}" if ho_rmse is not None else "n/a"
        print(
            f"  {label}: CV RMSE={cv_rmse_s} R^2={cv_r2_s}; "
            f"holdout RMSE={ho_rmse_s} R^2={ho_r2_s}"
        )
    if winner_name:
        row = metrics_table.get(winner_name) or {}
        cv_r2 = row.get("cv_r2_levels")
        ho_r2 = row.get("holdout_r2_levels")
        cv_s = f"{cv_r2:.4f}" if cv_r2 is not None else "n/a"
        ho_s = f"{ho_r2:.4f}" if ho_r2 is not None else "n/a"
        print(f"  winner ({winner_name}): CV R^2={cv_s} holdout R^2={ho_s}")


def _selection_score(res: ModelResult, config: CampaignOptConfig) -> float:
    """Lower is better for RMSE; higher is better for R2."""
    metric = config.model_policy.selection_metric
    val_cfg = config.model_policy.validation
    use_cv = val_cfg.scheme == "time_series_cv"
    cv_rmse = res.cv_rmse
    cv_r2 = res.cv_r2
    if "r2" in metric:
        val = (cv_r2 if use_cv and cv_r2 is not None else res.holdout_r2) or 0.0
        return -val
    return cv_rmse if use_cv and cv_rmse is not None else res.holdout_rmse


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
        cv_folds = time_series_cv_folds(train, n_folds, **validation_cv_kwargs(config))
        n_train_dates = train["date"].nunique()
        min_train_eff = effective_min_train_days(
            n_train_dates,
            min_train_days=val_cfg.min_train_days,
            min_train_fraction=val_cfg.min_train_fraction,
        )
        print(
            f"CV (expanding-window): {len(cv_folds)} folds on {n_train_dates} train days "
            f"(requested={n_folds}, min_train>={min_train_eff} days "
            f"[{val_cfg.min_train_fraction:.0%} of panel], min_val_days={val_cfg.min_val_days})"
        )

    for name in config.model_policy.candidates:
        fitter = FITTERS.get(name)
        if fitter is None:
            continue
        try:
            hyperparams: dict[str, Any] | None = None
            if is_ensemble_candidate(name):
                member_names = ENSEMBLE_MEMBER_GROUPS[name]
                member_hp = _ensure_member_hyperparams(
                    member_names,
                    train,
                    config,
                    feature_cols,
                    best_hyperparams_all,
                    tune=tune,
                    n_folds=n_folds,
                )
                # Optimizer / ridge_xgb_embed use equal ridge+XGB blend; only the
                # full 5-member eval ensemble uses inverse-CV-RMSE weights.
                weights = (
                    _cv_rmse_member_weights(metrics_table, member_names)
                    if config.evaluation.weight_by_cv_rmse
                    and name != "ensemble_ridge_xgb"
                    else None
                )

                def _fit_ensemble(
                    tr: pd.DataFrame,
                    ho: pd.DataFrame,
                    cfg: CampaignOptConfig,
                    fc: list[str],
                    *,
                    _name: str = name,
                    _members: list[str] = member_names,
                    _mhp: dict[str, dict[str, Any]] = member_hp,
                    _weights: dict[str, float] | None = weights,
                ) -> ModelResult:
                    return fit_ensemble_tournament(
                        _name,
                        _members,
                        tr,
                        ho,
                        cfg,
                        fc,
                        member_hyperparams=_mhp,
                        member_weights=_weights,
                    )

                if tune or run_cv:
                    cv_metrics = cross_validate_model(
                        _fit_ensemble, train, config, feature_cols, n_folds=n_folds
                    )
                    res = _fit_ensemble(train, holdout, config, feature_cols)
                    res.cv_rmse = cv_metrics["cv_rmse_levels"]
                    res.cv_r2 = cv_metrics["cv_r2_levels"]
                    res.cv_mae = cv_metrics["cv_mae_levels"]
                    res.best_hyperparams = member_hp or None
                    best_hyperparams_all[name] = member_hp
                else:
                    res = _fit_ensemble(train, holdout, config, feature_cols)
                    res.best_hyperparams = member_hp or None
                    best_hyperparams_all[name] = member_hp
            elif tune:
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
            shap_effects = report_model_fit_diagnostics(
                res, train, config, feature_cols
            )
            if shap_effects:
                metrics_table[name]["shap_mean_effects"] = shap_effects
        except Exception as exc:
            print(f"  {name}: skipped ({exc})")

    try:
        baseline_res = fit_mean_baseline(train, holdout, config, feature_cols)
        if run_cv:
            cv_metrics = cross_validate_model(
                fit_mean_baseline, train, config, feature_cols, n_folds=n_folds
            )
            baseline_res.cv_rmse = cv_metrics["cv_rmse_levels"]
            baseline_res.cv_r2 = cv_metrics["cv_r2_levels"]
            baseline_res.cv_mae = cv_metrics["cv_mae_levels"]
        results.append(baseline_res)
        metrics_table[MEAN_BASELINE_CANDIDATE] = {
            "holdout_rmse_levels": baseline_res.holdout_rmse,
            "holdout_r2_levels": baseline_res.holdout_r2,
            "holdout_mae_levels": baseline_res.holdout_mae,
        }
        if baseline_res.cv_rmse is not None:
            metrics_table[MEAN_BASELINE_CANDIDATE]["cv_rmse_levels"] = baseline_res.cv_rmse
            metrics_table[MEAN_BASELINE_CANDIDATE]["cv_r2_levels"] = baseline_res.cv_r2
        if baseline_res.extra and "train_mean" in baseline_res.extra:
            metrics_table[MEAN_BASELINE_CANDIDATE]["train_mean"] = baseline_res.extra[
                "train_mean"
            ]
        cv_str = ""
        if baseline_res.cv_rmse is not None:
            cv_str = f" CV_RMSE={baseline_res.cv_rmse:.4f}"
            if baseline_res.cv_r2 is not None:
                cv_str += f" CV_R^2={baseline_res.cv_r2:.4f}"
        print(
            f"  {MEAN_BASELINE_CANDIDATE}: holdout RMSE={baseline_res.holdout_rmse:.4f} "
            f"R^2={baseline_res.holdout_r2:.4f}{cv_str} "
            f"train_mean={baseline_res.extra.get('train_mean') if baseline_res.extra else 'n/a'}"
        )
        for line in model_feature_overview_lines(baseline_res):
            print(line)
    except Exception as exc:
        print(f"  {MEAN_BASELINE_CANDIDATE}: skipped ({exc})")

    competitive = [r for r in results if not is_mean_baseline_candidate(r.name)]
    if not competitive:
        raise RuntimeError("No models succeeded in tournament")

    ridge_res = next((r for r in competitive if r.name == "ridge"), None)
    winner = min(competitive, key=lambda r: _selection_score(r, config))
    backend = resolve_backend(winner)

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

    winner_shap: dict[str, float] | None = None
    if not is_mean_baseline_candidate(winner.name):
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
        "feature_cols": feature_cols,
        "linear_design": (
            "region + is_broad_match + budget×(region + is_broad_match) + context_features"
        ),
        "mean_baseline_candidate": MEAN_BASELINE_CANDIDATE,
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
