"""Campaign optimization experiment configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from campaign_opt.paths import campaign_config_path, exp_dir as paths_exp_dir


@dataclass
class ValidationConfig:
    # time_holdout: last N days for reporting; time_series_cv: expanding-window CV on train
    scheme: str = "time_series_cv"
    holdout_days: int = 75
    cv_folds: int = 3
    min_train_fraction: float = 0.5
    min_train_days: int = 0
    min_val_days: int = 21
    min_train_rows: int = 50
    min_val_rows: int = 20
    tune_hyperparams: bool = True
    refit_on_full_data: bool = True  # after selection, refit winner on train+holdout for optimization
    recency_half_life_days: float | None = None  # exponential sample weights; None = uniform


@dataclass
class EvaluationConfig:
    """How plan vs actual is scored (incremental lift vs same keyword set at baseline_budget)."""
    use_ensemble: bool = True  # if True, multi-member ensemble; if False, optimizer_winner — both fit on full panel
    baseline_budget: float = 0.0
    weight_by_cv_rmse: bool = True  # else equal-weight average
    objective: str = "incremental"  # "levels" | "incremental" — MILP maximizes total level or lift
    apply_observed_budget_floor: bool = False  # zero preds when budget < min observed cap (optimizer paths only)
    max_level_ub: float | None = None  # optional cap on McCormick level_ub per segment
    milp_external_level_tol: float = 0.01  # warn when |milp_pred - gated sklearn level| exceeds this
    budget_floor_atol: float = 0.01  # treat budget as at/above min observed when within this (dollars)


@dataclass
class ModelPolicy:
    candidates: list[str] = field(
        default_factory=lambda: [
            "ridge",
            "power_log",
            "power_level",
            "random_forest",
            "xgboost",
            "ensemble",
            "ensemble_ridge_xgb",
        ]
    )
    selection_metric: str = "holdout_rmse_levels"
    secondary_selection_metrics: list[str] = field(
        default_factory=lambda: ["holdout_r2_levels", "holdout_mae_levels"]
    )
    optimizer_backend: str = "auto"  # "auto" uses the winner's backend; else force linear|piecewise_linear|tree_embed
    optimizer_winner: str | None = None  # if set, optimize with this candidate (e.g. "xgboost") instead of manifest winner
    stability_check: bool = False
    validation: ValidationConfig = field(default_factory=ValidationConfig)


@dataclass
class BacktestConfig:
    """Walk-forward backtest strategy (default: two-stage)."""
    strategy: str = "two_stage"


@dataclass
class CampaignOptConfig:
    exp_name: str
    course: str
    target: str = "all_conv"
    secondary_metrics: list[str] = field(default_factory=lambda: ["clicks"])
    decision_variables: dict[str, Any] = field(default_factory=dict)
    context_features: dict[str, list[str]] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    model_policy: ModelPolicy = field(default_factory=ModelPolicy)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    debug_write_lp: bool = False
    piecewise_budget_knots: int = 8
    modeling_lookback_days: int | None = None

    def exp_dir(self, base: Path | None = None) -> Path:
        if base is not None:
            return base / self.exp_name
        return paths_exp_dir(self.exp_name)


def _parse_model_policy(raw: dict[str, Any]) -> ModelPolicy:
    validation_raw = raw.pop("validation", {}) or {}
    validation = ValidationConfig(
        scheme=validation_raw.get("scheme", "time_series_cv"),
        holdout_days=int(validation_raw.get("holdout_days", 75)),
        cv_folds=int(validation_raw.get("cv_folds", 3)),
        min_train_fraction=float(validation_raw.get("min_train_fraction", 0.5)),
        min_train_days=int(validation_raw.get("min_train_days", 0)),
        min_val_days=int(validation_raw.get("min_val_days", 21)),
        min_train_rows=int(validation_raw.get("min_train_rows", 50)),
        min_val_rows=int(validation_raw.get("min_val_rows", 20)),
        tune_hyperparams=bool(validation_raw.get("tune_hyperparams", True)),
        refit_on_full_data=bool(validation_raw.get("refit_on_full_data", True)),
        recency_half_life_days=(
            float(validation_raw["recency_half_life_days"])
            if validation_raw.get("recency_half_life_days") is not None
            else None
        ),
    )
    mp_keys = {f.name for f in fields(ModelPolicy)}
    return ModelPolicy(
        validation=validation,
        **{k: v for k, v in raw.items() if k in mp_keys and k != "validation"},
    )


def load_campaign_config(path: str | Path) -> CampaignOptConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    mp_raw = data.pop("model_policy", {}) or {}
    model_policy = _parse_model_policy(mp_raw)
    eval_raw = data.pop("evaluation", {}) or {}
    evaluation = EvaluationConfig(**{k: v for k, v in eval_raw.items() if k in EvaluationConfig.__annotations__})
    bt_raw = data.pop("backtest", {}) or {}
    backtest = BacktestConfig(
        **{k: v for k, v in bt_raw.items() if k in BacktestConfig.__annotations__}
    )
    return CampaignOptConfig(
        model_policy=model_policy,
        evaluation=evaluation,
        backtest=backtest,
        **data,
    )


def default_config_path(exp_name: str = "default") -> Path:
    return campaign_config_path(exp_name)
