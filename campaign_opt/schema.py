"""Campaign optimization experiment configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ValidationConfig:
    # time_holdout: last N days for reporting; time_series_cv: expanding-window CV on train
    scheme: str = "time_series_cv"
    holdout_days: int = 75
    cv_folds: int = 5


@dataclass
class EvaluationConfig:
    """How plan vs actual is scored (ensemble incremental lift)."""
    use_ensemble: bool = True
    baseline_budget: float = 0.0
    weight_by_cv_rmse: bool = True  # else equal-weight average


@dataclass
class ModelPolicy:
    candidates: list[str] = field(
        default_factory=lambda: ["ridge", "power_log", "power_level", "random_forest", "xgboost"]
    )
    selection_metric: str = "holdout_rmse_levels"
    secondary_selection_metrics: list[str] = field(
        default_factory=lambda: ["holdout_r2_levels", "holdout_mae_levels"]
    )
    min_holdout_gain_vs_ridge: float = 0.03
    optimizer_backend: str = "auto"
    stability_check: bool = False
    validation: ValidationConfig = field(default_factory=ValidationConfig)


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
    debug_write_lp: bool = False
    piecewise_budget_knots: int = 8

    def exp_dir(self, base: Path | None = None) -> Path:
        root = base or Path("opt_results")
        return root / self.course / "campaign" / self.exp_name


def _parse_model_policy(raw: dict[str, Any]) -> ModelPolicy:
    validation_raw = raw.pop("validation", {}) or {}
    validation = ValidationConfig(
        scheme=validation_raw.get("scheme", "time_series_cv"),
        holdout_days=int(validation_raw.get("holdout_days", 75)),
        cv_folds=int(validation_raw.get("cv_folds", 5)),
    )
    return ModelPolicy(validation=validation, **{k: v for k, v in raw.items() if k != "validation"})


def load_campaign_config(path: str | Path) -> CampaignOptConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    mp_raw = data.pop("model_policy", {}) or {}
    model_policy = _parse_model_policy(mp_raw)
    eval_raw = data.pop("evaluation", {}) or {}
    evaluation = EvaluationConfig(**{k: v for k, v in eval_raw.items() if k in EvaluationConfig.__annotations__})
    return CampaignOptConfig(model_policy=model_policy, evaluation=evaluation, **data)


def default_config_path(course: str, exp_name: str = "default") -> Path:
    return Path("opt_results") / course / "campaign" / exp_name / "campaign_config.json"
