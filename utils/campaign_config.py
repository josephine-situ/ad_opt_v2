"""Load campaign experiment config from JSON."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from utils.paths import campaign_config_path, exp_dir as paths_exp_dir

DEFAULTS: dict = {
    "exp_name": "default",
    "course": "sys_think",
    "target": "all_conv",
    "secondary_metrics": ["clicks"],
    "decision_variables": {},
    "context_features": {},
    "constraints": {},
    "debug_write_lp": False,
    "piecewise_budget_knots": 8,
    "modeling_lookback_days": None,
    "model_policy": {
        "candidates": [
            "ridge",
            "power_log",
            "power_level",
            "random_forest",
            "xgboost",
            "ensemble",
            "ensemble_ridge_xgb",
        ],
        "selection_metric": "holdout_rmse_levels",
        "secondary_selection_metrics": ["holdout_r2_levels", "holdout_mae_levels"],
        "optimizer_backend": "auto",
        "optimizer_winner": None,
        "stability_check": False,
        "validation": {
            "scheme": "time_series_cv",
            "holdout_days": 75,
            "cv_folds": 3,
            "min_train_fraction": 0.5,
            "min_train_days": 0,
            "min_val_days": 21,
            "min_train_rows": 50,
            "min_val_rows": 20,
            "tune_hyperparams": True,
            "refit_on_full_data": True,
            "recency_half_life_days": None,
        },
    },
    "evaluation": {
        "use_ensemble": True,
        "baseline_budget": 0.0,
        "weight_by_cv_rmse": True,
        "objective": "incremental",
        "apply_observed_budget_floor": False,
        "max_level_ub": None,
        "milp_external_level_tol": 0.01,
        "budget_floor_atol": 0.01,
    },
    "backtest": {"strategy": "two_stage"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


_DICT_FIELDS = frozenset({"constraints", "context_features", "decision_variables"})


def _to_ns(data: dict) -> SimpleNamespace:
    fields = {}
    for key, val in data.items():
        if key in _DICT_FIELDS:
            fields[key] = dict(val) if isinstance(val, dict) else {}
        elif isinstance(val, dict):
            fields[key] = _to_ns(val)
        else:
            fields[key] = val
    return SimpleNamespace(**fields)


def _attach_exp_dir(cfg: SimpleNamespace) -> SimpleNamespace:
    def _exp_dir(base: Path | None = None) -> Path:
        if base is not None:
            return base / "campaign" / cfg.exp_name
        return paths_exp_dir(cfg.course, cfg.exp_name)

    cfg.exp_dir = _exp_dir
    return cfg


def _ns_values(value) -> dict:
    if isinstance(value, SimpleNamespace):
        return {k: _ns_values(v) for k, v in vars(value).items() if k != "exp_dir"}
    return value


def ValidationConfig(**kwargs) -> SimpleNamespace:
    return _to_ns(_deep_merge(DEFAULTS["model_policy"]["validation"], kwargs))


def ModelPolicy(**kwargs) -> SimpleNamespace:
    raw = dict(kwargs)
    validation = raw.pop("validation", None)
    merged = _deep_merge(DEFAULTS["model_policy"], raw)
    if validation is not None:
        merged["validation"] = _deep_merge(
            DEFAULTS["model_policy"]["validation"], _ns_values(validation)
        )
    return _to_ns(merged)


def EvaluationConfig(**kwargs) -> SimpleNamespace:
    return _to_ns(_deep_merge(DEFAULTS["evaluation"], kwargs))


def CampaignOptConfig(**kwargs) -> SimpleNamespace:
    raw = {k: _ns_values(v) for k, v in kwargs.items()}
    for section in ("model_policy", "evaluation", "backtest"):
        if section in raw and isinstance(raw[section], dict):
            raw[section] = _deep_merge(DEFAULTS[section], raw[section])
    return _attach_exp_dir(_to_ns(_deep_merge(DEFAULTS, raw)))


def load_campaign_config(path: str | Path) -> SimpleNamespace:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return _attach_exp_dir(_to_ns(_deep_merge(DEFAULTS, raw)))


def default_config_path(course: str = "sys_think", exp_name: str = "default") -> Path:
    return campaign_config_path(course, exp_name)
