"""Load merged campaign config: config/default.yaml + optional <course>/course.yaml."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from utils.paths import DEFAULT_COURSE, REPO_ROOT, course_yaml_path, experiments_dir, prod_dir

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"
DEFAULT_SOLVER_TIME_LIMIT = 600


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _merged_dict(course: str, *, extra_path: str | Path | None = None) -> dict:
    if not DEFAULT_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing shared config: {DEFAULT_CONFIG_PATH}")
    raw = _load_yaml(DEFAULT_CONFIG_PATH)
    course_path = course_yaml_path(course)
    if course_path.is_file():
        raw = _deep_merge(raw, _load_yaml(course_path))
    if extra_path:
        path = Path(extra_path)
        if path.suffix in {".yaml", ".yml"}:
            extra = _load_yaml(path)
        else:
            with open(path, encoding="utf-8") as f:
                extra = json.load(f)
        raw = _deep_merge(raw, extra)
    return raw


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


def _attach_dirs(cfg: SimpleNamespace) -> SimpleNamespace:
    def _prod_dir(base: Path | None = None) -> Path:
        if base is not None:
            return Path(base)
        return prod_dir(cfg.course)

    def _experiments_dir(base: Path | None = None) -> Path:
        if base is not None:
            return Path(base)
        return experiments_dir(cfg.course)

    cfg.prod_dir = _prod_dir
    cfg.exp_dir = _prod_dir  # backward-compatible alias
    cfg.experiments_dir = _experiments_dir
    return cfg


def _ns_values(value) -> dict:
    if isinstance(value, SimpleNamespace):
        return {
            k: _ns_values(v)
            for k, v in vars(value).items()
            if k not in ("exp_dir", "prod_dir", "experiments_dir")
        }
    return value


def _build_config(raw: dict, *, course: str) -> SimpleNamespace:
    merged = dict(raw)
    merged["course"] = course
    return _attach_dirs(_to_ns(merged))


def load_config(course: str = DEFAULT_COURSE) -> SimpleNamespace:
    """Merge ``config/default.yaml`` with optional ``<course>/course.yaml``."""
    return _build_config(_merged_dict(course), course=course)


def resolve_config(course: str, config_path: str = "") -> SimpleNamespace:
    """Load merged course config, optionally layering a ``--config`` override file."""
    raw = _merged_dict(course, extra_path=config_path or None)
    return _build_config(raw, course=course)


def _section_defaults(section: str) -> dict:
    return dict(_load_yaml(DEFAULT_CONFIG_PATH).get(section, {}))


def ValidationConfig(**kwargs) -> SimpleNamespace:
    base = _section_defaults("model_policy").get("validation", {})
    return _to_ns(_deep_merge(base, kwargs))


def ModelPolicy(**kwargs) -> SimpleNamespace:
    raw = dict(kwargs)
    validation = raw.pop("validation", None)
    merged = _deep_merge(_section_defaults("model_policy"), raw)
    if validation is not None:
        val_base = merged.get("validation", {})
        merged["validation"] = _deep_merge(val_base, _ns_values(validation))
    return _to_ns(merged)


def EvaluationConfig(**kwargs) -> SimpleNamespace:
    return _to_ns(_deep_merge(_section_defaults("evaluation"), kwargs))


def MonitoringConfig(**kwargs) -> SimpleNamespace:
    return _to_ns(_deep_merge(_section_defaults("monitoring"), kwargs))


def solver_time_limit(config: SimpleNamespace) -> int:
    """Gurobi TimeLimit (seconds) for MILP solves."""
    return getattr(config, "solver_time_limit", DEFAULT_SOLVER_TIME_LIMIT)


def CampaignOptConfig(course: str = DEFAULT_COURSE, **overrides) -> SimpleNamespace:
    """Build config from YAML defaults plus test/programmatic overrides."""
    raw = _merged_dict(course)
    if overrides:
        raw = _deep_merge(raw, {k: _ns_values(v) for k, v in overrides.items()})
    return _build_config(raw, course=course)
