"""Per-course data, production, and backtest path helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_COURSE = "sys_think"
REPO_ROOT = Path(__file__).resolve().parents[1]
_ENROLLMENT_GLOB = "*Keywords*Enrollments*.xlsx"

_NON_COURSE_DIRS = frozenset(
    {
        ".git",
        ".venv",
        ".vscode",
        "__pycache__",
        "config",
        "docs",
        "figures",
        "logs",
        "scripts",
        "tests",
        "utils",
        "campaign_opt",
    }
)


def list_courses() -> list[str]:
    """Course bundles: top-level dirs with a ``data/`` folder."""
    courses = [
        p.name
        for p in REPO_ROOT.iterdir()
        if p.is_dir() and p.name not in _NON_COURSE_DIRS and (p / "data").is_dir()
    ]
    return sorted(courses)


def add_course_arg(parser: argparse.ArgumentParser, *, default: str = DEFAULT_COURSE) -> None:
    parser.add_argument(
        "--course",
        default=default,
        choices=list_courses(),
        help=f"Course key (default: {default})",
    )


def course_root(course: str = DEFAULT_COURSE) -> Path:
    return REPO_ROOT / course


def data_dir(course: str = DEFAULT_COURSE) -> Path:
    return course_root(course) / "data"


def data_path(course: str, *parts: str) -> Path:
    return data_dir(course).joinpath(*parts)


def prod_dir(course: str = DEFAULT_COURSE) -> Path:
    """Production pipeline outputs (model fit, two-stage plans)."""
    return course_root(course) / "prod"


def backtests_dir(course: str = DEFAULT_COURSE) -> Path:
    return course_root(course) / "backtests"


def backtest_window_dir(course: str, start: str, end: str) -> Path:
    return backtests_dir(course) / f"{start}_{end}"


def course_yaml_path(course: str) -> Path:
    return course_root(course) / "course.yaml"


def gkp_dir(course: str = DEFAULT_COURSE) -> Path:
    return data_dir(course) / "gkp"


def processed_dir(course: str = DEFAULT_COURSE) -> Path:
    return data_dir(course) / "processed"


def reports_dir(course: str = DEFAULT_COURSE) -> Path:
    return data_dir(course) / "reports"


def enrollment_allowlist_path(course: str = DEFAULT_COURSE) -> Path | None:
    gkp = gkp_dir(course)
    if not gkp.is_dir():
        return None
    matches = sorted(gkp.glob(_ENROLLMENT_GLOB), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def require_enrollment_allowlist(course: str = DEFAULT_COURSE) -> Path:
    path = enrollment_allowlist_path(course)
    if path is None:
        raise FileNotFoundError(
            f"Required enrollment allowlist not found under {gkp_dir(course)}. "
            f"Add a file matching {_ENROLLMENT_GLOB!r} before building keyword candidates."
        )
    return path


# Backward-compatible aliases (deprecated)
def opt_results_dir(course: str = DEFAULT_COURSE) -> Path:
    return prod_dir(course)


def exp_dir(course: str, exp_name: str = "default") -> Path:
    _ = exp_name
    return prod_dir(course)


def campaign_config_path(course: str, exp_name: str = "default") -> Path:
    _ = exp_name
    return course_yaml_path(course)
