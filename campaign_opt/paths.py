"""Central path resolution for per-course data and experiment bundles."""

from __future__ import annotations

from pathlib import Path

DEFAULT_COURSE = "sys_think"
REPO_ROOT = Path(__file__).resolve().parents[1]
_ENROLLMENT_FILE_GLOB = "*Keywords*Enrollments*.xlsx"


def validate_course(course: str) -> str:
    from config import COURSE_CONFIG

    if course not in COURSE_CONFIG:
        raise ValueError(
            f"Unknown course {course!r}. Configured courses: {sorted(COURSE_CONFIG)}"
        )
    return course


def course_root(course: str = DEFAULT_COURSE) -> Path:
    return REPO_ROOT / validate_course(course)


def data_dir(course: str = DEFAULT_COURSE) -> Path:
    return course_root(course) / "data"


def data_path(course: str, *parts: str) -> Path:
    return data_dir(course).joinpath(*parts)


def opt_results_dir(course: str = DEFAULT_COURSE) -> Path:
    return course_root(course) / "opt_results"


def opt_results_path(course: str, *parts: str) -> Path:
    return opt_results_dir(course).joinpath(*parts)


def gkp_dir(course: str = DEFAULT_COURSE) -> Path:
    return data_dir(course) / "gkp"


def processed_dir(course: str = DEFAULT_COURSE) -> Path:
    return data_dir(course) / "processed"


def reports_dir(course: str = DEFAULT_COURSE) -> Path:
    return data_dir(course) / "reports"


def campaign_config_path(course: str, exp_name: str = "default") -> Path:
    return opt_results_path(course, "campaign", exp_name, "campaign_config.json")


def exp_dir(course: str, exp_name: str = "default") -> Path:
    return opt_results_path(course, "campaign", exp_name)


def backtest_window_dir(course: str, exp_name: str, start: str, end: str) -> Path:
    return exp_dir(course, exp_name) / "backtest" / f"{start}_{end}"


def enrollment_allowlist_path(course: str = DEFAULT_COURSE) -> Path | None:
    """Return newest enrollment allowlist xlsx, or None if missing."""
    gkp = gkp_dir(course)
    if not gkp.is_dir():
        return None
    matches = sorted(gkp.glob(_ENROLLMENT_FILE_GLOB), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def require_enrollment_allowlist(course: str = DEFAULT_COURSE) -> Path:
    """Return enrollment allowlist path or raise with setup instructions."""
    path = enrollment_allowlist_path(course)
    if path is None:
        raise FileNotFoundError(
            f"Required enrollment allowlist not found under {gkp_dir(course)}. "
            f"Add a file matching {_ENROLLMENT_FILE_GLOB!r} before building keyword candidates."
        )
    return path
