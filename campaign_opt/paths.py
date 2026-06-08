"""Central path resolution for the sys_think course bundle."""

from __future__ import annotations

from pathlib import Path

COURSE = "sys_think"
REPO_ROOT = Path(__file__).resolve().parents[1]
COURSE_ROOT = REPO_ROOT / COURSE
DATA_DIR = COURSE_ROOT / "data"
OPT_RESULTS_DIR = COURSE_ROOT / "opt_results"
GKP_DIR = DATA_DIR / "gkp"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"

_ENROLLMENT_FILE_GLOB = "*Keywords*Enrollments*.xlsx"


def data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def opt_results_path(*parts: str) -> Path:
    return OPT_RESULTS_DIR.joinpath(*parts)


def campaign_config_path(exp_name: str = "default") -> Path:
    return opt_results_path("campaign", exp_name, "campaign_config.json")


def exp_dir(exp_name: str = "default") -> Path:
    return opt_results_path("campaign", exp_name)


def backtest_window_dir(exp_name: str, start: str, end: str) -> Path:
    return exp_dir(exp_name) / "backtest" / f"{start}_{end}"


def enrollment_allowlist_path() -> Path | None:
    """Return newest enrollment allowlist xlsx, or None if missing."""
    if not GKP_DIR.is_dir():
        return None
    matches = sorted(GKP_DIR.glob(_ENROLLMENT_FILE_GLOB), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def require_enrollment_allowlist() -> Path:
    """Return enrollment allowlist path or raise with setup instructions."""
    path = enrollment_allowlist_path()
    if path is None:
        raise FileNotFoundError(
            f"Required enrollment allowlist not found under {GKP_DIR}. "
            f"Add a file matching {_ENROLLMENT_FILE_GLOB!r} before building keyword candidates."
        )
    return path
