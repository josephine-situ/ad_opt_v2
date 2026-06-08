"""Shared pytest fixtures for campaign_opt tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_SYNTHETIC_PANEL_FILES = (
    "campaign-day-panel.csv",
    "campaign-summary.csv",
    "campaign-keyword-sets.csv",
    "kw-day-panel.csv",
)


def install_synthetic_sys_think_data(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_course: Path,
    tmp_path: Path,
    *,
    course: str = "sys_think",
) -> Path:
    """
    Copy synthetic processed CSVs into ``tmp_path`` and redirect ``data_paths``.

    Never writes into the repo's tracked ``sys_think/data/processed`` tree.
    """
    src = synthetic_course / course / "processed"
    dst = tmp_path / course / "data" / "processed"
    dst.mkdir(parents=True, exist_ok=True)
    for name in _SYNTHETIC_PANEL_FILES:
        path = src / name
        if path.exists():
            shutil.copy(path, dst / name)

    gkp_dst = tmp_path / course / "data" / "gkp"
    gkp_dst.mkdir(parents=True, exist_ok=True)

    def _data_paths(course_name: str) -> dict[str, Path]:
        base = tmp_path / course_name / "data"
        return {
            "processed": base / "processed",
            "gkp": base / "gkp",
            "cache": base / "cache",
        }

    def _processed_dir(course_name: str = "sys_think") -> Path:
        return tmp_path / course_name / "data" / "processed"

    def _gkp_dir(course_name: str = "sys_think") -> Path:
        return tmp_path / course_name / "data" / "gkp"

    def _data_dir(course_name: str = "sys_think") -> Path:
        return tmp_path / course_name / "data"

    monkeypatch.setattr("utils.campaign_features.data_paths", _data_paths)
    monkeypatch.setattr("utils.paths.processed_dir", _processed_dir)
    monkeypatch.setattr("utils.paths.gkp_dir", _gkp_dir)
    monkeypatch.setattr("utils.paths.data_dir", _data_dir)
    monkeypatch.setattr(
        "utils.keyword_allowlist.load_enrollment_keyword_allowlist",
        lambda _course="sys_think": None,
    )
    monkeypatch.setattr(
        "utils.keyword_allowlist.load_enrollment_keyword_allowlist_ordered",
        lambda _course="sys_think": None,
    )

    import utils.campaign_features as campaign_features

    campaign_features._version_start_cache.pop(course, None)
    return dst


@pytest.fixture
def synthetic_sys_think_data(monkeypatch, synthetic_course, tmp_path):
    """Isolated synthetic ``sys_think`` panels (does not touch repo ``data/``)."""
    install_synthetic_sys_think_data(monkeypatch, synthetic_course, tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def synthetic_course(tmp_path_factory):
    base = tmp_path_factory.mktemp("data") / "sys_think" / "processed"
    base.mkdir(parents=True)
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    segments = ["USA / Broad", "B / Phrase; Exact"]
    rows = []
    for d in dates:
        for i, seg in enumerate(segments):
            region, mt = seg.split(" / ", 1)
            rows.append(
                {
                    "date": d,
                    "campaign_version": f"v_{seg.replace(' ', '_')}",
                    "region": region,
                    "match_types": mt,
                    "daily_budget": 50.0 + i * 30 + (d.dayofyear % 20),
                    "clicks": max(0, int(5 + i * 3 + d.dayofyear % 7)),
                    "cost": 40.0,
                    "keyword_set_id": f"ks_{i}",
                }
            )
    panel = pd.DataFrame(rows)
    panel.to_csv(base / "campaign-day-panel.csv", index=False)

    summary = panel[
        ["campaign_version", "region", "match_types", "daily_budget", "keyword_set_id"]
    ].drop_duplicates()
    summary["campaign"] = "Test Campaign"
    summary["start_date"] = "2025-01-01"
    summary["end_date"] = ""
    summary.to_csv(base / "campaign-summary.csv", index=False)

    sets = pd.DataFrame(
        {
            "keyword_set_id": ["ks_0", "ks_1"],
            "positive_keywords": ["alpha; beta", "gamma; delta"],
        }
    )
    sets.to_csv(base / "campaign-keyword-sets.csv", index=False)

    kw = pd.DataFrame(
        {
            "date": dates[:30],
            "region": ["USA"] * 30,
            "keyword": ["alpha"] * 30,
            "campaign": ["Test"] * 30,
            "match_type": ["Broad"] * 30,
            "clicks": np.arange(30),
            "cost": np.linspace(1, 30, 30),
        }
    )
    (base.parent / "processed").mkdir(parents=True, exist_ok=True)
    kw.to_csv(base.parent / "processed" / "kw-day-panel.csv", index=False)

    return base.parent.parent
