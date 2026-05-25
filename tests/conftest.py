"""Shared pytest fixtures for campaign_opt tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def copy_synthetic_to_repo(synthetic_course: Path, root: Path) -> None:
    src = synthetic_course / "sys_think" / "processed"
    dst = root / "data" / "sys_think" / "processed"
    dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "campaign-day-panel.csv",
        "campaign-summary.csv",
        "campaign-keyword-sets.csv",
        "kw-day-panel.csv",
    ):
        p = src / name
        if p.exists():
            shutil.copy(p, dst / name)


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
