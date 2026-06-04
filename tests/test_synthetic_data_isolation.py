"""Ensure synthetic test setup never overwrites tracked repo panels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.conftest import install_synthetic_sys_think_data


def test_install_synthetic_data_does_not_touch_repo_panels(
    monkeypatch, synthetic_course, tmp_path
):
    repo_panel = Path("data/sys_think/processed/campaign-day-panel.csv")
    if not repo_panel.exists():
        import pytest

        pytest.skip("tracked panel missing")

    before = repo_panel.read_bytes()
    before_rows = len(pd.read_csv(repo_panel))

    install_synthetic_sys_think_data(monkeypatch, synthetic_course, tmp_path)

    assert repo_panel.read_bytes() == before
    assert len(pd.read_csv(repo_panel)) == before_rows
    assert len(pd.read_csv(tmp_path / "data/sys_think/processed/campaign-day-panel.csv")) == 240
