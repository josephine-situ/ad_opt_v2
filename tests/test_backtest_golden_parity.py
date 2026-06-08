"""Golden snapshot parity for the 2026-05-12 → 2026-05-25 two-stage backtest window."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

GOLDEN_DIR = Path("tests/fixtures/backtest_golden/2026-05-12_2026-05-25")
LIVE_DIR = Path("sys_think/backtests/2026-05-12_2026-05-25")

NUMERIC_TOL = 1e-4


def _load_csv_pair(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    golden = pd.read_csv(GOLDEN_DIR / name)
    live = pd.read_csv(LIVE_DIR / name)
    return golden, live


@pytest.mark.parametrize(
    "filename",
    ["daily_backtest_summary.csv", "backtest_summary.csv", "evaluation_results.csv"],
)
def test_golden_csv_matches_live(filename: str):
    if not (LIVE_DIR / filename).is_file():
        pytest.skip(f"Live backtest output missing: {LIVE_DIR / filename}")
    golden, live = _load_csv_pair(filename)
    assert list(golden.columns) == list(live.columns)
    assert len(golden) == len(live)
    for col in golden.columns:
        if pd.api.types.is_numeric_dtype(golden[col]):
            pd.testing.assert_series_equal(
                golden[col],
                live[col],
                check_names=False,
                rtol=NUMERIC_TOL,
                atol=NUMERIC_TOL,
            )
        else:
            assert golden[col].tolist() == live[col].tolist()


def test_golden_fixed_keyword_sets_match_live():
    if not (LIVE_DIR / "fixed_keyword_sets.json").is_file():
        pytest.skip("Live fixed_keyword_sets.json missing")
    with open(GOLDEN_DIR / "fixed_keyword_sets.json", encoding="utf-8") as f:
        golden = json.load(f)
    with open(LIVE_DIR / "fixed_keyword_sets.json", encoding="utf-8") as f:
        live = json.load(f)
    assert golden == live
