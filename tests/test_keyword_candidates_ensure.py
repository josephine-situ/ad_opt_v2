"""Tests for ensure_segment_keyword_candidates freshness checks."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from utils.keyword_candidates import (
    _segment_keyword_candidates_are_current,
    ensure_segment_keyword_candidates,
)


def _install_processed_tree(tmp_path: Path, course: str = "sys_think") -> Path:
    processed = tmp_path / course / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame({"campaign": ["c"], "region": ["USA"], "match_types": ["Broad"]}).to_csv(
        processed / "campaign-summary.csv",
        index=False,
    )
    pd.DataFrame({"keyword_set_id": ["ks_0"], "positive_keywords": ["alpha"]}).to_csv(
        processed / "campaign-keyword-sets.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "region": ["USA"],
            "keyword": ["alpha"],
            "match_type": ["Broad"],
            "clicks": [1],
        }
    ).to_csv(processed / "kw-day-panel.csv", index=False)
    return processed


def _write_candidate_outputs(processed: Path, *, mtime: float) -> None:
    pd.DataFrame(
        [{"segment": "USA / Broad", "keyword_set_id": "ks_0", "source": "historical"}]
    ).to_csv(processed / "segment-keyword-candidates.csv", index=False)
    pd.DataFrame(
        [{"keyword_set_id": "ks_0", "positive_keywords": "alpha", "broad_keywords": "alpha"}]
    ).to_csv(processed / "campaign-keyword-sets-extended.csv", index=False)
    for path in (
        processed / "segment-keyword-candidates.csv",
        processed / "campaign-keyword-sets-extended.csv",
    ):
        os.utime(path, (mtime, mtime))


@pytest.fixture
def isolated_course_paths(monkeypatch, tmp_path):
    course = "sys_think"
    processed = _install_processed_tree(tmp_path, course)
    allowlist = tmp_path / course / "data" / "gkp" / "Keywords Enrollments.xlsx"
    allowlist.parent.mkdir(parents=True)
    allowlist.write_bytes(b"placeholder")

    def _data_dir(course_name: str = "sys_think") -> Path:
        return tmp_path / course_name / "data"

    monkeypatch.setattr("utils.paths.data_dir", _data_dir)
    monkeypatch.setattr(
        "utils.paths.require_enrollment_allowlist",
        lambda _course="sys_think": allowlist,
    )
    monkeypatch.setattr(
        "utils.paths.enrollment_allowlist_path",
        lambda _course="sys_think": allowlist,
    )
    return course, processed, allowlist


def test_segment_keyword_candidates_are_current_when_outputs_are_newer(isolated_course_paths):
    course, processed, _ = isolated_course_paths
    _write_candidate_outputs(processed, mtime=2_000_000_000.0)

    assert _segment_keyword_candidates_are_current(course)


def test_segment_keyword_candidates_are_stale_when_input_is_newer(isolated_course_paths):
    course, processed, _ = isolated_course_paths
    _write_candidate_outputs(processed, mtime=2_000_000_000.0)
    os.utime(processed / "kw-day-panel.csv", (2_000_000_100.0, 2_000_000_100.0))

    assert not _segment_keyword_candidates_are_current(course)


def test_ensure_reuses_existing_candidates_without_rebuild(isolated_course_paths, monkeypatch):
    course, processed, _ = isolated_course_paths
    _write_candidate_outputs(processed, mtime=2_000_000_000.0)

    def _fail_build(*_args, **_kwargs):
        raise AssertionError("build_segment_candidates should not run when outputs are current")

    monkeypatch.setattr("utils.keyword_candidates.build_segment_candidates", _fail_build)

    path = ensure_segment_keyword_candidates(course)
    assert path == processed / "segment-keyword-candidates.csv"
