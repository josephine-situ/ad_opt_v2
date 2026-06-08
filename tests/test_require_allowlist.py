"""Tests for compulsory enrollment allowlist."""

from __future__ import annotations

import pytest

from campaign_opt.paths import GKP_DIR, require_enrollment_allowlist


def test_require_enrollment_allowlist_finds_file():
    path = require_enrollment_allowlist()
    assert path.is_file()
    assert "Enrollments" in path.name or "Keywords" in path.name
    assert path.parent == GKP_DIR


def test_require_enrollment_allowlist_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("campaign_opt.paths.GKP_DIR", tmp_path / "empty_gkp")
    with pytest.raises(FileNotFoundError, match="Required enrollment allowlist"):
        require_enrollment_allowlist()
