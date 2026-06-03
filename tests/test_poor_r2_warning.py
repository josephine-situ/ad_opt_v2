"""Tests for poor R² warnings."""

from __future__ import annotations

from campaign_opt.modeling import POOR_R2_THRESHOLD, warn_if_poor_r2


def test_warn_if_poor_r2_skips_good_and_nan(capsys):
    warn_if_poor_r2(0.5, scope="holdout")
    warn_if_poor_r2(float("nan"), scope="holdout")
    assert capsys.readouterr().out == ""


def test_warn_if_poor_r2_prints_for_low_r2(capsys):
    warn_if_poor_r2(0.1, scope="CV", label="ridge")
    out = capsys.readouterr().out
    assert "[Warn]" in out
    assert "Poor CV R²=0.1000" in out
    assert f"< {POOR_R2_THRESHOLD}" in out
    assert "(ridge)" in out


def test_warn_if_poor_r2_respects_custom_threshold(capsys):
    warn_if_poor_r2(0.4, scope="holdout", threshold=0.5)
    out = capsys.readouterr().out
    assert "[Warn]" in out
    assert "< 0.5" in out
