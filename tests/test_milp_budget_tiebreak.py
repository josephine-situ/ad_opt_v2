"""MILP budget tie-break penalty config."""

from __future__ import annotations

import pytest

from utils.backends.milp_core import _budget_tiebreak_penalty
from utils.campaign_config import CampaignOptConfig


def test_budget_tiebreak_penalty_default():
    config = CampaignOptConfig(exp_name="t", course="c")
    assert _budget_tiebreak_penalty(config) == 0.0001


def test_budget_tiebreak_penalty_override():
    config = CampaignOptConfig(
        exp_name="t",
        course="c",
        constraints={"budget_tiebreak_penalty": 0.0},
    )
    assert _budget_tiebreak_penalty(config) == 0.0


def test_budget_tiebreak_penalty_rejects_negative():
    config = CampaignOptConfig(
        exp_name="t",
        course="c",
        constraints={"budget_tiebreak_penalty": -1.0},
    )
    with pytest.raises(ValueError, match="non-negative"):
        _budget_tiebreak_penalty(config)
