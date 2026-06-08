"""MILP maximizes total predicted level."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from utils.backends.milp_core import make_linear_segment_predictor, solve_campaign_milp
from utils.campaign_config import CampaignOptConfig, EvaluationConfig

pytest.importorskip("gurobipy")


def _two_segment_linear_problem() -> tuple[CampaignOptConfig, pd.DataFrame, pd.DataFrame, dict]:
    config = CampaignOptConfig(
        course="sys_think",
        constraints={"budget_tiebreak_penalty": 0.0},
        evaluation=EvaluationConfig(apply_observed_budget_floor=False),
    )
    seg_a = "USA / Broad"
    seg_b = "A / Broad"
    candidates = pd.DataFrame(
        {
            "segment": [seg_a, seg_b],
            "keyword_set_id": ["set_a", "set_b"],
        }
    )
    panel = pd.DataFrame(
        {
            "segment": [seg_a, seg_b],
            "daily_budget": [10.0, 10.0],
        }
    )
    coeffs = {
        "segment_intercept": {seg_a: 0.0, seg_b: 0.0},
        "segment_budget_slope": {seg_a: 0.1, seg_b: 10.0},
        "static_context_lift": {"set_a": 100.0, "set_b": 0.0},
        "calendar_offset": 0.0,
    }
    return config, candidates, panel, coeffs


def test_linear_milp_objective_is_levels(tmp_path: Path):
    config, candidates, panel, coeffs = _two_segment_linear_problem()
    predictor = make_linear_segment_predictor(coeffs)
    plan = solve_campaign_milp(
        config,
        candidates,
        panel,
        predictor,
        total_budget=20.0,
        output_dir=tmp_path,
        model_name="campaign_linear_levels",
        time_limit=30,
        write_outputs=True,
        solver_coeffs=coeffs,
    )
    status = json.loads((tmp_path / "solver_status.json").read_text(encoding="utf-8"))
    level_total = float(pd.to_numeric(plan["milp_pred"], errors="coerce").sum())
    assert level_total == pytest.approx(status["obj_val"], rel=1e-5)
    assert status["objective"] == "levels"
