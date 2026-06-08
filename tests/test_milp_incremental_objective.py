"""MILP maximizes incremental lift, not level f(plan)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from utils.backends.milp_core import (
    baseline_levels_from_coeffs,
    make_linear_segment_predictor,
    solve_campaign_milp,
)
from utils.campaign_config import CampaignOptConfig

pytest.importorskip("gurobipy")


def _two_segment_linear_problem() -> tuple[CampaignOptConfig, pd.DataFrame, pd.DataFrame, dict]:
    """
    Two segments (one keyword set each).

    USA: high static lift, low budget slope.
    A: no static lift, high budget slope.
    """
    config = CampaignOptConfig(exp_name="t", course="c", constraints={"budget_tiebreak_penalty": 0.0})
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


def test_baseline_levels_linear_at_zero():
    _, _, _, coeffs = _two_segment_linear_problem()
    segments = ["USA / Broad", "A / Broad"]
    k_map = {"USA / Broad": ["set_a"], "A / Broad": ["set_b"]}
    levels = baseline_levels_from_coeffs(coeffs, segments, k_map, 0.0)
    assert levels[("USA / Broad", "set_a")] == pytest.approx(100.0)
    assert levels[("A / Broad", "set_b")] == pytest.approx(0.0)


def test_linear_milp_objective_is_incremental_lift(tmp_path: Path):
    config, candidates, panel, coeffs = _two_segment_linear_problem()
    predictor = make_linear_segment_predictor(coeffs)
    plan = solve_campaign_milp(
        config,
        candidates,
        panel,
        predictor,
        total_budget=20.0,
        output_dir=tmp_path,
        model_name="campaign_linear_test",
        time_limit=30,
        write_outputs=True,
        solver_coeffs=coeffs,
    )
    status = json.loads((tmp_path / "solver_status.json").read_text(encoding="utf-8"))
    lift_total = float(pd.to_numeric(plan["pred_over_base"], errors="coerce").sum())
    level_total = float(pd.to_numeric(plan["milp_pred"], errors="coerce").sum())
    assert lift_total == pytest.approx(status["obj_val"], rel=1e-5)
    assert level_total > lift_total + 50.0
    assert lift_total == pytest.approx(101.0, rel=1e-3)


def test_linear_milp_objective_is_levels(tmp_path: Path):
    config, candidates, panel, coeffs = _two_segment_linear_problem()
    config.evaluation.objective = "levels"
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
    lift_total = float(pd.to_numeric(plan["pred_over_base"], errors="coerce").sum())
    assert level_total == pytest.approx(status["obj_val"], rel=1e-5)
    assert level_total > lift_total
