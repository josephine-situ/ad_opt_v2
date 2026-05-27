"""Observed minimum budget floor for optimizer predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from campaign_opt.backends.milp_core import make_linear_segment_predictor, solve_campaign_milp
from campaign_opt.backends.prediction_gating import budget_big_m_from_bounds, gate_level_expr
from campaign_opt.decisions import observed_min_daily_budget
from campaign_opt.optimizer_prediction import apply_observed_budget_floor
from campaign_opt.schema import CampaignOptConfig, EvaluationConfig

pytest.importorskip("gurobipy")
import gurobipy as gp
from gurobipy import GRB


def test_observed_min_daily_budget():
    panel = pd.DataFrame(
        {
            "segment": ["A / Broad", "A / Broad", "B / Broad"],
            "daily_budget": [20.0, 13.5, 50.0],
        }
    )
    mins = observed_min_daily_budget(panel, ["A / Broad", "B / Broad"])
    assert mins["A / Broad"] == pytest.approx(13.5)
    assert mins["B / Broad"] == pytest.approx(50.0)


def test_apply_observed_budget_floor_numpy():
    levels = np.array([5.0, 3.0, 1.0])
    budgets = np.array([10.0, 12.0, 14.0])
    segments = np.array(["A / Broad", "A / Broad", "B / Broad"])
    mins = {"A / Broad": 13.5, "B / Broad": 50.0}
    out = apply_observed_budget_floor(levels, budgets, segments, mins)
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[2] == 0.0


def test_gate_level_expr_at_boundary():
    for budget, expect in ((25.0, 0.0), (35.0, 10.0 + 2.0 * 35.0)):
        model = gp.Model("gate_test")
        model.setParam("OutputFlag", 0)
        x = model.addVar(lb=budget, ub=budget, name="x")
        raw = 10.0 + 2.0 * x
        gated = gate_level_expr(
            model,
            raw,
            x,
            budget_min=30.0,
            level_ub=500.0,
            budget_big_m=budget_big_m_from_bounds(0, 100),
            name_prefix=f"t_{int(budget)}",
        )
        model.setObjective(gated, GRB.MAXIMIZE)
        model.optimize()
        assert float(gated.X) == pytest.approx(expect, rel=1e-5)


def test_linear_milp_levels_objective_with_floor(tmp_path: Path):
    seg_a = "USA / Broad"
    seg_b = "A / Broad"
    config = CampaignOptConfig(
        exp_name="t",
        course="c",
        target="clicks",
        constraints={"budget_tiebreak_penalty": 0.0},
        evaluation=EvaluationConfig(
            objective="levels",
            apply_observed_budget_floor=True,
            baseline_budget=0.0,
        ),
    )
    candidates = pd.DataFrame(
        {"segment": [seg_a, seg_b], "keyword_set_id": ["set_a", "set_b"]}
    )
    panel = pd.DataFrame(
        {
            "segment": [seg_a, seg_b],
            "daily_budget": [50.0, 40.0],
            "clicks": [100.0, 80.0],
        }
    )
    coeffs = {
        "segment_intercept": {seg_a: 0.0, seg_b: 0.0},
        "segment_budget_slope": {seg_a: 0.1, seg_b: 10.0},
        "static_context_lift": {"set_a": 100.0, "set_b": 0.0},
        "calendar_offset": 0.0,
    }
    predictor = make_linear_segment_predictor(coeffs)
    plan = solve_campaign_milp(
        config,
        candidates,
        panel,
        predictor,
        total_budget=20.0,
        output_dir=tmp_path,
        model_name="levels_floor_test",
        time_limit=30,
        write_outputs=True,
        solver_coeffs=coeffs,
    )
    status = json.loads((tmp_path / "solver_status.json").read_text(encoding="utf-8"))
    level_total = float(pd.to_numeric(plan["milp_pred"], errors="coerce").sum())
    assert level_total == pytest.approx(status["obj_val"], rel=1e-4)
    usa_budget = float(plan.loc[plan["segment"] == seg_a, "daily_budget"].iloc[0])
    usa_pred = float(plan.loc[plan["segment"] == seg_a, "milp_pred"].iloc[0])
    if usa_budget < 50.0:
        assert usa_pred == pytest.approx(0.0, abs=1e-4)
