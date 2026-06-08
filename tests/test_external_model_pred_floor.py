"""Post-solve external_model_pred uses the same observed-budget floor as the MILP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.backends.tree_embed import _external_incremental_pred_by_segment
from utils.decisions import observed_min_daily_budget
from utils.evaluation import build_plan_prediction_rows
from utils.campaign_config import CampaignOptConfig, EvaluationConfig


def test_external_pred_zeros_below_observed_min_budget():
    from pathlib import Path

    if not Path("sys_think/data/processed/campaign-summary.csv").exists():
        return
    config = CampaignOptConfig(
        exp_name="t",
        course="sys_think",
        target="clicks",
        evaluation=EvaluationConfig(
            apply_observed_budget_floor=True,
            baseline_budget=0.0,
            budget_floor_atol=0.01,
        ),
    )
    panel = pd.DataFrame(
        {
            "segment": ["USA / Phrase; Exact", "USA / Phrase; Exact"],
            "daily_budget": [200.0, 50.0],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "campaign_version": [1.0, 1.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "segment": ["USA / Phrase; Exact"],
            "keyword_set_id": ["ks_a"],
        }
    )
    set_features = pd.DataFrame(
        {
            "keyword_set_id": ["ks_a"],
            "embed_course_sim_mean": [0.5],
            "num_unique_keywords": [10],
            "last_month_searches_mean": [100.0],
            "competition_index_mean": [0.5],
            "embed_dispersion_broad": [0.1],
            "embed_dispersion_phrase": [0.2],
            "embed_dispersion_exact": [0.3],
        }
    )

    class StubPipeline:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.full(len(X), 1.007, dtype=float)

    plan = pd.DataFrame(
        {
            "segment": ["USA / Phrase; Exact"],
            "daily_budget": [0.0],
            "keyword_set_id": ["ks_a"],
            "milp_pred": [0.0],
        }
    )
    ext = _external_incremental_pred_by_segment(
        plan,
        StubPipeline(),
        panel,
        config,
        pd.Timestamp("2025-01-08"),
        set_features,
        candidates=candidates,
    )
    assert float(ext["external_model_pred"].iloc[0]) == 0.0
    assert float(ext["pred_over_base"].iloc[0]) == 0.0


def test_build_plan_prediction_rows_uses_embed_template():
    config = CampaignOptConfig(exp_name="t", course="sys_think", target="clicks")
    panel = pd.DataFrame(
        {
            "segment": ["B / Broad"],
            "daily_budget": [12.35],
            "date": pd.to_datetime(["2025-01-08"]),
            "campaign_version": [2.0],
        }
    )
    candidates = pd.DataFrame({"segment": ["B / Broad"], "keyword_set_id": ["ks_0002"]})
    set_features = pd.DataFrame(
        {
            "keyword_set_id": ["ks_0002"],
            "embed_course_sim_mean": [0.1],
            "num_unique_keywords": [5],
            "last_month_searches_mean": [50.0],
            "competition_index_mean": [0.4],
            "embed_dispersion_broad": [0.0],
            "embed_dispersion_phrase": [0.0],
            "embed_dispersion_exact": [0.0],
        }
    )
    plan_dec = pd.DataFrame(
        {
            "segment": ["B / Broad"],
            "keyword_set_id": ["ks_0002"],
            "daily_budget": [12.34],
        }
    )
    embed_path = build_plan_prediction_rows(
        plan_dec,
        config,
        pd.Timestamp("2025-01-08"),
        set_features,
        panel,
        candidates=candidates,
    )
    decision_path = build_plan_prediction_rows(
        plan_dec,
        config,
        pd.Timestamp("2025-01-08"),
        set_features,
        panel,
        candidates=None,
    )
    assert float(embed_path["daily_budget"].iloc[0]) == 12.34
    assert "is_broad_match" in embed_path.columns
    mins = observed_min_daily_budget(panel, ["B / Broad"])
    assert mins["B / Broad"] == 12.35
    assert embed_path["days_since_version_start"].notna().all()
    assert decision_path["days_since_version_start"].notna().all()
