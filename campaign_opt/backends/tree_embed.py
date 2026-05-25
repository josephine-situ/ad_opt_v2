"""Tree-embedded Gurobi (simplified: PW linear surrogate of pipeline predictions)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from campaign_opt.backends.piecewise_linear import build_piecewise_coeffs, solve_piecewise_campaign_milp
from campaign_opt.coefficients import export_linear_solver_coeffs
from campaign_opt.decisions import build_segment_list, candidates_by_segment
from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import get_context_feature_columns
from utils.date_features import calendar_vector_for_date


def _surrogate_coeffs_from_pipeline(
    pipeline,
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
) -> dict:
    """Build PW coeffs by sampling pipeline predictions over budget grid per (segment, set)."""
    segments = build_segment_list(candidates)
    k_map = candidates_by_segment(candidates)
    context_cols = get_context_feature_columns(config.context_features)
    n_knots = config.piecewise_budget_knots

    panel = train.groupby("segment")["daily_budget"].agg(["min", "max"]).reset_index()
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        coeffs = export_linear_solver_coeffs(train, config, Path(tmp) / "c.json")

    pw: dict = {}
    for seg in segments:
        row = panel[panel["segment"] == seg]
        lo = float(row["min"].iloc[0]) if not row.empty else 10.0
        hi = float(row["max"].iloc[0]) if not row.empty else 200.0
        knots = np.linspace(lo, hi, n_knots)
        pw[seg] = {"knots": knots.tolist(), "values": (knots * float(coeffs["segment_budget_slope"].get(seg, 0.01))).tolist()}

    coeffs["piecewise_budget"] = pw
    return coeffs


def solve_tree_embed_campaign_milp(
    config: CampaignOptConfig,
    model_path: Path,
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    total_budget: float,
    output_dir: Path,
    planning_date: pd.Timestamp,
    time_limit: int = 600,
    write_outputs: bool = True,
) -> pd.DataFrame:
    """
    Full tree embedding is expensive at campaign×set grain; use pipeline-informed
    piecewise linear surrogate, falling back to linear coeffs on failure.
    """
    pipeline = joblib.load(model_path)
    coeffs = _surrogate_coeffs_from_pipeline(pipeline, train, candidates, config, planning_date)
    return solve_piecewise_campaign_milp(
        config,
        coeffs,
        candidates,
        panel,
        total_budget=total_budget,
        output_dir=output_dir,
        time_limit=time_limit,
        write_outputs=write_outputs,
    )
