"""Dispatch optimization backend from manifest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from campaign_opt.backends.linear import solve_linear_campaign_milp
from campaign_opt.backends.piecewise_linear import solve_piecewise_campaign_milp
from campaign_opt.backends.tree_embed import solve_tree_embed_campaign_milp
from campaign_opt.coefficients import export_linear_solver_coeffs
from campaign_opt.schema import CampaignOptConfig


def run_optimizer(
    config: CampaignOptConfig,
    manifest: dict,
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    total_budget: float,
    output_dir: Path,
    model_path: Path | None = None,
    planning_date: pd.Timestamp | None = None,
    write_outputs: bool = True,
) -> pd.DataFrame:
    """Pick solver backend from manifest and return segment-level plan."""
    output_dir = Path(output_dir)
    coeffs_path = output_dir / "linear_coeffs.json"
    coeffs = export_linear_solver_coeffs(train, config, coeffs_path)

    backend = manifest.get("backend", "linear")
    common = dict(
        config=config,
        coeffs=coeffs,
        candidates=candidates,
        panel=panel,
        total_budget=total_budget,
        output_dir=output_dir,
        write_outputs=write_outputs,
    )

    if backend == "linear":
        return solve_linear_campaign_milp(**common)
    if backend == "piecewise_linear":
        return solve_piecewise_campaign_milp(**common)
    if backend == "tree_embed":
        if model_path is None:
            model_path = output_dir / "winner_model.joblib"
        return solve_tree_embed_campaign_milp(
            config,
            model_path,
            train,
            candidates,
            panel,
            total_budget=total_budget,
            output_dir=output_dir,
            planning_date=planning_date or pd.Timestamp(train["date"].max()),
            write_outputs=write_outputs,
        )
    raise ValueError(f"Unknown backend: {backend}")
