"""Linear MILP backend (thin wrapper over milp_core)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from campaign_opt.backends.milp_core import make_linear_segment_predictor, solve_campaign_milp
from campaign_opt.schema import CampaignOptConfig


def solve_linear_campaign_milp(
    config: CampaignOptConfig,
    coeffs: dict[str, Any],
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    total_budget: float,
    output_dir: Path,
    time_limit: int = 600,
    write_outputs: bool = True,
) -> pd.DataFrame:
    predictor = make_linear_segment_predictor(coeffs)
    return solve_campaign_milp(
        config,
        candidates,
        panel,
        predictor,
        total_budget=total_budget,
        output_dir=output_dir,
        model_name="campaign_linear",
        time_limit=time_limit,
        write_outputs=write_outputs,
    )
