"""Shared inputs for planning, backtest, and optimization."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from campaign_opt.decisions import apply_candidate_region_policy, parse_allowed_match_types, parse_excluded_regions
from campaign_opt.features import prepare_modeling_data
from campaign_opt.optimize import require_optimizer_winner
from campaign_opt.schema import CampaignOptConfig
from utils.campaign_features import add_segment_column, load_campaign_day_panel
from utils.keyword_candidates import ensure_segment_keyword_candidates


def load_fit_manifest(config: CampaignOptConfig) -> dict:
    """Alias for :func:`optimizer_manifest_for_backtest`."""
    return optimizer_manifest_for_backtest(config)


def optimizer_manifest_for_backtest(config: CampaignOptConfig) -> dict:
    """Load ``model_manifest.json``; required for backtest and optimizer metadata."""
    require_optimizer_winner(config)
    path = config.exp_dir() / "model_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run fit-models before backtest."
        )
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    if not manifest.get("feature_cols"):
        raise ValueError(f"{path} missing feature_cols")
    return manifest


def load_planning_inputs(config: CampaignOptConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load modeling frame, campaign-day panel, and keyword-set candidates."""
    df = prepare_modeling_data(config)
    if config.target not in df.columns or df[config.target].isna().all():
        print(f"[Warn] target={config.target} missing; using clicks")
        config.target = "clicks"

    panel = add_segment_column(load_campaign_day_panel(config.course))
    allowed_match_types = parse_allowed_match_types(config.constraints)
    excluded_regions = parse_excluded_regions(config.constraints)
    cand_path = ensure_segment_keyword_candidates(
        config.course,
        allowed_match_types=allowed_match_types,
        excluded_regions=excluded_regions or None,
    )
    candidates = apply_candidate_region_policy(pd.read_csv(cand_path), config.constraints)
    return df, panel, candidates
