"""Feature matrix construction and planning/backtest input loading."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from utils.campaign_config import CampaignOptConfig
from utils.campaign_features import (
    add_segment_column,
    build_modeling_frame,
    get_context_feature_columns,
    load_campaign_day_panel,
    merge_match_type_set_features,
)
from utils.decisions import (
    apply_candidate_region_policy,
    parse_allowed_match_types,
    parse_excluded_regions,
    region_of_segment,
)
from utils.keyword_candidates import ensure_segment_keyword_candidates

def filter_training_scope(df: pd.DataFrame, config: CampaignOptConfig) -> pd.DataFrame:
    """
    Keep rows in optimization scope: ``allowed_match_types`` and non-``excluded_regions``.
    """
    out = df.copy()
    if "segment" not in out.columns:
        return out

    excluded = parse_excluded_regions(config.constraints)
    if excluded:
        regions = out["segment"].map(region_of_segment)
        out = out[~regions.isin(excluded)]

    allowed = parse_allowed_match_types(config.constraints)
    if allowed:
        if "match_types" in out.columns:
            mt = out["match_types"].astype(str)
        else:
            mt = out["segment"].astype(str).str.split(" / ", n=1).str[1]
        out = out[mt.isin(allowed)]

    return out.reset_index(drop=True)


def filter_modeling_lookback(
    df: pd.DataFrame,
    lookback_days: int | None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Keep rows with ``date_col`` in the last ``lookback_days`` through panel max date."""
    if not lookback_days or lookback_days <= 0:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    max_date = out[date_col].max()
    cutoff = max_date - pd.Timedelta(days=int(lookback_days))
    return out[out[date_col] >= cutoff].copy()


def prepare_modeling_data(config: SimpleNamespace | str) -> pd.DataFrame:
    if isinstance(config, str):
        course = config
        target = "all_conv"
        context_features: dict[str, list[str]] = {}
        lookback_days = None
    else:
        course = config.course
        target = config.target
        context_features = config.context_features
        lookback_days = config.modeling_lookback_days

    df = build_modeling_frame(course, target_col=target)
    if isinstance(config, SimpleNamespace) and config.context_features.get("match_type_set"):
        df = merge_match_type_set_features(df, course)
    context_cols = get_context_feature_columns(context_features) if context_features else []
    if context_cols:
        for col in context_cols:
            if col not in df.columns:
                df[col] = pd.NA
    df = df.dropna(subset=["daily_budget", "segment"])
    if isinstance(config, SimpleNamespace):
        df = filter_training_scope(df, config)
    return filter_modeling_lookback(df, lookback_days)


def train_holdout_split(
    df: pd.DataFrame,
    holdout_days: int,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split last ``holdout_days`` for static evaluation (fit_response_models)."""
    df = df.sort_values(date_col)
    cutoff = df[date_col].max() - pd.Timedelta(days=holdout_days)
    train = df[df[date_col] <= cutoff].copy()
    holdout = df[df[date_col] > cutoff].copy()
    return train, holdout


def train_before_date(
    df: pd.DataFrame,
    before: pd.Timestamp,
    date_col: str = "date",
) -> pd.DataFrame:
    """Training rows strictly before ``before`` (for walk-forward backtest)."""
    before = pd.Timestamp(before)
    return df[pd.to_datetime(df[date_col]) < before].copy()


def load_fit_manifest(config: CampaignOptConfig) -> dict:
    return optimizer_manifest_for_backtest(config)


def optimizer_manifest_for_backtest(config: CampaignOptConfig) -> dict:
    from utils.optimize import require_optimizer_winner

    require_optimizer_winner(config)
    path = config.exp_dir() / "model_manifest.json"
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest


def load_planning_inputs(config: CampaignOptConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
