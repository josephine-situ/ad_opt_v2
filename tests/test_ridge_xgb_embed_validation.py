"""Ridge+XGB embed: ensemble baselines and external validation after incremental objective."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from campaign_opt.backends.tree_embed import solve_ridge_xgb_embed_campaign_milp
from campaign_opt.evaluation import baseline_levels_for_candidate_sets
from campaign_opt.features import prepare_modeling_data, train_holdout_split
from campaign_opt.optimize import _fit_and_save_embed_model
from campaign_opt.schema import default_config_path, load_campaign_config
from utils.campaign_features import (
    add_segment_column,
    build_keyword_set_feature_table,
    load_campaign_day_panel,
)
from campaign_opt.decisions import apply_candidate_region_policy, candidates_by_segment

pytest.importorskip("gurobipy")


def test_ridge_xgb_embed_matches_ensemble_with_incremental_objective(tmp_path: Path):
    config_path = default_config_path("sys_think", "default")
    if not config_path.exists():
        pytest.skip("sys_think default config missing")
    config = load_campaign_config(config_path)
    out_dir = config.exp_dir()
    manifest_path = out_dir / "model_manifest.json"
    if not manifest_path.exists():
        pytest.skip("model manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("optimizer_winner") not in (None, "ensemble_ridge_xgb"):
        pytest.skip("manifest winner is not ensemble_ridge_xgb")

    data_root = Path("data") / config.course / "processed"
    candidates_path = data_root / "segment-keyword-candidates.csv"
    if not candidates_path.exists():
        pytest.skip("segment-keyword-candidates.csv missing")

    import pandas as pd

    df = prepare_modeling_data(config)
    train, holdout = train_holdout_split(df, config.model_policy.validation.holdout_days)
    production = (
        pd.concat([train, holdout], ignore_index=True).sort_values("date")
        if len(holdout)
        else train
    )
    panel = add_segment_column(load_campaign_day_panel(config.course))
    candidates = apply_candidate_region_policy(
        pd.read_csv(candidates_path),
        config.constraints,
    )
    model_path = _fit_and_save_embed_model(
        config, manifest, production, tmp_path, tune=False
    )
    planning_date = pd.Timestamp(production["date"].max())
    set_features = build_keyword_set_feature_table(config.course)
    k_map = candidates_by_segment(candidates)
    pipeline = joblib.load(model_path)

    baselines = baseline_levels_for_candidate_sets(
        pipeline, k_map, config, planning_date, set_features
    )
    assert baselines

    total_budget = float(production["daily_budget"].median()) * len(k_map)
    solve_ridge_xgb_embed_campaign_milp(
        config,
        model_path,
        production,
        candidates,
        panel,
        total_budget=total_budget,
        output_dir=tmp_path / "milp",
        planning_date=planning_date,
        time_limit=120,
        write_outputs=True,
    )
