"""Persist modeling frames and design matrices for debugging."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils.linear_design import build_linear_milp_design_matrix
from utils.training_matrix import prep_xy
from utils.campaign_config import CampaignOptConfig
from utils.campaign_features import get_context_feature_columns


def save_modeling_artifacts(
    out_dir: Path,
    config: CampaignOptConfig,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> dict[str, str]:
    """Write wide modeling frames and design matrices under ``out_dir/features/``."""
    out_dir = Path(out_dir)
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    train_path = feat_dir / "modeling_frame_train.csv"
    holdout_path = feat_dir / "modeling_frame_holdout.csv"
    train.to_csv(train_path, index=False)
    holdout.to_csv(holdout_path, index=False)
    paths["modeling_frame_train"] = str(train_path)
    paths["modeling_frame_holdout"] = str(holdout_path)

    train_design = build_linear_milp_design_matrix(train, config)
    holdout_design = build_linear_milp_design_matrix(
        holdout, config, columns=train_design.x_columns
    )
    linear_train_path = feat_dir / "linear_milp_design_train.csv"
    linear_holdout_path = feat_dir / "linear_milp_design_holdout.csv"
    train_design.X.to_csv(linear_train_path, index=False)
    holdout_design.X.to_csv(linear_holdout_path, index=False)
    paths["linear_milp_design_train"] = str(linear_train_path)
    paths["linear_milp_design_holdout"] = str(linear_holdout_path)

    with open(feat_dir / "linear_milp_design_columns.json", "w", encoding="utf-8") as f:
        json.dump(train_design.x_columns, f, indent=2)
    paths["linear_milp_design_columns"] = str(feat_dir / "linear_milp_design_columns.json")

    feature_cols = get_context_feature_columns(config.context_features)
    if feature_cols and config.target in train.columns:
        ctx_train, _ = prep_xy(train, config.target, feature_cols)
        ctx_holdout, _ = prep_xy(holdout, config.target, feature_cols)
        ctx_train_path = feat_dir / "context_design_train.csv"
        ctx_holdout_path = feat_dir / "context_design_holdout.csv"
        ctx_train.to_csv(ctx_train_path, index=False)
        ctx_holdout.to_csv(ctx_holdout_path, index=False)
        paths["context_design_train"] = str(ctx_train_path)
        paths["context_design_holdout"] = str(ctx_holdout_path)

    with open(feat_dir / "artifact_manifest.json", "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2)
    return paths
