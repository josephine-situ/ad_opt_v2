"""Export linear solver coefficients for Gurobi MILP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from campaign_opt.decisions import region_of_segment
from campaign_opt.linear_design import (
    SEGMENT_STRUCTURAL_EXACT,
    SEGMENT_STRUCTURAL_PREFIXES,
    LinearMilpDesign,
    LinearMilpRidgeModel,
    build_linear_milp_design_matrix,
    fit_linear_milp_ridge,
    segment_budget_response_from_artifact,
    segment_intercept_from_model,
    segment_slope_from_model,
    static_context_columns,
    template_row_for_segment,
)
from campaign_opt.schema import CampaignOptConfig
from utils.date_features import calendar_vector_for_date
from utils.shrinkage import shrink_segment_slopes


def extract_context_feature_coefs(model: Ridge, x_columns: list[str]) -> dict[str, float]:
    """Ridge coefficients on context feature columns (excludes segment / budget terms)."""
    return {
        col: float(model.coef_[i])
        for i, col in enumerate(x_columns)
        if col not in SEGMENT_STRUCTURAL_EXACT
        and not any(col.startswith(p) for p in SEGMENT_STRUCTURAL_PREFIXES)
    }


def context_contribution(
    values: pd.Series | dict,
    context_coefs: dict[str, float],
    cols: list[str],
) -> float:
    """Linear contribution from a row of context feature values."""
    effect = 0.0
    for col in cols:
        if isinstance(values, dict):
            val = values.get(col)
        elif col in values.index:
            val = values[col]
        else:
            val = np.nan
        dummy_cols = [c for c in context_coefs if c.startswith(f"{col}_")]
        if dummy_cols:
            for dc in dummy_cols:
                level = dc[len(col) + 1 :]
                if str(val) == level:
                    effect += context_coefs[dc]
            continue
        if col not in context_coefs:
            continue
        if pd.isna(val):
            continue
        try:
            effect += context_coefs[col] * float(val)
        except (TypeError, ValueError):
            continue
    return effect


def static_context_lift_from_features(
    context_coefs: dict[str, float],
    static_cols: list[str],
    set_features: pd.DataFrame,
    set_ids: list[str] | pd.Index,
) -> dict[str, float]:
    """Map keyword_set_id -> static context-feature contribution for MILP set selection."""
    if not context_coefs or not static_cols:
        return {}
    indexed = set_features.set_index("keyword_set_id") if "keyword_set_id" in set_features.columns else set_features
    effects: dict[str, float] = {}
    for set_id in set_ids:
        sid = str(set_id)
        if sid in indexed.index:
            effects[sid] = context_contribution(indexed.loc[sid], context_coefs, static_cols)
        else:
            effects[sid] = 0.0
    return effects


def refresh_static_context_lift(
    coeffs: dict,
    config: CampaignOptConfig,
    candidates: pd.DataFrame,
    set_features: pd.DataFrame,
) -> dict:
    """Recompute per-set static context lifts from context coefs and candidate set features."""
    context_coefs = coeffs.get("context_feature_coefs")
    static_cols = coeffs.get("static_context_columns") or static_context_columns(config)
    if not context_coefs or not static_cols:
        return coeffs
    out = dict(coeffs)
    set_ids = candidates["keyword_set_id"].astype(str).unique()
    out["static_context_lift"] = static_context_lift_from_features(
        context_coefs, static_cols, set_features, set_ids
    )
    return out


def _fit_ridge_design(
    train: pd.DataFrame,
    config: CampaignOptConfig,
) -> tuple[LinearMilpRidgeModel, pd.DataFrame, pd.Series, list[str], list[str]]:
    """Fit ridge on the shared MILP-linear design matrix."""
    design = build_linear_milp_design_matrix(train, config)
    artifact = fit_linear_milp_ridge(design, config, alpha=1.0)
    return artifact, design.X, pd.Series(design.y), design.cal_cols, design.x_columns


def calendar_offset_for_date(
    model: Ridge | LinearMilpRidgeModel,
    X_columns: list[str],
    cal_cols: list[str],
    planning_date: pd.Timestamp,
    region: str,
    course: str,
    *,
    template: pd.Series | None = None,
    config: CampaignOptConfig | None = None,
) -> float:
    """Calendar-only contribution from fitted ridge for one date/region."""
    if isinstance(model, LinearMilpRidgeModel) and model.uses_scaled_fit():
        if template is None or config is None:
            raise ValueError("template and config required for scaled LinearMilpRidgeModel")
        return _calendar_offset_from_artifact(
            model, template, cal_cols, planning_date, region, course, config
        )
    cal = calendar_vector_for_date(planning_date, region, course)
    context_coefs = extract_context_feature_coefs(model, X_columns)
    return context_contribution(cal, context_coefs, cal_cols)


def _calendar_offset_from_artifact(
    artifact: LinearMilpRidgeModel,
    template: pd.Series,
    cal_cols: list[str],
    planning_date: pd.Timestamp,
    region: str,
    course: str,
    config: CampaignOptConfig,
) -> float:
    """Calendar effect via predict (scaled ridge); template supplies segment/static context."""
    cal = calendar_vector_for_date(planning_date, region, course)
    row_cal = template.copy()
    row_cal["daily_budget"] = 0.0
    for col in cal_cols:
        if col in cal:
            row_cal[col] = cal[col]
    row_zero_cal = row_cal.copy()
    for col in cal_cols:
        if col in row_zero_cal.index and pd.api.types.is_numeric_dtype(type(row_zero_cal[col])):
            row_zero_cal[col] = 0.0
        elif col in row_zero_cal.index:
            row_zero_cal[col] = ""
    level_with = float(artifact.predict_design_frame(pd.DataFrame([row_cal]))[0])
    level_without = float(artifact.predict_design_frame(pd.DataFrame([row_zero_cal]))[0])
    return level_with - level_without


def calendar_offset_from_context_coefs(
    context_coefs: dict[str, float],
    cal_cols: list[str],
    planning_date: pd.Timestamp,
    region: str,
    course: str,
) -> float:
    """Calendar contribution from saved context coefficients."""
    cal = calendar_vector_for_date(planning_date, region, course)
    return context_contribution(cal, context_coefs, cal_cols)


def calendar_offsets_for_planning(
    train: pd.DataFrame,
    config: CampaignOptConfig,
    planning_dates: list[pd.Timestamp],
    segments: list[str],
) -> dict[tuple[str, int], float]:
    """Per-segment, per-date-index calendar offsets for multi-day MILP objectives."""
    artifact, _, _, cal_cols, x_columns = _fit_ridge_design(train, config)
    design = build_linear_milp_design_matrix(train, config, columns=x_columns)
    offsets: dict[tuple[str, int], float] = {}
    for seg in segments:
        region = region_of_segment(seg)
        template = template_row_for_segment(design.sub, seg)
        for idx, d in enumerate(planning_dates):
            offsets[(seg, idx)] = calendar_offset_for_date(
                artifact,
                x_columns,
                cal_cols,
                pd.Timestamp(d),
                region,
                config.course,
                template=template,
                config=config,
            )
    return offsets


def _static_lift_from_artifact(
    artifact: LinearMilpRidgeModel,
    sub: pd.DataFrame,
    static_cols: list[str],
) -> dict[str, float]:
    """Per keyword_set_id static context contribution (prediction-based for scaled ridge)."""
    if not static_cols or "keyword_set_id" not in sub.columns:
        return {}
    out: dict[str, float] = {}
    for set_id in sub["keyword_set_id"].dropna().unique():
        row = sub[sub["keyword_set_id"] == set_id].iloc[0]
        row_zero = row.copy()
        for col in static_cols:
            if col in row_zero.index:
                row_zero[col] = 0.0
        level = float(artifact.predict_design_frame(pd.DataFrame([row]))[0])
        level_zero = float(artifact.predict_design_frame(pd.DataFrame([row_zero]))[0])
        out[str(set_id)] = level - level_zero
    return out


def coeffs_from_linear_milp_design(
    model: Ridge | LinearMilpRidgeModel,
    design: LinearMilpDesign,
    config: CampaignOptConfig,
    *,
    shrink_weight: float = 0.5,
    min_budget_levels: int = 3,
    calendar_date: pd.Timestamp | None = None,
    calendar_region: str | None = None,
) -> dict:
    """Extract MILP coefficients from a fitted ridge on ``design``."""
    sub = design.sub
    x_columns = design.x_columns
    artifact = model if isinstance(model, LinearMilpRidgeModel) else LinearMilpRidgeModel(model, x_columns, config)
    ridge = artifact.model

    seg_slopes: dict[str, float] = {}
    seg_intercepts: dict[str, float] = {}

    if artifact.uses_scaled_fit():
        for seg in sub["segment"].unique():
            seg_str = str(seg)
            template = template_row_for_segment(sub, seg_str)
            intercept, slope = segment_budget_response_from_artifact(
                artifact, template, seg_str, budget_lo=0.0, budget_hi=1.0
            )
            seg_intercepts[seg_str] = intercept
            seg_slopes[seg_str] = slope
        global_slope = float(np.mean(list(seg_slopes.values()))) if seg_slopes else 0.0
        context_coefs = extract_context_feature_coefs(ridge, x_columns)
        static_lift = _static_lift_from_artifact(artifact, sub, design.static_cols)
    else:
        context_coefs = extract_context_feature_coefs(ridge, x_columns)
        global_slope = float(ridge.coef_[x_columns.index("daily_budget")])
        for seg in sub["segment"].unique():
            seg_str = str(seg)
            seg_slopes[seg_str] = segment_slope_from_model(ridge, x_columns, seg_str)
            seg_intercepts[seg_str] = segment_intercept_from_model(ridge, x_columns, seg_str)
        static_lift = {}
        if design.static_cols and "keyword_set_id" in sub.columns:
            for set_id in sub["keyword_set_id"].dropna().unique():
                row = sub[sub["keyword_set_id"] == set_id].iloc[0]
                static_lift[str(set_id)] = context_contribution(row, context_coefs, design.static_cols)

    seg_slopes = shrink_segment_slopes(
        pd.Series(seg_slopes),
        global_slope=global_slope,
        min_levels=min_budget_levels,
        weight=shrink_weight,
    ).to_dict()

    if calendar_date is not None and calendar_region is not None:
        if artifact.uses_scaled_fit():
            reg_rows = sub[
                sub["segment"].astype(str).map(region_of_segment) == calendar_region
            ]
            template = reg_rows.iloc[0] if not reg_rows.empty else sub.iloc[0]
            cal_effect = _calendar_offset_from_artifact(
                artifact,
                template,
                design.cal_cols,
                pd.Timestamp(calendar_date),
                calendar_region,
                config.course,
                config,
            )
        else:
            cal_effect = calendar_offset_from_context_coefs(
                context_coefs,
                design.cal_cols,
                pd.Timestamp(calendar_date),
                calendar_region,
                config.course,
            )
    else:
        last = sub.iloc[-1]
        last_region = region_of_segment(str(last["segment"]))
        if "date" in sub.columns:
            last_date = pd.Timestamp(last["date"]) if "date" in last.index else pd.Timestamp(sub["date"].max())
        else:
            last_date = pd.Timestamp("2026-01-01")
        if artifact.uses_scaled_fit():
            cal_effect = _calendar_offset_from_artifact(
                artifact,
                last,
                design.cal_cols,
                last_date,
                last_region,
                config.course,
                config,
            )
        else:
            cal_effect = calendar_offset_from_context_coefs(
                context_coefs, design.cal_cols, last_date, last_region, config.course
            )

    return {
        "segment_intercept": seg_intercepts,
        "segment_budget_slope": seg_slopes,
        "context_feature_coefs": context_coefs,
        "static_context_columns": design.static_cols,
        "calendar_context_columns": design.cal_cols,
        "static_context_lift": static_lift,
        "calendar_offset": cal_effect,
        "global_intercept": float(ridge.intercept_),
    }


def ridge_embed_coeffs(
    artifact: LinearMilpRidgeModel,
    train: pd.DataFrame,
    config: CampaignOptConfig,
    candidates: pd.DataFrame,
    set_features: pd.DataFrame,
    planning_date: pd.Timestamp,
    segments: list[str],
) -> dict:
    """MILP ridge coeffs with per-segment calendar for ``ridge_xgb_embed`` (matches ``predict_design_frame``)."""
    design = build_linear_milp_design_matrix(train, config, columns=artifact.x_columns)
    # No slope shrinkage: coeffs must match ``predict_design_frame`` on the fitted ridge.
    coeffs = coeffs_from_linear_milp_design(
        artifact, design, config, shrink_weight=0.0, min_budget_levels=1
    )
    cal_cols = coeffs["calendar_context_columns"]
    coeffs["calendar_offset_by_segment"] = {}
    for seg in segments:
        template = template_row_for_segment(design.sub, seg)
        coeffs["calendar_offset_by_segment"][str(seg)] = calendar_offset_for_date(
            artifact,
            artifact.x_columns,
            cal_cols,
            pd.Timestamp(planning_date),
            region_of_segment(seg),
            config.course,
            template=template,
            config=config,
        )
    return refresh_static_context_lift(coeffs, config, candidates, set_features)


def export_linear_solver_coeffs(
    train: pd.DataFrame,
    config: CampaignOptConfig,
    output_path: Path,
    *,
    shrink_weight: float = 0.5,
    min_budget_levels: int = 3,
    prefit_model: Ridge | LinearMilpRidgeModel | None = None,
    prefit_design: LinearMilpDesign | None = None,
) -> dict:
    """
    Fit (or reuse) interpretable linear model for MILP:
      y ~ region + match_type + daily_budget×(region + match) + context_features
    """
    target = config.target
    if target not in train.columns:
        raise ValueError(f"Target column {target} missing from training frame")

    if prefit_model is not None and prefit_design is not None:
        model, design = prefit_model, prefit_design
    else:
        design = build_linear_milp_design_matrix(train, config)
        model = fit_linear_milp_ridge(design, config, alpha=1.0)

    coeffs = coeffs_from_linear_milp_design(
        model,
        design,
        config,
        shrink_weight=shrink_weight,
        min_budget_levels=min_budget_levels,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coeffs, f, indent=2)
    return coeffs


def load_linear_solver_coeffs(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
