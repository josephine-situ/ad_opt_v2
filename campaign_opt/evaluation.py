"""Ensemble predictions and incremental evaluation f(decision) - f(0)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from campaign_opt.decisions import region_of_segment
from campaign_opt.modeling import _build_preprocessor, _prep_xy, pipeline_feature_overview_lines
from campaign_opt.schema import CampaignOptConfig
from campaign_opt.train_specs import TrainSpec, get_train_spec
from utils.campaign_features import build_keyword_set_feature_table, get_context_feature_columns
from utils.date_features import calendar_vector_for_date


@dataclass
class FittedMember:
    name: str
    pipeline: Any
    spec: TrainSpec
    weight: float = 1.0


@dataclass
class EnsembleModel:
    """Average of level predictions across fitted members."""

    members: list[FittedMember]
    feature_cols: list[str]
    target: str
    baseline_budget: float = 0.0

    def predict_levels(self, rows: pd.DataFrame) -> np.ndarray:
        """Full level prediction f(decision context) per row."""
        if not self.members:
            raise RuntimeError("Ensemble has no fitted members")
        preds = np.zeros(len(rows))
        total_w = sum(m.weight for m in self.members)
        for member in self.members:
            preds += member.weight * _predict_member_levels(member, rows, self.target, self.feature_cols)
        return np.clip(preds / total_w, 0, None)

    def predict_incremental(self, decision_rows: pd.DataFrame, baseline_rows: pd.DataFrame) -> np.ndarray:
        """f(decision) - f(0) per row; removes intercept/calendar level inflation."""
        f_dec = self.predict_levels(decision_rows)
        f_zero = self.predict_levels(baseline_rows)
        return np.clip(f_dec - f_zero, 0, None)


def _predict_member_levels(
    member: FittedMember,
    rows: pd.DataFrame,
    target: str,
    feature_cols: list[str],
) -> np.ndarray:
    df = rows.copy()
    spec = member.spec
    y_name = spec.fit_y_col or target
    if y_name not in df.columns:
        df[y_name] = 0.0
    if target not in df.columns:
        df[target] = 0.0
    if spec.transform is not None:
        df = spec.transform(df, target)
    if hasattr(member.pipeline, "predict_design_frame"):
        pred = member.pipeline.predict_design_frame(df)
    else:
        X, _ = _prep_xy(df, target, feature_cols, y_col=spec.fit_y_col)
        if spec.budget_col != "daily_budget" and spec.budget_col in df.columns:
            X = X.rename(columns={spec.budget_col: "daily_budget"})
        pred = member.pipeline.predict(X)
    if spec.inverse_pred is not None:
        pred = spec.inverse_pred(pred)
    return pred


def fit_member_on_train(
    spec: TrainSpec,
    train: pd.DataFrame,
    config: CampaignOptConfig,
    feature_cols: list[str],
    *,
    hyperparams: dict[str, Any] | None = None,
) -> FittedMember:
    """Fit one candidate on all training rows (no holdout split)."""
    from sklearn.pipeline import Pipeline

    if spec.name == "ridge":
        from campaign_opt.modeling import fit_ridge_full

        pipe = fit_ridge_full(train, config, hyperparams=hyperparams)
        return FittedMember(spec.name, pipe, spec)

    target = config.target
    tr = spec.transform(train, target) if spec.transform else train
    sub = tr.dropna(subset=[target, "daily_budget", "segment"])
    X, y = _prep_xy(sub, target, feature_cols, y_col=spec.fit_y_col)
    if spec.budget_col != "daily_budget":
        X = X.rename(columns={spec.budget_col: "daily_budget"})

    from campaign_opt.train_specs import build_estimator

    estimator = build_estimator(spec.name, hyperparams) if hyperparams else spec.estimator
    pipe = Pipeline([("prep", _build_preprocessor(feature_cols, tr)), ("model", estimator)])
    pipe.fit(X, y)
    return FittedMember(spec.name, pipe, spec)


def fit_ensemble(
    train: pd.DataFrame,
    config: CampaignOptConfig,
    *,
    member_weights: dict[str, float] | None = None,
    member_hyperparams: dict[str, dict[str, Any]] | None = None,
) -> EnsembleModel:
    """
    Fit every candidate in ``model_policy`` on all training data.
    Weights default to equal; pass CV-based weights for RMSE-weighted blend.
    """
    feature_cols = get_context_feature_columns(config.context_features)
    members: list[FittedMember] = []

    for name in config.model_policy.candidates:
        hp = (member_hyperparams or {}).get(name)
        spec = get_train_spec(name, hp)
        if spec is None:
            continue
        try:
            member = fit_member_on_train(spec, train, config, feature_cols, hyperparams=hp)
            w = 1.0 if not member_weights else member_weights.get(name, 0.0)
            if w > 0:
                member.weight = w
                members.append(member)
                print(f"    ensemble member fitted: {name} (w={w:.3f})")
                for line in pipeline_feature_overview_lines(member.pipeline):
                    print(line)
        except Exception as exc:
            print(f"    ensemble member skipped {name}: {exc}")

    if not members:
        raise RuntimeError("No ensemble members could be fitted")

    return EnsembleModel(
        members=members,
        feature_cols=feature_cols,
        target=config.target,
        baseline_budget=float(config.evaluation.baseline_budget),
    )


def baseline_keyword_sets(train: pd.DataFrame) -> pd.Series:
    """Reference keyword set per segment (modal on train) for f(0)."""
    return train.groupby("segment")["keyword_set_id"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]
    )


def observed_by_segment(day_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Realized target summed on one calendar day."""
    if day_df.empty:
        return pd.DataFrame(columns=["segment", f"observed_{target}"])
    cols = {f"observed_{target}": (target, "sum"), "observed_clicks": ("clicks", "sum")}
    cols = {k: v for k, v in cols.items() if v[0] in day_df.columns}
    return day_df.groupby("segment", as_index=False).agg(**cols)


def actual_decisions_by_segment(day_df: pd.DataFrame) -> pd.DataFrame:
    """Budget + keyword set actually in market on that day."""
    if day_df.empty:
        return pd.DataFrame(columns=["segment", "daily_budget", "keyword_set_id"])
    return (
        day_df.groupby("segment", as_index=False)
        .agg(
            daily_budget=("daily_budget", "median"),
            keyword_set_id=("keyword_set_id", lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]),
        )
    )


def build_segment_decision_rows(
    decisions: pd.DataFrame,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
    course: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """One feature row per segment for ensemble prediction."""
    rows: list[dict[str, Any]] = []
    set_feats = set_features.set_index("keyword_set_id")

    for _, dec in decisions.iterrows():
        seg = str(dec["segment"])
        region = region_of_segment(seg)
        cal = calendar_vector_for_date(planning_date, region, course)
        row: dict[str, Any] = {
            "segment": seg,
            "daily_budget": float(dec["daily_budget"]),
            "keyword_set_id": str(dec["keyword_set_id"]),
            **cal,
        }
        kid = row["keyword_set_id"]
        if kid in set_feats.index:
            for col in feature_cols:
                if col in set_feats.columns:
                    row[col] = set_feats.loc[kid, col]
        rows.append(row)

    out = pd.DataFrame(rows)
    for col in feature_cols:
        if col not in out.columns:
            out[col] = np.nan
    return out


def build_baseline_rows(
    segments: list[str],
    baseline_sets: pd.Series,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
    course: str,
    feature_cols: list[str],
    baseline_budget: float,
) -> pd.DataFrame:
    """f(0): zero budget + modal reference keyword set per segment."""
    base_dec = pd.DataFrame(
        {
            "segment": segments,
            "daily_budget": baseline_budget,
            "keyword_set_id": [str(baseline_sets.get(s, baseline_sets.iloc[0])) for s in segments],
        }
    )
    return build_segment_decision_rows(base_dec, planning_date, set_features, course, feature_cols)


def compare_plan_and_actual(
    ensemble: EnsembleModel,
    plan: pd.DataFrame,
    day_df: pd.DataFrame,
    train: pd.DataFrame,
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score plan vs actual using the same ensemble:
      pred_lift = f(plan) - f(0)
      actual_model_lift = f(actual decisions) - f(0)
    Plus observed levels for reference.
    """
    target = config.target
    actual_dec = actual_decisions_by_segment(day_df)
    if actual_dec.empty:
        return pd.DataFrame()

    baseline_sets = baseline_keyword_sets(train)
    segments = sorted(plan["segment"].astype(str).unique())

    plan_dec = plan[["segment", "daily_budget", "keyword_set_id"]].copy()
    plan_dec["segment"] = plan_dec["segment"].astype(str)
    plan_dec["keyword_set_id"] = plan_dec["keyword_set_id"].astype(str)
    plan_dec["daily_budget"] = pd.to_numeric(plan_dec["daily_budget"], errors="coerce")

    # Actual market decisions on this day (same segments as plan)
    actual_dec = actual_dec.copy()
    actual_dec["segment"] = actual_dec["segment"].astype(str)
    actual_dec = actual_dec[actual_dec["segment"].isin(segments)]
    for seg in segments:
        if seg not in actual_dec["segment"].values:
            actual_dec = pd.concat(
                [
                    actual_dec,
                    pd.DataFrame(
                        {
                            "segment": [seg],
                            "daily_budget": [ensemble.baseline_budget],
                            "keyword_set_id": [str(baseline_sets.get(seg, baseline_sets.iloc[0]))],
                        }
                    ),
                ],
                ignore_index=True,
            )

    baseline_rows = build_baseline_rows(
        segments,
        baseline_sets,
        planning_date,
        set_features,
        config.course,
        ensemble.feature_cols,
        ensemble.baseline_budget,
    )
    plan_rows = build_segment_decision_rows(
        plan_dec, planning_date, set_features, config.course, ensemble.feature_cols
    )
    actual_rows = build_segment_decision_rows(
        actual_dec, planning_date, set_features, config.course, ensemble.feature_cols
    )

    out = plan_dec.copy()
    out["pred_lift"] = ensemble.predict_incremental(plan_rows, baseline_rows)
    out["actual_model_lift"] = ensemble.predict_incremental(actual_rows, baseline_rows)
    out["f_plan_level"] = ensemble.predict_levels(plan_rows)
    out["f_zero"] = ensemble.predict_levels(baseline_rows)

    obs = observed_by_segment(day_df, target)
    out = out.merge(obs, on="segment", how="left")
    out = out.merge(
        actual_dec.rename(
            columns={"daily_budget": "actual_budget", "keyword_set_id": "actual_keyword_set_id"}
        ),
        on="segment",
        how="left",
    )
    return out


def week_planning_dates(
    week_start: pd.Timestamp,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Mon–Sun dates for one budget week, clipped to the backtest window."""
    week_start = pd.Timestamp(week_start).normalize()
    dates = pd.date_range(week_start, week_start + pd.Timedelta(days=6), freq="D")
    return [pd.Timestamp(d) for d in dates if window_start <= d <= window_end]


def week_starts_in_window(
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str = "W-MON",
) -> list[pd.Timestamp]:
    """Mondays (or cadence anchor) in [start, end], plus start if the window opens mid-week."""
    starts = [pd.Timestamp(d) for d in pd.date_range(start, end, freq=freq)]
    if not starts or starts[0] > start:
        starts = [pd.Timestamp(start).normalize()] + starts
    return starts


def compare_plan_and_actual_week(
    ensemble: EnsembleModel,
    plan: pd.DataFrame,
    df: pd.DataFrame,
    train: pd.DataFrame,
    config: CampaignOptConfig,
    week_dates: list[pd.Timestamp],
    set_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score a constant weekly plan against each day in week_dates.
    Returns (weekly_agg_by_segment, daily_diagnostics).
    """
    daily_parts: list[pd.DataFrame] = []
    for d in week_dates:
        day_df = df[df["date"] == d]
        if day_df.empty:
            continue
        comp = compare_plan_and_actual(
            ensemble, plan, day_df, train, config, d, set_features
        )
        if comp.empty:
            continue
        comp = comp.copy()
        comp["date"] = d.date().isoformat()
        daily_parts.append(comp)

    if not daily_parts:
        return pd.DataFrame(), pd.DataFrame()

    daily = pd.concat(daily_parts, ignore_index=True)
    target = config.target
    obs_col = f"observed_{target}"
    agg_cols: dict[str, tuple[str, str]] = {
        "pred_lift": ("pred_lift", "sum"),
        "actual_model_lift": ("actual_model_lift", "sum"),
        "f_plan_level": ("f_plan_level", "sum"),
        "f_zero": ("f_zero", "sum"),
        "n_days": ("date", "count"),
    }
    if obs_col in daily.columns:
        agg_cols[obs_col] = (obs_col, "sum")
    if "observed_clicks" in daily.columns:
        agg_cols["observed_clicks"] = ("observed_clicks", "sum")

    weekly = daily.groupby("segment", as_index=False).agg(**agg_cols)
    for col in ("daily_budget", "keyword_set_id", "region"):
        if col in plan.columns:
            weekly = weekly.merge(plan[["segment", col]], on="segment", how="left")
    return weekly, daily


def metrics_from_comparison(comp: pd.DataFrame, target: str) -> dict[str, float | None]:
    """RMSE/R² for incremental model-on-model and lift vs observed."""
    metrics: dict[str, float | None] = {}
    obs_col = f"observed_{target}"
    mask = np.isfinite(comp["pred_lift"]) & np.isfinite(comp["actual_model_lift"])
    if mask.any():
        metrics["rmse_pred_vs_actual_model_lift"] = float(
            np.sqrt(mean_squared_error(comp.loc[mask, "actual_model_lift"], comp.loc[mask, "pred_lift"]))
        )
        if mask.sum() > 1:
            metrics["r2_pred_vs_actual_model_lift"] = float(
                r2_score(comp.loc[mask, "actual_model_lift"], comp.loc[mask, "pred_lift"])
            )
    m2 = np.isfinite(comp["pred_lift"]) & comp[obs_col].notna() if obs_col in comp.columns else pd.Series(False)
    if m2.any():
        metrics["rmse_pred_lift_vs_observed"] = float(
            np.sqrt(mean_squared_error(comp.loc[m2, obs_col], comp.loc[m2, "pred_lift"]))
        )
    m3 = np.isfinite(comp["actual_model_lift"]) & comp[obs_col].notna() if obs_col in comp.columns else pd.Series(False)
    if m3.any():
        metrics["rmse_actual_model_lift_vs_observed"] = float(
            np.sqrt(mean_squared_error(comp.loc[m3, obs_col], comp.loc[m3, "actual_model_lift"]))
        )
    return metrics


def save_ensemble(ensemble: EnsembleModel, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, path)
