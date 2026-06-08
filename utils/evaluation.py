"""Ensemble predictions and plan-vs-actual evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from utils.decisions import build_segment_list, observed_min_daily_budget, region_of_segment
from utils.modeling_prep import train_holdout_split
from utils.modeling import (
    _cv_rmse_member_weights,
    FITTERS,
    base_tournament_candidates,
    configured_evaluation_model_name,
    hyperparams_from_manifest,
    pipeline_feature_overview_lines,
    report_model_fit_diagnostics,
    warn_if_not_tournament_winner,
)
from utils.training_matrix import (
    build_preprocessor,
    prep_xy,
)
from utils.campaign_config import CampaignOptConfig
from utils.train_specs import TrainSpec, get_train_spec
from utils.campaign_features import (
    build_keyword_set_feature_table,
    get_context_feature_columns,
)
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

    def predict_levels(self, rows: pd.DataFrame) -> np.ndarray:
        """Full level prediction f(decision context) per row."""
        if not self.members:
            raise RuntimeError("Ensemble has no fitted members")
        preds = np.zeros(len(rows))
        total_w = sum(m.weight for m in self.members)
        for member in self.members:
            preds += member.weight * _predict_member_levels(member, rows, self.target, self.feature_cols)
        return np.clip(preds / total_w, 0, None)


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
        X, _ = prep_xy(df, target, feature_cols, y_col=spec.fit_y_col)
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
        from utils.modeling import fit_ridge_full

        pipe = fit_ridge_full(train, config, hyperparams=hyperparams)
        return FittedMember(spec.name, pipe, spec)

    target = config.target
    tr = spec.transform(train, target) if spec.transform else train
    sub = tr.dropna(subset=[target, "daily_budget", "segment"])
    X, y = prep_xy(sub, target, feature_cols, y_col=spec.fit_y_col)
    if spec.budget_col != "daily_budget":
        X = X.rename(columns={spec.budget_col: "daily_budget"})

    from utils.train_specs import build_estimator

    estimator = build_estimator(spec.name, hyperparams) if hyperparams else spec.estimator
    pipe = Pipeline([("prep", build_preprocessor(feature_cols, tr)), ("model", estimator)])
    from utils.recency_weights import training_row_recency_weights

    fit_kw: dict[str, Any] = {}
    sample_weight = training_row_recency_weights(
        tr, config, y_col=spec.fit_y_col, date_col="date"
    )
    if sample_weight is not None:
        fit_kw["model__sample_weight"] = sample_weight
    pipe.fit(X, y, **fit_kw)
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

    for name in base_tournament_candidates(config.model_policy.candidates):
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
    )


def load_holdout_metrics(config: CampaignOptConfig) -> dict[str, dict[str, float]]:
    """Tournament CV/holdout metrics (``holdout_metrics.json``)."""
    path = config.exp_dir() / "holdout_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run fit_response_models.py before fitting the evaluation ensemble."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluation_ensemble_weights(
    config: CampaignOptConfig,
    metrics_table: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Inverse-CV-RMSE weights over base tournament members only."""
    member_names = base_tournament_candidates(config.model_policy.candidates)
    for name in member_names:
        rmse = (metrics_table.get(name) or {}).get("cv_rmse_levels")
        if rmse is None:
            raise ValueError(
                f"holdout_metrics.json entry {name!r} missing cv_rmse_levels "
                "(required when evaluation.weight_by_cv_rmse is true)"
            )
    return _cv_rmse_member_weights(metrics_table, member_names)


def fit_evaluation_model(
    config: CampaignOptConfig,
    df: pd.DataFrame,
    manifest: dict,
    out_dir: Path,
) -> EnsembleModel:
    """
    Fit the plan-vs-actual scorer on the full modeling panel (backtest + production).

    When ``evaluation.use_ensemble`` is true, uses CV-RMSE weights from
    ``holdout_metrics.json`` when ``evaluation.weight_by_cv_rmse`` is set.
    Saves ``ensemble_model.joblib`` (or ``evaluation_{name}.joblib``) under ``out_dir``.
    """
    out_dir = Path(out_dir)
    eval_model_name = configured_evaluation_model_name(config)
    warn_if_not_tournament_winner(eval_model_name, manifest, role="Evaluation")
    if not config.evaluation.use_ensemble:
        eval_name = optimizer_winner_name(config)
        print(f"Fitting evaluation model {eval_name!r} on full panel: {len(df)} rows")
        model = fit_single_model_evaluation(
            df,
            config,
            manifest,
            model_name=eval_name,
        )
        save_evaluation_model(model, out_dir / f"evaluation_{eval_name}.joblib")
        return model

    static_metrics = load_holdout_metrics(config)
    weights = (
        evaluation_ensemble_weights(config, static_metrics)
        if config.evaluation.weight_by_cv_rmse
        else None
    )
    dmin = pd.to_datetime(df["date"]).min().date()
    dmax = pd.to_datetime(df["date"]).max().date()
    print(
        f"Fitting evaluation ensemble on full panel: "
        f"{len(df)} rows ({dmin} → {dmax})"
    )
    if weights:
        for name, w in sorted(weights.items()):
            print(f"    member weight: {name}={w:.3f}")
    ensemble = fit_ensemble(
        df,
        config,
        member_weights=weights,
        member_hyperparams=manifest.get("best_hyperparams"),
    )
    path = out_dir / "ensemble_model.joblib"
    save_ensemble(ensemble, path)
    _write_ensemble_meta(ensemble, config, out_dir / "ensemble_meta.json", weights=weights)
    print(f"Saved {path}")
    return ensemble


def _write_ensemble_meta(
    ensemble: EnsembleModel,
    config: CampaignOptConfig,
    path: Path,
    *,
    weights: dict[str, float] | None = None,
) -> None:
    path = Path(path)
    meta: dict[str, Any] = {
        "n_members": len(ensemble.members),
        "members": [m.name for m in ensemble.members],
        "member_weights": {m.name: m.weight for m in ensemble.members},
        "target": config.target,
        "weight_by_cv_rmse": config.evaluation.weight_by_cv_rmse,
    }
    if weights is not None:
        meta["cv_rmse_weights"] = weights
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_or_fit_evaluation_model(
    config: CampaignOptConfig,
    df: pd.DataFrame,
    manifest: dict,
    out_dir: Path,
) -> EnsembleModel:
    """Load saved evaluation ensemble or fit with :func:`fit_evaluation_model`."""
    out_dir = Path(out_dir)
    if config.evaluation.use_ensemble:
        path = out_dir / "ensemble_model.joblib"
        if path.exists():
            return joblib.load(path)
    else:
        eval_name = optimizer_winner_name(config)
        path = out_dir / f"evaluation_{eval_name}.joblib"
        if path.exists():
            return joblib.load(path)
    return fit_evaluation_model(config, df, manifest, out_dir)


def optimizer_winner_name(config: CampaignOptConfig) -> str:
    """Configured optimizer / single-model evaluation candidate."""
    from utils.optimize import require_optimizer_winner

    return require_optimizer_winner(config)


def optimizer_model_path(output_dir: Path, winner_name: str) -> Path:
    """Path to the optimizer pipeline saved by ``run_optimizer``."""
    if not winner_name:
        raise ValueError("winner_name is required")
    path = Path(output_dir) / f"optimizer_{winner_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing optimizer artifact: {path}")
    return path


def fit_single_model_evaluation(
    fit_df: pd.DataFrame,
    config: CampaignOptConfig,
    manifest: dict,
    *,
    model_name: str | None = None,
    feature_cols: list[str] | None = None,
) -> EnsembleModel:
    """
    Fit one response model for plan-vs-actual scoring (default: ``optimizer_winner``).

    Separate from the MILP optimizer (``optimizer_*.joblib``): the optimizer is refit
    walk-forward on ``date < t`` each day; this model is typically fit **once** on the
    full modeling panel passed into the backtest (all rows in ``fit_df``).

    Diagnostics (holdout R², SHAP) use the same train/holdout split and reporting as
    ``run_tournament``; the returned model is always refit on all of ``fit_df``.
    """
    from utils.modeling import (
        ENSEMBLE_MEMBER_GROUPS,
        fit_ensemble_tournament,
        is_ensemble_candidate,
    )

    model_name = model_name or optimizer_winner_name(config)
    feature_cols = feature_cols or manifest.get("feature_cols") or get_context_feature_columns(
        config.context_features
    )
    holdout_days = config.model_policy.validation.holdout_days
    train, holdout = train_holdout_split(fit_df, holdout_days)

    if model_name == "ensemble_ridge_xgb":
        member_names = ENSEMBLE_MEMBER_GROUPS[model_name]
        member_hp = {
            m: hyperparams_from_manifest(manifest, m) or {}
            for m in member_names
        }
        diag = fit_ensemble_tournament(
            model_name,
            member_names,
            train,
            holdout,
            config,
            feature_cols,
            member_hyperparams=member_hp,
        )
        report_model_fit_diagnostics(diag, train, config, feature_cols)
        return fit_ensemble_tournament(
            model_name,
            member_names,
            fit_df,
            fit_df.iloc[0:0],
            config,
            feature_cols,
            member_hyperparams=member_hp,
        ).pipeline
    if is_ensemble_candidate(model_name):
        raise ValueError(f"Evaluation model {model_name!r} is not supported for plan_vs_actual")
    hp = hyperparams_from_manifest(manifest, model_name)
    fitter = FITTERS.get(model_name)
    if fitter is None:
        raise ValueError(f"Unknown evaluation model: {model_name!r}")
    diag = fitter(train, holdout, config, feature_cols, hyperparams=hp)
    report_model_fit_diagnostics(diag, train, config, feature_cols)
    spec = get_train_spec(model_name, hp)
    if spec is None:
        raise ValueError(f"Unknown evaluation model: {model_name!r}")
    member = fit_member_on_train(spec, fit_df, config, feature_cols, hyperparams=hp)
    return EnsembleModel(
        members=[member],
        feature_cols=feature_cols,
        target=config.target,
    )


def save_evaluation_model(ensemble: EnsembleModel, path: Path, *, model_name: str = "") -> Path:
    """Persist the evaluation model (not the optimizer artifact)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not ensemble.members:
        raise RuntimeError("Evaluation model has no fitted members")
    name = model_name or (
        ensemble.members[0].name
        if len(ensemble.members) == 1
        else "ensemble_ridge_xgb"
    )
    if not path.name.startswith("evaluation_"):
        path = path.parent / f"evaluation_{name}.joblib"
    artifact = ensemble if len(ensemble.members) > 1 else ensemble.members[0].pipeline
    joblib.dump(artifact, path)
    return path


def plan_vs_actual_row_metrics(comp: pd.DataFrame, target: str) -> dict[str, Any]:
    """Summary fields for backtest daily/weekly rows from a plan_vs_actual frame."""
    if comp.empty:
        return {}
    plan_rows, market_rows = _split_plan_and_market_rows(comp)
    mets = metrics_from_comparison(plan_rows, target)
    obs_col = f"observed_{target}"
    observed_total = None
    if obs_col in comp.columns:
        obs_src = market_rows if not market_rows.empty else plan_rows
        observed_total = float(pd.to_numeric(obs_src[obs_col], errors="coerce").sum())
    act_budget = None
    if "actual_budget" in comp.columns:
        budget_src = market_rows if not market_rows.empty else plan_rows
        act_budget = float(pd.to_numeric(budget_src["actual_budget"], errors="coerce").sum())
    elif not market_rows.empty and "daily_budget" in market_rows.columns:
        act_budget = float(pd.to_numeric(market_rows["daily_budget"], errors="coerce").sum())
    out: dict[str, Any] = {
        "pred_lift_total": float(pd.to_numeric(plan_rows["pred_lift"], errors="coerce").sum()),
        "actual_model_lift_total": float(
            pd.to_numeric(market_rows["actual_model_lift"], errors="coerce").sum()
        )
        if not market_rows.empty
        else float(pd.to_numeric(plan_rows["actual_model_lift"], errors="coerce").sum()),
        "observed_total": observed_total,
        **mets,
    }
    if "pred_lift_raw" in plan_rows.columns:
        out["pred_lift_raw_total"] = float(pd.to_numeric(plan_rows["pred_lift_raw"], errors="coerce").sum())
    if not market_rows.empty and "actual_model_lift_raw" in market_rows.columns:
        out["actual_model_lift_raw_total"] = float(
            pd.to_numeric(market_rows["actual_model_lift_raw"], errors="coerce").sum()
        )
    if act_budget is not None:
        out["act_budget_total"] = act_budget
    return out


def observed_by_segment(day_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Realized target summed on one calendar day."""
    if day_df.empty:
        return pd.DataFrame(columns=["segment", f"observed_{target}"])
    cols = {f"observed_{target}": (target, "sum"), "observed_clicks": ("clicks", "sum")}
    cols = {k: v for k, v in cols.items() if v[0] in day_df.columns}
    return day_df.groupby("segment", as_index=False).agg(**cols)


def _ensure_region_column(df: pd.DataFrame) -> pd.DataFrame:
    if "region" in df.columns:
        return df
    out = df.copy()
    if "segment" in out.columns:
        out["region"] = out["segment"].astype(str).map(region_of_segment)
    return out


def _split_plan_and_market_rows(comp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plan rows (optimizer segments) vs market rows (panel campaigns)."""
    if "row_kind" in comp.columns:
        plan = comp[comp["row_kind"] == "plan"]
        market = comp[comp["row_kind"] == "market"]
        return plan, market
    return comp, comp.iloc[0:0]


def _require_campaign_budget_column(day_df: pd.DataFrame) -> None:
    """Plan vs actual uses configured campaign caps, not observed ``cost``."""
    if day_df.empty:
        return
    if "daily_budget" not in day_df.columns:
        raise ValueError(
            "Holdout panel must include daily_budget (configured campaign budget cap). "
            "Observed cost is not used for plan-vs-actual scoring."
        )


def _aggregate_campaign_decisions(
    day_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """``daily_budget`` (campaign cap) + keyword set per panel campaign row."""
    _require_campaign_budget_column(day_df)
    budget = pd.to_numeric(day_df["daily_budget"], errors="coerce")
    if budget.isna().all():
        raise ValueError("daily_budget is all missing on holdout day; cannot score actual campaigns.")
    agg: dict[str, tuple[str, Any]] = {
        "daily_budget": ("daily_budget", "median"),
        "keyword_set_id": (
            "keyword_set_id",
            lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0],
        ),
    }
    if "campaign_version" in day_df.columns:
        agg["campaign_version"] = (
            "campaign_version",
            lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0],
        )
    return day_df.groupby(group_cols, as_index=False).agg(**agg)


def actual_decisions_by_segment(day_df: pd.DataFrame) -> pd.DataFrame:
    """Campaign budget cap + keyword set per panel row (segment labels as in the modeling frame)."""
    if day_df.empty:
        return pd.DataFrame(columns=["segment", "daily_budget", "keyword_set_id"])
    return _aggregate_campaign_decisions(day_df, ["segment"])


def region_actual_lookup(day_df: pd.DataFrame) -> pd.DataFrame:
    """Campaign budget cap + keyword set by region (one row per region per day)."""
    if day_df.empty:
        return pd.DataFrame(columns=["region", "daily_budget", "keyword_set_id"])
    df = _ensure_region_column(day_df)
    return _aggregate_campaign_decisions(df, ["region"])


def _raw_predict_levels(
    model: EnsembleModel | Any,
    rows: pd.DataFrame,
    *,
    target: str,
    feature_cols: list[str],
) -> np.ndarray:
    """Unfloored level predictions for ensemble or single sklearn pipeline."""
    if isinstance(model, EnsembleModel):
        return model.predict_levels(rows)
    X, _ = prep_xy(rows, target, feature_cols)
    return np.asarray(model.predict(X), dtype=float)


def build_segment_decision_rows(
    decisions: pd.DataFrame,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
    course: str,
    feature_cols: list[str],
    *,
    panel: pd.DataFrame | None = None,
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
            "region": region,
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


def feature_rows_at_plan_budgets(
    plan_dec: pd.DataFrame,
    embed_rows: pd.DataFrame,
    keys: list[tuple[str, str]],
    target: str,
) -> pd.DataFrame:
    """One feature row per plan decision using the tree-MILP embed row template."""
    key_to_idx = {k: i for i, k in enumerate(keys)}
    chunks: list[pd.DataFrame] = []
    for _, dec in plan_dec.iterrows():
        seg = str(dec["segment"])
        kid = str(dec["keyword_set_id"])
        key = (seg, kid)
        if key not in key_to_idx:
            raise KeyError(f"Plan keyword set {key!r} not in embed candidate keys")
        row = embed_rows.iloc[key_to_idx[key]: key_to_idx[key] + 1].copy()
        row["daily_budget"] = float(dec["daily_budget"])
        if target not in row.columns:
            row[target] = 0.0
        chunks.append(row)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def build_plan_prediction_rows(
    plan_dec: pd.DataFrame,
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Feature rows for post-solve sklearn checks.

    When ``candidates`` is provided, rows match the tree embed MILP path
    (``build_candidate_feature_rows`` + solved budgets). Otherwise falls back
    to ``build_segment_decision_rows``.
    """
    feature_cols = get_context_feature_columns(config.context_features)
    target = config.target
    if candidates is not None and not candidates.empty:
        from utils.backends.tree_embed import build_candidate_feature_rows

        embed_rows, keys = build_candidate_feature_rows(
            candidates, config, planning_date, set_features, panel=panel
        )
        return feature_rows_at_plan_budgets(plan_dec, embed_rows, keys, target)
    return build_segment_decision_rows(
        plan_dec,
        planning_date,
        set_features,
        config.course,
        feature_cols,
        panel=panel,
    )


def _predict_levels_for_scoring(
    model: EnsembleModel | Any,
    rows: pd.DataFrame,
    panel: pd.DataFrame,
    config: CampaignOptConfig,
    *,
    floor_panel: pd.DataFrame | None = None,
) -> np.ndarray:
    if config.evaluation.apply_observed_budget_floor:
        from utils.optimizer_prediction import predict_levels_optimizer

        return predict_levels_optimizer(
            model, rows, panel, config, floor_panel=floor_panel
        )
    feature_cols = (
        model.feature_cols if isinstance(model, EnsembleModel) else get_context_feature_columns(config.context_features)
    )
    return _raw_predict_levels(model, rows, target=config.target, feature_cols=feature_cols)


def add_optimizer_plan_columns(
    plan: pd.DataFrame,
    panel: pd.DataFrame,
    model: Any,
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
    *,
    candidates: pd.DataFrame | None = None,
    level_tol: float | None = None,
    gating_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Attach gated ``external_model_pred`` and warn on MILP mismatch.

    ``milp_pred`` is the per-segment value that enters the MILP objective (already gated:
    contribution is 0 when ``daily_budget`` is below the segment's minimum observed
    ``daily_budget`` on ``panel``, same rule as ``solve_campaign_milp``).

    ``external_model_pred`` is the same quantity from sklearn: raw ensemble level at the
    plan row, then the **identical** floor using ``observed_min_daily_budget`` (same
    segment universe as the MILP when ``candidates`` is passed).

    Called after MILP solve (e.g. from ``run_optimizer``); does not modify tree embedding.
    """
    if not config.evaluation.apply_observed_budget_floor:
        return plan
    if plan.empty:
        return plan

    feature_cols = (
        model.feature_cols if isinstance(model, EnsembleModel) else get_context_feature_columns(config.context_features)
    )
    target = config.target
    plan_dec = plan[["segment", "daily_budget", "keyword_set_id"]].copy()
    plan_dec["segment"] = plan_dec["segment"].astype(str)
    plan_dec["keyword_set_id"] = plan_dec["keyword_set_id"].astype(str)
    plan_dec["daily_budget"] = pd.to_numeric(plan_dec["daily_budget"], errors="coerce")

    decision_rows = build_plan_prediction_rows(
        plan_dec,
        config,
        planning_date,
        set_features,
        panel,
        candidates=candidates,
    )
    if target not in decision_rows.columns:
        decision_rows[target] = 0.0

    from utils.optimizer_prediction import apply_observed_budget_floor

    # Same floor map as milp_core.gate_pred_vars_if_enabled when candidates match MILP input.
    floor_segments = (
        build_segment_list(candidates)
        if candidates is not None and not candidates.empty
        else sorted(plan_dec["segment"].astype(str).unique().tolist())
    )
    min_budget_by_seg = observed_min_daily_budget(
        gating_panel if gating_panel is not None else panel, floor_segments
    )

    raw_dec = _raw_predict_levels(model, decision_rows, target=target, feature_cols=feature_cols)
    floor_atol = float(config.evaluation.budget_floor_atol)
    pred_dec = apply_observed_budget_floor(
        raw_dec,
        plan_dec["daily_budget"].to_numpy(),
        plan_dec["segment"].to_numpy(),
        min_budget_by_seg,
        budget_atol=floor_atol,
    )
    ext = pd.DataFrame(
        {
            "segment": plan_dec["segment"].tolist(),
            "external_model_pred": pred_dec,
        }
    )
    out = plan.drop(columns=["external_model_pred"], errors="ignore")
    out = out.merge(ext, on="segment", how="left")

    if "milp_pred" in out.columns:
        tol = float(
            config.evaluation.milp_external_level_tol if level_tol is None else level_tol
        )
        milp = pd.to_numeric(out["milp_pred"], errors="coerce")
        ext_lev = pd.to_numeric(out["external_model_pred"], errors="coerce")
        mask = milp.notna() & ext_lev.notna()
        if mask.any():
            max_diff = float((milp.loc[mask] - ext_lev.loc[mask]).abs().max())
            if max_diff > tol:
                print(
                    "[Warn] MILP objective term vs gated sklearn level mismatch "
                    "(per-segment level summed in objective; 0 below min observed budget): "
                    f"max|milp_pred - external_model_pred| = {max_diff:.6g} > {tol}"
                )
                if candidates is not None and not candidates.empty:
                    from utils.backends.tree_embed import diagnose_plan_sklearn_path_gap

                    diagnose_plan_sklearn_path_gap(
                        out,
                        model,
                        config,
                        planning_date,
                        set_features,
                        panel,
                        candidates,
                        tol=tol,
                        gating_panel=gating_panel,
                    )
    return out


def compare_plan_and_actual(
    ensemble: EnsembleModel,
    plan: pd.DataFrame,
    day_df: pd.DataFrame,
    train: pd.DataFrame,
    config: CampaignOptConfig,
    planning_date: pd.Timestamp,
    set_features: pd.DataFrame,
    *,
    market_ensemble: EnsembleModel | None = None,
    scoring_panel: pd.DataFrame | None = None,
    floor_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Score optimizer plan vs historical market campaigns on this day.

    Uses ``ensemble`` for optimizer plan rows. Market rows use ``market_ensemble`` when
    provided (defaults to ``ensemble``), so panel counterfactuals stay stable when the
    plan scorer is a wider tournament ensemble.

    Walk-forward ``scoring_panel`` supplies feature/version context; ``floor_panel`` (when
    set) supplies observed-min budget floors — typically the full campaign panel so
    evaluation gating matches all historical spend levels.

      - ``pred_lift``: f(plan budget, plan keyword set)
      - ``actual_model_lift``: f(actual campaign budget, actual keyword set) on panel rows

    Actual budgets are ``daily_budget`` from the campaign-day panel (configured cap),
    **not** observed ``cost``. Plan rows include region-level ``actual_budget`` for
    reference. Market rows use ``row_kind='market'``.
    """
    target = config.target
    feature_panel = scoring_panel if scoring_panel is not None else train
    floor_panel = floor_panel if floor_panel is not None else feature_panel
    market_model = market_ensemble or ensemble
    market_dec = actual_decisions_by_segment(day_df)
    if market_dec.empty:
        return pd.DataFrame()

    plan_dec = plan[["segment", "daily_budget", "keyword_set_id"]].copy()
    plan_dec["segment"] = plan_dec["segment"].astype(str)
    plan_dec["keyword_set_id"] = plan_dec["keyword_set_id"].astype(str)
    plan_dec["daily_budget"] = pd.to_numeric(plan_dec["daily_budget"], errors="coerce")

    plan_rows = build_segment_decision_rows(
        plan_dec, planning_date, set_features, config.course, ensemble.feature_cols, panel=feature_panel
    )

    out = plan_dec.copy()
    out["row_kind"] = "plan"
    out["pred_lift"] = _predict_levels_for_scoring(
        ensemble, plan_rows, feature_panel, config, floor_panel=floor_panel
    )
    out["f_plan_level"] = out["pred_lift"]
    out["actual_model_lift"] = np.nan

    region_actual = region_actual_lookup(day_df)
    if not region_actual.empty:
        reg_budget = region_actual.set_index("region")["daily_budget"]
        reg_kw = region_actual.set_index("region")["keyword_set_id"]
        out["actual_budget"] = out["segment"].map(lambda s: reg_budget.get(region_of_segment(s), np.nan))
        out["actual_keyword_set_id"] = out["segment"].map(
            lambda s: reg_kw.get(region_of_segment(s), np.nan)
        )

    out["region"] = out["segment"].map(region_of_segment)
    obs = observed_by_segment(day_df, target)
    if not obs.empty:
        obs_cols = [c for c in obs.columns if c.startswith("observed_")]
        obs_by_region = obs.assign(
            region=obs["segment"].astype(str).map(region_of_segment)
        ).groupby("region", as_index=False)[obs_cols].sum()
        out = out.merge(obs_by_region, on="region", how="left")

    market_dec = market_dec.copy()
    market_dec["segment"] = market_dec["segment"].astype(str)
    market_rows = build_segment_decision_rows(
        market_dec,
        planning_date,
        set_features,
        config.course,
        market_model.feature_cols,
        panel=feature_panel,
    )
    market_scored = market_dec.copy()
    market_scored["row_kind"] = "market"
    market_scored["pred_lift"] = np.nan
    market_scored["actual_model_lift"] = _predict_levels_for_scoring(
        market_model, market_rows, feature_panel, config, floor_panel=floor_panel
    )
    market_scored["f_plan_level"] = market_scored["actual_model_lift"]
    market_scored["campaign_budget"] = market_scored["daily_budget"]
    market_scored["actual_budget"] = market_scored["daily_budget"]
    market_scored["actual_keyword_set_id"] = market_scored["keyword_set_id"]
    market_scored = market_scored.merge(obs, on="segment", how="left")

    shared_cols = sorted(set(out.columns) | set(market_scored.columns))
    out = pd.concat(
        [out.reindex(columns=shared_cols), market_scored.reindex(columns=shared_cols)],
        ignore_index=True,
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
    """RMSE/R² for model plan levels vs actual campaign levels, and vs observed."""
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
