"""Dispatch optimization backend from manifest."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from campaign_opt.backends.linear import solve_linear_campaign_milp
from campaign_opt.backends.piecewise_linear import solve_piecewise_campaign_milp
from campaign_opt.backends.tree_embed import (
    solve_ridge_xgb_embed_campaign_milp,
    solve_ridge_xgb_embed_multiday_campaign_milp,
    solve_tree_embed_campaign_milp,
)
from campaign_opt.coefficients import export_linear_solver_coeffs
from campaign_opt.modeling import (
    configured_evaluation_model_name,
    refit_optimizer_model,
    warn_if_not_tournament_winner,
)
from campaign_opt.evaluation import add_optimizer_plan_columns
from campaign_opt.schema import CampaignOptConfig
from campaign_opt.train_specs import get_train_spec
from utils.campaign_features import build_keyword_set_feature_table

_LINEAR_BACKENDS = frozenset({"linear", "piecewise_linear"})


def require_optimizer_winner(config: CampaignOptConfig) -> str:
    winner = config.model_policy.optimizer_winner
    if not winner:
        raise ValueError("model_policy.optimizer_winner must be set in campaign_config.json")
    return winner


def _resolve_backend(config: CampaignOptConfig, manifest: dict) -> str:
    policy = config.model_policy
    winner = require_optimizer_winner(config)
    if policy.optimizer_backend != "auto":
        return policy.optimizer_backend
    if winner == "ensemble_ridge_xgb":
        return "ridge_xgb_embed"
    if is_ensemble_candidate(winner):
        raise ValueError(
            f"optimizer_winner={winner!r} has no MILP backend; "
            "use ensemble_ridge_xgb or set optimizer_backend explicitly."
        )
    spec = get_train_spec(winner)
    if spec is None:
        raise ValueError(f"Unknown optimizer_winner: {winner!r}")
    return spec.backend


def _fit_and_save_embed_model(
    config: CampaignOptConfig,
    manifest: dict,
    train: pd.DataFrame,
    output_dir: Path,
    *,
    tune: bool,
) -> Path:
    """Fit optimizer model on ``train``, persist, and return path for MILP embedding."""
    winner_name = require_optimizer_winner(config)
    print(
        f"[Info] Fitting optimizer {winner_name!r} on {len(train)} rows "
        f"(tune={tune})"
    )
    pipeline = refit_optimizer_model(winner_name, train, config, manifest, tune=tune)
    path = output_dir / f"optimizer_{winner_name}.joblib"
    joblib.dump(pipeline, path)
    return path


def run_optimizer(
    config: CampaignOptConfig,
    manifest: dict,
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    total_budget: float,
    output_dir: Path,
    planning_date: pd.Timestamp | None = None,
    planning_dates: list[pd.Timestamp] | None = None,
    fixed_keyword_sets: dict[str, str] | None = None,
    fixed_budgets: dict[str, float] | None = None,
    write_outputs: bool = True,
    tune_optimizer: bool = False,
) -> pd.DataFrame:
    """Pick solver backend from manifest and return segment-level plan."""
    output_dir = Path(output_dir)
    backend = _resolve_backend(config, manifest)
    optimizer_winner = require_optimizer_winner(config)

    warn_if_not_tournament_winner(optimizer_winner, manifest, role="Optimizer")

    print(
        f"[Info] optimizer_winner={optimizer_winner!r} "
        f"(manifest winner={manifest.get('winner')!r}, backend={backend})"
    )

    dates = planning_dates
    if dates is None and planning_date is not None:
        dates = [pd.Timestamp(planning_date)]
    multi_day = dates is not None and len(dates) > 1
    _MULTIDAY_BACKENDS = _LINEAR_BACKENDS | frozenset({"ridge_xgb_embed"})
    if multi_day or fixed_budgets is not None:
        if backend not in _MULTIDAY_BACKENDS:
            raise ValueError(
                f"backend {backend!r} does not support multi-day or fixed-budget optimization; "
                "use strategy two_stage or a linear/ridge_xgb_embed backend."
            )

    if planning_date is None and (dates is None or len(dates) == 0):
        raise ValueError("planning_date or planning_dates is required")

    milp_kwargs = dict(
        fixed_keyword_sets=fixed_keyword_sets,
        fixed_budgets=fixed_budgets,
        planning_dates=dates,
        train=train,
    )

    if backend in _LINEAR_BACKENDS:
        coeffs_path = output_dir / "linear_coeffs.json"
        coeffs = export_linear_solver_coeffs(train, config, coeffs_path)
        print(f"[Info] Exported linear coeffs to {coeffs_path}")

    plan_date = pd.Timestamp(planning_date) if planning_date is not None else pd.Timestamp(dates[0])

    def _finalize_plan(plan: pd.DataFrame, embed_path: Path | None = None) -> pd.DataFrame:
        if not config.evaluation.apply_observed_budget_floor:
            return plan
        model_path = embed_path
        if model_path is None and backend in _LINEAR_BACKENDS:
            winner = require_optimizer_winner(config)
            candidate = output_dir / f"optimizer_{winner}.joblib"
            if candidate.exists():
                model_path = candidate
        if model_path is None or not Path(model_path).exists():
            return plan
        pipeline = joblib.load(model_path)
        set_features = build_keyword_set_feature_table(config.course)
        plan = add_optimizer_plan_columns(
            plan,
            panel,
            pipeline,
            config,
            plan_date,
            set_features,
            candidates=candidates,
        )
        if write_outputs:
            plan.to_csv(output_dir / "campaign_plan.csv", index=False)
        return plan

    if backend == "linear":
        plan = solve_linear_campaign_milp(
            config=config,
            coeffs=coeffs,
            candidates=candidates,
            panel=panel,
            total_budget=total_budget,
            output_dir=output_dir,
            write_outputs=write_outputs,
            **milp_kwargs,
        )
        return _finalize_plan(plan)
    if backend == "piecewise_linear":
        plan = solve_piecewise_campaign_milp(
            config=config,
            coeffs=coeffs,
            candidates=candidates,
            panel=panel,
            total_budget=total_budget,
            output_dir=output_dir,
            write_outputs=write_outputs,
            **milp_kwargs,
        )
        return _finalize_plan(plan)
    if backend == "tree_embed":
        embed_path = _fit_and_save_embed_model(
            config, manifest, train, output_dir, tune=tune_optimizer
        )
        plan = solve_tree_embed_campaign_milp(
            config,
            embed_path,
            train,
            candidates,
            panel,
            total_budget=total_budget,
            output_dir=output_dir,
            planning_date=plan_date,
            write_outputs=write_outputs,
            fixed_keyword_sets=fixed_keyword_sets,
            fixed_budgets=fixed_budgets,
        )
        return _finalize_plan(plan, embed_path)
    if backend == "ridge_xgb_embed":
        embed_path = _fit_and_save_embed_model(
            config, manifest, train, output_dir, tune=tune_optimizer
        )
        if multi_day:
            if fixed_budgets:
                raise ValueError(
                    "fixed_budgets is not supported with multi-day ridge_xgb_embed; "
                    "per-day budgets are optimized jointly in the MILP."
                )
            plan = solve_ridge_xgb_embed_multiday_campaign_milp(
                config,
                embed_path,
                train,
                candidates,
                panel,
                total_budget=total_budget,
                output_dir=output_dir,
                planning_dates=dates,
                write_outputs=write_outputs,
                fixed_keyword_sets=fixed_keyword_sets,
            )
            return _finalize_plan(plan, embed_path)
        plan = solve_ridge_xgb_embed_campaign_milp(
            config,
            embed_path,
            train,
            candidates,
            panel,
            total_budget=total_budget,
            output_dir=output_dir,
            planning_date=plan_date,
            write_outputs=write_outputs,
            fixed_keyword_sets=fixed_keyword_sets,
            fixed_budgets=fixed_budgets,
        )
        return _finalize_plan(plan, embed_path)
    raise ValueError(f"Unknown backend: {backend}")
