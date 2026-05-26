"""Dispatch optimization backend from manifest."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from campaign_opt.backends.linear import solve_linear_campaign_milp
from campaign_opt.backends.piecewise_linear import solve_piecewise_campaign_milp
from campaign_opt.backends.tree_embed import solve_tree_embed_campaign_milp
from campaign_opt.coefficients import export_linear_solver_coeffs, load_linear_solver_coeffs
from campaign_opt.modeling import refit_optimizer_model
from campaign_opt.schema import CampaignOptConfig
from campaign_opt.train_specs import get_train_spec

_LINEAR_BACKENDS = frozenset({"linear", "piecewise_linear"})


def _resolve_backend(config: CampaignOptConfig, manifest: dict) -> str:
    policy = config.model_policy
    if policy.optimizer_backend != "auto":
        return policy.optimizer_backend
    if policy.optimizer_winner:
        spec = get_train_spec(policy.optimizer_winner)
        if spec is not None:
            return spec.backend
    return manifest.get("backend", "linear")


def _resolve_optimizer_winner(config: CampaignOptConfig, manifest: dict) -> str:
    return config.model_policy.optimizer_winner or manifest.get("winner", "ridge")


def _load_linear_coeffs(
    config: CampaignOptConfig,
    train: pd.DataFrame,
    output_dir: Path,
    *,
    refit_coeffs: bool,
) -> dict:
    coeffs_path = output_dir / "linear_coeffs.json"
    if not refit_coeffs and coeffs_path.exists():
        coeffs = load_linear_solver_coeffs(coeffs_path)
        print(f"[Info] Using saved linear coeffs from {coeffs_path}")
        return coeffs
    coeffs = export_linear_solver_coeffs(train, config, coeffs_path)
    print(f"[Info] Exported linear coeffs to {coeffs_path}")
    return coeffs


def _tree_embed_model_path(
    config: CampaignOptConfig,
    manifest: dict,
    train: pd.DataFrame,
    output_dir: Path,
    model_path: Path | None,
) -> Path:
    path = Path(model_path or output_dir / "winner_model.joblib")
    winner_name = _resolve_optimizer_winner(config, manifest)
    manifest_winner = manifest.get("winner")
    if winner_name == manifest_winner and path.exists():
        if config.model_policy.optimizer_winner:
            print(f"[Info] Using winner_model.joblib for optimizer_winner={winner_name!r}")
        return path

    print(
        f"[Info] Refitting {winner_name!r} on {len(train)} rows for optimization "
        f"(manifest winner={manifest_winner!r})"
    )
    pipeline = refit_optimizer_model(winner_name, train, config, manifest)
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
    model_path: Path | None = None,
    planning_date: pd.Timestamp | None = None,
    planning_dates: list[pd.Timestamp] | None = None,
    fixed_keyword_sets: dict[str, str] | None = None,
    fixed_budgets: dict[str, float] | None = None,
    write_outputs: bool = True,
    refit_coeffs: bool = False,
) -> pd.DataFrame:
    """Pick solver backend from manifest and return segment-level plan."""
    output_dir = Path(output_dir)
    backend = _resolve_backend(config, manifest)
    optimizer_winner = _resolve_optimizer_winner(config, manifest)

    if config.model_policy.optimizer_winner:
        print(
            f"[Info] optimizer_winner={optimizer_winner!r} "
            f"(manifest winner={manifest.get('winner')!r}, backend={backend})"
        )

    dates = planning_dates
    if dates is None and planning_date is not None:
        dates = [pd.Timestamp(planning_date)]
    multi_day = dates is not None and len(dates) > 1
    if multi_day or fixed_budgets is not None:
        if backend not in _LINEAR_BACKENDS:
            print(
                f"[Info] Using linear backend for multi-day/fixed-budget solve "
                f"(requested backend={backend})"
            )
            backend = "linear"

    milp_kwargs = dict(
        fixed_keyword_sets=fixed_keyword_sets,
        fixed_budgets=fixed_budgets,
        planning_dates=dates,
        train=train,
    )

    if backend in _LINEAR_BACKENDS:
        coeffs = _load_linear_coeffs(config, train, output_dir, refit_coeffs=refit_coeffs)

    if backend == "linear":
        return solve_linear_campaign_milp(
            config=config,
            coeffs=coeffs,
            candidates=candidates,
            panel=panel,
            total_budget=total_budget,
            output_dir=output_dir,
            write_outputs=write_outputs,
            **milp_kwargs,
        )
    if backend == "piecewise_linear":
        return solve_piecewise_campaign_milp(
            config=config,
            coeffs=coeffs,
            candidates=candidates,
            panel=panel,
            total_budget=total_budget,
            output_dir=output_dir,
            write_outputs=write_outputs,
            **milp_kwargs,
        )
    if backend == "tree_embed":
        embed_path = _tree_embed_model_path(
            config, manifest, train, output_dir, model_path
        )
        return solve_tree_embed_campaign_milp(
            config,
            embed_path,
            train,
            candidates,
            panel,
            total_budget=total_budget,
            output_dir=output_dir,
            planning_date=planning_date or pd.Timestamp(train["date"].max()),
            write_outputs=write_outputs,
            fixed_keyword_sets=fixed_keyword_sets,
            fixed_budgets=fixed_budgets,
        )
    raise ValueError(f"Unknown backend: {backend}")
