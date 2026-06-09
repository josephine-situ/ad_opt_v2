"""Production plan-vs-actual monitoring for daily pipeline runs."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from utils.campaign_config import CampaignOptConfig
from utils.campaign_features import build_keyword_set_feature_table
from utils.evaluation import (
    compare_plan_and_actual,
    load_or_fit_evaluation_model,
    plan_vs_actual_row_metrics,
)
from utils.metrics import get_metrics_client
from utils.modeling_prep import load_planning_inputs, optimizer_manifest_for_backtest, train_before_date
from utils.paths import prod_monitoring_dir


def plan_path_for_date(config: CampaignOptConfig, score_date: pd.Timestamp) -> Path:
    return (
        config.prod_dir()
        / "two_stage_plan"
        / "stage2_budgets"
        / pd.Timestamp(score_date).strftime("%Y%m%d")
        / "campaign_plan.csv"
    )


def _scored_dates(metrics_path: Path) -> set[str]:
    if not metrics_path.is_file():
        return set()
    df = pd.read_csv(metrics_path)
    if df.empty or "score_date" not in df.columns:
        return set()
    return set(df["score_date"].astype(str))


def find_unscored_dates(
    config: CampaignOptConfig,
    *,
    lookback_days: int = 7,
    metrics_path: Path | None = None,
) -> list[pd.Timestamp]:
    """Dates in [yesterday - lookback + 1, yesterday] with a saved plan and no metrics row yet."""
    yesterday = pd.Timestamp(date.today() - timedelta(days=1))
    start = yesterday - pd.Timedelta(days=max(lookback_days, 1) - 1)
    scored = _scored_dates(metrics_path) if metrics_path else set()

    dates: list[pd.Timestamp] = []
    for d in pd.date_range(start, yesterday, freq="D"):
        ds = d.date().isoformat()
        if ds in scored:
            continue
        if plan_path_for_date(config, d).is_file():
            dates.append(pd.Timestamp(d))
    return dates


def append_daily_metrics(metrics_path: Path, row: dict[str, Any], *, overwrite: bool = False) -> None:
    """Append one daily summary row; skip when score_date already present unless overwrite."""
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    score_date = str(row.get("score_date", ""))
    if metrics_path.is_file():
        existing = pd.read_csv(metrics_path)
        if not overwrite and score_date and "score_date" in existing.columns:
            if score_date in existing["score_date"].astype(str).values:
                return
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])
    updated.to_csv(metrics_path, index=False)


def update_rolling_summary(
    metrics_path: Path,
    *,
    windows: list[int] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Compute rolling mean bias_pct and nrmse over persisted daily metrics."""
    windows = windows or [7, 30]
    metrics_path = Path(metrics_path)
    if not metrics_path.is_file():
        return {}

    df = pd.read_csv(metrics_path)
    if df.empty:
        return {}

    df["score_date"] = pd.to_datetime(df["score_date"], errors="coerce")
    df = df.dropna(subset=["score_date"]).sort_values("score_date")
    summary: dict[str, Any] = {"as_of": df["score_date"].max().date().isoformat(), "windows": {}}

    for window in windows:
        tail = df.tail(window)
        entry: dict[str, float | None] = {"n_days": int(len(tail))}
        if "total_bias_pct" in tail.columns:
            entry["mean_bias_pct"] = _safe_mean(tail["total_bias_pct"])
        if "nrmse" in tail.columns:
            entry["mean_nrmse"] = _safe_mean(tail["nrmse"])
        if "rmse_pred_vs_observed" in tail.columns:
            entry["mean_rmse"] = _safe_mean(tail["rmse_pred_vs_observed"])
        summary["windows"][str(window)] = entry

    out_path = output_path or metrics_path.parent / "rolling_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _safe_mean(series: pd.Series) -> float | None:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def format_monitoring_report(row: dict[str, Any]) -> str:
    score_date = row.get("score_date", "?")
    observed = row.get("observed_total")
    pred = row.get("pred_total")
    bias = row.get("total_bias_pct")
    rmse = row.get("rmse_pred_vs_observed")
    nrmse = row.get("nrmse")
    n_segments = row.get("n_segments", "?")

    def _fmt(val: Any, digits: int = 1) -> str:
        if val is None or (isinstance(val, float) and not pd.notna(val)):
            return "n/a"
        return f"{float(val):.{digits}f}"

    bias_str = f"{bias:+.1f}%" if bias is not None and pd.notna(bias) else "n/a"
    return (
        f"[monitoring] {score_date}: observed={_fmt(observed)}, pred={_fmt(pred)}, "
        f"bias={bias_str}, RMSE={_fmt(rmse)}, nRMSE={_fmt(nrmse, 2)}, segments={n_segments}"
    )


def format_rolling_report(summary: dict[str, Any]) -> str:
    if not summary or "windows" not in summary:
        return ""
    parts: list[str] = []
    for window, stats in summary["windows"].items():
        bias = stats.get("mean_bias_pct")
        nrmse = stats.get("mean_nrmse")
        if bias is None and nrmse is None:
            continue
        bias_str = f"{bias:+.1f}%" if bias is not None else "n/a"
        nrmse_str = f"{nrmse:.2f}" if nrmse is not None else "n/a"
        parts.append(f"{window}d rolling bias={bias_str}, nRMSE={nrmse_str}")
    return "[monitoring] " + "; ".join(parts) if parts else ""


def score_production_day(
    config: CampaignOptConfig,
    score_date: pd.Timestamp,
    *,
    monitoring_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Score one saved production plan against realized panel outcomes."""
    score_date = pd.Timestamp(score_date).normalize()
    plan_path = plan_path_for_date(config, score_date)
    if not plan_path.is_file():
        print(f"[monitoring] Skip {score_date.date()}: no plan at {plan_path}")
        return None

    try:
        df, panel, _candidates = load_planning_inputs(config)
    except Exception as exc:
        print(f"[monitoring] Skip {score_date.date()}: failed to load planning inputs ({exc})")
        return None

    holdout = df[pd.to_datetime(df["date"]).dt.normalize() == score_date]
    if holdout.empty:
        print(f"[monitoring] Skip {score_date.date()}: no modeling-panel rows")
        return None

    train = train_before_date(df, score_date)
    min_train = config.model_policy.validation.min_train_rows
    if len(train) < min_train:
        print(
            f"[monitoring] Skip {score_date.date()}: "
            f"insufficient train rows ({len(train)} < {min_train})"
        )
        return None

    try:
        manifest = optimizer_manifest_for_backtest(config)
    except FileNotFoundError:
        print(f"[monitoring] Skip {score_date.date()}: model_manifest.json not found (run fit-models first)")
        return None

    eval_model = load_or_fit_evaluation_model(config, df, manifest, config.prod_dir())
    set_features = build_keyword_set_feature_table(config.course)
    plan = pd.read_csv(plan_path)

    comp = compare_plan_and_actual(
        eval_model,
        plan,
        holdout,
        df,
        config,
        score_date,
        set_features,
        scoring_panel=train,
        floor_panel=panel,
    )
    if comp.empty:
        print(f"[monitoring] Skip {score_date.date()}: compare_plan_and_actual returned no rows")
        return None

    out_dir = (monitoring_dir or prod_monitoring_dir(config.course)) / "plan_vs_actual" / score_date.strftime(
        "%Y%m%d"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out_dir / "plan_vs_actual.csv", index=False)

    metrics = plan_vs_actual_row_metrics(comp, config.target)
    plan_rows = comp[comp["row_kind"] == "plan"] if "row_kind" in comp.columns else comp
    row: dict[str, Any] = {
        "score_date": score_date.date().isoformat(),
        "target": config.target,
        "n_segments": int(len(plan_rows)),
        **metrics,
    }
    return row


def _emit_grafana_metrics(config: CampaignOptConfig, row: dict[str, Any]) -> None:
    client = get_metrics_client()
    if not hasattr(client, "emit_production_monitoring_metrics"):
        return
    client.emit_production_monitoring_metrics(config.course, str(row.get("score_date", "")), row)


def run_production_monitoring(config: CampaignOptConfig) -> list[dict[str, Any]]:
    """Score unscored production days within the configured lookback window."""
    mon = getattr(config, "monitoring", None)
    if mon is not None and not getattr(mon, "enabled", True):
        print("[monitoring] Disabled in config")
        return []

    lookback = int(getattr(mon, "lookback_days", 7) if mon else 7)
    windows = list(getattr(mon, "rolling_windows", [7, 30]) if mon else [7, 30])

    monitoring_dir = prod_monitoring_dir(config.course)
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = monitoring_dir / "daily_metrics.csv"

    dates = find_unscored_dates(config, lookback_days=lookback, metrics_path=metrics_path)
    if not dates:
        print("[monitoring] No unscored dates with saved plans in lookback window")
    else:
        print(f"[monitoring] Scoring {len(dates)} day(s): {', '.join(d.date().isoformat() for d in dates)}")

    results: list[dict[str, Any]] = []
    for score_date in dates:
        row = score_production_day(config, score_date, monitoring_dir=monitoring_dir)
        if not row:
            continue
        append_daily_metrics(metrics_path, row)
        print(format_monitoring_report(row))
        _emit_grafana_metrics(config, row)
        results.append(row)

    rolling = update_rolling_summary(metrics_path, windows=windows, output_path=monitoring_dir / "rolling_summary.json")
    rolling_msg = format_rolling_report(rolling)
    if rolling_msg:
        print(rolling_msg)

    return results
