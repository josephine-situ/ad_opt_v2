"""Tests for production plan-vs-actual monitoring."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from utils.campaign_config import CampaignOptConfig, MonitoringConfig
from utils.metrics import CampaignOptMonitoringClient
from utils.production_monitoring import (
    append_daily_metrics,
    find_unscored_dates,
    format_monitoring_report,
    format_rolling_report,
    plan_path_for_date,
    run_production_monitoring,
    score_production_day,
    update_rolling_summary,
)


def _sample_row(score_date: str = "2026-06-08") -> dict:
    return {
        "score_date": score_date,
        "target": "conv_scaled_clicks",
        "n_segments": 6,
        "pred_total": 158.1,
        "observed_total": 142.3,
        "total_bias_pct": 11.1,
        "rmse_pred_vs_observed": 18.4,
        "nrmse": 0.13,
    }


def test_append_daily_metrics_idempotent(tmp_path: Path):
    metrics_path = tmp_path / "daily_metrics.csv"
    row = _sample_row()
    append_daily_metrics(metrics_path, row)
    append_daily_metrics(metrics_path, row)
    df = pd.read_csv(metrics_path)
    assert len(df) == 1


def test_update_rolling_summary(tmp_path: Path):
    metrics_path = tmp_path / "daily_metrics.csv"
    rows = []
    for i in range(10):
        d = (date(2026, 6, 1) + timedelta(days=i)).isoformat()
        row = _sample_row(d)
        row["total_bias_pct"] = float(i - 5)
        row["nrmse"] = 0.1 + i * 0.01
        rows.append(row)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    summary = update_rolling_summary(metrics_path, windows=[7, 30], output_path=tmp_path / "rolling.json")
    assert summary["windows"]["7"]["n_days"] == 7
    assert summary["windows"]["30"]["n_days"] == 10
    assert (tmp_path / "rolling.json").is_file()


def test_find_unscored_dates(tmp_path: Path, monkeypatch):
    config = CampaignOptConfig(course="sys_think", monitoring=MonitoringConfig(lookback_days=3))
    yesterday = date.today() - timedelta(days=1)
    plan_dir = (
        tmp_path
        / "prod"
        / "two_stage_plan"
        / "stage2_budgets"
        / yesterday.strftime("%Y%m%d")
    )
    plan_dir.mkdir(parents=True)
    (plan_dir / "campaign_plan.csv").write_text("segment,daily_budget,keyword_set_id\n", encoding="utf-8")

    monkeypatch.setattr(
        "utils.production_monitoring.plan_path_for_date",
        lambda cfg, d: tmp_path / "prod" / "two_stage_plan" / "stage2_budgets" / d.strftime("%Y%m%d") / "campaign_plan.csv",
    )
    monkeypatch.setattr(
        config,
        "prod_dir",
        lambda base=None: tmp_path / "prod" if base is None else Path(base),
    )

    dates = find_unscored_dates(config, lookback_days=3, metrics_path=tmp_path / "missing.csv")
    assert any(d.date() == yesterday for d in dates)


def test_format_monitoring_report():
    msg = format_monitoring_report(_sample_row())
    assert "[monitoring]" in msg
    assert "bias=+11.1%" in msg
    assert "nRMSE=0.13" in msg


def test_format_rolling_report():
    summary = {
        "windows": {
            "7": {"mean_bias_pct": 6.2, "mean_nrmse": 0.11, "n_days": 7},
        }
    }
    msg = format_rolling_report(summary)
    assert "7d rolling bias=+6.2%" in msg


def test_emit_production_monitoring_metrics(monkeypatch):
    emitted: list[tuple[str, float, dict[str, str], str | None]] = []

    def _capture(self, metric_name, value, labels, metric_prefix=None):
        emitted.append((metric_name, value, labels, metric_prefix))

    monkeypatch.setattr(CampaignOptMonitoringClient, "emit_metric", _capture)
    client = CampaignOptMonitoringClient("http://example", "user", "token")
    client.emit_production_monitoring_metrics("sys_think", "2026-06-08", _sample_row())

    assert emitted
    assert emitted[0][3] == "campaign_opt_monitoring"
    assert emitted[0][2]["course"] == "sys_think"
    fields = {name for name, _val, _labels, _prefix in emitted}
    assert "rmse" in fields
    assert "bias_pct" in fields


def test_score_production_day_skips_without_plan(tmp_path: Path, monkeypatch):
    config = CampaignOptConfig(course="sys_think", monitoring=MonitoringConfig())
    monkeypatch.setattr(
        config,
        "prod_dir",
        lambda base=None: tmp_path / "prod" if base is None else Path(base),
    )
    result = score_production_day(config, pd.Timestamp("2026-06-08"), monitoring_dir=tmp_path / "monitoring")
    assert result is None


def test_run_production_monitoring_disabled(tmp_path: Path, monkeypatch):
    config = CampaignOptConfig(course="sys_think", monitoring=MonitoringConfig(enabled=False))
    monkeypatch.setattr("utils.production_monitoring.prod_monitoring_dir", lambda _course: tmp_path / "monitoring")
    assert run_production_monitoring(config) == []


def test_plan_path_for_date(tmp_path: Path, monkeypatch):
    config = CampaignOptConfig(course="sys_think")
    monkeypatch.setattr(
        config,
        "prod_dir",
        lambda base=None: tmp_path / "prod" if base is None else Path(base),
    )
    path = plan_path_for_date(config, pd.Timestamp("2026-06-08"))
    assert path == tmp_path / "prod" / "two_stage_plan" / "stage2_budgets" / "20260608" / "campaign_plan.csv"
