"""Tests for backtest result compilation and summarization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from campaign_opt.backtest_analysis import (
    analyze_backtest_run,
    compile_evaluation_results,
    regional_breakdown,
    summarize_performance,
)


def test_compile_and_summarize_from_plan_vs_actual(tmp_path: Path):
    bt_dir = tmp_path / "backtest" / "2025-10-01_2025-10-03"
    day_dir = bt_dir / "plans" / "20251001"
    day_dir.mkdir(parents=True)

    plan_vs = pd.DataFrame(
        {
            "segment": ["USA / Broad", "A / Phrase; Exact"],
            "daily_budget": [100.0, 50.0],
            "actual_budget": [90.0, 55.0],
            "pred_lift": [1.0, 2.0],
            "actual_model_lift": [0.8, 1.5],
            "observed_clicks": [10.0, 5.0],
            "date": ["2025-10-01", "2025-10-01"],
        }
    )
    plan_vs.to_csv(day_dir / "plan_vs_actual.csv", index=False)

    eval_df = compile_evaluation_results(bt_dir, target="clicks")
    assert len(eval_df) == 1
    assert eval_df.iloc[0]["pred_lift_total"] == 3.0
    assert eval_df.iloc[0]["observed_total"] == 15.0

    summary = summarize_performance(eval_df, target="clicks")
    model_row = summary.loc[summary["scenario"] == "Model"].iloc[0]
    actual_row = summary.loc[summary["scenario"] == "Actual"].iloc[0]
    assert model_row["conversions"] == 3.0
    assert actual_row["conversions"] == 2.3
    assert model_row["improvement_conversions_pct"] == pytest.approx(100 * (3.0 - 2.3) / 2.3)

    plan_vs["row_kind"] = "plan"
    plan_vs["pred_lift_raw"] = plan_vs["pred_lift"]
    market = pd.DataFrame(
        {
            "segment": ["USA / Broad; Phrase; Exact"],
            "daily_budget": [100.0],
            "actual_model_lift": [0.0],
            "actual_model_lift_raw": [-2.0],
            "observed_clicks": [10.0],
            "date": ["2025-10-01"],
            "row_kind": ["market"],
        }
    )
    combined = pd.concat([plan_vs, market], ignore_index=True)
    regional = regional_breakdown(combined, target="clicks")
    assert len(regional) == 2
    assert set(regional["scenario"]) == {"Model", "Actual"}
    actual_row = regional.loc[regional["scenario"] == "Actual"].iloc[0]
    assert actual_row["Conversions USA"] == pytest.approx(1.0)


def test_summarize_prefers_raw_lift_totals():
    eval_df = pd.DataFrame(
        {
            "Day": ["2025-01-01", "2025-01-02"],
            "pred_lift_total": [1.0, 1.0],
            "pred_lift_raw_total": [2.0, 4.0],
            "actual_model_lift_total": [0.0, 0.0],
            "actual_model_lift_raw_total": [-1.0, -3.0],
            "opt_budget_total": [100.0, 100.0],
            "act_budget_total": [90.0, 90.0],
        }
    )
    summary = summarize_performance(eval_df, target="clicks")
    assert summary.loc[summary["scenario"] == "Model", "conversions"].iloc[0] == 3.0
    assert summary.loc[summary["scenario"] == "Actual", "conversions"].iloc[0] == -2.0
    assert summary["lift_source"].iloc[0] == "raw"


def test_compile_empty_summary_falls_back_to_campaign_plans(tmp_path: Path):
    bt_dir = tmp_path / "backtest" / "2026-03-01_2026-03-02"
    day_dir = bt_dir / "plans" / "20260301"
    day_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "segment": ["A / Broad"],
            "daily_budget": [40.0],
            "keyword_set_id": ["ks1"],
            "opt_date": ["2026-03-01"],
        }
    ).to_csv(day_dir / "campaign_plan.csv", index=False)
    (bt_dir / "daily_backtest_summary.csv").write_text("", encoding="utf-8")

    eval_df = compile_evaluation_results(bt_dir, target="clicks")
    assert len(eval_df) == 1
    assert eval_df.iloc[0]["Day"] == "2026-03-01"
    assert eval_df.iloc[0]["opt_budget_total"] == 40.0


def test_analyze_backtest_run_writes_outputs(tmp_path: Path):
    bt_dir = tmp_path / "backtest" / "2025-10-06_2025-10-12"
    day_dir = bt_dir / "plans" / "20251006"
    day_dir.mkdir(parents=True)
    (day_dir / "campaign_plan.csv").write_text("segment,daily_budget,keyword_set_id\n", encoding="utf-8")

    plan_vs = pd.DataFrame(
        {
            "segment": ["USA / Broad"],
            "daily_budget": [120.0],
            "actual_budget": [100.0],
            "pred_lift": [0.5],
            "actual_model_lift": [0.4],
            "observed_clicks": [3.0],
            "date": ["2025-10-06"],
        }
    )
    plan_vs.to_csv(day_dir / "plan_vs_actual.csv", index=False)
    (bt_dir / "backtest_config.json").write_text(
        '{"start_day":"2025-10-06","end_day":"2025-10-12","target":"clicks"}',
        encoding="utf-8",
    )

    result = analyze_backtest_run(bt_dir, target="clicks", write_latex=True)
    assert (bt_dir / "backtest_summary.csv").exists()
    assert (bt_dir / "evaluation_results.csv").exists()
    assert result["n_days_evaluated"] == 1
