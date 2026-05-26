"""Tests for backtest result compilation and summarization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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
    assert summary.iloc[0]["avg pred_lift (opt)"] == 3.0

    regional = regional_breakdown(plan_vs, target="clicks")
    assert len(regional) == 2
    assert set(regional["scenario"]) == {"Opt", "Act"}


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
