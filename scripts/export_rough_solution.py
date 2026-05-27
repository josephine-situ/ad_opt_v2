#!/usr/bin/env python3
"""Export stage-1 (or daily) campaign plan as a rough deployable solution folder."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.keyword_sets_display import (
    keyword_set_display_frame,
    load_keyword_sets_table,
)


def _segment_filename(segment: str) -> str:
    """Filesystem-safe name derived from segment label."""
    name = str(segment).strip()
    name = name.replace(" / ", "_")
    name = re.sub(r"[;:/\\]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    return f"{name}.csv"


def export_rough_solution(
    plan_path: Path,
    output_dir: Path,
    *,
    course: str,
) -> Path:
    """
    Write ``segment_budgets.csv`` and one keyword CSV per segment (Broad / Phrase / Exact).

    Uses ``daily_budget`` from the plan (mean over planning days for multi-day stage-1).
    """
    plan = pd.read_csv(plan_path)
    for col in ("segment", "daily_budget", "keyword_set_id"):
        if col not in plan.columns:
            raise ValueError(f"{plan_path} is missing required column: {col}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    budgets = plan[["segment", "daily_budget"]].copy()
    budgets["avg_daily_budget"] = pd.to_numeric(budgets["daily_budget"], errors="coerce").round(2)
    budgets = budgets[["segment", "avg_daily_budget"]]
    budgets.to_csv(output_dir / "segment_budgets.csv", index=False)

    sets_df = load_keyword_sets_table(course)
    sets_df = sets_df.set_index(sets_df["keyword_set_id"].astype(str))

    for _, row in plan.iterrows():
        segment = str(row["segment"])
        set_id = str(row["keyword_set_id"])
        if set_id not in sets_df.index:
            raise ValueError(
                f"Keyword set {set_id!r} for segment {segment!r} not found in keyword sets table"
            )
        display_df = keyword_set_display_frame(sets_df.loc[set_id])
        display_df.to_csv(output_dir / _segment_filename(segment), index=False)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export segment budgets and per-segment keyword CSVs from a campaign plan."
    )
    parser.add_argument("--course", default="sys_think")
    parser.add_argument(
        "--campaign-plan",
        required=True,
        help="campaign_plan.csv or keyword_set_plan.csv from stage-1 or a daily plan folder",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output folder (default: <plan_parent>/rough_solution)",
    )
    args = parser.parse_args()

    plan_path = Path(args.campaign_plan)
    out = Path(args.output_dir) if args.output_dir else plan_path.parent / "rough_solution"
    out_dir = export_rough_solution(plan_path, out, course=args.course)
    n_segments = len(pd.read_csv(out_dir / "segment_budgets.csv"))
    print(f"Wrote rough solution for {n_segments} segment(s) to {out_dir}")


if __name__ == "__main__":
    main()
