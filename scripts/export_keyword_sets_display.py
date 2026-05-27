#!/usr/bin/env python3
"""Export one CSV per keyword set with Broad / Phrase / Exact columns for spreadsheet display."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.keyword_sets_display import (
    export_keyword_sets_display,
    keyword_set_ids_from_fixed,
    keyword_set_ids_from_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write one display CSV per keyword set (Broad / Phrase / Exact columns)."
    )
    parser.add_argument("--course", default="sys_think")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Override output directory (default: data/<course>/processed/keyword-sets-display)",
    )
    parser.add_argument(
        "--keyword-set-plan",
        default="",
        help="Stage-1 keyword_set_plan.csv; export only its keyword_set_id values",
    )
    parser.add_argument(
        "--fixed-keyword-sets",
        default="",
        help="fixed_keyword_sets.json from a two-stage backtest (alternative to --keyword-set-plan)",
    )
    parser.add_argument(
        "--from-candidates",
        action="store_true",
        help="Export only keyword_set_id rows in segment-keyword-candidates.csv",
    )
    args = parser.parse_args()

    keyword_set_ids: list[str] | None = None
    segment_plan: pd.DataFrame | None = None
    if args.from_candidates:
        cand_path = Path("data") / args.course / "processed" / "segment-keyword-candidates.csv"
        segment_plan = pd.read_csv(cand_path)
        keyword_set_ids = keyword_set_ids_from_plan(cand_path)
    elif args.keyword_set_plan:
        plan_path = Path(args.keyword_set_plan)
        keyword_set_ids = keyword_set_ids_from_plan(plan_path)
        segment_plan = pd.read_csv(plan_path)
    elif args.fixed_keyword_sets:
        keyword_set_ids = keyword_set_ids_from_fixed(Path(args.fixed_keyword_sets))

    out = Path(args.output_dir) if args.output_dir else None
    out_dir = export_keyword_sets_display(
        args.course,
        output_dir=out,
        keyword_set_ids=keyword_set_ids,
        segment_plan=segment_plan,
    )
    set_files = [p for p in out_dir.glob("*.csv") if p.name != "segment_index.csv"]
    print(f"Wrote {len(set_files)} keyword-set display CSV(s) to {out_dir}")


if __name__ == "__main__":
    main()
