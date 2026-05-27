#!/usr/bin/env python3
"""Export one CSV per keyword set with Broad / Phrase / Exact columns for spreadsheet display."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.campaign_features import MATCH_TYPE_LIST_COLS, data_paths


def _parse_keyword_list(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return sorted(k.strip() for k in str(value).split(";") if k.strip())


def keyword_set_display_frame(row: pd.Series) -> pd.DataFrame:
    """One row per keyword index; columns are match types (Broad, Phrase, Exact)."""
    by_type = {mt: _parse_keyword_list(row[col]) for mt, col in MATCH_TYPE_LIST_COLS.items()}
    n_rows = max((len(words) for words in by_type.values()), default=0)
    return pd.DataFrame({mt: (by_type[mt] + [""] * n_rows)[:n_rows] for mt in MATCH_TYPE_LIST_COLS})


def load_keyword_sets_table(course: str) -> pd.DataFrame:
    processed = data_paths(course)["processed"]
    ext_path = processed / "campaign-keyword-sets-extended.csv"
    base_path = processed / "campaign-keyword-sets.csv"
    path = ext_path if ext_path.exists() else base_path
    if not path.exists():
        raise FileNotFoundError(f"No keyword sets file at {ext_path} or {base_path}")
    return pd.read_csv(path)


def keyword_set_ids_from_plan(plan_path: Path) -> list[str]:
    plan = pd.read_csv(plan_path)
    if "keyword_set_id" not in plan.columns:
        raise ValueError(f"{plan_path} is missing keyword_set_id column")
    return sorted(plan["keyword_set_id"].dropna().astype(str).unique().tolist())


def keyword_set_ids_from_fixed(path: Path) -> list[str]:
    import json

    with open(path, encoding="utf-8") as f:
        mapping = json.load(f)
    return sorted({str(v) for v in mapping.values() if v})


def export_keyword_sets_display(
    course: str,
    *,
    output_dir: Path | None = None,
    keyword_set_ids: list[str] | None = None,
    segment_plan: pd.DataFrame | None = None,
) -> Path:
    sets_df = load_keyword_sets_table(course)
    missing = [col for col in MATCH_TYPE_LIST_COLS.values() if col not in sets_df.columns]
    if missing:
        raise ValueError(f"Keyword sets table missing columns: {', '.join(missing)}")

    if keyword_set_ids is not None:
        wanted = {str(k) for k in keyword_set_ids}
        sets_df = sets_df[sets_df["keyword_set_id"].astype(str).isin(wanted)].copy()
        missing_ids = wanted - set(sets_df["keyword_set_id"].astype(str))
        if missing_ids:
            raise ValueError(f"Keyword set id(s) not found in keyword sets table: {sorted(missing_ids)}")

    out_dir = output_dir or (data_paths(course)["processed"] / "keyword-sets-display")
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, str]] = []
    for _, row in sets_df.iterrows():
        set_id = str(row["keyword_set_id"])
        display_df = keyword_set_display_frame(row)
        display_path = out_dir / f"{set_id}.csv"
        display_df.to_csv(display_path, index=False)
        if segment_plan is not None and "segment" in segment_plan.columns:
            seg_rows = segment_plan[segment_plan["keyword_set_id"].astype(str) == set_id]
            for _, seg_row in seg_rows.iterrows():
                index_rows.append(
                    {
                        "segment": str(seg_row["segment"]),
                        "keyword_set_id": set_id,
                        "display_file": display_path.name,
                    }
                )
        else:
            index_rows.append({"segment": "", "keyword_set_id": set_id, "display_file": display_path.name})

    if index_rows:
        pd.DataFrame(index_rows).to_csv(out_dir / "segment_index.csv", index=False)

    return out_dir


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
    args = parser.parse_args()

    keyword_set_ids: list[str] | None = None
    segment_plan: pd.DataFrame | None = None
    if args.keyword_set_plan:
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
    n_files = len(list(out_dir.glob("*.csv")))
    print(f"Wrote {n_files} keyword-set display CSV(s) to {out_dir}")


if __name__ == "__main__":
    main()
