#!/usr/bin/env python3
"""Export one CSV per keyword set with Broad / Phrase / Exact columns for spreadsheet display."""

from __future__ import annotations

import argparse
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


def export_keyword_sets_display(
    course: str,
    *,
    output_dir: Path | None = None,
) -> Path:
    sets_df = load_keyword_sets_table(course)
    missing = [col for col in MATCH_TYPE_LIST_COLS.values() if col not in sets_df.columns]
    if missing:
        raise ValueError(f"Keyword sets table missing columns: {', '.join(missing)}")

    out_dir = output_dir or (data_paths(course)["processed"] / "keyword-sets-display")
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in sets_df.iterrows():
        set_id = str(row["keyword_set_id"])
        display_df = keyword_set_display_frame(row)
        display_df.to_csv(out_dir / f"{set_id}.csv", index=False)

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
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else None
    out_dir = export_keyword_sets_display(args.course, output_dir=out)
    n_files = len(list(out_dir.glob("*.csv")))
    print(f"Wrote {n_files} keyword-set display CSV(s) to {out_dir}")


if __name__ == "__main__":
    main()
