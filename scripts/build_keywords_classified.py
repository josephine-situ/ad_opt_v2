#!/usr/bin/env python3
"""Build keywords_classified.csv from existing Search campaign keywords only."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COURSE_CONFIG
from utils.keyword_classification import (
    EXISTING_ORIGIN,
    write_keywords_classified,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write data/<course>/gkp/keywords_classified.csv using only keywords "
            "already in Search campaigns (Origin='existing'). "
            "Does not include search terms or Semrush/new candidates."
        )
    )
    parser.add_argument(
        "--course",
        type=str,
        choices=sorted(COURSE_CONFIG.keys()),
        default="sys_think",
        help="Course key (default: sys_think).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output CSV path (default: data/<course>/gkp/keywords_classified.csv).",
    )
    parser.add_argument(
        "--min-clicks",
        type=int,
        default=1,
        help="Keep only keyword rows with more than this many clicks (default: 1).",
    )
    parser.add_argument(
        "--no-keyword-sets",
        action="store_true",
        help="Do not merge keywords from processed/campaign-keyword-sets.csv.",
    )
    args = parser.parse_args()

    output = write_keywords_classified(
        args.course,
        args.output or None,
        min_clicks=args.min_clicks,
        include_keyword_sets=not args.no_keyword_sets,
    )
    import pandas as pd

    frame = pd.read_csv(output)
    print(f"Wrote {len(frame)} keywords to {output}")
    print(f"  Origin={EXISTING_ORIGIN!r}: {int((frame['Origin'] == EXISTING_ORIGIN).sum())}")


if __name__ == "__main__":
    main()
