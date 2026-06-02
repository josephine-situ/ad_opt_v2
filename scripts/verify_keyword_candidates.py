#!/usr/bin/env python3
"""Verify segment-keyword-candidates.csv matches backtest expectations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.keyword_candidates import DEFAULT_TOP_N_VALUES, verify_segment_keyword_candidates
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify keyword-set candidate tables.")
    parser.add_argument("--course", default="sys_think")
    parser.add_argument(
        "--top-n-values",
        default=",".join(str(n) for n in DEFAULT_TOP_N_VALUES),
        help="Expected synthetic caps (default: 10,20,40)",
    )
    args = parser.parse_args()

    setup_tee_logging(log_file=None, default_log_prefix=f"verify_candidates_{args.course}")
    caps = tuple(int(x.strip()) for x in args.top_n_values.split(",") if x.strip())
    issues = verify_segment_keyword_candidates(args.course, top_n_values=caps)
    if issues:
        print("Verification failed:")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)
    print(f"Verification passed ({args.course}, caps={caps}).")


if __name__ == "__main__":
    main()
