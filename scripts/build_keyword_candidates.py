#!/usr/bin/env python3
"""Build per-segment keyword-set candidates."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.keyword_candidates import build_segment_candidates
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build segment keyword-set candidates.")
    parser.add_argument("--course", default="sys_think")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    setup_tee_logging(log_file=None, default_log_prefix=f"build_candidates_{args.course}")

    candidates, extended_sets = build_segment_candidates(args.course, top_n=args.top_n)
    out_dir = Path("data") / args.course / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_dir / "segment-keyword-candidates.csv", index=False)
    extended_sets.to_csv(out_dir / "campaign-keyword-sets-extended.csv", index=False)
    print(f"Wrote {len(candidates)} candidate rows across {candidates['segment'].nunique()} segments.")


if __name__ == "__main__":
    main()
