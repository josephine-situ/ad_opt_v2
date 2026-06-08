#!/usr/bin/env python3
"""Aggregate GKP keyword stats to keyword-set level (run before modeling)."""

from __future__ import annotations

import argparse

from config import COURSE, COURSE_CONFIG
from utils.paths import data_path, gkp_dir
from utils.campaign_features import build_keyword_set_feature_table
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GKP + semantic keyword-set features.")
    parser.add_argument(
        "--course",
        default=COURSE,
        choices=sorted(COURSE_CONFIG.keys()),
        help=f"Course key (default: {COURSE})",
    )
    args = parser.parse_args()

    setup_tee_logging(log_file=None, default_log_prefix="gkp_features")

    gkp = gkp_dir(args.course)
    if not gkp.exists():
        print(f"[Warn] GKP dir missing: {gkp}. Run pull_input_data with keyword_planning dataset.")
        print("       Writing semantic-only keyword-set features.")

    table = build_keyword_set_feature_table(args.course)
    out = data_path(args.course, "processed", "keyword-set-features.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {len(table)} keyword-set feature rows to {out}")


if __name__ == "__main__":
    main()
