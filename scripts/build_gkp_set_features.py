#!/usr/bin/env python3
"""Aggregate GKP keyword stats to keyword-set level (run before modeling)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.campaign_features import build_keyword_set_feature_table, load_keyword_sets
from utils.gkp_features import aggregate_gkp_to_keyword_sets, load_gkp_keyword_stats
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GKP + semantic keyword-set features.")
    parser.add_argument("--course", default="sys_think")
    args = parser.parse_args()

    setup_tee_logging(log_file=None, default_log_prefix=f"gkp_features_{args.course}")

    gkp_dir = Path("data") / args.course / "gkp"
    if not gkp_dir.exists():
        print(f"[Warn] GKP dir missing: {gkp_dir}. Run pull_input_data.py --datasets keyword_planning")
        print("       Writing semantic-only keyword-set features.")

    table = build_keyword_set_feature_table(args.course)
    out = Path("data") / args.course / "processed" / "keyword-set-features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {len(table)} keyword-set feature rows to {out}")


if __name__ == "__main__":
    main()
