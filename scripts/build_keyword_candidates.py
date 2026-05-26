#!/usr/bin/env python3
"""Build per-segment keyword-set candidates."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.decisions import parse_allowed_match_types, parse_excluded_regions
from campaign_opt.schema import default_config_path, load_campaign_config
from utils.keyword_candidates import build_segment_candidates
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build segment keyword-set candidates.")
    parser.add_argument("--course", default="sys_think")
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--set-size", type=int, default=0, help="Target keywords per synthetic set (0 = segment median)")
    parser.add_argument("--no-performance-synthetic", action="store_true")
    parser.add_argument("--no-semantic-synthetic", action="store_true")
    parser.add_argument("--no-dispersion-synthetic", action="store_true")
    parser.add_argument("--no-composite-synthetic", action="store_true")
    args = parser.parse_args()

    setup_tee_logging(log_file=None, default_log_prefix=f"build_candidates_{args.course}")

    config_path = Path(args.config) if args.config else default_config_path(args.course, args.exp_name)
    config = load_campaign_config(config_path)
    allowed_match_types = parse_allowed_match_types(config.constraints)
    excluded_regions = parse_excluded_regions(config.constraints)

    candidates, extended_sets = build_segment_candidates(
        args.course,
        top_n=args.top_n,
        set_size=args.set_size or None,
        allowed_match_types=allowed_match_types,
        excluded_regions=excluded_regions or None,
        include_performance_synthetic=not args.no_performance_synthetic,
        include_semantic_synthetic=not args.no_semantic_synthetic,
        include_dispersion_synthetic=not args.no_dispersion_synthetic,
        include_composite_synthetic=not args.no_composite_synthetic,
    )
    out_dir = Path("data") / args.course / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_dir / "segment-keyword-candidates.csv", index=False)
    extended_sets.to_csv(out_dir / "campaign-keyword-sets-extended.csv", index=False)
    print(f"Wrote {len(candidates)} candidate rows across {candidates['segment'].nunique()} segments.")


if __name__ == "__main__":
    main()
