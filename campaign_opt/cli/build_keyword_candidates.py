#!/usr/bin/env python3
"""Build per-segment keyword-set candidates."""

from __future__ import annotations

import argparse

from campaign_opt.decisions import parse_allowed_match_types, parse_excluded_regions
from campaign_opt.schema import default_config_path, load_campaign_config
from config import COURSE
from utils.keyword_candidates import (
    DEFAULT_TOP_N_VALUES,
    build_segment_candidates,
    verify_segment_keyword_candidates,
    write_segment_keyword_candidate_files,
)
from utils.tee_logging import setup_tee_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build segment keyword-set candidates.")
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--top-n", type=int, default=30, help="Panel rank cap when --top-n-values is not set")
    parser.add_argument(
        "--top-n-values",
        default=",".join(str(n) for n in DEFAULT_TOP_N_VALUES),
        help="Comma-separated caps for separate synthetic sets (default: 10,20,40)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After writing, run verify_segment_keyword_candidates and exit non-zero on failure",
    )
    parser.add_argument("--set-size", type=int, default=0, help="Target keywords per synthetic set (0 = segment median)")
    parser.add_argument("--no-top-conv-synthetic", action="store_true")
    parser.add_argument("--no-allowlist-synthetic", action="store_true")
    parser.add_argument("--no-semantic-synthetic", action="store_true")
    parser.add_argument("--no-dispersion-synthetic", action="store_true")
    parser.add_argument("--no-composite-synthetic", action="store_true")
    args = parser.parse_args()

    setup_tee_logging(log_file=None, default_log_prefix="build_candidates")

    config_path = default_config_path(args.exp_name) if not args.config else args.config
    config = load_campaign_config(config_path)
    allowed_match_types = parse_allowed_match_types(config.constraints)
    excluded_regions = parse_excluded_regions(config.constraints)

    top_n_values = [int(x.strip()) for x in args.top_n_values.split(",") if x.strip()]
    candidates, extended_sets = build_segment_candidates(
        COURSE,
        top_n=args.top_n,
        top_n_values=top_n_values or None,
        set_size=args.set_size or None,
        allowed_match_types=allowed_match_types,
        excluded_regions=excluded_regions or None,
        include_top_conv_synthetic=not args.no_top_conv_synthetic,
        include_allowlist_synthetic=not args.no_allowlist_synthetic,
        include_semantic_synthetic=not args.no_semantic_synthetic,
        include_dispersion_synthetic=not args.no_dispersion_synthetic,
        include_composite_synthetic=not args.no_composite_synthetic,
    )
    _, _, display_dir = write_segment_keyword_candidate_files(COURSE, candidates, extended_sets)
    print(f"Wrote {len(candidates)} candidate rows across {candidates['segment'].nunique()} segments.")
    print(f"Updated keyword-sets-display at {display_dir}")

    if args.verify:
        caps = tuple(top_n_values or list(DEFAULT_TOP_N_VALUES))
        issues = verify_segment_keyword_candidates(COURSE, candidates, extended_sets, top_n_values=caps)
        if issues:
            print("Verification failed:")
            for issue in issues:
                print(f"  - {issue}")
            raise SystemExit(1)
        print("Verification passed.")


if __name__ == "__main__":
    main()
