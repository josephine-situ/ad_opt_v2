#!/usr/bin/env python3
"""Generate campaign-day panel and campaign summary files."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COURSE_CONFIG
from utils.data_processing import generate_campaign_day_panel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate active campaign-day aggregates and campaign summary."
    )
    parser.add_argument(
        "--output-course",
        type=str,
        choices=sorted(COURSE_CONFIG.keys()),
        default="sys_think",
        help="Course key used to determine default input and output locations.",
    )
    parser.add_argument(
        "--kw-day-panel-file",
        type=str,
        default="",
        help="Processed kw-day-panel CSV. Defaults to data/<course>/processed/kw-day-panel.csv.",
    )
    parser.add_argument(
        "--campaign-summary-file",
        type=str,
        default="",
        help="Campaign summary CSV. Defaults to data/<course>/processed/campaign-summary.csv.",
    )
    parser.add_argument(
        "--campaign-day-output-file",
        type=str,
        default="",
        help="Campaign-day panel CSV. Defaults to data/<course>/processed/campaign-day-panel.csv.",
    )
    parser.add_argument(
        "--campaign-summary-output-file",
        type=str,
        default="",
        help="Optional campaign summary output CSV. Defaults to not rewriting it.",
    )

    args = parser.parse_args()
    course = args.output_course
    kw_day_panel_file = args.kw_day_panel_file or f"data/{course}/processed/kw-day-panel.csv"
    campaign_summary_file = (
        args.campaign_summary_file or f"data/{course}/processed/campaign-summary.csv"
    )
    campaign_day_output_file = (
        args.campaign_day_output_file or f"data/{course}/processed/campaign-day-panel.csv"
    )
    campaign_summary_output_file = args.campaign_summary_output_file or None

    print(f"Generating campaign-day panel from: {kw_day_panel_file}")
    print(f"Using campaign summary from: {campaign_summary_file}")
    campaign_day, campaign_summary = generate_campaign_day_panel(
        kw_day_panel_file=kw_day_panel_file,
        campaign_summary_file=campaign_summary_file,
        campaign_day_output_file=campaign_day_output_file,
        campaign_summary_output_file=campaign_summary_output_file,
    )
    print(f"Generated {len(campaign_day):,} campaign-day row(s): {campaign_day_output_file}")
    if campaign_summary_output_file:
        print(
            f"Generated {len(campaign_summary):,} campaign summary row(s): "
            f"{campaign_summary_output_file}"
        )


if __name__ == "__main__":
    main()
