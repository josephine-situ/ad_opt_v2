#!/usr/bin/env python3
"""Generate campaign-day panel from kw-day-panel and campaign summary."""

from __future__ import annotations

import argparse

from utils.campaign_features import (
    export_segment_conv_per_click_rates,
    segment_conv_per_click_rates_path,
)
from utils.paths import add_course_arg
from utils.paths import data_path, processed_dir
from utils.data_processing import generate_campaign_day_panel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate active campaign-day aggregates and campaign summary."
    )
    add_course_arg(parser)
    parser.add_argument(
        "--kw-day-panel-file",
        default="",
        help="Processed kw-day-panel CSV.",
    )
    parser.add_argument(
        "--campaign-summary-file",
        default="",
        help="Campaign summary CSV.",
    )
    parser.add_argument(
        "--campaign-day-output-file",
        default="",
        help="Campaign-day panel CSV.",
    )
    parser.add_argument(
        "--campaign-summary-output-file",
        default="",
        help="Optional campaign summary output CSV. Defaults to not rewriting it.",
    )

    args = parser.parse_args()
    kw_path = args.kw_day_panel_file or str(data_path(args.course, "processed", "kw-day-panel.csv"))
    summary_path = args.campaign_summary_file or str(processed_dir(args.course) / "campaign-summary.csv")
    panel_out = args.campaign_day_output_file or str(
        data_path(args.course, "processed", "campaign-day-panel.csv")
    )
    summary_out = args.campaign_summary_output_file or None

    campaign_day, _ = generate_campaign_day_panel(
        kw_path,
        summary_path,
        panel_out,
        campaign_summary_output_file=summary_out,
    )
    print(f"Generated campaign-day panel: {panel_out}")

    if "all_conv" in campaign_day.columns:
        rates_path = segment_conv_per_click_rates_path(args.course)
        export_segment_conv_per_click_rates(campaign_day, rates_path)
        print(f"Exported segment conv/click rates: {rates_path}")


if __name__ == "__main__":
    main()
