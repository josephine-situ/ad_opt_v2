#!/usr/bin/env python3
"""Generate campaign-day panel from kw-day-panel and campaign summary."""

from __future__ import annotations

import argparse

from campaign_opt.paths import PROCESSED_DIR, data_path
from utils.data_processing import generate_campaign_day_panel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate active campaign-day aggregates and campaign summary."
    )
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
    kw_path = args.kw_day_panel_file or str(data_path("processed", "kw-day-panel.csv"))
    summary_path = args.campaign_summary_file or str(PROCESSED_DIR / "campaign-summary.csv")
    panel_out = args.campaign_day_output_file or str(data_path("processed", "campaign-day-panel.csv"))
    summary_out = args.campaign_summary_output_file or None

    generate_campaign_day_panel(
        kw_path,
        summary_path,
        panel_out,
        campaign_summary_output_file=summary_out,
    )
    print(f"Generated campaign-day panel: {panel_out}")


if __name__ == "__main__":
    main()
