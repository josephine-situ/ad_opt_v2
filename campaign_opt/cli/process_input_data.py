#!/usr/bin/env python3
"""Clean pulled Google Ads kw-day-panel into the processed panel."""

from __future__ import annotations

import argparse

from campaign_opt.cli.course_arg import add_course_arg
from campaign_opt.paths import data_path, reports_dir
from utils.data_processing import clean_kw_day_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean pulled Google Ads kw-day-panel")
    add_course_arg(parser)
    parser.add_argument(
        "--input-file",
        default="",
        help="API kw-day-panel CSV. Defaults to <course>/data/reports/kw-day-panel.csv.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Cleaned kw-day-panel CSV. Defaults to <course>/data/processed/kw-day-panel.csv.",
    )

    args = parser.parse_args()
    input_file = args.input_file or str(reports_dir(args.course) / "kw-day-panel.csv")
    output_file = args.output_file or str(data_path(args.course, "processed", "kw-day-panel.csv"))

    print(f"Cleaning kw-day-panel: {input_file}")
    clean_kw_day_panel(input_file, output_file)
    print(f"Generated: {output_file}")


if __name__ == "__main__":
    main()
