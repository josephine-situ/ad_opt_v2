#!/usr/bin/env python3
"""Clean pulled Google Ads input data."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COURSE_CONFIG
from utils.data_processing import clean_kw_day_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Process pulled Google Ads input data")
    parser.add_argument(
        "--output-course",
        type=str,
        choices=sorted(COURSE_CONFIG.keys()),
        required=True,
        help="Course key used to determine default input and output locations.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="",
        help=(
            "Raw Search keyword CSV. Defaults to "
            "data/<course>/reports/Search keyword - raw input to models.csv."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Cleaned kw-day-panel CSV. Defaults to data/<course>/processed/kw-day-panel.csv.",
    )

    args = parser.parse_args()
    input_file = (
        args.input_file
        or f"data/{args.output_course}/reports/Search keyword - raw input to models.csv"
    )
    output_file = args.output_file or f"data/{args.output_course}/processed/kw-day-panel.csv"

    print(f"Cleaning kw-day-panel: {input_file}")
    clean_kw_day_panel(input_file, output_file)
    print(f"Generated: {output_file}")


if __name__ == "__main__":
    main()
