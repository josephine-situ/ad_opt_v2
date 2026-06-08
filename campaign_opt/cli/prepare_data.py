#!/usr/bin/env python3
"""Orchestrate input data preparation for the sys_think campaign pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys

from campaign_opt.paths import PROCESSED_DIR, REPORTS_DIR, data_path
from utils.tee_logging import setup_tee_logging


def _run(cmd: list[str]) -> None:
    print(f"[prepare-data] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sys_think campaign input data.")
    parser.add_argument(
        "--skip-pull",
        action="store_true",
        help="Skip Google Ads API pull (use existing reports/)",
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Skip kw-day-panel cleaning",
    )
    parser.add_argument(
        "--skip-panel",
        action="store_true",
        help="Skip campaign-day panel generation",
    )
    parser.add_argument("--datasets", default="campaign_opt", help="Comma-separated datasets for pull")
    parser.add_argument("--google-ads-yaml", default="", help="Google Ads YAML config (required unless --skip-pull)")
    parser.add_argument("--customer-id", default="", help="Google Ads customer ID (required unless --skip-pull)")
    parser.add_argument("--start-date", default="", help="YYYY-MM-DD start for API pull")
    parser.add_argument("--end-date", default="", help="YYYY-MM-DD end for API pull")
    args = parser.parse_args()

    setup_tee_logging(log_file=None, default_log_prefix="prepare_data")
    py = sys.executable

    if not args.skip_pull:
        if not args.google_ads_yaml or not args.customer_id:
            parser.error("--google-ads-yaml and --customer-id are required unless --skip-pull")
        pull_cmd = [
            py,
            "-m",
            "campaign_opt.cli.pull_input_data",
            "--datasets",
            args.datasets,
            "--google-ads-yaml",
            args.google_ads_yaml,
            "--customer-id",
            args.customer_id,
        ]
        if args.start_date:
            pull_cmd.extend(["--start-date", args.start_date])
        if args.end_date:
            pull_cmd.extend(["--end-date", args.end_date])
        _run(pull_cmd)

    if not args.skip_process:
        reports_kw = REPORTS_DIR / "kw-day-panel.csv"
        if not reports_kw.exists():
            raise FileNotFoundError(f"Missing {reports_kw}; run pull step first or pass --skip-pull only after pulling")
        _run([py, "-m", "campaign_opt.cli.process_input_data"])

    if not args.skip_panel:
        kw_processed = data_path("processed", "kw-day-panel.csv")
        summary = PROCESSED_DIR / "campaign-summary.csv"
        if not kw_processed.exists():
            raise FileNotFoundError(f"Missing {kw_processed}; run process step first")
        if not summary.exists():
            raise FileNotFoundError(
                f"Missing {summary}; parse change-history HTML to build campaign-summary.csv first"
            )
        _run([py, "-m", "campaign_opt.cli.generate_campaign_day_panel"])

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
