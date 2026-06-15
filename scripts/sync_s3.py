#!/usr/bin/env python3
"""Sync production artifacts between local disk and S3.

Pull mode: download daily_metrics.csv and the last 7 days of campaign_plan.csv
           files from S3 into the expected local locations for the specified course.

Push mode: upload daily_metrics.csv, the last 7 days of campaign_plan.csv files,
           all plan_vs_actual CSVs, and rolling_summary.json to S3.

S3 key layout mirrors <course>/prod/:
  <course>/monitoring/daily_metrics.csv
  <course>/monitoring/rolling_summary.json
  <course>/monitoring/plan_vs_actual/YYYYMMDD/plan_vs_actual.csv
  <course>/two_stage_plan/stage2_budgets/YYYYMMDD/campaign_plan.csv

Usage:
  uv run python -m scripts.sync_s3 --course sys_think push my-s3-bucket
  uv run python -m scripts.sync_s3 --course sys_think pull my-s3-bucket
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import boto3

from utils.paths import add_course_arg, prod_dir, prod_monitoring_dir

_CAMPAIGN_PLAN_SUBPATH = "two_stage_plan/stage2_budgets"
_LOOKBACK_DAYS = 7


def _recent_date_strings(lookback_days: int = _LOOKBACK_DAYS) -> set[str]:
    """Return a set of YYYYMMDD strings for the last `lookback_days` days (inclusive of today)."""
    today = date.today()
    return {(today - timedelta(days=i)).strftime("%Y%m%d") for i in range(lookback_days)}


def _local_campaign_plan_dirs(course: str, *, lookback_days: int = _LOOKBACK_DAYS) -> list[Path]:
    """Return local YYYYMMDD dirs with a campaign_plan.csv within the lookback window."""
    root = prod_dir(course) / _CAMPAIGN_PLAN_SUBPATH
    if not root.is_dir():
        return []
    cutoff = _recent_date_strings(lookback_days)
    return sorted(
        d
        for d in root.iterdir()
        if d.is_dir() and d.name in cutoff and (d / "campaign_plan.csv").is_file()
    )


def _s3_campaign_plan_dates(
    s3_client, bucket: str, course: str, *, lookback_days: int = _LOOKBACK_DAYS
) -> list[str]:
    """Return YYYYMMDD date strings present on S3 within the lookback window, sorted ascending."""
    prefix = f"{course}/{_CAMPAIGN_PLAN_SUBPATH}/"
    paginator = s3_client.get_paginator("list_objects_v2")
    cutoff = _recent_date_strings(lookback_days)
    dates: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            date_part = cp["Prefix"].rstrip("/").split("/")[-1]
            if date_part in cutoff:
                dates.append(date_part)
    return sorted(dates)


def _upload(s3_client, local_path: Path, bucket: str, key: str) -> None:
    print(f"  upload  {local_path}  →  s3://{bucket}/{key}")
    s3_client.upload_file(str(local_path), bucket, key)


def _download(s3_client, bucket: str, key: str, local_path: Path) -> None:
    print(f"  download  s3://{bucket}/{key}  →  {local_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, str(local_path))


def push(course: str, bucket: str) -> None:
    """Upload local prod artifacts to S3."""
    s3 = boto3.client("s3")
    monitoring_dir = prod_monitoring_dir(course)

    # daily_metrics.csv
    daily_metrics = monitoring_dir / "daily_metrics.csv"
    if daily_metrics.is_file():
        _upload(s3, daily_metrics, bucket, f"{course}/monitoring/daily_metrics.csv")
    else:
        print(f"  [skip] {daily_metrics} not found")

    # rolling_summary.json
    rolling_summary = monitoring_dir / "rolling_summary.json"
    if rolling_summary.is_file():
        _upload(s3, rolling_summary, bucket, f"{course}/monitoring/rolling_summary.json")
    else:
        print(f"  [skip] {rolling_summary} not found")

    # Last 7 days of campaign_plan.csv files
    campaign_plan_dirs = _local_campaign_plan_dirs(course)
    if campaign_plan_dirs:
        for plan_dir in campaign_plan_dirs:
            key = f"{course}/{_CAMPAIGN_PLAN_SUBPATH}/{plan_dir.name}/campaign_plan.csv"
            _upload(s3, plan_dir / "campaign_plan.csv", bucket, key)
    else:
        print(f"  [skip] no recent campaign plans found under {prod_dir(course)}")

    # All plan_vs_actual CSVs
    pva_root = monitoring_dir / "plan_vs_actual"
    if pva_root.is_dir():
        for date_dir in sorted(pva_root.iterdir()):
            if not date_dir.is_dir():
                continue
            pva_csv = date_dir / "plan_vs_actual.csv"
            if pva_csv.is_file():
                key = f"{course}/monitoring/plan_vs_actual/{date_dir.name}/plan_vs_actual.csv"
                _upload(s3, pva_csv, bucket, key)
    else:
        print(f"  [skip] {pva_root} not found")

    print("Push complete.")


def pull(course: str, bucket: str) -> None:
    """Download S3 artifacts into the local prod tree for the given course."""
    s3 = boto3.client("s3")
    monitoring_dir = prod_monitoring_dir(course)

    # daily_metrics.csv
    _download(
        s3,
        bucket,
        f"{course}/monitoring/daily_metrics.csv",
        monitoring_dir / "daily_metrics.csv",
    )

    # Last 7 days of campaign_plan.csv files
    dates = _s3_campaign_plan_dates(s3, bucket, course)
    if dates:
        for date_str in dates:
            key = f"{course}/{_CAMPAIGN_PLAN_SUBPATH}/{date_str}/campaign_plan.csv"
            local_path = prod_dir(course) / _CAMPAIGN_PLAN_SUBPATH / date_str / "campaign_plan.csv"
            _download(s3, bucket, key, local_path)
    else:
        print(
            f"  [skip] no recent campaign plan dates found in "
            f"s3://{bucket}/{course}/{_CAMPAIGN_PLAN_SUBPATH}/"
        )

    print("Pull complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync production artifacts between local disk and S3.",
    )
    add_course_arg(parser)
    parser.add_argument(
        "mode",
        choices=["push", "pull"],
        help="push: local → S3; pull: S3 → local",
    )
    parser.add_argument("bucket", help="S3 bucket name")
    args = parser.parse_args()

    if args.mode == "push":
        push(args.course, args.bucket)
    else:
        pull(args.course, args.bucket)


if __name__ == "__main__":
    main()
