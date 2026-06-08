"""Shared --course CLI argument."""

from __future__ import annotations

import argparse

from config import COURSE, COURSE_CONFIG


def add_course_arg(parser: argparse.ArgumentParser, *, default: str = COURSE) -> None:
    parser.add_argument(
        "--course",
        default=default,
        choices=sorted(COURSE_CONFIG.keys()),
        help=f"Course key (default: {default})",
    )
