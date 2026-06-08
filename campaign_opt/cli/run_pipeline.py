#!/usr/bin/env python3
"""End-to-end two-stage campaign optimization pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys

from campaign_opt.paths import PROCESSED_DIR
from campaign_opt.schema import default_config_path, load_campaign_config
from config import COURSE_CONFIG
from utils.keyword_candidates import DEFAULT_TOP_N_VALUES
from utils.tee_logging import setup_tee_logging


def _run(cmd: list[str]) -> None:
    print(f"[Pipeline] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full two-stage campaign optimization pipeline.")
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--planning-date", default="")
    parser.add_argument("--window-start", required=True, help="Stage-1 window start YYYY-MM-DD")
    parser.add_argument("--window-end", required=True, help="Stage-1 window end YYYY-MM-DD")
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Reuse fixed_keyword_sets.json from exp_dir/two_stage_plan",
    )
    parser.add_argument("--skip-gkp", action="store_true")
    parser.add_argument("--skip-candidates", action="store_true")
    args = parser.parse_args()

    config_path = default_config_path(args.exp_name) if not args.config else args.config
    config = load_campaign_config(config_path)
    py = sys.executable
    top_n_arg = ",".join(str(n) for n in DEFAULT_TOP_N_VALUES)

    log_path = setup_tee_logging(log_file=None, default_log_prefix="run_campaign_pipeline")
    print(f"Log file: {log_path}")
    print(f"Config: {config_path}")

    panel_path = PROCESSED_DIR / "campaign-day-panel.csv"
    if not panel_path.exists():
        _run([py, "-m", "campaign_opt.cli.generate_campaign_day_panel"])

    if not args.skip_candidates:
        _run(
            [
                py,
                "-m",
                "campaign_opt.cli.build_keyword_candidates",
                "--top-n-values",
                top_n_arg,
                "--verify",
            ]
        )

    if not args.skip_gkp:
        _run([py, "-m", "campaign_opt.cli.build_gkp_set_features"])

    _run([py, "-m", "campaign_opt.cli.fit_models", "--config", str(config_path)])

    default_b = COURSE_CONFIG[config.course].get("campaign_budget", 400.0)
    budget = args.budget or default_b

    plan_cmd = [
        py,
        "-m",
        "campaign_opt.cli.plan_two_stage_campaign",
        "--config",
        str(config_path),
        "--window-start",
        args.window_start,
        "--window-end",
        args.window_end,
        "--budget",
        str(budget),
    ]
    if args.planning_date:
        plan_cmd.extend(["--planning-date", args.planning_date])
    if args.skip_stage1:
        plan_cmd.append("--skip-stage1")
    _run(plan_cmd)
    print(f"Two-stage pipeline finished. Budget cap: ${budget:.0f}")


if __name__ == "__main__":
    main()
