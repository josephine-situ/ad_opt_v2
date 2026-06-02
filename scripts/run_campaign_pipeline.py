#!/usr/bin/env python3
"""End-to-end campaign optimization pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaign_opt.schema import default_config_path, load_campaign_config
from config import COURSE_CONFIG
from utils.keyword_candidates import DEFAULT_TOP_N_VALUES
from utils.tee_logging import setup_tee_logging


def _run(cmd: list[str]) -> None:
    print(f"[Pipeline] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full campaign optimization pipeline.")
    parser.add_argument("--course", default="sys_think")
    parser.add_argument("--config", default="")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--planning-date", default="")
    parser.add_argument(
        "--strategy",
        choices=("single", "two_stage"),
        default="single",
        help="single: one MILP for planning date; two_stage: fix sets then daily budgets",
    )
    parser.add_argument(
        "--window-start",
        default="",
        help="Two-stage stage-1 window start YYYY-MM-DD (required for two_stage)",
    )
    parser.add_argument(
        "--window-end",
        default="",
        help="Two-stage stage-1 window end YYYY-MM-DD (required for two_stage)",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Two-stage: reuse fixed_keyword_sets.json from exp_dir/two_stage_plan",
    )
    parser.add_argument("--skip-gkp", action="store_true")
    parser.add_argument("--skip-candidates", action="store_true")
    parser.add_argument(
        "--skip-verify-candidates",
        action="store_true",
        help="Skip verify_keyword_candidates after building candidates",
    )
    args = parser.parse_args()

    if args.strategy == "two_stage" and not args.skip_stage1:
        if not args.window_start or not args.window_end:
            parser.error("--window-start and --window-end are required for --strategy two_stage")

    config_path = Path(args.config) if args.config else default_config_path(args.course, args.exp_name)
    config = load_campaign_config(config_path)
    py = sys.executable
    top_n_arg = ",".join(str(n) for n in DEFAULT_TOP_N_VALUES)

    log_path = setup_tee_logging(
        log_file=None,
        default_log_prefix=f"run_campaign_pipeline_{config.course}",
    )
    print(f"Log file: {log_path}")
    print(f"Config: {config_path}")
    print(f"Strategy: {args.strategy}")

    panel_script = [py, "scripts/generate_campaign_day_panel.py", "--output-course", config.course]
    if not (Path("data") / config.course / "processed" / "campaign-day-panel.csv").exists():
        _run(panel_script)

    if not args.skip_candidates:
        _run(
            [
                py,
                "scripts/build_keyword_candidates.py",
                "--course",
                config.course,
                "--top-n-values",
                top_n_arg,
                "--verify",
            ]
        )
    elif not args.skip_verify_candidates:
        _run([py, "scripts/verify_keyword_candidates.py", "--course", config.course])

    if not args.skip_gkp:
        _run([py, "scripts/build_gkp_set_features.py", "--course", config.course])

    _run([py, "scripts/fit_response_models.py", "--course", config.course, "--config", str(config_path)])

    default_b = COURSE_CONFIG.get(config.course, {}).get("campaign_budget", 400.0)
    budget = args.budget or default_b

    if args.strategy == "two_stage":
        plan_cmd = [
            py,
            "scripts/plan_two_stage_campaign.py",
            "--course",
            config.course,
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
        return

    opt_cmd = [
        py,
        "scripts/optimize_campaign.py",
        "--course",
        config.course,
        "--config",
        str(config_path),
        "--budget",
        str(budget),
    ]
    if args.planning_date:
        opt_cmd.extend(["--planning-date", args.planning_date])
    _run(opt_cmd)

    print(f"Pipeline finished. Budget used: {budget}")


if __name__ == "__main__":
    main()
