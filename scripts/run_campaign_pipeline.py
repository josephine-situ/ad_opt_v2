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
    parser.add_argument("--skip-gkp", action="store_true")
    parser.add_argument("--skip-candidates", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path(args.course, args.exp_name)
    config = load_campaign_config(config_path)
    py = sys.executable

    log_path = setup_tee_logging(
        log_file=None,
        default_log_prefix=f"run_campaign_pipeline_{config.course}",
    )
    print(f"Log file: {log_path}")
    print(f"Config: {config_path}")

    panel_script = [py, "scripts/generate_campaign_day_panel.py", "--output-course", config.course]
    if not (Path("data") / config.course / "processed" / "campaign-day-panel.csv").exists():
        _run(panel_script)

    if not args.skip_gkp:
        _run([py, "scripts/build_gkp_set_features.py", "--course", config.course])

    if not args.skip_candidates:
        _run([py, "scripts/build_keyword_candidates.py", "--course", config.course])

    _run([py, "scripts/fit_response_models.py", "--course", config.course, "--config", str(config_path)])

    opt_cmd = [
        py,
        "scripts/optimize_campaign.py",
        "--course",
        config.course,
        "--config",
        str(config_path),
    ]
    if args.budget is not None:
        opt_cmd.extend(["--budget", str(args.budget)])
    if args.planning_date:
        opt_cmd.extend(["--planning-date", args.planning_date])
    _run(opt_cmd)

    default_b = COURSE_CONFIG.get(config.course, {}).get("campaign_budget", 400.0)
    print(f"Pipeline finished. Budget used: {args.budget or default_b}")


if __name__ == "__main__":
    main()
