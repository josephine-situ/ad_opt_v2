#!/bin/bash
#SBATCH --job-name=adopt-v2-backtest-missing
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/backtest_missing_%j.out
#SBATCH --error=logs/backtest_missing_%j.err

# Rerun missing backtest days (inferred from plans/YYYYMMDD/campaign_plan.csv) and re-summarize.

EXP_NAME="${EXP_NAME:-default}"
COURSES="${COURSES:-sys_think}"
START_DAY="${START_DAY:-2025-10-01}"
END_DAY="${END_DAY:-2026-02-10}"
STRATEGY="${STRATEGY:-daily}"
MISSING_DATES="${MISSING_DATES:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "========== Missing-day backtest rerun =========="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Courses: $COURSES"
echo "Missing dates: ${MISSING_DATES:-<auto-detect>}"
echo "==============================================="

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif command -v module >/dev/null 2>&1; then
  module load miniforge 2>/dev/null || true
  conda activate adopt_env 2>/dev/null || true
fi

PY="${PY:-uv run python}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

get_missing_dates() {
  local course="$1"
  $PY - <<PY
from campaign_opt.backtest_analysis import backtest_window_dir, _find_missing_days

course = "$course"
exp = "$EXP_NAME"
start = "$START_DAY"
end = "$END_DAY"
bt_dir = backtest_window_dir(course, exp, start, end)
missing = _find_missing_days(bt_dir, start, end)
print(" ".join(missing))
PY
}

for COURSE in $COURSES; do
  echo ""
  echo "############################################################"
  echo "  Course: $COURSE"
  echo "############################################################"

  if [ -n "$MISSING_DATES" ]; then
    DATE_RANGE="$MISSING_DATES"
  else
    DATE_RANGE=$(get_missing_dates "$COURSE")
  fi

  if [ -z "$DATE_RANGE" ]; then
    echo "No missing dates for $COURSE; skipping rerun."
    continue
  fi

  echo "Missing dates: $DATE_RANGE"

  for DAY in $DATE_RANGE; do
    echo "Running day $DAY ..."
    $PY -u scripts/backtest_campaign.py \
      --course "$COURSE" \
      --exp-name "$EXP_NAME" \
      --start "$START_DAY" \
      --end "$END_DAY" \
      --day "$DAY" \
      --strategy "$STRATEGY" \
      $EXTRA_ARGS
  done

  echo "Re-analyzing $COURSE ..."
  $PY -u scripts/analyze_backtest_results.py \
    --course "$COURSE" \
    --exp-name "$EXP_NAME" \
    --start "$START_DAY" \
    --end "$END_DAY"
done

echo "Missing-date rerun complete: $(date)"
