#!/bin/bash
#SBATCH --job-name=adopt-v2-backtest
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-132%4
#SBATCH --output=logs/backtest_%A_%a.out
#SBATCH --error=logs/backtest_%A_%a.err

# Walk-forward campaign backtest: one Slurm array task per calendar day.
# Override via environment variables before sbatch, e.g.:
#   EXP_NAME=default COURSES="sys_think ml" START_DAY=2025-10-01 END_DAY=2026-02-10 sbatch submit_backtest.sh

EXP_NAME="${EXP_NAME:-default}"
COURSES="${COURSES:-sys_think}"
START_DAY="${START_DAY:-2025-10-01}"
END_DAY="${END_DAY:-2026-02-10}"
STRATEGY="${STRATEGY:-daily}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "========== Campaign backtest (ad_opt_v2) =========="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Courses: $COURSES | Exp: $EXP_NAME | Window: $START_DAY → $END_DAY"
echo "Strategy: $STRATEGY"
echo "==================================================="

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

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
DAY=$(python - <<PY
import datetime as dt
start = dt.date.fromisoformat("$START_DAY")
print((start + dt.timedelta(days=int("$TASK_ID"))).isoformat())
PY
)

MAX_TASK_ID=$(python - <<PY
import datetime as dt
start = dt.date.fromisoformat("$START_DAY")
end = dt.date.fromisoformat("$END_DAY")
print((end - start).days)
PY
)

for COURSE in $COURSES; do
  echo ""
  echo "############################################################"
  echo "  Course: $COURSE | Day: $DAY (task $TASK_ID)"
  echo "############################################################"

  CMD="$PY -u scripts/backtest_campaign.py \
    --course $COURSE \
    --exp-name $EXP_NAME \
    --start $START_DAY \
    --end $END_DAY \
    --day $DAY \
    --strategy $STRATEGY \
    $EXTRA_ARGS"

  echo "Running: $CMD"
  eval "$CMD"
  echo "Day $DAY complete for $COURSE: $(date)"
done

if [ "$TASK_ID" -eq "$MAX_TASK_ID" ]; then
  echo ""
  echo "========================================"
  echo "  Last array task — running analysis"
  echo "========================================"
  for COURSE in $COURSES; do
    echo "Analyzing $COURSE ..."
    $PY -u scripts/analyze_backtest_results.py \
      --course "$COURSE" \
      --exp-name "$EXP_NAME" \
      --start "$START_DAY" \
      --end "$END_DAY"
  done
  echo "Post-processing complete: $(date)"
fi

echo "End: $(date)"
