#!/bin/bash
# Run the profiling pipeline end to end: scores, reference labels, router training, and router export.
# Run from anywhere; paths are resolved from the repository root (three levels up from this script).

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"

if [ "${RUN_GPT_JUDGE:-0}" = "1" ]; then
  echo "=== [0/6] GPT-4o judge ==="
  python profiling/code/pipeline/00_gpt_judge.py
else
  echo "=== [0/6] GPT-4o judge skipped (set RUN_GPT_JUDGE=1 to make the paid API calls) ==="
fi

echo "=== [1/6] Quality components ==="
python profiling/code/pipeline/01_quality_components.py

echo "=== [2/6] Cost components ==="
python profiling/code/pipeline/02_cost_components.py

echo "=== [3/6] Reference labels ==="
python profiling/code/pipeline/03_oracle_labels.py

echo "=== [4/6] Train and evaluate the two-stage router ==="
python profiling/code/pipeline/04_train_eval_router.py

echo "=== [5/6] Sensitivity to the reference-label settings ==="
python profiling/code/pipeline/05_sensitivity.py

echo "=== [6/6] Export the routers ==="
python profiling/code/pipeline/06_export_routers.py

echo ""
echo "Pipeline outputs: $ROOT/profiling/results/pipeline_outputs/"
echo "Routers:          $ROOT/deployment/routers/"
