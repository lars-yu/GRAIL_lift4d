#!/bin/bash
# ============================================================================
# Run a pipeline command in parallel on the local machine.
#
# Each worker gets a different --job_chunk_idx so categories are split evenly.
# All workers share the same GPU — set CUDA_VISIBLE_DEVICES to control which.
#
# Usage:
#   bash scripts/batch/local.sh <num_workers> "<command>"
#
# Examples:
#   # 4 parallel workers for terrain rendering
#   bash scripts/batch/local.sh 4 "python -m grail.pipelines.gen_2dhoi \
#       --config configs/gen_2dhoi/terrain_curbs.yaml \
#       --dataset Terrain --results_dir results_terrain \
#       --skip_step4 --skip_done"
#
#   # 2 workers for ComAsset reconstruction
#   bash scripts/batch/local.sh 2 "python -m grail.pipelines.recon_4dhoi \
#       --dataset ComAsset --category cordless_drill --results_dir results"
# ============================================================================
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/batch/local.sh <num_workers> \"<command>\""
    echo ""
    echo "  num_workers  Number of parallel processes"
    echo "  command      Pipeline command (--job_chunk_idx and --num_job_chunks are appended)"
    exit 1
fi

NUM_WORKERS=$1
JOB_CMD=$2

echo "Starting $NUM_WORKERS parallel workers..."
echo "Command: $JOB_CMD"
echo ""

PIDS=()

for (( i=0; i<NUM_WORKERS; i++ )); do
    echo "[Worker $i/$NUM_WORKERS] Starting..."
    $JOB_CMD --job_chunk_idx $i --num_job_chunks $NUM_WORKERS \
        > >(sed "s/^/[w$i] /") 2>&1 &
    PIDS+=($!)
    sleep 1
done

echo ""
echo "All $NUM_WORKERS workers launched. PIDs: ${PIDS[*]}"
echo "Waiting for completion... (Ctrl+C to stop all)"

# Wait and track results
FAILURES=0
for (( i=0; i<NUM_WORKERS; i++ )); do
    if wait ${PIDS[$i]}; then
        echo "[Worker $i] Finished successfully"
    else
        echo "[Worker $i] Failed (exit code $?)"
        FAILURES=$((FAILURES + 1))
    fi
done

echo ""
if [ $FAILURES -eq 0 ]; then
    echo "All $NUM_WORKERS workers completed successfully."
else
    echo "$FAILURES/$NUM_WORKERS workers failed."
    exit 1
fi
