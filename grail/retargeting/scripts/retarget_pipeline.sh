#!/bin/bash
# End-to-end retarget pipeline: retarget -> process -> compute_bps.
#
# Usage:  bash grail/retargeting/scripts/retarget_pipeline.sh <data_dir> <output_folder> [extra_retarget_args...]
# Example: bash grail/retargeting/scripts/retarget_pipeline.sh \
#              data/genhoi/benchmark_v3/generation/4dhoi_recon_valid/Hunyuan \
#              benchmark_v3_0203

set -euo pipefail

DATA_DIR="${1:?data directory}"
OUTPUT_FOLDER="${2:?output folder name under data/motion_lib/}"
shift 2

SCRIPT_DIR="$(dirname "$0")"
RETARGET_ARGS=()
PROCESS_ARGS=()

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --treat_hands_equally|--treat-hands-equally)
            PROCESS_ARGS+=("$1")
            ;;
        *)
            RETARGET_ARGS+=("$1")
            ;;
    esac
    shift
done

echo ">>> [1/3] retarget"
bash "${SCRIPT_DIR}/retarget.sh" "${DATA_DIR}" "${OUTPUT_FOLDER}" "${RETARGET_ARGS[@]}"

echo ">>> [2/3] process (hand actions + table geometry)"
bash "${SCRIPT_DIR}/process.sh" "${OUTPUT_FOLDER}" "${PROCESS_ARGS[@]}"

# Only run BPS if the dataset has multiple object USDs.
USD_COUNT=$(find "data/motion_lib/${OUTPUT_FOLDER}/object_usd" -maxdepth 1 -name '*.usd' 2>/dev/null | wc -l)
if [[ "${USD_COUNT}" -gt 1 ]]; then
    echo ">>> [3/3] compute_bps (${USD_COUNT} objects)"
    bash "${SCRIPT_DIR}/compute_bps.sh" "${OUTPUT_FOLDER}"
else
    echo ">>> [3/3] compute_bps: skipped (single-object dataset, ${USD_COUNT} USDs)"
fi

echo "Pipeline complete. See data/motion_lib/${OUTPUT_FOLDER}{,_ha}/"
