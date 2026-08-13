#!/bin/bash
# Retarget GRAIL HOI motions to the G1 robot using GMR.
#
# Usage:  bash grail/retargeting/scripts/retarget.sh <data_dir> <output_folder> [--zero_out_wrist]
# Example: bash grail/retargeting/scripts/retarget.sh data/genhoi/benchmark_v3 benchmark_v3_0203
#
# Extra args after <output_folder> are forwarded to grail.retargeting.retarget (e.g., --zero_out_wrist
# for terrain/sitting data, --no_g1_proportions for non-G1-proportioned SMPLX).

set -euo pipefail

DATA_DIR="${1:?data directory (e.g. data/genhoi/<dataset>/generation/4dhoi_recon_valid/Hunyuan)}"
OUTPUT_FOLDER="${2:?output folder name under data/motion_lib/}"
shift 2

OUTPUT_BASE="data/motion_lib/${OUTPUT_FOLDER}"

# Activate the retargeting env (ships GMR + mujoco + smplx + isaaclab + pxr).
# Override with GRAIL_SONIC_ENV=<name> when using a non-default environment.
eval "$(conda shell.bash hook)"
conda activate "${GRAIL_SONIC_ENV:-sonic}"

# GMR opens a mujoco viewer — ensure DISPLAY is set for headless runs.
export DISPLAY="${DISPLAY:-:1}"

python -m grail.retargeting.retarget \
    --data_dir "${DATA_DIR}" \
    --all \
    --robot unitree_g1 \
    --output_dir "${OUTPUT_BASE}" \
    --no_viewer \
    "$@"

echo "Retarget output: ${OUTPUT_BASE}/{robot,objects,object_usd,meta}/"
