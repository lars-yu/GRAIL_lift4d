#!/bin/bash
# Step 3 (optional) — compute BPS shape encoding for multi-object datasets.
#
# Usage: bash grail/retargeting/scripts/compute_bps.sh <output_folder>
#
# Reads   data/motion_lib/<output_folder>/object_usd/*.usd
# Writes  data/motion_lib/<output_folder>/bps/<stem>.npy + _basis.npy
#
# Skip when the dataset has a single object — BPS is only useful for
# multi-object HOI training (e.g. ComAsset, RoboCasa).

set -euo pipefail

OUTPUT_FOLDER="${1:?output folder name under data/motion_lib/}"
DATA_DIR="data/motion_lib/${OUTPUT_FOLDER}"

eval "$(conda shell.bash hook)"
conda activate "${GRAIL_SONIC_ENV:-sonic}"

python -u -m grail.retargeting.compute_bps \
    --object_usd_dir "${DATA_DIR}/object_usd" \
    --output_dir "${DATA_DIR}/bps"

echo "BPS output: ${DATA_DIR}/bps/"
