#!/bin/bash
# Render a single robot pkl from a motion-lib-shaped directory into <data_dir>/vis/.
#
# Usage: ./visualize_single.sh <robot_pkl_path> [cam_offset_x,cam_offset_y,cam_offset_z] [quat_convention]
# Example:
#   ./visualize_single.sh data/release/dataset/pickup_table/robot/pickup_table__alcohol_5__000.pkl
#   ./visualize_single.sh /abs/path/to/robot/my_motion.pkl -3.5,0.0,1.2
#   QUAT_CONVENTION=wxyz ./visualize_single.sh data/motion_lib/<name>/robot/<seq>.pkl
#
# Arguments:
#   robot_pkl_path   Full path to a robot/*.pkl file inside a motion-lib-shaped dir.
#   cam_offset       Camera position as x,y,z (default: 1.5,-1.5,1.0).
#   quat_convention  root_rot convention in the input pkl: auto|wxyz|xyzw
#                    (default: $QUAT_CONVENTION env var, else 'xyzw'). The
#                    default matches data-export merged / public-release pkls.
#                    Pass 'wxyz' for retargeting output (data/motion_lib/<name>/),
#                    or 'auto' for magnitude-based detection.
#
# Hand DOFs: gripper is driven from hand_dof_pos in the robot pkl when
# present (data-export writes this as of 2026-06-01). Older pkls without
# that field render with the gripper open.

set -euo pipefail
export DISPLAY=${DISPLAY:-:0}

ROBOT_PKL="${1:?Error: Please provide path to a robot/*.pkl file}"
CAM_OFFSET="${2:-1.5,-1.5,1.0}"
QUAT_CONVENTION="${3:-${QUAT_CONVENTION:-xyzw}}"

case "${QUAT_CONVENTION}" in
    auto|wxyz|xyzw) ;;
    *) echo "Error: quat_convention must be auto|wxyz|xyzw (got '${QUAT_CONVENTION}')"; exit 1;;
esac

if [ ! -f "${ROBOT_PKL}" ]; then
    echo "Error: file not found: ${ROBOT_PKL}"
    exit 1
fi

ROBOT_PKL="$(realpath "${ROBOT_PKL}")"
MOTION_KEY="$(basename "${ROBOT_PKL}" .pkl)"
ROBOT_DIR="$(dirname "${ROBOT_PKL}")"
DATA_DIR="$(dirname "${ROBOT_DIR}")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

VIDEO_DIR="${DATA_DIR}/vis"
SHARD_DIR="/tmp/vis_single_${MOTION_KEY}"

# Activate sonic conda environment (needed for IsaacSim/IsaacLab)
eval "$(conda shell.bash hook)"
conda activate sonic

echo "Rendering single motion"
echo "  Robot pkl:       ${ROBOT_PKL}"
echo "  Motion key:      ${MOTION_KEY}"
echo "  Data dir:        ${DATA_DIR}"
echo "  Video output:    ${VIDEO_DIR}/${MOTION_KEY}.mp4"
echo "  Camera:          ${CAM_OFFSET}"
echo "  Quat convention: ${QUAT_CONVENTION}"
echo ""

# --- Step 1: Prepare shard for this single motion ---
echo "[Step 1/2] Preparing trajectory data..."

cd "${REPO_ROOT}"
python -m grail.visualization.prepare_vis_shard \
    --data_dir "${DATA_DIR}" \
    --shard_dir "${SHARD_DIR}" \
    --motion_keys "${MOTION_KEY}" \
    --quat_convention "${QUAT_CONVENTION}"

echo ""

# --- Step 2: Render ---
echo "[Step 2/2] Rendering video..."

IFS=',' read -r CAM_X CAM_Y CAM_Z <<< "${CAM_OFFSET}"

mkdir -p "${VIDEO_DIR}"

python -u -m grail.visualization.batch_render_replay \
    --shard_dir "${SHARD_DIR}" \
    --traj_dir "${SHARD_DIR}/trajectories" \
    --object_usd_dir "${DATA_DIR}/object_usd" \
    --output_dir "${VIDEO_DIR}" \
    --camera_offset ${CAM_X} ${CAM_Y} ${CAM_Z} \
    --camera_target 0.0 0.0 0.8 \
    --start_frame_skip 0 \
    --headless

# Cleanup
rm -rf "${SHARD_DIR}"

echo ""
echo "Done!"
echo "  Video: ${VIDEO_DIR}/${MOTION_KEY}.mp4"
