#!/bin/bash
# Generate kinematic-replay visualization videos for any motion library directory.
#
# Works on any folder that contains the standard subdirs:
#     <motion_lib>/robot/*.pkl       (required)
#     <motion_lib>/objects/*.pkl     (required)
#     <motion_lib>/object_usd/*.usd  (required)
#     <motion_lib>/meta/*.pkl        (optional, used for table_pos)
#
# Compatible layouts:
#   - Retarget output:     data/motion_lib/<name>/
#   - Data-export merged:  <exp>/exported/step_*/merged/
#   - Public-release:      data/release/dataset/<name>/
#
# Renders into <motion_lib>/vis/ (kept separate from the release `video/` dir
# which holds the original 2D HOI render).
#
# Uses a single IsaacSim session via grail/visualization/batch_render_replay.py
# instead of spawning one process per motion.
#
# Usage: ./visualize.sh <motion_lib_path> [max_videos] [cam_offset_x,cam_offset_y,cam_offset_z] [quat_convention]
# Example:
#   ./visualize.sh data/release/dataset/pickup_table
#   ./visualize.sh data/release/dataset/pickup_table 0          # render all motions
#   ./visualize.sh /abs/path/to/motion_lib 16 -3.5,0.0,1.2 xyzw
#   QUAT_CONVENTION=wxyz ./visualize.sh data/motion_lib/<name> 16
#
# Arguments:
#   motion_lib_path  Full path (absolute or repo-relative) to the motion library dir.
#   max_videos       Max videos to render (default=16). Pass 0 to render all
#                    motions; 0 also skips grid/combined post-processing.
#   cam_offset       Camera position as x,y,z (default: 1.5,-1.5,1.0).
#   quat_convention  root_rot convention in the input pkls: auto|wxyz|xyzw
#                    (default: $QUAT_CONVENTION env var, else 'xyzw'). The
#                    default matches data-export merged / public-release dirs.
#                    Pass 'wxyz' for retargeting output (data/motion_lib/<name>/),
#                    or 'auto' for magnitude-based detection (may mis-classify
#                    motions that don't start near-upright).
#
# Hand DOFs: the renderer drives the gripper from the per-motion (T, 14)
# hand_dof_pos array when present in the robot pkl (data-export pipeline
# writes this as of 2026-06-01). Older pkls without that field render with
# the gripper open.

set -euo pipefail
export DISPLAY=${DISPLAY:-:0}

MOTION_LIB_ARG="${1:?Error: Please provide motion library path as argument}"
MAX_VIDEOS="${2:-16}"
CAM_OFFSET="${3:-1.5,-1.5,1.0}"
QUAT_CONVENTION="${4:-${QUAT_CONVENTION:-xyzw}}"

case "${QUAT_CONVENTION}" in
    auto|wxyz|xyzw) ;;
    *) echo "Error: quat_convention must be auto|wxyz|xyzw (got '${QUAT_CONVENTION}')"; exit 1;;
esac

MOTION_LIB="$(realpath "${MOTION_LIB_ARG}")"
if [ ! -d "${MOTION_LIB}" ]; then
    echo "Error: not a directory: ${MOTION_LIB}"
    exit 1
fi
for sub in robot objects object_usd; do
    if [ ! -d "${MOTION_LIB}/${sub}" ]; then
        echo "Error: required subdir missing: ${MOTION_LIB}/${sub}"
        exit 1
    fi
done

SKIP_POSTPROCESS=false
if [ "${MAX_VIDEOS}" -eq 0 ]; then
    SKIP_POSTPROCESS=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

KEY="$(basename "${MOTION_LIB}")"
VIDEO_DIR="${MOTION_LIB}/vis"
SHARD_DIR="/tmp/vis_shard_${KEY}"

# Activate sonic conda environment (needed for IsaacSim/IsaacLab)
eval "$(conda shell.bash hook)"
conda activate sonic

echo "Generating visualization videos"
echo "  Motion library:   ${MOTION_LIB}"
echo "  Video output:     ${VIDEO_DIR}"
echo "  Max videos:       ${MAX_VIDEOS} (0=all, skips grid/combined)"
echo "  Camera offset:    ${CAM_OFFSET}"
echo "  Quat convention:  ${QUAT_CONVENTION}"
echo ""

cd "${REPO_ROOT}"

# --- Step 1: Convert motion_lib format to shard format ---
TOTAL_STEPS=$( [ "${SKIP_POSTPROCESS}" = true ] && echo 2 || echo 3 )
echo "[Step 1/${TOTAL_STEPS}] Preparing trajectory data..."

python -m grail.visualization.prepare_vis_shard \
    --data_dir "${MOTION_LIB}" \
    --shard_dir "${SHARD_DIR}" \
    --max_motions "${MAX_VIDEOS}" \
    --quat_convention "${QUAT_CONVENTION}"

echo ""

# --- Step 2: Render with batch_render_replay (single IsaacSim session) ---
echo "[Step 2/${TOTAL_STEPS}] Rendering videos (single IsaacSim session)..."

IFS=',' read -r CAM_X CAM_Y CAM_Z <<< "${CAM_OFFSET}"

mkdir -p "${VIDEO_DIR}"

python -u -m grail.visualization.batch_render_replay \
    --shard_dir "${SHARD_DIR}" \
    --traj_dir "${SHARD_DIR}/trajectories" \
    --object_usd_dir "${MOTION_LIB}/object_usd" \
    --output_dir "${VIDEO_DIR}" \
    --camera_offset ${CAM_X} ${CAM_Y} ${CAM_Z} \
    --camera_target 0.0 0.0 0.8 \
    --skip_existing \
    --start_frame_skip 0 \
    --headless

echo ""

total=$(ls "${VIDEO_DIR}"/*.mp4 2>/dev/null | wc -l)

if [ "$total" -eq 0 ]; then
    echo "No videos rendered."
    rm -rf "${SHARD_DIR}"
    exit 0
fi

if [ "${SKIP_POSTPROCESS}" = true ]; then
    echo "Skipping post-processing (grid/combined videos) for full render"
else
    echo "[Step 3/${TOTAL_STEPS}] Post-processing videos..."

    # --- Step 3: ffmpeg labeling + combined/grid videos ---
    COMBINED_VIDEO="${VIDEO_DIR}/all_motions_combined.mp4"
    TEMP_LABELED_DIR="/tmp/vis_labeled_${KEY}"
    mkdir -p "${TEMP_LABELED_DIR}"

    echo "  Adding labels to ${total} videos..."

    for VIDEO_FILE in "${VIDEO_DIR}"/*.mp4; do
        DATA_NAME=$(basename "$VIDEO_FILE" .mp4)
        LABELED_FILE="${TEMP_LABELED_DIR}/${DATA_NAME}.mp4"

        # Create a short display label (strip common long prefixes for readability)
        DISPLAY_NAME=$(echo "$DATA_NAME" | sed 's/_start[0-9]*-end[0-9]*//' | tail -c 60)

        ffmpeg -i "$VIDEO_FILE" \
            -vf "drawtext=text='${DISPLAY_NAME}':fontsize=20:fontcolor=white:borderw=2:bordercolor=black:x=20:y=20" \
            -c:v libx264 -preset fast -crf 23 -c:a copy \
            "$LABELED_FILE" -y -loglevel error

        # Replace original with labeled version
        mv "$LABELED_FILE" "$VIDEO_FILE"
    done

    # Create concatenated video
    echo "  Creating combined video..."
    ls "${VIDEO_DIR}"/*.mp4 | sort | grep -v "/all_motions_combined.mp4$" | grep -v "/examples_grid.mp4$" \
        | while read f; do echo "file '$(realpath "$f")'"; done > /tmp/video_list_${KEY}.txt
    ffmpeg -f concat -safe 0 -i /tmp/video_list_${KEY}.txt -c copy "$COMBINED_VIDEO" -y -loglevel error

    # Create grid video
    GRID_VIDEO="${VIDEO_DIR}/examples_grid.mp4"
    GRID_VIDEOS=($(ls "${VIDEO_DIR}"/*.mp4 | sort | grep -v "/all_motions_combined.mp4$" | grep -v "/examples_grid.mp4$" | head -16))
    GRID_COUNT=${#GRID_VIDEOS[@]}

    if [ "$GRID_COUNT" -ge 16 ]; then
        echo "  Creating 4x4 grid video..."
        ffmpeg -y -loglevel error \
            -i "${GRID_VIDEOS[0]}" -i "${GRID_VIDEOS[1]}" -i "${GRID_VIDEOS[2]}" -i "${GRID_VIDEOS[3]}" \
            -i "${GRID_VIDEOS[4]}" -i "${GRID_VIDEOS[5]}" -i "${GRID_VIDEOS[6]}" -i "${GRID_VIDEOS[7]}" \
            -i "${GRID_VIDEOS[8]}" -i "${GRID_VIDEOS[9]}" -i "${GRID_VIDEOS[10]}" -i "${GRID_VIDEOS[11]}" \
            -i "${GRID_VIDEOS[12]}" -i "${GRID_VIDEOS[13]}" -i "${GRID_VIDEOS[14]}" -i "${GRID_VIDEOS[15]}" \
            -filter_complex "
                [0:v]scale=480:270[v0];[1:v]scale=480:270[v1];[2:v]scale=480:270[v2];[3:v]scale=480:270[v3];
                [4:v]scale=480:270[v4];[5:v]scale=480:270[v5];[6:v]scale=480:270[v6];[7:v]scale=480:270[v7];
                [8:v]scale=480:270[v8];[9:v]scale=480:270[v9];[10:v]scale=480:270[v10];[11:v]scale=480:270[v11];
                [12:v]scale=480:270[v12];[13:v]scale=480:270[v13];[14:v]scale=480:270[v14];[15:v]scale=480:270[v15];
                [v0][v1][v2][v3]hstack=inputs=4[row0];
                [v4][v5][v6][v7]hstack=inputs=4[row1];
                [v8][v9][v10][v11]hstack=inputs=4[row2];
                [v12][v13][v14][v15]hstack=inputs=4[row3];
                [row0][row1][row2][row3]vstack=inputs=4[out]
            " \
            -map "[out]" -c:v libx264 -preset fast -crf 23 "$GRID_VIDEO"
    elif [ "$GRID_COUNT" -ge 4 ]; then
        echo "  Creating 2x2 grid video (only $GRID_COUNT videos)..."
        ffmpeg -y -loglevel error \
            -i "${GRID_VIDEOS[0]}" -i "${GRID_VIDEOS[1]}" -i "${GRID_VIDEOS[2]}" -i "${GRID_VIDEOS[3]}" \
            -filter_complex "
                [0:v]scale=640:360[v0];[1:v]scale=640:360[v1];
                [2:v]scale=640:360[v2];[3:v]scale=640:360[v3];
                [v0][v1]hstack=inputs=2[row0];
                [v2][v3]hstack=inputs=2[row1];
                [row0][row1]vstack=inputs=2[out]
            " \
            -map "[out]" -c:v libx264 -preset fast -crf 23 "$GRID_VIDEO"
    fi

    # Cleanup temp files
    rm -f /tmp/video_list_${KEY}.txt
    rm -rf "${TEMP_LABELED_DIR}"
fi

rm -rf "${SHARD_DIR}"

echo ""
echo "Done!"
echo "  Individual videos: ${VIDEO_DIR}/"
if [ "${SKIP_POSTPROCESS}" = false ]; then
    echo "  Combined video:    ${COMBINED_VIDEO}"
    if [ -f "$GRID_VIDEO" ]; then
        echo "  Grid video:        ${GRID_VIDEO}"
    fi
fi
echo "  Total videos: ${total}"
