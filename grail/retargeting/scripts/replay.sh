#!/bin/bash
# Kinematic replay of a retargeted motion library in Isaac Lab.
#
# Invokes imports/SONIC/gear_sonic/train_agent_trl.py in replay mode (no
# training, no policy — just plays the motion through the scene). Two modes:
#
#   Headless (default)  — renders offscreen via eval_camera, writes an MP4.
#   Live / --gui        — opens the Isaac Sim viewer window; no MP4 written.
#                          Loops the motion indefinitely so you can watch
#                          (hit the viewer's ESC / close button to exit).
#                          Requires DISPLAY (X server on the machine).
#
# Usage:
#   bash grail/retargeting/scripts/replay.sh [--gui] <motion_lib_name> [object_usd_filename] [output_mp4]
#
# Examples:
#   # Headless: produce out/replay_synstairs_test_001.mp4
#   bash grail/retargeting/scripts/replay.sh synstairs_test_001
#
#   # Live viewer (needs DISPLAY):
#   bash grail/retargeting/scripts/replay.sh --gui synstairs_test_001
#
#   # Headless, explicit USD + custom output:
#   bash grail/retargeting/scripts/replay.sh benchmark_v3_0203 \
#       mug_white_jason_rigged_001_indoor1-v7_rand00042.usd \
#       /tmp/replay_mug.mp4
#
# Expects motion library layout (produced by retarget_pipeline.sh):
#   data/motion_lib_genhoi/<name>/
#     ├── robot/*.pkl
#     ├── objects/*.pkl
#     └── object_usd/*.usd
#
# Override the sweep config (tnf is the default because it fits terrain+object
# data) by setting GRAIL_REPLAY_SWEEP=bp2|bp3|g21|tnch before running.

set -eo pipefail

# Parse --gui flag (accepted anywhere in argv; positional args keep their order)
GUI=0
POSITIONAL=()
for a in "$@"; do
    case "$a" in
        --gui|--live) GUI=1 ;;
        -h|--help)
            grep '^#' "$0" | head -35 | sed 's/^# \?//'
            exit 0
            ;;
        *) POSITIONAL+=("$a") ;;
    esac
done
set -- "${POSITIONAL[@]}"

MOTION_LIB="${1:?motion lib name under data/motion_lib_genhoi/}"
OBJECT_USD_FILE="${2:-}"
OUTPUT_MP4="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SONIC_DIR="${REPO_ROOT}/imports/SONIC"
DATA_DIR="${REPO_ROOT}/data/motion_lib_genhoi/${MOTION_LIB}"
SWEEP="${GRAIL_REPLAY_SWEEP:-tnf}"
OUTPUT_MP4="${OUTPUT_MP4:-${REPO_ROOT}/out/replay_${MOTION_LIB}.mp4}"

MODE_LABEL="headless, MP4"
[[ "${GUI}" == "1" ]] && MODE_LABEL="live GUI"
echo ">>> replay: ${MOTION_LIB}  (sweep=${SWEEP}, env=${GRAIL_SONIC_ENV:-sonic}, mode=${MODE_LABEL})"
[[ "${GUI}" == "1" ]] || echo ">>> output: ${OUTPUT_MP4}"

if [[ "${GUI}" == "1" && -z "${DISPLAY:-}" ]]; then
    echo "ERROR: --gui requires DISPLAY to be set (Isaac Sim viewer needs an X server)." >&2
    echo "       Run without --gui for headless MP4 output, or 'export DISPLAY=:0' if you have a local session." >&2
    exit 1
fi

# Resolve object USD: if caller didn't name one, take the first *.usd in the
# object_usd dir that isn't the flat_placeholder.
if [[ -z "${OBJECT_USD_FILE}" ]]; then
    OBJECT_USD_FILE="$(cd "${DATA_DIR}/object_usd" && \
        ls *.usd 2>/dev/null | grep -v '^flat_placeholder.usd$' | head -n1 || true)"
    [[ -z "${OBJECT_USD_FILE}" ]] && \
        { echo "ERROR: no object USD found in ${DATA_DIR}/object_usd/" >&2; exit 1; }
fi
OBJECT_USD_PATH="${DATA_DIR}/object_usd/${OBJECT_USD_FILE}"

# Preflight
[[ -d "${DATA_DIR}/robot" ]]      || { echo "ERROR: ${DATA_DIR}/robot missing" >&2; exit 1; }
[[ -d "${DATA_DIR}/objects" ]]    || { echo "ERROR: ${DATA_DIR}/objects missing" >&2; exit 1; }
[[ -f "${OBJECT_USD_PATH}" ]]     || { echo "ERROR: ${OBJECT_USD_PATH} missing" >&2; exit 1; }

# Shell defensives: PYTHONPATH from a parent shell can poison sonic env's
# sys.path (see grail/training/scripts/train_*_local_debug.sh for the same
# treatment). Start clean.
unset PYTHONPATH

echo ">>> activating conda env: ${GRAIL_SONIC_ENV:-sonic}"
eval "$(conda shell.bash hook)"
conda activate "${GRAIL_SONIC_ENV:-sonic}"

export OMNI_KIT_ACCEPT_EULA=Yes
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"

# Pull the sweep's canonical arg bundle (COMMON_ARGS / RANDOM_OFF / rewards).
# Each sweep's batch_sweep.sh has a define-only early-return path triggered
# by GRAIL_DEFS_ONLY=1; it populates ARGS + HYDRA_CONFIG without submitting.
#
# Note: batch_sweep.sh has its own arg parser reading $@ of the sourcing
# shell. If we don't clear positional args first, our motion-lib name (our
# $1) gets treated as an unknown option and the script does `exit 1` — which
# in sourced context kills our shell. Save and restore around the source.
echo ">>> sourcing sweep arg bundle: ${SWEEP}"
cluster="04"
NUM_ENVS=1
_saved_argv=("$@")
set --
GRAIL_DEFS_ONLY=1 source "${REPO_ROOT}/grail/training/sweeps/${SWEEP}/batch_sweep.sh" >/dev/null
set -- ${_saved_argv[@]+"${_saved_argv[@]}"}

# batch_sweep.sh wraps each arg in single quotes for the cluster submitter;
# strip the outer pair so Hydra parses them correctly.
UNQ_ARGS=()
for x in "${ARGS[@]}"; do
    x="${x#\'}"; x="${x%\'}"
    UNQ_ARGS+=("$x")
done

mkdir -p "$(dirname "${OUTPUT_MP4}")"

echo ">>> motion lib:   ${DATA_DIR}"
echo ">>> object USD:   ${OBJECT_USD_PATH}"
echo ">>> exp config:   ${HYDRA_CONFIG}"
if [[ "${GUI}" == "1" ]]; then
    echo ">>> launching Isaac Sim viewer (DISPLAY=${DISPLAY}); close the window or Ctrl-C to exit..."
else
    echo ">>> launching Isaac Sim (first log line ~15-20s away)..."
fi

# Mode-specific flags:
#   Headless: save an MP4, play the motion once (replay_loop_num=1) and exit.
#   Live    : open the viewer, loop forever so the user can watch / pause.
MODE_ARGS=()
if [[ "${GUI}" == "1" ]]; then
    MODE_ARGS=(
        "headless=False"
        "num_envs=1"
        "++manager_env.config.render_results=False"
        "++replay=true"
    )
else
    MODE_ARGS=(
        "headless=True"
        "num_envs=1"
        "++manager_env.config.render_results=True"
        "++replay=true"
        "++replay_loop_num=1"
        "++replay_save_video=${OUTPUT_MP4}"
    )
fi

cd "${SONIC_DIR}/gear_sonic"
python -u train_agent_trl.py \
    "+exp=${HYDRA_CONFIG}" \
    "${UNQ_ARGS[@]}" \
    "++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=${SONIC_DIR}/gear_sonic/data/assets/robot_description/mjcf/" \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=${DATA_DIR}/robot" \
    "++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${DATA_DIR}/objects" \
    "++manager_env.config.terrain_motion_dir=${DATA_DIR}" \
    "++manager_env.config.object_usd_path=${OBJECT_USD_PATH}" \
    "${MODE_ARGS[@]}"

echo ""
if [[ "${GUI}" == "1" ]]; then
    echo "Replay session closed."
else
    echo "Replay complete: ${OUTPUT_MP4}"
fi
