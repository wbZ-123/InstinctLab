#!/usr/bin/env bash
set -euo pipefail

# Play a trained foothold policy on stair-only terrain.
#
# This is a diagnostic wrapper around scripts/instinct_rl/play.py. It does not
# change training configs; it only restricts the play-time terrain generator to
# one stair terrain family so visual/debug output is not mixed with rough/gap/box
# terrains.
#
# Examples:
#   ./scripts/foothold_play_step.sh
#   LOAD_RUN=20260724_192653_foothold_amp_aligned_curriculum_30000 ./scripts/foothold_play_step.sh
#   STEP_TERRAIN_NAME=pyramid_stairs_high STEP_TERRAIN_LEVEL=0 ./scripts/foothold_play_step.sh
#   STEP_TERRAIN_NAME=pyramid_stairs,pyramid_stairs_inv NUM_ENVS=2 ./scripts/foothold_play_step.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${REPO_ROOT}/third_party/IsaacLab}"
INSTINCT_RL_ROOT="${INSTINCT_RL_ROOT:-${REPO_ROOT}/third_party/instinct_rl}"

export PYTHONPATH="${REPO_ROOT}/source/instinctlab:${INSTINCT_RL_ROOT}:${PYTHONPATH:-}"
export PARKOUR_MOTION_REFERENCE_DIR="${PARKOUR_MOTION_REFERENCE_DIR:-/home/zhangweibo/Datasets/hiking_in_the_wild/data&model/parkour_motion_reference}"
export PARKOUR_MOTION_SELECTION_FILE="${PARKOUR_MOTION_SELECTION_FILE:-${PARKOUR_MOTION_REFERENCE_DIR}/parkour_motion_without_run.yaml}"

TASK="${TASK:-Instinct-Parkour-Target-Amp-G1-Play-v0}"
NUM_ENVS="${NUM_ENVS:-2}"
LOAD_RUN="${LOAD_RUN:-20260724_192653_foothold_amp_aligned_curriculum_30000}"
STEP_TERRAIN_NAME="${STEP_TERRAIN_NAME:-pyramid_stairs,pyramid_stairs_inv}"
STEP_TERRAIN_ROWS="${STEP_TERRAIN_ROWS:-1}"
STEP_TERRAIN_COLS="${STEP_TERRAIN_COLS:-2}"
STEP_TERRAIN_LEVEL="${STEP_TERRAIN_LEVEL:-0}"
FOOTHOLD_CURRICULUM_SCALE_OVERRIDE="${FOOTHOLD_CURRICULUM_SCALE_OVERRIDE:-1.0}"
FOOTHOLD_DEBUG_INTERVAL="${FOOTHOLD_DEBUG_INTERVAL:-30}"
FOOTHOLD_DEBUG_ENV_ID="${FOOTHOLD_DEBUG_ENV_ID:-0}"
FOOTHOLD_DEBUG_ENV_IDS="${FOOTHOLD_DEBUG_ENV_IDS:-all}"
FOOTHOLD_TRAJECTORY_SAMPLES="${FOOTHOLD_TRAJECTORY_SAMPLES:-8}"

if [[ ! -f "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
    echo "[foothold_play_step] IsaacLab launcher not found: ${ISAACLAB_ROOT}/isaaclab.sh" >&2
    echo "[foothold_play_step] Run: git submodule update --init --recursive" >&2
    echo "[foothold_play_step] Set ISAACLAB_ROOT=/path/to/IsaacLab if needed." >&2
    exit 1
fi

if [[ ! -f "${INSTINCT_RL_ROOT}/instinct_rl/__init__.py" ]]; then
    echo "[foothold_play_step] Instinct-RL source not found: ${INSTINCT_RL_ROOT}" >&2
    echo "[foothold_play_step] Run: git submodule update --init --recursive" >&2
    echo "[foothold_play_step] Set INSTINCT_RL_ROOT=/path/to/instinct_rl if needed." >&2
    exit 1
fi

echo "[foothold_play_step] repo=${REPO_ROOT}"
echo "[foothold_play_step] isaaclab_root=${ISAACLAB_ROOT}"
echo "[foothold_play_step] instinct_rl_root=${INSTINCT_RL_ROOT}"
echo "[foothold_play_step] task=${TASK} load_run=${LOAD_RUN}"
echo "[foothold_play_step] terrain=${STEP_TERRAIN_NAME} rows=${STEP_TERRAIN_ROWS} cols=${STEP_TERRAIN_COLS} level=${STEP_TERRAIN_LEVEL}"

cd "${REPO_ROOT}"

"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/instinct_rl/play.py \
    --task "${TASK}" \
    --num_envs "${NUM_ENVS}" \
    --load_run "${LOAD_RUN}" \
    --print_reset_debug \
    --print_foothold_debug \
    --print_foothold_marker_debug \
    --print_foothold_debug_on_plan_event \
    --print_foothold_debug_interval "${FOOTHOLD_DEBUG_INTERVAL}" \
    --print_foothold_debug_env_id "${FOOTHOLD_DEBUG_ENV_ID}" \
    --print_foothold_debug_env_ids "${FOOTHOLD_DEBUG_ENV_IDS}" \
    --show_foothold_debug_markers \
    --foothold_debug_trajectory_samples "${FOOTHOLD_TRAJECTORY_SAMPLES}" \
    --foothold_curriculum_scale_override "${FOOTHOLD_CURRICULUM_SCALE_OVERRIDE}" \
    --foothold_step_terrain_only \
    --foothold_step_terrain_name "${STEP_TERRAIN_NAME}" \
    --foothold_step_terrain_rows "${STEP_TERRAIN_ROWS}" \
    --foothold_step_terrain_cols "${STEP_TERRAIN_COLS}" \
    --foothold_step_terrain_level "${STEP_TERRAIN_LEVEL}" \
    "$@"
