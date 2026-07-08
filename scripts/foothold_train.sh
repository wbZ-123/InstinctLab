#!/usr/bin/env bash
set -euo pipefail

# Canonical entry point for foothold-planner parkour training.
#
# Why this wrapper exists:
# - The machine also has ~/InstinctLab. Without PYTHONPATH, Python may import
#   that older checkout instead of this ~/InstinctLab-foothold repository.
# - The parkour AMP task needs the motion-reference selection YAML, whose path
#   contains an ampersand. Keeping it here avoids shell quoting mistakes.
#
# Examples:
#   ./scripts/foothold_train.sh
#   NUM_ENVS=4 MAX_ITERATIONS=1 RUN_NAME=foothold_reward_tag_check ./scripts/foothold_train.sh
#   NUM_ENVS=64 MAX_ITERATIONS=500 RUN_NAME=foothold_planner_500it_real ./scripts/foothold_train.sh
#   DRY_RUN=1 ./scripts/foothold_train.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${REPO_ROOT}/../IsaacLab}"

export PYTHONPATH="${REPO_ROOT}/source/instinctlab:${PYTHONPATH:-}"
export PARKOUR_MOTION_REFERENCE_DIR="${PARKOUR_MOTION_REFERENCE_DIR:-/home/zhangweibo/Datasets/hiking_in_the_wild/data&model/parkour_motion_reference}"
export PARKOUR_MOTION_SELECTION_FILE="${PARKOUR_MOTION_SELECTION_FILE:-${PARKOUR_MOTION_REFERENCE_DIR}/parkour_motion_without_run.yaml}"

TASK="${TASK:-Instinct-Parkour-Target-Amp-G1-v0}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITERATIONS="${MAX_ITERATIONS:-500}"
RUN_NAME="${RUN_NAME:-foothold_planner}"

if [[ ! -f "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
    echo "[foothold_train] IsaacLab launcher not found: ${ISAACLAB_ROOT}/isaaclab.sh" >&2
    echo "[foothold_train] Set ISAACLAB_ROOT=/path/to/IsaacLab if needed." >&2
    exit 1
fi

if [[ ! -f "${PARKOUR_MOTION_SELECTION_FILE}" ]]; then
    echo "[foothold_train] Motion selection file not found:" >&2
    echo "  ${PARKOUR_MOTION_SELECTION_FILE}" >&2
    echo "[foothold_train] Set PARKOUR_MOTION_REFERENCE_DIR or PARKOUR_MOTION_SELECTION_FILE." >&2
    exit 1
fi

cd "${REPO_ROOT}"

cmd=(
    "${ISAACLAB_ROOT}/isaaclab.sh"
    -p scripts/instinct_rl/train.py
    --headless
    --task "${TASK}"
    --num_envs "${NUM_ENVS}"
    --max_iterations "${MAX_ITERATIONS}"
    --run_name "${RUN_NAME}"
    "$@"
)

echo "[foothold_train] repo: ${REPO_ROOT}"
echo "[foothold_train] task: ${TASK}"
echo "[foothold_train] num_envs: ${NUM_ENVS}"
echo "[foothold_train] max_iterations: ${MAX_ITERATIONS}"
echo "[foothold_train] run_name: ${RUN_NAME}"
echo "[foothold_train] motion_reference_dir: ${PARKOUR_MOTION_REFERENCE_DIR}"
echo "[foothold_train] motion_selection_file: ${PARKOUR_MOTION_SELECTION_FILE}"
echo "[foothold_train] pythonpath_prefix: ${REPO_ROOT}/source/instinctlab"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[foothold_train] command:'
    printf ' %q' "${cmd[@]}"
    printf '
'
    exit 0
fi

"${cmd[@]}"
