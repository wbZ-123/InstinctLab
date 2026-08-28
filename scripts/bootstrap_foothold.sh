#!/usr/bin/env bash
set -euo pipefail

# Initialize the repository-owned source dependencies and verify the remaining
# external runtime/data requirements. This script deliberately does not install
# Isaac Sim because its installation method is machine-specific.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_ISAACLAB_COMMIT="f73c33173801f5f8afea4142482e47b7710c2b75"
EXPECTED_INSTINCT_RL_COMMIT="f870ead0953fa0e3c3da3349b0aece1c74bfb421"
ISAACLAB_ROOT="${REPO_ROOT}/third_party/IsaacLab"
INSTINCT_RL_ROOT="${REPO_ROOT}/third_party/instinct_rl"
PYTHON_BIN="${PYTHON_BIN:-python}"
PARKOUR_MOTION_REFERENCE_DIR="${PARKOUR_MOTION_REFERENCE_DIR:-${HOME}/Datasets/hiking_in_the_wild/data&model/parkour_motion_reference}"
PARKOUR_MOTION_SELECTION_FILE="${PARKOUR_MOTION_SELECTION_FILE:-${PARKOUR_MOTION_REFERENCE_DIR}/parkour_motion_without_run.yaml}"

CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
    CHECK_ONLY=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--check-only]" >&2
    exit 2
fi

if [[ "${CHECK_ONLY}" == "0" ]]; then
    echo "[bootstrap] Initializing pinned Git submodules..."
    git -C "${REPO_ROOT}" submodule update --init --recursive
fi

failures=0

check_submodule() {
    local label="$1"
    local path="$2"
    local expected_commit="$3"

    if [[ ! -e "${path}/.git" ]]; then
        echo "[bootstrap] ERROR: ${label} is not initialized at ${path}" >&2
        echo "[bootstrap] Run: git submodule update --init --recursive" >&2
        failures=$((failures + 1))
        return
    fi

    local actual_commit
    actual_commit="$(git -C "${path}" rev-parse HEAD)"
    if [[ "${actual_commit}" != "${expected_commit}" ]]; then
        echo "[bootstrap] ERROR: ${label} commit mismatch" >&2
        echo "  expected: ${expected_commit}" >&2
        echo "  actual:   ${actual_commit}" >&2
        failures=$((failures + 1))
        return
    fi

    echo "[bootstrap] OK: ${label} ${actual_commit}"
}

check_submodule "IsaacLab" "${ISAACLAB_ROOT}" "${EXPECTED_ISAACLAB_COMMIT}"
check_submodule "Instinct-RL" "${INSTINCT_RL_ROOT}" "${EXPECTED_INSTINCT_RL_COMMIT}"

if "${PYTHON_BIN}" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("isaacsim") else 1)' >/dev/null 2>&1; then
    echo "[bootstrap] OK: Isaac Sim Python package is visible to ${PYTHON_BIN}"
else
    echo "[bootstrap] ERROR: Isaac Sim 5.1 is not visible to ${PYTHON_BIN}" >&2
    echo "[bootstrap] Activate the project environment (normally: conda activate hiking)" >&2
    echo "[bootstrap] and install Isaac Sim 5.1 before training or Play." >&2
    failures=$((failures + 1))
fi

if [[ -f "${PARKOUR_MOTION_SELECTION_FILE}" ]]; then
    echo "[bootstrap] OK: motion selection file ${PARKOUR_MOTION_SELECTION_FILE}"
else
    echo "[bootstrap] ERROR: motion selection file not found" >&2
    echo "  ${PARKOUR_MOTION_SELECTION_FILE}" >&2
    echo "[bootstrap] Set PARKOUR_MOTION_REFERENCE_DIR or PARKOUR_MOTION_SELECTION_FILE." >&2
    failures=$((failures + 1))
fi

if [[ "${failures}" -ne 0 ]]; then
    echo "[bootstrap] Setup check failed with ${failures} problem(s)." >&2
    exit 1
fi

echo "[bootstrap] Foothold project dependencies are ready."
