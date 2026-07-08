#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${LOGDIR:-logs/instinct_rl/g1_parkour}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"

cd "${REPO_ROOT}"

cmd=(
    tensorboard
    --logdir "${LOGDIR}"
    --host "${HOST}"
    --port "${PORT}"
)

echo "[tensorboard] logdir: ${REPO_ROOT}/${LOGDIR}"
echo "[tensorboard] url: http://${HOST}:${PORT}"
echo "[tensorboard] If opening in a browser, use localhost/forwarded URL instead of 0.0.0.0."

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[tensorboard] command:'
    printf ' %q' "${cmd[@]}"
    printf '
'
    exit 0
fi

"${cmd[@]}"
