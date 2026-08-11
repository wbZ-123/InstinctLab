cd ~/InstinctLab-foothold

RUN_DIR="logs/instinct_rl/g1_parkour/20260724_160239_foothold_amp_aligned_curriculum_30000it_v2"
TARGET_ITER=30000

LATEST_CKPT="$(ls -v "$RUN_DIR"/model_*.pt | tail -1)"
LATEST_ITER="$(basename "$LATEST_CKPT" .pt | sed 's/model_//')"
REMAINING=$((TARGET_ITER - LATEST_ITER))

echo "latest checkpoint: $LATEST_CKPT"
echo "latest iter: $LATEST_ITER"
echo "remaining: $REMAINING"

if [ "$REMAINING" -gt 0 ]; then
  FOOTHOLD_DEBUG_EVENT_MAX_COUNT=100 \
  RUN_NAME=foothold_amp_aligned_curriculum_resume_to_30000it \
  MAX_ITERATIONS="$REMAINING" \
  NUM_ENVS=4096 \
  ./scripts/foothold_train.sh \
    --resume \
    --load_run "$(basename "$RUN_DIR")" \
    --checkpoint "$(basename "$LATEST_CKPT")"
else
  echo "Already reached ${TARGET_ITER} iterations."
fi