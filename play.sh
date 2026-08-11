cd ~/InstinctLab-foothold

export PARKOUR_MOTION_REFERENCE_DIR="/home/zhangweibo/Datasets/hiking_in_the_wild/data&model/parkour_motion_reference"
export PARKOUR_MOTION_SELECTION_FILE="${PARKOUR_MOTION_REFERENCE_DIR}/parkour_motion_without_run.yaml"

PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
../IsaacLab/isaaclab.sh -p scripts/instinct_rl/play.py \
  --task Instinct-Parkour-Target-Amp-G1-Play-v0 \
  --num_envs 1 \
  --load_run 20260724_192653_foothold_amp_aligned_curriculum_30000it_v3 \
  --print_reset_debug \
  --print_foothold_debug \
  --print_foothold_marker_debug \
  --print_foothold_debug_interval 10 \
  --print_foothold_debug_env_id 0 \
  --show_foothold_debug_markers \
  --foothold_debug_trajectory_samples 12 \
