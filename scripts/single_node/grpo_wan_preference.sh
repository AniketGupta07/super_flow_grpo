#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-1}"
CONFIG="${CONFIG:-config/grpo.py:preference_wan2_1_test}"
# vps: VPS PairMLP probability vs cached reference | random: uniform 0-1 | clip: softmax of CLIP scores vs reference video
REWARD_FN="${REWARD_FN:-vps}"

case "$REWARD_FN" in
    vps|random|clip) ;;
    *)
        echo "[grpo_wan_preference] Invalid REWARD_FN='$REWARD_FN'. Expected one of: vps, random, clip." >&2
        exit 1
        ;;
esac

if [[ "$REWARD_FN" == "clip" && -z "${WAN_REF_VIDEO_DIR:-}" ]]; then
    echo "[grpo_wan_preference] REWARD_FN=clip requires WAN_REF_VIDEO_DIR pointing at the ref_XXXXXXX.mp4 reference videos." >&2
    exit 1
fi

echo "[grpo_wan_preference] Launching training with $NUM_TRAIN_GPUS GPUs (config=$CONFIG, reward_fn=$REWARD_FN) ..."
echo "[grpo_wan_preference] Qwen vision embedder runs on CPU per process (Wan stays on GPU). For local test, NUM_TRAIN_GPUS=1 avoids loading multiple 7B copies in RAM."

accelerate launch \
    --config_file "$SCRIPT_DIR/accelerate_configs/multi_gpu.yaml" \
    --num_processes="$NUM_TRAIN_GPUS" \
    --main_process_port 29501 \
    "$SCRIPT_DIR/train_wan2_1.py" \
    --config "$CONFIG" \
    --reward_fn "$REWARD_FN"
