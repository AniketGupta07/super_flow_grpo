#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-1}"
CONFIG="${CONFIG:-config/grpo.py:preference_wan2_1_test}"

echo "[grpo_wan_preference] Launching training with $NUM_TRAIN_GPUS GPUs (config=$CONFIG) ..."
echo "[grpo_wan_preference] Qwen vision embedder runs on CPU per process (Wan stays on GPU). For local test, NUM_TRAIN_GPUS=1 avoids loading multiple 7B copies in RAM."

accelerate launch \
    --config_file "$SCRIPT_DIR/accelerate_configs/multi_gpu.yaml" \
    --num_processes="$NUM_TRAIN_GPUS" \
    --main_process_port 29501 \
    "$SCRIPT_DIR/train_wan2_1.py" \
    --config "$CONFIG"
