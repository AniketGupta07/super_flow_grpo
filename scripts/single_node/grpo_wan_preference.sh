#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-1}"
CONFIG="${CONFIG:-config/grpo.py:preference_wan2_1_test}"
REWARD_FN="${REWARD_FN:-vps}"
OUT_DIR="${OUT_DIR:-}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [REWARD_FN] [options]

Reward functions (give as the first argument or via -r):
  vps         VPS PairMLP probability against the cached reference video (default)
  random      Uniform 0-1, as a control baseline
  clip        Softmax of CLIP scores against the reference video (needs WAN_REF_VIDEO_DIR)
  clip2       CLIP score of the video against its prompt alone, reference-free
  videoscore  VideoScore2 visual quality / alignment / physics blend, reference-free

Options:
  -r, --reward-fn NAME  Reward function (env: REWARD_FN)
  -c, --config SPEC     Training config spec (env: CONFIG)
  -g, --gpus N          Number of training processes (env: NUM_TRAIN_GPUS)
  -o, --out-dir DIR     Checkpoint output dir; saved as DIR/checkpoint-<step>/lora (env: OUT_DIR)
  -p, --port N          accelerate main_process_port; use 0 to auto-pick a free port (env: MAIN_PROCESS_PORT, default 29501)
  -h, --help            Show this message

Examples:
  $(basename "$0") clip
  $(basename "$0") random --gpus 4
  $(basename "$0") vps -c config/grpo.py:preference_wan2_1
EOF
}

positional_seen=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--reward-fn|--reward_fn) REWARD_FN="${2:?--reward-fn needs a value}"; shift 2 ;;
        --reward-fn=*|--reward_fn=*) REWARD_FN="${1#*=}"; shift ;;
        -c|--config) CONFIG="${2:?--config needs a value}"; shift 2 ;;
        --config=*) CONFIG="${1#*=}"; shift ;;
        -g|--gpus|--num-gpus) NUM_TRAIN_GPUS="${2:?--gpus needs a value}"; shift 2 ;;
        --gpus=*|--num-gpus=*) NUM_TRAIN_GPUS="${1#*=}"; shift ;;
        -o|--out-dir|--out_dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --out-dir=*|--out_dir=*) OUT_DIR="${1#*=}"; shift ;;
        -p|--port|--main-process-port|--main_process_port) MAIN_PROCESS_PORT="${2:?--port needs a value}"; shift 2 ;;
        --port=*|--main-process-port=*|--main_process_port=*) MAIN_PROCESS_PORT="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        -*)
            echo "[grpo_wan_preference] Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -n "$positional_seen" ]]; then
                echo "[grpo_wan_preference] Unexpected argument: $1" >&2
                usage >&2
                exit 1
            fi
            REWARD_FN="$1"
            positional_seen=1
            shift
            ;;
    esac
done

case "$REWARD_FN" in
    vps|random|clip|clip2|videoscore) ;;
    *)
        echo "[grpo_wan_preference] Invalid reward function '$REWARD_FN'. Expected one of: vps, random, clip, clip2, videoscore." >&2
        usage >&2
        exit 1
        ;;
esac

if [[ "$REWARD_FN" == "clip" && -z "${WAN_REF_VIDEO_DIR:-}" ]]; then
    echo "[grpo_wan_preference] reward_fn=clip requires WAN_REF_VIDEO_DIR pointing at the ref_XXXXXXX.mp4 reference videos." >&2
    exit 1
fi

echo "[grpo_wan_preference] Launching training with $NUM_TRAIN_GPUS GPUs (config=$CONFIG, reward_fn=$REWARD_FN) ..."
if [[ "$REWARD_FN" == "videoscore" ]]; then
    echo "[grpo_wan_preference] VideoScore2 loads a 7B scorer on the training GPU (~16 GiB) and takes ~10-30s per clip (it writes a rationale before scoring; sampling makes the length vary). Lower VIDEOSCORE2_FPS / VIDEOSCORE2_MAX_PIXELS / VIDEOSCORE2_MAX_NEW_TOKENS to speed it up, VIDEOSCORE2_TEMPERATURE=0 for a deterministic reward, or VIDEOSCORE2_DEVICE=cpu to keep the GPU free."
else
    echo "[grpo_wan_preference] Qwen vision embedder runs on CPU per process (Wan stays on GPU). For local test, NUM_TRAIN_GPUS=1 avoids loading multiple 7B copies in RAM."
fi

train_args=(--config "$CONFIG" --reward_fn "$REWARD_FN")
if [[ -n "$OUT_DIR" ]]; then
    echo "[grpo_wan_preference] Checkpoints -> $OUT_DIR/checkpoint-<step>/lora"
    train_args+=(--out_dir "$OUT_DIR")
fi

# Make the in-repo flow_grpo package win over any editable/site-packages install that
# points at a different checkout. `accelerate launch scripts/train_wan2_1.py` sets
# sys.path[0] to scripts/ (not the repo root), so without this the import can resolve
# to a stale editable install and use the wrong reward functions.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[grpo_wan_preference] accelerate main_process_port=$MAIN_PROCESS_PORT"
echo "[grpo_wan_preference] PYTHONPATH prefixed with repo root: $REPO_ROOT"
accelerate launch \
    --config_file "$SCRIPT_DIR/accelerate_configs/multi_gpu.yaml" \
    --num_processes="$NUM_TRAIN_GPUS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    "$SCRIPT_DIR/train_wan2_1.py" \
    "${train_args[@]}"
