#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-1}"
CONFIG="${CONFIG:-config/grpo.py:preference_wan2_1_test}"
REWARD_FN="${REWARD_FN:-vps}"
OUT_DIR="${OUT_DIR:-}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"
VPS_P="${VPS_P:-${VPS_PROB:-}}"
VPS_PROB_SWEEP_ROOT="${VPS_PROB_SWEEP_ROOT:-/simurgh2/projects/super_video/exps/prob_sweep/runs}"

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
      --vps-p VALUE     For reward_fn=vps: use PairMLP trained at social_win_prob=VALUE
                        from \$VPS_PROB_SWEEP_ROOT/pVALUE/models (env: VPS_P).
                        Examples: 0.5, 0.85, p0.5. Overrides VPS_MODEL_DIR.
  -h, --help            Show this message

Examples:
  $(basename "$0") clip
  $(basename "$0") random --gpus 4
  $(basename "$0") vps -c config/grpo.py:preference_wan2_1
  $(basename "$0") vps --vps-p 0.5
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
        --vps-p|--vps_p|--vps-prob|--vps_prob) VPS_P="${2:?--vps-p needs a value}"; shift 2 ;;
        --vps-p=*|--vps_p=*|--vps-prob=*|--vps_prob=*) VPS_P="${1#*=}"; shift ;;
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

if [[ -n "$VPS_P" ]]; then
    if [[ "$REWARD_FN" != "vps" ]]; then
        echo "[grpo_wan_preference] --vps-p / VPS_P only applies to reward_fn=vps (got '$REWARD_FN')." >&2
        exit 1
    fi
    # Strip optional leading "p" and canonicalize via python so 0.50 -> p0.5.
    _vps_tag="$(python3 -c 'import sys; v=sys.argv[1].strip(); v=v[1:] if v[:1] in "pP" else v; print(f"p{float(v):g}")' "$VPS_P")" || {
        echo "[grpo_wan_preference] Invalid --vps-p='$VPS_P'. Expected e.g. 0.5 or p0.5." >&2
        exit 1
    }
    _vps_model_dir="${VPS_PROB_SWEEP_ROOT}/${_vps_tag}/models"
    if [[ ! -f "${_vps_model_dir}/best.pth" || ! -f "${_vps_model_dir}/config.json" ]]; then
        echo "[grpo_wan_preference] No PairMLP for VPS_P=$VPS_P at ${_vps_model_dir}." >&2
        echo "[grpo_wan_preference] Available:" >&2
        for _d in "${VPS_PROB_SWEEP_ROOT}"/p*/models; do
            [[ -d "$_d" ]] && echo "  $(basename "$(dirname "$_d")")" >&2
        done
        exit 1
    fi
    export VPS_P="$VPS_P"
    export VPS_PROB_SWEEP_ROOT
    echo "[grpo_wan_preference] VPS_P=$VPS_P -> PairMLP ${_vps_model_dir}"
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
