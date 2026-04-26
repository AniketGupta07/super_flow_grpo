#!/bin/bash
#SBATCH --job-name=wan_pref_grpo
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --exclude=simurgh5,simurgh3
#SBATCH --output=/simurgh/group/super_video/api_test/logs/wan_pref_grpo_%j.log
#SBATCH --error=/simurgh/group/super_video/api_test/logs/wan_pref_grpo_%j.err

set -eo pipefail

FLOW_GRPO_ROOT="${FLOW_GRPO_ROOT:-/simurgh/group/super_video/flow_grpo}"
CONFIG="${CONFIG:-config/grpo.py:preference_wan2_1}"
NUM_GPUS="${NUM_GPUS:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"
MODEL_TYPE="${MODEL_TYPE:-both}"
USE_REPHRASED="${USE_REPHRASED:-0}"
OUT_DIR="${OUT_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-type)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --model-type (expected: social|both|aa)"
        exit 1
      fi
      MODEL_TYPE="$2"
      shift 2
      ;;
    --rephrased)
      USE_REPHRASED=1
      shift
      ;;
    --out-dir)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --out-dir (expected: /path/to/output)"
        exit 1
      fi
      OUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: sbatch $0 [--model-type social|both|aa] [--rephrased] [--out-dir /path/to/output]"
      exit 1
      ;;
  esac
done

if [[ "$MODEL_TYPE" != "social" && "$MODEL_TYPE" != "both" && "$MODEL_TYPE" != "aa" ]]; then
  echo "Invalid MODEL_TYPE='$MODEL_TYPE'. Expected one of: social, both, aa"
  exit 1
fi

echo "Job $SLURM_JOB_ID | GPUs $NUM_GPUS | $(date)"
echo "Config: $CONFIG"
echo "MODEL_TYPE: $MODEL_TYPE"
echo "USE_REPHRASED: $USE_REPHRASED"
echo "OUT_DIR: ${OUT_DIR:-<default from config>}"

mkdir -p /simurgh/group/super_video/api_test/logs

source /simurgh/group/super_video/conda/bin/activate rl || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$FLOW_GRPO_ROOT"

accelerate launch \
  --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes="$NUM_GPUS" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  scripts/train_wan2_1.py \
  --config "$CONFIG" \
  --model_type "$MODEL_TYPE" \
  --rephrased_only "$USE_REPHRASED" \
  --out_dir "$OUT_DIR"

echo "Done $(date)"
