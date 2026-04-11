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

echo "Job $SLURM_JOB_ID | GPUs $NUM_GPUS | $(date)"
echo "Config: $CONFIG"

mkdir -p /simurgh/group/super_video/api_test/logs

source /simurgh/group/super_video/conda/bin/activate rl || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$FLOW_GRPO_ROOT"

accelerate launch \
  --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes="$NUM_GPUS" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  scripts/train_wan2_1.py \
  --config "$CONFIG"

echo "Done $(date)"
