#!/bin/bash
#SBATCH --job-name=darling-train
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=logs/darling_train_%j.out
#SBATCH --error=logs/darling_train_%j.err

set -euo pipefail

##############################
# ENVIRONMENT
##############################
module purge
module load 2023
module load Anaconda3/2023.07-2
eval "$(conda shell.bash hook)"
conda activate verlenv

echo "========== ENV INFO =========="
nvidia-smi
echo "=============================="

##############################
# PATHS
##############################
TRAIN_DS="/home/abaielli/darling/datasets/wildchat10k.parquet"
VAL_DS="/home/abaielli/darling/datasets/wildchat_valid.parquet"

LLAMA_PATH="/home/abaielli/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
ATHENE_PATH="/home/abaielli/darling/models/athene_rm_8b"
PARTITION_REWARD="/home/abaielli/darling/verl/verl/utils/reward_score/partition_reward_vllm_serve.py"

##############################
# START LOCAL VLLM SERVER (GPU 0)
##############################
export VLLM_PORT=8000
CONTAINER=/projects/2/managed_datasets/containers/vllm/vllm_25.09.sif
MODEL="dogtooth/similarity-classifier-f168-hf"

echo "Starting local VLLM classifier on GPU0..."

CUDA_VISIBLE_DEVICES=0 \
apptainer exec --nv -B $PWD $CONTAINER bash -lc "
    python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --dtype float16 \
        --max-model-len 4096 \
        --port $VLLM_PORT \
        --host 0.0.0.0
" > vllm_classifier.log 2>&1 &

VLLM_PID=$!

echo "Waiting for VLLM..."
for i in {1..40}; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$VLLM_PORT/health || true)
    [[ "$STATUS" == "200" ]] && echo "VLLM READY" && break
    sleep 2
done

[[ "$STATUS" != "200" ]] && echo "VLLM FAILED" && kill $VLLM_PID && exit 1


##############################
# PPO TRAINING (GPU 1,2)
##############################
export CUDA_VISIBLE_DEVICES=1,2

B=32
N=4
L=512

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DS \
    data.val_files=$VAL_DS \
    data.prompt_key="prompt" \
    data.train_batch_size=$B \
    data.max_prompt_length=256 \
    data.max_response_length=$L \
    actor_rollout_ref.model.path=$LLAMA_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.endpoint="http://localhost:${VLLM_PORT}" \
    reward_model.enable=True \
    reward_model.reward_manager=diversity \
    reward_model.model.path=$ATHENE_PATH \
    reward_model.model.input_tokenizer=$ATHENE_PATH \
    reward_model.micro_batch_size_per_gpu=16 \
    +reward_model.custom_diversity_function.path=$PARTITION_REWARD \
    +reward_model.custom_diversity_function.name=partition \
    trainer.project_name="darling_llama32_3b" \
    trainer.experiment_name="local_vllm" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=2 \
    trainer.logger="[console]" \
    trainer.default_local_dir="/home/abaielli/darling/checkpoints"

echo "Stopping VLLM..."
kill $VLLM_PID || true
