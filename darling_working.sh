#!/bin/bash
#SBATCH --job-name=darling-train
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
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
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
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

##############################################
# START LOCAL VLLM CLASSIFIER SERVER ON GPU 0
##############################################
export VLLM_SERVER_HOSTNAME="localhost"
export VLLM_PORT=8000
export PYTHONUNBUFFERED=1

MODEL="dogtooth/similarity-classifier-f168-hf"
CONTAINER=/projects/2/managed_datasets/containers/vllm/vllm_25.09.sif

echo "Starting VLLM classifier on GPU0..."

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
echo "VLLM PID = $VLLM_PID"

echo "Waiting for LOCAL VLLM server at http://localhost:${VLLM_PORT}/health"
for i in {1..60}; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${VLLM_PORT}/health" || true)
    if [[ "$STATUS" == "200" ]]; then
        echo "LOCAL VLLM READY!"
        break
    fi
    sleep 2
done

if [[ "$STATUS" != "200" ]]; then
    echo "ERROR: LOCAL VLLM failed to start"
    tail -n 50 vllm_classifier.log
    kill $VLLM_PID || true
    exit 1
fi

##############################################
# GPU USAGE MONITOR (BACKGROUND)
##############################################
echo "Training will use GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

(
    while true; do
        echo "------ GPU USAGE @ $(date) ------"
        nvidia-smi
        sleep 20
    done
) &

GPU_MONITOR_PID=$!

##############################
# HYPERPARAMETERS
##############################
B=32
N=4
L=512

##############################
# TRAINING
##############################
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
\
    data.train_files=$TRAIN_DS \
    data.val_files=$VAL_DS \
    data.prompt_key="prompt" \
    data.train_batch_size=$B \
    data.max_prompt_length=256 \
    data.max_response_length=$L \
    data.filter_overlong_prompts=True \
    data.truncation=error \
\
    actor_rollout_ref.model.path=$LLAMA_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
\
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
\
    reward_model.enable=True \
    reward_model.reward_manager=diversity \
    reward_model.model.path=$ATHENE_PATH \
    reward_model.model.input_tokenizer=$ATHENE_PATH \
    reward_model.micro_batch_size_per_gpu=16 \
    +reward_model.custom_diversity_function.path=$PARTITION_REWARD \
    +reward_model.custom_diversity_function.name=partition \
\
    trainer.project_name="darling_llama32_3b" \
    trainer.experiment_name="darling_partition" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.logger="[console]" \
    trainer.default_local_dir="/home/abaielli/darling/checkpoints" \
    trainer.validation_data_dir="/home/abaielli/darling/checkpoints/rollouts" \
    trainer.critic_warmup=0 \
    trainer.save_freq=2 \
    trainer.test_freq=2 \


##############################################
# CLEANUP
##############################################
echo "Stopping GPU monitor..."
kill $GPU_MONITOR_PID || true

echo "Killing classifier server..."
kill $VLLM_PID || true
