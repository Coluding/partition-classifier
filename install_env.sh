#!/bin/bash
#SBATCH --job-name=install-darling-env
#SBATCH --partition=gpu_h100       
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/install_darling_env_%j.out
#SBATCH --error=logs/install_darling_env_%j.err

set -euo pipefail

module purge
module load 2023
module load Anaconda3/2023.07-2

ENV_NAME=verlenv

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# PyTorch (CUDA 12.8 per README)
pip install --upgrade pip
pip install torch torchvision torchaudio 

# Install verl (editable)
pip install -e ./verl

# vLLM and deps
USE_MEGATRON=0 bash verl/scripts/install_vllm_sglang_mcore.sh
pip install vllm==0.8.3
pip install flash-attn --no-build-isolation

echo "Environment '${ENV_NAME}' is ready."