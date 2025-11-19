#!/bin/bash
#SBATCH --job-name=sample-wildchat
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sample_wildchat_%j.out

set -euo pipefail
module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"
conda activate verlenv

echo "=============== WILDCHAT SAMPLING JOB ==============="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Output dir: /home/abaielli/darling/datasets"
echo "======================================================"

python3 /home/abaielli/darling/sample_wildchat.py
