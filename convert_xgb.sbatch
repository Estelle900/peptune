#!/bin/bash
#SBATCH --job-name=convert_xgb
#SBATCH --output=logs/convert_xgb_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --error=logs/convert_xgb_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# Run from submission directory
cd "$SLURM_SUBMIT_DIR"

# Activate your environment (adjust as needed)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate peptune

python src/convert_xgb.py \
    --repo . \
    --temperature 0.1
