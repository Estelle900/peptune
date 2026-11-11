#!/bin/bash
#SBATCH --job-name=peptune-train-xgb
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/peptune-train-xgb-%j.out
#SBATCH --error=logs/peptune-train-xgb-%j.err
#
## Default input directory:  peptune/PeptideBERT/data/
# Results:                  peptune/xgboost_results

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
TRAIN_SCRIPT="${SUBMIT_DIR}/src/train_peptune_xgb.py"

# Adjust this if you keep PeptideBERT/data elsewhere.
if [[ -d "${SUBMIT_DIR}/PeptideBERT/data" ]]; then
    DATA_DIR="${SUBMIT_DIR}/PeptideBERT/data"
else
    echo "ERROR: Missing PeptideBERT/data under the repository root or src/." >&2
    exit 1
fi

OUTPUT_DIR="${SUBMIT_DIR}/xgboost_results"
mkdir -p "${OUTPUT_DIR}"

source .venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

printf 'Job ID: %s\nHost: %s\nRepository: %s\nScript: %s\nData: %s\nOutput: %s\n' \
    "${SLURM_JOB_ID:-local}" "$(hostname)" "${SUBMIT_DIR}" "${TRAIN_SCRIPT}" "${DATA_DIR}" "${OUTPUT_DIR}"
python --version

# PeptideCLM embedding extraction uses CUDA; XGBoost uses 8 CPU threads.
# The script saves training logs, plots, predictions, metrics and JSON weights.
srun python -u "${TRAIN_SCRIPT}" \
    --repo-root "${SUBMIT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --tasks solubility hemolysis nonfouling \
    --encoding auto \
    --device cpu \
    --batch-size 16 \
    --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
    --n-trials 50 \
    --max-rounds 1000 \
    --early-stopping-rounds 40 \
    --threshold-mode validation_f1 \
    --seed 42

echo "Training finished. Held-out test metrics: ${OUTPUT_DIR}/combined_test_metrics.json"
