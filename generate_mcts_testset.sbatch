#!/bin/bash
#SBATCH --job-name=peptune-mcts
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/peptune-mcts-%j.out
#SBATCH --error=logs/peptune-mcts-%j.err

set -euo pipefail

# PepTune repository location
HOME_LOC="${SLURM_SUBMIT_DIR}"
SCRIPT_LOC="${HOME_LOC}/src"
CKPT="${HOME_LOC}/checkpoints/epoch13-new-tokenizer.ckpt"

export PEPTUNE_HOME="$HOME_LOC"

sed 's|${HOME}|'"$PEPTUNE_HOME"'|g' \
    "$SCRIPT_LOC/config.yaml.template" \
    > "$SCRIPT_LOC/config.yaml"

# Original PepTune test arguments
PROT_NAME2="${1:-}"
MODE="${2:-2}"
MODEL="${3:-mcts}"
LENGTH="${4:-100}"
EPOCH="${5:-7}"

cd "$HOME_LOC"

# Use uv environment instead of Conda
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed"
    exit 1
fi

if [[ ! -x "$HOME_LOC/.venv/bin/python" ]]; then
    echo "Error: .venv not found; create it first"
    exit 1
fi


echo "SLURM Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Model: $MODEL"
echo "Length: $LENGTH"
echo "Epoch: $EPOCH"

nvidia-smi

# Preserve the original Hydra arguments
for PROT_NAME1 in $(python -c "from src.LNR_Sequences import LNR_Sequences as r; print(*r.keys())"); do
  mkdir -p "$HOME_LOC/logs" "$HOME_LOC/results/$PROT_NAME1"
  HYDRA_ARGS=(
      "+prot_name1=$PROT_NAME1"
      "++mode=$MODE"
      "+model_type=$MODEL"
      "+length=$LENGTH"
      "+epoch=$EPOCH"
  )

  if [[ -n "$PROT_NAME2" ]]; then
      HYDRA_ARGS+=("+prot_name2=$PROT_NAME2")
  fi

  cd "$SCRIPT_LOC"

  uv run --no-project \
      --python "$HOME_LOC/.venv/bin/python" \
      python generate_mcts.py \
      --config-path "$SCRIPT_LOC" \
      --config-name config \
      "base_path=$HOME_LOC" \
      "eval.checkpoint_path=$CKPT" \
      "${HYDRA_ARGS[@]}"
  
done
echo "MCTS generation completed"
