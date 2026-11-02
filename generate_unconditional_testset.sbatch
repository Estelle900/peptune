#!/bin/bash
#SBATCH --job-name=peptune-unconditional
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/peptune-unconditional-%j.out
#SBATCH --error=logs/peptune-unconditional-%j.err

set -euo pipefail

HOME_LOC="${SLURM_SUBMIT_DIR}"
SCRIPT_LOC="${HOME_LOC}/src"
CKPT="${HOME_LOC}/checkpoints/epoch13-new-tokenizer.ckpt"
export PEPTUNE_HOME="$HOME_LOC"

# Same positional conventions as generate_mcts_all_supported.sbatch.
PROT_NAME2="${1:-}"
MODE="${2:-2}"
MODEL="${3:-unconditional}"
LENGTH="${4:-100}"
EPOCH="${5:-7}"

cd "$HOME_LOC"
mkdir -p "$HOME_LOC/logs"
sed 's|${HOME}|'"$PEPTUNE_HOME"'|g' \
    "$SCRIPT_LOC/config.yaml.template" > "$SCRIPT_LOC/config.yaml"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
if ! command -v uv >/dev/null 2>&1; then
    echo 'Error: uv is not installed' >&2
    exit 1
fi
if [[ ! -x "$HOME_LOC/.venv/bin/python" ]]; then
    echo 'Error: .venv not found; create it first' >&2
    exit 1
fi
if [[ ! -f "$CKPT" ]]; then
    echo "Error: checkpoint missing: $CKPT" >&2
    exit 1
fi

echo "Job: ${SLURM_JOB_ID:-unknown}; Node: $(hostname)"
echo "Mode: $MODE; Model: $MODEL; Length: $LENGTH; Epoch: $EPOCH"
nvidia-smi

for PROT_NAME1 in $(python -c "from src.LNR_Sequences import LNR_Sequences as r; print(*r.keys())"); do
    echo "Generating for target: $PROT_NAME1"
    mkdir -p "$HOME_LOC/results_unconditional/$PROT_NAME1"
    HYDRA_ARGS=(
        "+prot_name1=$PROT_NAME1"
        "++mode=$MODE"
        "+model_type=$MODEL"
        "+length=$LENGTH"
        "+epoch=$EPOCH"
        "sampling.seq_length=$LENGTH"
    )
    if [[ -n "$PROT_NAME2" ]]; then
        HYDRA_ARGS+=("+prot_name2=$PROT_NAME2")
    fi
    cd "$SCRIPT_LOC"
    uv run --no-project \
        --python "$HOME_LOC/.venv/bin/python" \
        python generate_unconditional.py \
        --config-path "$SCRIPT_LOC" \
        --config-name config \
        "base_path=$HOME_LOC" \
        "eval.checkpoint_path=$CKPT" \
        "${HYDRA_ARGS[@]}"
done

echo 'Unconditional generation completed for all supported targets.'
