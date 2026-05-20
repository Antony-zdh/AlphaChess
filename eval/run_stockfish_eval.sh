#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

STOCKFISH_PATH="${STOCKFISH_PATH:-stockfish}"
SKILL_LEVELS="${SKILL_LEVELS:-0,2,4,6,8,10}"
GAMES="${GAMES:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OUTPUT="${OUTPUT:-eval/stockfish_results.json}"

# Override MODELS to evaluate specific checkpoints, e.g.:
#   MODELS="models/sft_models/alpha_chess_epoch_16.pth,models/ppo_models/ppo_model_itr_1000/ppo_model_itr_1000.pth"
# Leave unset to evaluate the SFT baseline + all PPO checkpoints.
MODELS="${MODELS:-}"
MODEL_GLOB="${MODEL_GLOB:-}"

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

ARGS=(
    --stockfish_path "$STOCKFISH_PATH"
    --skill_levels "$SKILL_LEVELS"
    --games "$GAMES"
    --batch_size "$BATCH_SIZE"
    --output "$OUTPUT"
)
[ -n "$MODELS" ]     && ARGS+=(--models "$MODELS")
[ -n "$MODEL_GLOB" ] && ARGS+=(--model_glob "$MODEL_GLOB")

cd "$PROJECT_ROOT"
$PYTHON_CMD eval/evaluate_vs_stockfish.py "${ARGS[@]}"
