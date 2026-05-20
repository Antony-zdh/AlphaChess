#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

PPO_ROOT="models/ppo_models"
BASELINE_PATH="models/alpha_chess_epoch_16.pth"
GAMES=128
BATCH_SIZE=16
OUTPUT_JSON="eval/ppo_eval_results.json"
OPENING_PLIES=4
SEED=1234
BASELINE_TEMPERATURE=0.2

if command -v py &> /dev/null; then
    PYTHON_CMD="py"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

$PYTHON_CMD eval/evaluate_ppo_vs_sft.py \
    --ppo_root "$PPO_ROOT" \
    --baseline_path "$BASELINE_PATH" \
    --games "$GAMES" \
    --batch_size "$BATCH_SIZE" \
    --output_json "$OUTPUT_JSON" \
    --opening_plies "$OPENING_PLIES" \
    --seed "$SEED" \
    --baseline_temperature "$BASELINE_TEMPERATURE"
