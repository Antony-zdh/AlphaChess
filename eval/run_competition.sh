#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

MODEL_GLOB="models/*.pth"
SFT_MODEL_LIST="models/sft_models/alpha_chess_epoch_16.pth"
PPO_MODEL_LIST="models/ppo_models/ppo_model_itr_1000/ppo_model_itr_1000.pth"
CHESS_AI_MODEL_LIST="chess_ai:models/chess_ai/elo700, chess_ai:models/chess_ai/elo1100, chess_ai:models/chess_ai/elo1200"
CHESS_DEEP_RL_MODEL_LIST="chess_deep_rl:models/chess_deep_rl/model.h5"
MODELS_LIST="$SFT_MODEL_LIST,$PPO_MODEL_LIST,$CHESS_AI_MODEL_LIST,$CHESS_DEEP_RL_MODEL_LIST"
GAMES_PER_PAIR=16
BATCH_SIZE=16
GAMES_LOG="eval/elo_games.jsonl"
RATINGS_OUT="eval/elo_ratings.json"
K_FACTOR=24.0

if command -v py &> /dev/null; then
    PYTHON_CMD="py"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

$PYTHON_CMD eval/competition.py \
    --model_glob "$MODEL_GLOB" \
    ${MODELS_LIST:+--models_list "$MODELS_LIST"} \
    --games_per_pair "$GAMES_PER_PAIR" \
    --batch_size "$BATCH_SIZE" \
    --games_log "$GAMES_LOG" \
    --ratings_out "$RATINGS_OUT" \
    --k_factor "$K_FACTOR"
