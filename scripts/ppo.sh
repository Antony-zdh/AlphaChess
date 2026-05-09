# 1. Set Project Root for Python Imports
# Get the directory where this script sits (scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Go one level up to get f:\alpha_chess
PROJECT_ROOT="$SCRIPT_DIR/.."
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

ITERATIONS=1000
LR=1e-4
MODEL_PATH="models/alpha_chess_epoch_16.pth"
LOG_INTERVAL=1
SAVE_INTERVAL=100
OUTPUT_PATH="models/ppo_models/"
USE_WANDB="${USE_WANDB:-true}"

echo "🚀 Starting Alpha Chess PPO Training..."
echo "iterations: $ITERATIONS"
echo "lr: $LR"
echo "model_path: $MODEL_PATH"
echo "log_interval: $LOG_INTERVAL"
echo "save_interval: $SAVE_INTERVAL"
echo "output_path: $OUTPUT_PATH"
echo "Wandb logging: $USE_WANDB"

if command -v py &> /dev/null; then
    PYTHON_CMD="py"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

$PYTHON_CMD src/ppo.py \
    --iterations $ITERATIONS \
    --lr $LR \
    --model_path $MODEL_PATH \
    --log_interval $LOG_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --output_path $OUTPUT_PATH \
    $( [ "$USE_WANDB" = true ] && echo "--wandb" )


echo "✅ PPO Training finished."