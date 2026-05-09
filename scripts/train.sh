# 1. Set Project Root for Python Imports
# Get the directory where this script sits (scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Go one level up to get f:\alpha_chess
PROJECT_ROOT="$SCRIPT_DIR/.."
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

BATCH_SIZE=256
EPOCHS=20
LR=0.001

echo "🚀 Starting Alpha Chess Training..."
echo "batch_size: $BATCH_SIZE"
echo "epochs: $EPOCHS"
echo "lr: $LR"

if command -v py &> /dev/null; then
    PYTHON_CMD="py"
else
    PYTHON_CMD="python"
fi

$PYTHON_CMD src/train.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --log_interval 4 \
    --save_interval 1 \
    --wandb 


echo "✅ Training finished."