# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaChess is a chess AI trained first via Supervised Fine-Tuning (SFT) on grandmaster games from Lichess, then via Proximal Policy Optimization (PPO) self-play. The model is a ResNet-style CNN with policy and value heads.

## Common Commands

```bash
# Install dependencies
uv sync

# SFT training (20 epochs, ~batch_size=256)
bash scripts/train.sh

# PPO training (1000 iterations)
bash scripts/ppo.sh

# PPO pilot run (20 iterations, quick test)
bash scripts/ppo_pilot_run.sh

# Run a game with GUI (human vs AI or AI vs AI)
bash scripts/run.sh

# Collect training data from Lichess API
python src/data_collector.py
```

All training scripts pass arguments to `src/train.py` or `src/ppo.py` directly. To run without scripts, check the shell files for the full argument lists.

## Architecture

### Neural Network (`src/model.py`)

`AlphaChess` is a CNN with:
- **Input**: 19×8×8 tensor — channels 0-5 are current player's pieces (P,N,B,R,Q,K), 6-11 opponent's, 12 side-to-move, 13-16 castling rights, 17 en-passant square, 18 50-move clock (normalized)
- **Backbone**: One CNN input block (19→128 channels), then 10 residual blocks (128 channels)
- **Policy head**: 2-channel 1×1 conv → FC(304→4096) → 4096 logits for all 64×64 from→to move combinations
- **Value head**: 1-channel 1×1 conv → FC(64→128→1) → tanh output ∈ (-1, 1)

Board is always encoded from the current player's perspective (white or black flips the board). Illegal moves are masked to `-inf` before softmax. Promotions default to queen.

### Training Pipeline

**SFT** (`src/train.py`): Reads `data/raw/GM_games.pgn` (3988 GM games) via a streaming `ChessDataset`, cross-entropy on policy logits + MSE on value. Uses Adam + CosineAnnealingLR. Best checkpoint: `models/sft_models/alpha_chess_epoch_16.pth`.

**PPO** (`src/ppo.py`): Self-play with trajectory collection, GAE advantage estimation (γ=0.99, λ=0.95), clipped policy loss (ε=0.2), VF coefficient=0.5, entropy coefficient=0.01. Batch size: 32 trajectories, mini-batch: 8. Checkpoints saved every 100 iterations in `models/ppo_models/`.

`src/buffer.py` provides `OnPolicyBuffer` (capacity=1, used by PPO) and `OffPolicyBuffer`. `src/grpo.py` is a stub for a future GRPO algorithm.

### Game Engine & Inference (`src/alpha_chess.py`, `src/ui.py`)

`ChessGame` supports human-vs-AI, AI-vs-AI, and human-vs-human. `AIPlayer` loads a checkpoint, encodes board state via `BoardEncoder`, masks illegal moves, and selects argmax during inference (sampling during PPO training). Pygame GUI is in `src/ui.py`. Piece graphics are in `assets/`.

### Evaluation (`eval/`)

`eval/opponents.py` defines 5+ opponent classes including `AlphaChessOpponent` (this project's models), plus ports of external Keras/PyTorch chess AI baselines. `eval/competition.py` runs batch games (up to 32 parallel). `eval/elo_system.py` computes ELO. `eval/evaluate_ppo_vs_sft.py` is the main evaluation script comparing PPO vs SFT checkpoints.

## Key File Map

| File | Role |
|------|------|
| `src/model.py` | AlphaChess network, BoardEncoder, PolicyHead, ValueHead |
| `src/alpha_chess.py` | ChessGame loop, HumanPlayer, AIPlayer |
| `src/train.py` | SFT training loop |
| `src/ppo.py` | PPO training loop |
| `src/ui.py` | Pygame GUI |
| `eval/opponents.py` | Opponent wrappers for evaluation |
| `eval/competition.py` | Batch game runner |
| `scripts/run.sh` | Inference entry point (edit model path here) |

## Experiment Tracking

All training runs log to Weights & Biases. The project uses gradient tracking during SFT and logs policy accuracy, loss components, and learning rate each epoch/iteration.
