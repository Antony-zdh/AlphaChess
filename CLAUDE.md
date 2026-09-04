# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaChess is a chess AI trained first via Supervised Fine-Tuning (SFT) on grandmaster games from Lichess, then via Proximal Policy Optimization (PPO) self-play. The model is a ResNet-style CNN with policy and value heads. **The trained model currently performs poorly**: PPO historically peaks around iteration 300 then regresses below the SFT baseline (see `eval/ppo_flaws.md`). Several root causes were fixed in recent commits (GAE sign bug, gradient clipping, opening diversity, draw penalty, opponent pool); the model is still not strong.

## Common Commands

```bash
# Install dependencies (uv is the package manager; no requirements.txt)
uv sync

# SFT training (20 epochs, batch 256, LR 1e-3)
bash scripts/train.sh

# PPO training (1000 iterations, LR 1e-4 from SFT checkpoint)
bash scripts/ppo.sh

# PPO pilot run (20 iterations, save every 10) — quick smoke test
bash scripts/ppo_pilot_run.sh

# Evaluation: PPO checkpoints vs SFT baseline (128 games, batch 16)
bash eval/run_eval.sh

# Evaluation: round-robin competition across model checkpoints
bash eval/run_competition.sh

# Evaluation: model vs Stockfish across skill levels
bash eval/run_stockfish_eval.sh
```

There is **no test suite and no linter/formatter configured** (no pytest/ruff/black/mypy in `pyproject.toml`). Verification is done via the eval scripts above. All training/eval shell scripts set `PROJECT_ROOT`/`PYTHONPATH` and pick `py`/`python3`/`python` as the interpreter.

To run a single training path without the shell wrapper, pass the args directly, e.g.:
```bash
python src/ppo.py --iterations 20 --lr 1e-4 --model_path models/sft_models/alpha_chess_epoch_16.pth --save_interval 10 --output_path models/ppo_models/
```

### `scripts/run.sh` caveat

`scripts/run.sh` invokes `python src/main.py ...`, but `src/main.py` is a stub (`print("Hello from alphachess!")`) — it does **not** launch the GUI. The real game loop and CLI live in `src/alpha_chess.py` (`ChessGame`, with `HumanPlayer`/`AIPlayer`). To actually run a game, invoke `src/alpha_chess.py` directly with the same `--player0/--player1/--model_path/--use_gui` flags (edit the script or call it yourself).

## Architecture

### Neural Network (`src/model.py`)

`AlphaChess` is a CNN with:
- **Input**: 19×8×8 tensor — channels 0-5 current player's pieces (P,N,B,R,Q,K), 6-11 opponent's, 12 side-to-move, 13-16 castling rights (current K, current Q, opponent K, opponent Q), 17 en-passant target square, 18 50-move clock (normalized `/50.0`). Board is always encoded from the **current player's perspective** (black-to-move flips the board).
- **Backbone**: One CNN input block (19→128 channels, 3×3 conv + BN + ReLU), then 10 residual blocks (128 channels, 3×3 convs + BN, skip connection + ReLU).
- **Policy head**: 1×1 conv (128→2) + BN + ReLU → flatten to `2*8*8 = 128` → `Linear(128, 4096)` producing raw logits for all 64×64 from→to move combinations. No softmax here.
- **Value head**: 1×1 conv (128→1) + BN + ReLU → flatten to 64 → `Linear(64,128)` + ReLU → `Linear(128,1)` → `tanh`, output ∈ (-1, 1).

The 4096-logit policy cannot distinguish underpromotions (one logit per from/to pair). Illegal moves are masked to `-inf` downstream (not in the model). Move index = `from_square * 64 + to_square`. Promotions default to queen and are resolved at the engine layer.

### Training Pipeline

**SFT** (`src/train.py`): Reads `data/raw/GM_games.pgn` (3988 GM games) via a streaming `ChessDataset`, cross-entropy on policy logits + MSE on value. Adam + CosineAnnealingLR. Best checkpoint: `models/sft_models/alpha_chess_epoch_16.pth` (overfit after epoch 16; the path `models/alpha_chess_epoch_16.pth` referenced by some defaults is stale).

**PPO** (`src/ppo.py`): Self-play with trajectory collection, GAE advantage estimation, clipped policy loss. Each iteration: collect 32 trajectories (16 boards played as both colors → 16 white + 16 black trajectories), compute GAE, then run **4 epochs** over the batch in mini-batches of 8 trajectories. Module-level constants (in `src/ppo.py`):

| Constant | Value | Meaning |
|---|---|---|
| `PPO_BATCH_SIZE` | 32 | trajectories per iteration |
| `PPO_MINI_BATCH_SIZE` | 8 | trajectories per gradient step |
| `PPO_EPOCHS` | 4 | passes over collected data |
| `PPO_GAMMA` | 0.99 | discount |
| `PPO_LAMBDA` | 0.95 | GAE lambda |
| `PPO_EPSILON` | 0.2 | clip ratio |
| `PPO_VF_LOSS_COEF` | 0.5 | value loss weight |
| `PPO_ENTROPY_COEF` | 0.01 | entropy bonus |
| `PPO_GRADIENT_CLIP` | 0.5 | max grad norm |
| `PPO_DRAW_PENALTY` | −0.05 | terminal reward for a draw (both sides) |
| `PPO_OPENING_PLIES` | 8 | `random.randint(0,8)` random opening plies per board for diversity |
| `PPO_POOL_FRACTION` | 0.3 | fraction of iterations playing vs a frozen-pool opponent |
| `PPO_POOL_MAX_SIZE` | 3 | max frozen checkpoints kept in opponent pool |

**Reward**: intermediate steps = 0; terminal = +1 win / −1 loss / −0.05 draw (draws penalize both sides equally). 
**GAE cross-color bootstrapping**: `next_value` for a step uses the **negated opponent value** for the same position (zero-sum sign flip) — white step `i`'s `next_value = -black step i value`, black step `i`'s `next_value = -white step (i+1) value`. Advantages are normalized per-trajectory (zero mean / unit std).
**Opponent pool**: a FIFO list of frozen checkpoint paths (max 3). Each iteration, 30% chance of playing 32 games against a random pool checkpoint (pool opponent plays **greedy argmax**, learner samples); otherwise pure self-play. New checkpoints are appended to the pool on each save.

Checkpoints saved every `args.save_interval` (default 100) under `models/ppo_models/ppo_model_itr_{N}/` with model, optimizer, scheduler state, and a JSON config. CLI: `--iterations` (1000), `--model_path` (SFT path), `--lr` (default **1e-3**, but `scripts/ppo.sh` overrides to **1e-4**), `--wandb`, `--log_interval`, `--save_interval`, `--output_path`. Note the cosine scheduler steps once per mini-batch update, with `T_max = iterations * 4 * 4`.

`src/buffer.py` provides `OnPolicyBuffer` (capacity=1, returns latest) and `OffPolicyBuffer` (capacity=1000, random sampling). `src/grpo.py` is a **stub** — trajectory/group containers exist but advantage computation and the GRPO objective are unimplemented (`PPOAdvantageManager.compute_advantages` is `pass`).

### Game Engine & Inference (`src/alpha_chess.py`, `src/ui.py`)

`ChessGame` supports human-vs-AI, AI-vs-AI, and human-vs-human. `AIPlayer` loads a checkpoint, encodes board state via `BoardEncoder`, masks illegal moves (sets logits to `-inf`), softmaxes, and selects **argmax** during inference (greedy — no sampling/temperature path exists in `AIPlayer`; PPO training uses a separate sampling path in `ActorCritic`). Pygame GUI is in `src/ui.py`. Piece graphics are in `assets/`.

Note: `AIPlayer.make_move` has a redundant final move reconstruction that re-creates `chess.Move(from, to)` without the promotion flag, discarding the promotion resolved earlier — promotion correctness is fragile there.

### Evaluation (`eval/`)

- `eval/opponents.py` — opponent wrappers under a common `BaseOpponent` interface (`select_move`/`select_moves_batch`, `supports_batch`). Includes `AlphaChessOpponent` (this project's models, greedy or sampled), `StockfishOpponent` (UCI, skill 0-20 → ~800-3500 Elo), and PyTorch ports of two external Keras baselines (`ChessDeepRLTorchOpponent`, `ChessAITorchOpponent`) with weight-transfer helpers. `OpponentSpec` + `create_opponent(spec, device)` parse spec strings like `alpha:path`, `chess_ai:dir`, `chess_deep_rl:path`, `stockfish:skill`.
- `eval/competition.py` — round-robin batch game runner (plays both color assignments per pair, batches moves for `supports_batch` opponents), appends JSONL game logs, recomputes Elo. CLI: `--model_glob`, `--models_list`, `--games_per_pair` (16), `--batch_size` (16), `--k_factor` (24).
- `eval/elo_system.py` — plain sequential Elo (k=24, initial 1500, no logistic optimization).
- `eval/evaluate_ppo_vs_sft.py` — head-to-head PPO checkpoint vs SFT baseline with per-game value tracking (PPO greedy, baseline temperature-scaled). Globs all `ppo_model_itr_*` checkpoints.
- `eval/evaluate_vs_stockfish.py` — model vs Stockfish across skill levels.
- `eval/ppo_flaws.md` — diagnosis of why PPO regresses (GAE sign bug, single epoch, no grad clip, draw starvation, no opponent pool). Most were addressed in recent commits; `ppo_eval_results.json` and `elo_*.json` hold the numbers.

## Key File Map

| File | Role |
|------|------|
| `src/model.py` | AlphaChess network, BoardEncoder, PolicyHead, ValueHead |
| `src/alpha_chess.py` | ChessGame loop, HumanPlayer, AIPlayer (greedy inference) |
| `src/train.py` | SFT training loop |
| `src/ppo.py` | PPO training loop, ActorCritic, PPOGroup, PPOTrajectory |
| `src/ui.py` | Pygame GUI |
| `eval/opponents.py` | Opponent wrappers + external baseline ports |
| `eval/competition.py` | Round-robin batch game runner |
| `eval/evaluate_ppo_vs_sft.py` | PPO-vs-SFT head-to-head evaluator |

## Experiment Tracking

All training runs log to Weights & Biases (project `AlphaChess_PPO` for PPO). SFT logs gradient tracking; PPO logs policy loss, value loss, entropy, total loss, and LR. Enable with `--wandb`.
