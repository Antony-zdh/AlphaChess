"""AlphaZero-style self-play training loop for AlphaChess.

Closed loop: MCTS (using the current net) generates self-play games -> games
train the net (policy imitates the MCTS visit distribution, value predicts the
game outcome) -> stronger net -> stronger MCTS -> iterate.

Behavioral cloning of MCTS (not PPO). More sample-efficient and stable for
full-game chess than policy-gradient. The resulting net+MCTS can later feed a
GRPO stage.

Run:
  cd <project root>
  PYTHONPATH=. uv run python src/az_train.py \
      --model_path models/sft_v2/alpha_chess_best.pth \
      --output_dir models/az --iterations 50 --games_per_iter 128 \
      --mcts_sims 200 --wandb
"""
import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import chess
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from src.model import AlphaChess, BoardEncoder
from src.mcts import BatchMCTS, action_to_move
from src.buffer import OffPolicyBuffer
from eval.opponents import AlphaZeroOpponent, StockfishOpponent
from eval.competition import play_games


def self_play_games(model, mcts, n_games, opening_plies=8, max_plies=200, seed=None):
    """Play n_games concurrent self-play games with MCTS. Returns
    (samples, results) where samples is a list of (state(19,8,8), pi(4096), z(1))
    and results is the list of game-result strings.

    Games are adjudicated as a draw once max_plies is reached (weak nets can
    otherwise drag on with shuffle/repetition)."""
    model.eval()
    rng = random.Random(seed)
    boards = [chess.Board() for _ in range(n_games)]
    # Random opening jitter for diversity.
    for b in boards:
        for _ in range(rng.randint(0, opening_plies)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
    encoder = BoardEncoder()
    # Per-game list of (state, pi, turn) collected before each push.
    records = [[] for _ in range(n_games)]
    ply = [0] * n_games

    while True:
        active_idx = [i for i, b in enumerate(boards)
                      if not b.is_game_over() and ply[i] < max_plies]
        if not active_idx:
            break
        active_boards = [boards[i] for i in active_idx]
        actions, pis = mcts.search_with_pi(active_boards)
        for offset, i in enumerate(active_idx):
            b = boards[i]
            turn = b.turn  # side to move, captured BEFORE push
            state = encoder.encode(b)        # (19, 8, 8)
            pi = pis[offset]                  # (4096,)
            records[i].append((state, pi.clone(), turn))
            a = actions[offset]
            b.push(action_to_move(b, a))
            ply[i] += 1

    # Assign terminal z (from each step's side-to-move perspective) and emit.
    samples = []
    results = []
    for i, rec in enumerate(records):
        # Game over -> real result; ply-capped -> adjudicate as draw.
        result = boards[i].result() if boards[i].is_game_over() else "1/2-1/2"
        results.append(result)
        if result == "1-0":
            gw = 1.0
        elif result == "0-1":
            gw = -1.0
        else:
            gw = 0.0  # draw / unfinished -> 0
        for state, pi, turn in rec:
            z = gw * (1.0 if turn == chess.WHITE else -1.0)
            samples.append((state, pi, torch.tensor([z], dtype=torch.float32)))
    return samples, results


def train_step(model, optimizer, buffer, batch_size, device):
    """One gradient step on a batch sampled from the replay buffer.
    Policy loss = soft cross-entropy vs MCTS visit distribution (masked to legal).
    Value loss = MSE vs game outcome z. Total = policy + 0.5 * value."""
    model.train()
    batch = buffer.sample(batch_size)
    # Samples may be CPU tensors or numpy arrays (mp.Queue path uses numpy to
    # avoid torch shared-memory exhaustion). np.asarray handles both.
    states = torch.as_tensor(np.stack([np.asarray(s[0]) for s in batch])).to(device)
    pis = torch.as_tensor(np.stack([np.asarray(s[1]) for s in batch])).to(device)
    zs = torch.as_tensor(np.stack([np.asarray(s[2]) for s in batch])).to(device)

    policy_logits, value = model(states)
    legal = pis > 0
    masked = policy_logits.masked_fill(~legal, float("-inf"))
    logp = torch.log_softmax(masked, dim=1)
    logp = torch.nan_to_num(logp, neginf=0.0, posinf=0.0)  # kill -inf on illegal slots
    policy_loss = -(pis * logp).sum(dim=1).mean()
    value_loss = F.mse_loss(value, zs)
    loss = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return policy_loss.item(), value_loss.item()


@torch.no_grad()
def evaluate_az(model, args, device):
    """Play the current net+MCTS vs Stockfish skill 0,2,4. Returns mean score."""
    tmp = os.path.join(args.output_dir, "_eval_tmp.pth")
    torch.save(model.state_dict(), tmp)
    az = AlphaZeroOpponent(tmp, device, n_simulations=getattr(args, "eval_sims", None) or args.mcts_sims, greedy=True)
    total_score = 0.0
    n = 0
    for skill in [0, 2, 4]:
        sf = StockfishOpponent(args.stockfish_path, skill)
        half = args.eval_games // 2
        try:
            rw = play_games(az, sf, "az", f"sf{skill}", half, args.eval_batch,
                            opening_plies=8, max_plies=240, seed=42)
            rb = play_games(sf, az, f"sf{skill}", "az", args.eval_games - half,
                            args.eval_batch, opening_plies=8, max_plies=240, seed=43)
        finally:
            sf.close()
        wins = sum(1 for r in rw if r.result == "1-0") + \
            sum(1 for r in rb if r.result == "0-1")
        draws = sum(1 for r in rw if r.result not in ("1-0", "0-1")) + \
            sum(1 for r in rb if r.result not in ("1-0", "0-1"))
        total = len(rw) + len(rb)
        score = (wins + 0.5 * draws) / total if total else 0.0
        total_score += score
        n += 1
        logging.info(f"    skill{skill} (~{sf.approx_elo}): W{wins} D{draws} "
                     f"L{total - wins - draws} score={score:.2f}")
    try:
        os.remove(tmp)
    except OSError:
        pass
    return total_score / n if n else 0.0


def train_az(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    print(f"Training on device: {device}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = AlphaChess().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    logging.info(f"Loaded start checkpoint: {args.model_path}")

    encoder = BoardEncoder()
    mcts = BatchMCTS(model, encoder, device,
                     n_simulations=args.mcts_sims, c_puct=args.c_puct,
                     temperature=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iterations, eta_min=1e-5)
    buffer = OffPolicyBuffer(capacity=args.buffer_capacity)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        wandb.init(project="AlphaChess_AZ", config=vars(args))

    best_score = -1.0
    for it in range(1, args.iterations + 1):
        # Temperature schedule: explore early, exploit later.
        mcts.temperature = 1.0 if it <= args.iterations * 0.6 else 0.5

        # 1. Self-play data generation (play games_per_iter in selfplay_batch rounds).
        t0 = time.time()
        all_samples = []
        all_results = []
        remaining = args.games_per_iter
        while remaining > 0:
            batch = min(args.selfplay_batch, remaining)
            samples, results = self_play_games(
                model, mcts, batch, opening_plies=args.opening_plies,
                max_plies=args.max_plies,
                seed=args.seed + it * 1000 + remaining)
            for s in samples:
                buffer.add(s)
            all_samples.extend(samples)
            all_results.extend(results)
            remaining -= batch
        # 2. Training.
        pl = vl = 0.0
        if len(buffer.buffer) >= args.train_batch:
            for _ in range(args.train_steps):
                pl, vl = train_step(model, optimizer, buffer, args.train_batch, device)
            scheduler.step()
        # 3. Logging.
        n = len(all_results)
        wins = sum(1 for r in all_results if r == "1-0")
        losses = sum(1 for r in all_results if r == "0-1")
        draws = n - wins - losses
        lr = scheduler.get_last_lr()[0]
        logging.info(f"iter {it}/{args.iterations} buffer={len(buffer.buffer)} "
                     f"pl={pl:.4f} vl={vl:.4f} games={n} W/D/L={wins}/{draws}/{losses} "
                     f"lr={lr:.2e} time={time.time()-t0:.0f}s")
        if args.wandb:
            wandb.log({"policy_loss": pl, "value_loss": vl,
                       "buffer_size": len(buffer.buffer),
                       "wins": wins, "draws": draws, "losses": losses, "lr": lr})
        # 4. Checkpoint.
        if it % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, f"az_iter_{it}.pth"))
        # 5. Eval.
        if it % args.eval_every == 0:
            score = evaluate_az(model, args, device)
            logging.info(f"  eval mean score (skill0,2,4) = {score:.3f}")
            if args.wandb:
                wandb.log({"stockfish_score": score})
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, "az_best.pth"))
                logging.info(f"  ✅ new best score={best_score:.3f} -> az_best.pth")

    logging.info(f"Training done. best score={best_score:.3f}.")
    if args.wandb:
        wandb.finish()


def main():
    p = argparse.ArgumentParser(description="AlphaZero-style MCTS self-play training")
    p.add_argument("--model_path", default="models/sft_v2/alpha_chess_best.pth")
    p.add_argument("--output_dir", default="models/az")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--games_per_iter", type=int, default=128)
    p.add_argument("--selfplay_batch", type=int, default=32)
    p.add_argument("--train_steps", type=int, default=600)
    p.add_argument("--train_batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--mcts_sims", type=int, default=200)
    p.add_argument("--c_puct", type=float, default=4.0)
    p.add_argument("--opening_plies", type=int, default=8)
    p.add_argument("--max_plies", type=int, default=200, help="Adjudicate self-play games as draw after N plies.")
    p.add_argument("--buffer_capacity", type=int, default=100000)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--eval_games", type=int, default=12)
    p.add_argument("--eval_batch", type=int, default=12)
    p.add_argument("--stockfish_path", default="tools/stockfish_bin")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()
    train_az(args)


if __name__ == "__main__":
    main()
