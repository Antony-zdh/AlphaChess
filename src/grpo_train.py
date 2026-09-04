"""Multi-GPU GRPO self-play training for AlphaChess.

GRPO: no critic. Advantage = group-normalized Monte-Carlo discounted return
(gamma^(T-1-t) * terminal_reward), normalized across the whole group. Update =
PPO clipped surrogate (ratio clip + entropy), NO value loss. Pure policy-
sampling self-play (no MCTS during self-play); eval uses MCTS for true strength.

Starts from the strong SFT base (models/sft_large/alpha_chess_best.pth, ~1150 ELO).
Reuses: ppo.self_play_episodes data-collection pattern, ppo train_ppo clip,
az_train_mp actor-learner plumbing (mtime reload, _atomic_save, numpy queue).

Run:
  cd <project root>
  PYTHONPATH=. uv run python src/grpo_train.py \
      --model_path models/sft_large/alpha_chess_best.pth --output_dir models/grpo \
      --iterations 50 --games_per_iter 64 --selfplay_batch 16 \
      --grpo_epochs 4 --mini_batch 8 --lr 1e-4 \
      --actor_gpus 2,3,4,5,6,7 --learner_gpu 1 --wandb
"""
import argparse
import logging
import os
import queue
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import chess
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from src.model import AlphaChess, BoardEncoder
from src.grpo import GRPOTrajectory, GRPOGroup, GRPOAdvantageManager
from src.mcts import action_to_move
from eval.opponents import AlphaZeroOpponent, StockfishOpponent
from eval.competition import play_games


def self_play_episodes_grpo(model, device, n_games, opening_plies=8,
                            max_plies=160, draw_penalty=0.0, seed=None):
    """Play n_games concurrent self-play games with pure policy sampling.
    Returns list of GRPOTrajectory (2 per game: white steps + black steps).
    draw_penalty: reward for draws (0 = neutral/AlphaZero-style, negative = penalize)."""
    model.eval()
    rng = random.Random(seed)
    boards = [chess.Board() for _ in range(n_games)]
    for b in boards:
        for _ in range(rng.randint(0, opening_plies)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
    trajs_white = [GRPOTrajectory() for _ in range(n_games)]
    trajs_black = [GRPOTrajectory() for _ in range(n_games)]
    encoder = BoardEncoder()
    ply = [0] * n_games

    with torch.no_grad():
        while any(not boards[i].is_game_over() and ply[i] < max_plies
                 for i in range(n_games)):
            active = [i for i in range(n_games)
                      if not boards[i].is_game_over() and ply[i] < max_plies]
            if not active:
                break
            active_boards = [boards[i] for i in active]
            states = torch.stack([encoder.encode(b) for b in active_boards]).to(device)
            policy_logits, _ = model(states)
            for offset, i in enumerate(active):
                b = boards[i]
                legal = list(b.legal_moves)
                idx = [m.from_square * 64 + m.to_square for m in legal]
                mask = torch.zeros(4096, device=device)
                mask[idx] = 1.0
                masked = policy_logits[offset].masked_fill(mask == 0, float("-inf"))
                dist = torch.distributions.Categorical(logits=masked)
                action = dist.sample()
                log_prob = dist.log_prob(action).item()
                legal_mask_cpu = mask.bool()
                turn = b.turn
                state_cpu = encoder.encode(b)
                if turn == chess.WHITE:
                    trajs_white[i].add_step(state_cpu, action.item(), log_prob, turn, legal_mask_cpu)
                else:
                    trajs_black[i].add_step(state_cpu, action.item(), log_prob, turn, legal_mask_cpu)
                b.push(action_to_move(b, action.item()))
                ply[i] += 1

    # Terminal reward per color (this trajectory player's perspective).
    for i in range(n_games):
        result = boards[i].result() if boards[i].is_game_over() else "1/2-1/2"
        if result == "1-0":
            rw, rb = 1.0, -1.0
        elif result == "0-1":
            rw, rb = -1.0, 1.0
        else:
            rw = rb = draw_penalty
        trajs_white[i].set_terminal_reward(rw)
        trajs_black[i].set_terminal_reward(rb)
    return trajs_white + trajs_black


def grpo_update(model, optimizer, group, device, eps=0.2, entropy_coef=0.01,
                grad_clip=0.5, epochs=4, mini_batch=8):
    """Clipped-surrogate GRPO update over the group (no value loss).
    Multi-epoch + mini-batch like PPO. Returns (mean policy_loss, mean entropy)."""
    model.train()
    states, actions, old_log_probs, advantages, legal_masks = group.get_data_tensors()
    N = len(actions)
    if N == 0:
        return 0.0, 0.0
    idx = np.arange(N)
    total_pl = total_ent = 0.0
    n_updates = 0
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, N, mini_batch):
            mb = idx[start:start + mini_batch]
            mb_t = torch.as_tensor(mb, dtype=torch.long)
            s = states[mb_t].to(device)
            a = actions[mb_t].to(device)
            olp = old_log_probs[mb_t].to(device)
            adv = advantages[mb_t].to(device)
            lm = legal_masks[mb_t].to(device)
            policy_logits, _ = model(s)
            masked = policy_logits.masked_fill(~lm, float("-inf"))
            dist = torch.distributions.Categorical(logits=masked)
            nlp = dist.log_prob(a)
            ratio = torch.exp(nlp - olp)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
            pl = -torch.min(s1, s2).mean()
            ent = dist.entropy().mean()
            loss = pl - entropy_coef * ent
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_pl += pl.item()
            total_ent += ent.item()
            n_updates += 1
    return total_pl / max(n_updates, 1), total_ent / max(n_updates, 1)


def actor_worker(rank, gpu_id, model_path, latest_path, sample_queue, args):
    """Self-play actor: generate GRPOTrajectory lists, push to queue.
    Reloads latest model on mtime change (on-policy: old_log_probs match the
    generating policy). GRPOTrajectory stores numpy so pickling uses pipe,
    NOT torch shared memory (avoids /dev/shm exhaustion)."""
    device = torch.device(f"cuda:{gpu_id}")
    model = AlphaChess().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    last_mtime = os.path.getmtime(model_path)
    while True:
        try:
            if os.path.exists(latest_path):
                m = os.path.getmtime(latest_path)
                if m > last_mtime:
                    model.load_state_dict(torch.load(latest_path, map_location=device))
                    last_mtime = m
        except Exception:
            pass
        try:
            trajs = self_play_episodes_grpo(
                model, device, args.selfplay_batch,
                opening_plies=args.opening_plies, max_plies=args.max_plies,
                draw_penalty=args.draw_penalty,
                seed=random.randint(0, 2**31 - 1))
            sample_queue.put(trajs, timeout=600)
        except Exception as e:
            logging.info(f"actor {rank} (gpu {gpu_id}) error: {e}")
            time.sleep(1)


def _atomic_save(model, path):
    tmp = path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate_grpo(model, args, device):
    """Eval GRPO model + MCTS vs Stockfish skill 0,2,4. Returns mean score."""
    tmp = os.path.join(args.output_dir, "_eval_tmp.pth")
    torch.save(model.state_dict(), tmp)
    az = AlphaZeroOpponent(tmp, device, n_simulations=args.mcts_sims, greedy=True)
    total_score = 0.0
    n = 0
    for skill in [0, 2, 4]:
        sf = StockfishOpponent(args.stockfish_path, skill)
        half = args.eval_games // 2
        try:
            rw = play_games(az, sf, "grpo", f"sf{skill}", half, args.eval_batch,
                            opening_plies=8, max_plies=240, seed=42)
            rb = play_games(sf, az, f"sf{skill}", "grpo", args.eval_games - half,
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


def train_grpo(args):
    device = torch.device(f"cuda:{args.learner_gpu}")
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)
    latest_path = os.path.join(args.output_dir, "latest.pth")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = AlphaChess().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    _atomic_save(model, latest_path)
    logging.info(f"Loaded start checkpoint: {args.model_path}")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    actor_gpus = [int(g) for g in args.actor_gpus.split(",") if g.strip()]
    sample_queue = mp.Queue(maxsize=200)
    procs = []
    for rank, g in enumerate(actor_gpus):
        p = mp.Process(target=actor_worker,
                       args=(rank, g, args.model_path, latest_path, sample_queue, args))
        p.start()
        procs.append(p)
    logging.info(f"Started {len(actor_gpus)} actors on GPUs {actor_gpus}; learner GPU {args.learner_gpu}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iterations, eta_min=1e-5)
    adv_manager = GRPOAdvantageManager(gamma=args.gamma)

    if args.wandb:
        wandb.init(project="AlphaChess_GRPO", config=vars(args))

    best_score = -1.0
    for it in range(1, args.iterations + 1):
        t0 = time.time()
        # 1. Collect a group: games_per_iter games = 2*games_per_iter trajectories.
        trajs = []
        target = args.games_per_iter * 2
        while len(trajs) < target:
            try:
                t = sample_queue.get(timeout=60)
                trajs.extend(t)
            except queue.Empty:
                if not any(p.is_alive() for p in procs):
                    logging.info("All actors dead, stopping collection.")
                    break
                continue
        # 2. GRPO update (on-policy: consume once).
        pl = ent = 0.0
        if trajs:
            group = GRPOGroup(trajs)
            adv_manager.compute_advantages(group)
            pl, ent = grpo_update(model, optimizer, group, device,
                                  eps=args.eps, entropy_coef=args.entropy_coef,
                                  epochs=args.grpo_epochs, mini_batch=args.mini_batch)
            scheduler.step()
        # 3. Save latest so actors reload the improved policy.
        _atomic_save(model, latest_path)
        # 4. Periodic checkpoint.
        if it % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, f"grpo_iter_{it}.pth"))
        # 5. Eval vs Stockfish (GRPO policy + MCTS).
        score = -1.0
        if it % args.eval_every == 0:
            score = evaluate_grpo(model, args, device)
            logging.info(f"  eval mean score (skill0,2,4) = {score:.3f}")
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, "grpo_best.pth"))
                logging.info(f"  ✅ new best score={best_score:.3f} -> grpo_best.pth")
        lr = scheduler.get_last_lr()[0]
        logging.info(f"iter {it}/{args.iterations} trajs={len(trajs)} "
                     f"pl={pl:.4f} ent={ent:.4f} lr={lr:.2e} score={score:.3f} "
                     f"time={time.time()-t0:.0f}s")
        if args.wandb:
            wandb.log({"policy_loss": pl, "entropy": ent, "lr": lr,
                       "stockfish_score": score})

    for p in procs:
        p.terminate()
        p.join(timeout=5)
    logging.info(f"Training done. best score={best_score:.3f}.")
    if args.wandb:
        wandb.finish()


def main():
    p = argparse.ArgumentParser(description="Multi-GPU GRPO self-play training")
    p.add_argument("--model_path", default="models/sft_large/alpha_chess_best.pth")
    p.add_argument("--output_dir", default="models/grpo")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--games_per_iter", type=int, default=64)
    p.add_argument("--selfplay_batch", type=int, default=16, help="concurrent boards per actor round")
    p.add_argument("--grpo_epochs", type=int, default=4)
    p.add_argument("--mini_batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--opening_plies", type=int, default=8)
    p.add_argument("--max_plies", type=int, default=160)
    p.add_argument("--draw_penalty", type=float, default=0.0, help="0=neutral, negative=penalize draws")
    p.add_argument("--mcts_sims", type=int, default=100, help="MCTS sims for eval only")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--eval_games", type=int, default=8)
    p.add_argument("--eval_batch", type=int, default=8)
    p.add_argument("--stockfish_path", default="tools/stockfish_bin")
    p.add_argument("--actor_gpus", default="2,3,4,5,6,7")
    p.add_argument("--learner_gpu", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()
    train_grpo(args)


if __name__ == "__main__":
    main()
