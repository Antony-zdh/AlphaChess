"""DAPO (clip-higher + dynamic sampling) variant of GRPO for chess self-play.

Two changes vs grpo_train:
1. Clip-higher: asymmetric ratio clip — clamp(ratio, 1-eps_low, 1+eps_high),
   eps_high > eps_low, encouraging exploration (prevents entropy stagnation).
2. Dynamic sampling: filter trajectories whose terminal reward == 0 (draws /
   no-signal games) — only train on decisive games so the group baseline has
   signal. (DAPO filters all-zero-advantage groups; here == draw trajectories.)

Otherwise identical to grpo_train (no critic, group-normalized MC return, pure
policy self-play). Runs on 2 GPUs alongside the main GRPO run.

Run:
  PYTHONPATH=. uv run python src/dapo_train.py \
      --model_path models/sft_large/alpha_chess_best.pth --output_dir models/dapo \
      --iterations 50 --games_per_iter 32 --selfplay_batch 16 \
      --eps_low 0.2 --eps_high 0.28 --actor_gpus 2 --learner_gpu 0 --wandb
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
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from src.model import AlphaChess, BoardEncoder
from src.grpo import GRPOTrajectory, GRPOGroup, GRPOAdvantageManager
from src.mcts import action_to_move
from eval.opponents import AlphaZeroOpponent, StockfishOpponent
from eval.competition import play_games

# Reuse self-play + actor + eval from grpo_train (identical mechanics).
from src.grpo_train import self_play_episodes_grpo, actor_worker, _atomic_save, evaluate_grpo


def dapo_update(model, optimizer, group, device, eps_low=0.2, eps_high=0.28,
                entropy_coef=0.001, grad_clip=0.5, epochs=4, mini_batch=8):
    """DAPO clipped-surrogate update: clip-HIGHER (asymmetric). No value loss."""
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
            # Clip-HIGHER: upper bound eps_high > eps_low -> more room to increase prob.
            s2 = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv
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


def train_dapo(args):
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
    logging.info(f"[DAPO] Loaded start: {args.model_path}")

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
    logging.info(f"[DAPO] {len(actor_gpus)} actors on {actor_gpus}; learner GPU {args.learner_gpu}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iterations, eta_min=1e-5)
    adv_manager = GRPOAdvantageManager(gamma=args.gamma)

    if args.wandb:
        wandb.init(project="AlphaChess_DAPO", config=vars(args))

    best_score = -1.0
    for it in range(1, args.iterations + 1):
        t0 = time.time()
        trajs = []
        target = args.games_per_iter * 2
        while len(trajs) < target:
            try:
                t = sample_queue.get(timeout=60)
                trajs.extend(t)
            except queue.Empty:
                if not any(p.is_alive() for p in procs):
                    break
                continue
        # Dynamic sampling: drop draw / no-signal trajectories (terminal reward == 0).
        kept = [t for t in trajs if t.steps and t.steps[-1]["reward"] != 0.0]
        n_draw = len(trajs) - len(kept)
        pl = ent = 0.0
        if kept:
            group = GRPOGroup(kept)
            adv_manager.compute_advantages(group)
            pl, ent = dapo_update(model, optimizer, group, device,
                                   eps_low=args.eps_low, eps_high=args.eps_high,
                                   entropy_coef=args.entropy_coef,
                                   epochs=args.grpo_epochs, mini_batch=args.mini_batch)
            scheduler.step()
        _atomic_save(model, latest_path)
        if it % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, f"dapo_iter_{it}.pth"))
        score = -1.0
        if it % args.eval_every == 0:
            score = evaluate_grpo(model, args, device)  # reuse GRPO eval (MCTS vs Stockfish)
            logging.info(f"[DAPO] eval mean score (skill0,2,4) = {score:.3f}")
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, "dapo_best.pth"))
                logging.info(f"[DAPO] ✅ new best={best_score:.3f} -> dapo_best.pth")
        lr = scheduler.get_last_lr()[0]
        logging.info(f"[DAPO] iter {it}/{args.iterations} trajs={len(trajs)} "
                     f"kept={len(kept)} (draw_drop={n_draw}) pl={pl:.4f} ent={ent:.4f} "
                     f"lr={lr:.2e} score={score:.3f} time={time.time()-t0:.0f}s")
        if args.wandb:
            wandb.log({"policy_loss": pl, "entropy": ent, "lr": lr,
                       "stockfish_score": score, "draw_dropped": n_draw})

    for p in procs:
        p.terminate()
        p.join(timeout=5)
    logging.info(f"[DAPO] Training done. best score={best_score:.3f}.")
    if args.wandb:
        wandb.finish()


def main():
    p = argparse.ArgumentParser(description="DAPO (clip-higher + dynamic sampling) chess RL")
    p.add_argument("--model_path", default="models/sft_large/alpha_chess_best.pth")
    p.add_argument("--output_dir", default="models/dapo")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--games_per_iter", type=int, default=32)
    p.add_argument("--selfplay_batch", type=int, default=16)
    p.add_argument("--grpo_epochs", type=int, default=4)
    p.add_argument("--mini_batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps_low", type=float, default=0.2, help="lower clip (clip-higher: eps_high>eps_low)")
    p.add_argument("--eps_high", type=float, default=0.28, help="upper clip (looser, encourages exploration)")
    p.add_argument("--entropy_coef", type=float, default=0.001)
    p.add_argument("--opening_plies", type=int, default=8)
    p.add_argument("--max_plies", type=int, default=240)
    p.add_argument("--draw_penalty", type=float, default=0.0)
    p.add_argument("--mcts_sims", type=int, default=100, help="MCTS sims for eval only")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--eval_games", type=int, default=8)
    p.add_argument("--eval_batch", type=int, default=8)
    p.add_argument("--stockfish_path", default="tools/stockfish_bin")
    p.add_argument("--actor_gpus", default="2", help="actor GPUs (comma-separated)")
    p.add_argument("--learner_gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()
    train_dapo(args)


if __name__ == "__main__":
    main()
