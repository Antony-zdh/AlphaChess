"""Multi-GPU AlphaZero self-play: N actor processes (one GPU each) run MCTS
self-play and push samples to a queue; one learner process consumes, trains,
and writes the latest model back for actors to reload. Near-linear speedup
across GPUs since the MCTS Python bottleneck parallelizes.

Run (from project root):
  PYTHONPATH=. uv run python src/az_train_mp.py \
      --model_path models/sft_v2/alpha_chess_best.pth --output_dir models/az \
      --iterations 50 --games_per_iter 192 --selfplay_batch 32 \
      --mcts_sims 100 --max_plies 160 --train_steps 400 \
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
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from src.model import AlphaChess, BoardEncoder
from src.mcts import BatchMCTS
from src.buffer import OffPolicyBuffer
from src.az_train import self_play_games, train_step, evaluate_az


def actor_worker(rank, gpu_id, model_path, latest_path, sample_queue, args):
    """Self-play actor: load model onto GPU `gpu_id`, run self-play forever,
    pushing each game's samples (list of (state,pi,z)) to the queue. Reloads
    the latest model when its mtime changes."""
    device = torch.device(f"cuda:{gpu_id}")
    model = AlphaChess().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    encoder = BoardEncoder()
    mcts = BatchMCTS(model, encoder, device,
                     n_simulations=args.mcts_sims, c_puct=args.c_puct,
                     temperature=1.0)  # exploration throughout self-play
    last_mtime = os.path.getmtime(model_path)
    while True:
        # Reload latest model if the learner updated it.
        try:
            if os.path.exists(latest_path):
                m = os.path.getmtime(latest_path)
                if m > last_mtime:
                    model.load_state_dict(torch.load(latest_path, map_location=device))
                    last_mtime = m
        except Exception:
            pass
        try:
            samples, _ = self_play_games(
                model, mcts, args.selfplay_batch,
                opening_plies=args.opening_plies, max_plies=args.max_plies,
                seed=random.randint(0, 2**31 - 1))
            # Convert to numpy so mp.Queue pickles via pipe (NOT torch shared
            # memory, which exhausts /dev/shm — only ~1.4G free here).
            np_samples = [(s[0].cpu().numpy(), s[1].cpu().numpy(), s[2].cpu().numpy())
                          for s in samples]
            sample_queue.put(np_samples, timeout=600)
        except Exception as e:
            logging.info(f"actor {rank} (gpu {gpu_id}) error: {e}")
            time.sleep(1)


def _atomic_save(model, path):
    tmp = path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


def train_az_mp(args):
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
    _atomic_save(model, latest_path)  # initial latest for actors to load
    logging.info(f"Loaded start checkpoint: {args.model_path}")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    actor_gpus = [int(g) for g in args.actor_gpus.split(",") if g.strip()]
    n_actors = len(actor_gpus)
    sample_queue = mp.Queue(maxsize=500)
    procs = []
    for rank, g in enumerate(actor_gpus):
        p = mp.Process(target=actor_worker,
                       args=(rank, g, args.model_path, latest_path, sample_queue, args))
        p.start()
        procs.append(p)
    logging.info(f"Started {n_actors} actors on GPUs {actor_gpus}; learner on GPU {args.learner_gpu}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iterations, eta_min=1e-5)
    buffer = OffPolicyBuffer(capacity=args.buffer_capacity)

    if args.wandb:
        wandb.init(project="AlphaChess_AZ", config=vars(args))

    best_score = -1.0
    for it in range(1, args.iterations + 1):
        t0 = time.time()
        # 1. Drain queue: collect at least games_per_iter games worth of samples.
        games = 0
        while games < args.games_per_iter:
            try:
                samples = sample_queue.get(timeout=60)
                for s in samples:
                    buffer.add(s)
                games += args.selfplay_batch
            except queue.Empty:
                # Keep waiting as long as actors are alive (a batch can take
                # several minutes to generate; a short timeout must not abort).
                if not any(p.is_alive() for p in procs):
                    logging.info("All actors dead, stopping collection.")
                    break
                continue
        # 2. Train.
        pl = vl = 0.0
        if len(buffer.buffer) >= args.train_batch:
            for _ in range(args.train_steps):
                pl, vl = train_step(model, optimizer, buffer, args.train_batch, device)
            scheduler.step()
        # 3. Save latest so actors reload the improved model.
        _atomic_save(model, latest_path)
        # 4. Periodic checkpoint.
        if it % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, f"az_iter_{it}.pth"))
        # 5. Eval vs Stockfish.
        score = -1.0
        if it % args.eval_every == 0:
            score = evaluate_az(model, args, device)
            logging.info(f"  eval mean score (skill0,2,4) = {score:.3f}")
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, "az_best.pth"))
                logging.info(f"  ✅ new best score={best_score:.3f} -> az_best.pth")
        lr = scheduler.get_last_lr()[0]
        logging.info(f"iter {it}/{args.iterations} buffer={len(buffer.buffer)} "
                     f"pl={pl:.4f} vl={vl:.4f} games={games} lr={lr:.2e} "
                     f"score={score:.3f} time={time.time()-t0:.0f}s")
        if args.wandb:
            wandb.log({"policy_loss": pl, "value_loss": vl,
                       "buffer_size": len(buffer.buffer), "games": games,
                       "lr": lr, "stockfish_score": score})

    for p in procs:
        p.terminate()
        p.join(timeout=5)
    logging.info(f"Training done. best score={best_score:.3f}.")
    if args.wandb:
        wandb.finish()


def main():
    p = argparse.ArgumentParser(description="Multi-GPU AlphaZero self-play")
    p.add_argument("--model_path", default="models/sft_v2/alpha_chess_best.pth")
    p.add_argument("--output_dir", default="models/az")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--games_per_iter", type=int, default=192)
    p.add_argument("--selfplay_batch", type=int, default=32)
    p.add_argument("--train_steps", type=int, default=400)
    p.add_argument("--train_batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--mcts_sims", type=int, default=100)
    p.add_argument("--eval_sims", type=int, default=200, help="MCTS sims for eval (separate from self-play)")
    p.add_argument("--c_puct", type=float, default=4.0)
    p.add_argument("--opening_plies", type=int, default=8)
    p.add_argument("--max_plies", type=int, default=160)
    p.add_argument("--buffer_capacity", type=int, default=100000)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--eval_games", type=int, default=8)
    p.add_argument("--eval_batch", type=int, default=8)
    p.add_argument("--stockfish_path", default="tools/stockfish_bin")
    p.add_argument("--actor_gpus", default="2,3,4,5,6,7", help="Comma-separated actor GPUs.")
    p.add_argument("--learner_gpu", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()
    train_az_mp(args)


if __name__ == "__main__":
    main()
