"""SFT training v2: validation split + board-mirror augmentation + weight decay
+ early stopping + best-checkpoint selection.

Run:
  python src/train_v2.py --epochs 35 --batch_size 256 --lr 1e-3 \
      --weight_decay 1e-4 --augment --val_ratio 0.1 --early_stop_patience 8 \
      --data_path data/GM_games.pgn --output_dir models/sft_v2 --wandb
"""
import chess
import chess.pgn
import os
import copy
import torch
import argparse
import logging
import wandb
import random
from torch.utils.data import IterableDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import BoardEncoder, AlphaChess


class ChessDatasetV2(IterableDataset):
    """Streaming PGN dataset with train/val split and optional mirror augmentation.

    Split: games are partitioned by their 0-based index. A game belongs to the
    validation set iff (index % val_modulus == 0). This is deterministic, so the
    train and val passes over the same file see disjoint, stable partitions.

    Augmentation: chess is left-right symmetric (file a<->h), so mirroring a
    position yields an equivalent position. We mirror ~50% of training samples;
    mirror transforms a square as `sq ^ 7` (flips the low 3 file bits), so the
    move index `from*64+to` becomes `(from^7)*64 + (to^7)`. Val samples are
    never augmented (clean measurement).
    """

    def __init__(self, pgn_file, split="train", val_modulus=10,
                 max_games=None, buffer_size=10000, augment=False, seed=0):
        self.pgn_file = pgn_file
        self.split = split  # "train" or "val"
        self.val_modulus = val_modulus
        self.max_games = max_games
        self.encoder = BoardEncoder()
        self.buffer_size = buffer_size
        self.augment = augment and (split == "train")
        self.rng = random.Random(seed)

    def _is_val(self, game_index):
        return game_index % self.val_modulus == 0

    def __iter__(self):
        buffer = []
        with open(self.pgn_file, "r", encoding="utf-8") as pgn_handle:
            count = 0
            while True:
                try:
                    game = chess.pgn.read_game(pgn_handle)
                except ValueError:
                    continue
                if game is None:
                    break

                is_val = self._is_val(count)
                keep = (is_val and self.split == "val") or \
                       (not is_val and self.split == "train")
                count += 1
                if not keep:
                    continue
                if self.max_games and count > self.max_games:
                    break

                result_map = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}
                game_result = result_map.get(game.headers.get("Result", "*"), 0.0)
                board = game.board()

                for move in game.mainline_moves():
                    try:
                        # Decide augmentation per position.
                        do_mirror = self.augment and (self.rng.random() < 0.5)

                        if do_mirror:
                            enc_board = board.mirror()
                            move_idx = (move.from_square ^ 7) * 64 + (move.to_square ^ 7)
                        else:
                            enc_board = board
                            move_idx = move.from_square * 64 + move.to_square

                        X = self.encoder.encode(enc_board)
                        # Value target is from the side-to-move's perspective;
                        # mirroring does not change whose turn it is.
                        current_turn_val = 1.0 if board.turn == chess.WHITE else -1.0
                        y_val = game_result * current_turn_val

                        buffer.append((X, move_idx,
                                       torch.tensor([y_val], dtype=torch.float32)))
                        if len(buffer) >= self.buffer_size:
                            idx = self.rng.randint(0, len(buffer) - 1)
                            yield buffer[idx]
                            buffer[idx] = buffer[-1]
                            buffer.pop()
                    except Exception as e:
                        logging.info(f"Exception during data loading: {e}")
                    board.push(move)

        # Yield remaining buffer (val pass uses a smaller buffer but still drains).
        self.rng.shuffle(buffer)
        for item in buffer:
            yield item


@torch.no_grad()
def evaluate(model, val_loader, ce_loss, mse_loss, device, max_batches=200):
    """Compute val loss and policy accuracy. Caps batches to keep it fast.

    Note: val_loader wraps an IterableDataset, which has no __len__, so we must
    not call len(val_loader); we count batches ourselves instead.
    """
    model.eval()
    total_loss, total_ce, total_mse, total_acc = 0.0, 0.0, 0.0, 0.0
    n = 0
    b = 0
    for (X, y_p, y_v) in val_loader:
        if b >= max_batches:
            break
        X = X.to(device)
        y_p = y_p.to(device)
        y_v = y_v.to(device)
        pp, pv = model(X)
        pl = ce_loss(pp, y_p)
        vl = mse_loss(pv, y_v)
        total_loss += (pl + 0.5 * vl).item()
        total_ce += pl.item()
        total_mse += vl.item()
        total_acc += (pp.argmax(dim=1) == y_p).sum().item()
        n += y_p.size(0)
        b += 1
    model.train()
    if n == 0 or b == 0:
        return 0.0, 0.0, 0.0, 0.0
    return total_loss / b, total_ce / b, total_mse / b, total_acc / n


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    data_path = args.data_path
    model_save_dir = args.output_dir
    os.makedirs(model_save_dir, exist_ok=True)
    if not os.path.exists(data_path):
        print(f"❌ Error: PGN file not found at {data_path}.")
        return

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model = AlphaChess().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    val_mod = max(2, round(1.0 / args.val_ratio))  # 0.1 -> 10
    train_ds = ChessDatasetV2(data_path, split="train", val_modulus=val_mod,
                              max_games=args.max_games, augment=args.augment, seed=args.seed)
    val_ds = ChessDatasetV2(data_path, split="val", val_modulus=val_mod,
                            max_games=args.max_games, augment=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    run_name = "sft_v2_aug_wd_val"
    if args.wandb:
        wandb.init(project="AlphaChess_Training", name=run_name, config=vars(args))
        wandb.watch(model, log="gradients", log_freq=args.log_interval)

    print("Starting SFT v2 training loop (train/val split + aug + weight decay + early stop)...")
    model.train()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_left = args.early_stop_patience

    for epoch in range(args.epochs):
        interval_loss = interval_ce = interval_mse = interval_acc = 0.0
        interval_batches = 0
        total_batches = 0

        for batch_idx, (X, y_policy, y_value) in enumerate(train_loader):
            X = X.to(device)
            y_policy = y_policy.to(device)
            y_value = y_value.to(device)

            pred_policy, pred_value = model(X)
            policy_ce = ce_loss(pred_policy, y_policy)
            value_mse = mse_loss(pred_value, y_value)
            loss = policy_ce + value_mse * 0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_batches += 1
            policy_acc = (pred_policy.argmax(dim=1) == y_policy).float().mean().item()
            interval_loss += loss.item()
            interval_ce += policy_ce.item()
            interval_mse += value_mse.item()
            interval_acc += policy_acc
            interval_batches += 1

            if total_batches % args.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                logging.info(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}] "
                             f"Loss: {interval_loss/interval_batches:.4f} "
                             f"CE: {interval_ce/interval_batches:.4f} "
                             f"MSE: {interval_mse/interval_batches:.4f} "
                             f"Acc: {interval_acc/interval_batches:.2%} "
                             f"LR: {lr:.2e}")
                if args.wandb:
                    wandb.log({
                        "train/loss": interval_loss / interval_batches,
                        "train/ce_loss": interval_ce / interval_batches,
                        "train/mse_loss": interval_mse / interval_batches,
                        "train/policy_accuracy": interval_acc / interval_batches,
                        "lr": lr,
                    })
                interval_loss = interval_ce = interval_mse = interval_acc = 0.0
                interval_batches = 0

        scheduler.step()

        # Validation.
        v_loss, v_ce, v_mse, v_acc = evaluate(model, val_loader, ce_loss, mse_loss, device)
        lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Epoch [{epoch+1}/{args.epochs}] VAL Loss: {v_loss:.4f} "
                     f"CE: {v_ce:.4f} MSE: {v_mse:.4f} Acc: {v_acc:.2%} LR: {lr:.2e}")
        if args.wandb:
            wandb.log({"val/loss": v_loss, "val/ce_loss": v_ce,
                        "val/mse_loss": v_mse, "val/policy_accuracy": v_acc, "epoch": epoch + 1})

        # Best checkpoint by val loss (lower is better).
        improved = v_loss < best_val_loss - 1e-4
        if improved:
            best_val_loss = v_loss
            best_val_acc = v_acc
            patience_left = args.early_stop_patience
            best_path = os.path.join(model_save_dir, "alpha_chess_best.pth")
            torch.save(model.state_dict(), best_path)
            logging.info(f"  ✅ new best val_loss={v_loss:.4f} acc={v_acc:.2%} -> {best_path}")
        else:
            patience_left -= 1
            logging.info(f"  no improvement ({patience_left} patience left). best so far: "
                         f"val_loss={best_val_loss:.4f} acc={best_val_acc:.2%}")

        # Also save per-epoch checkpoint.
        if (epoch + 1) % args.save_interval == 0:
            ep_path = os.path.join(model_save_dir, f"alpha_chess_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ep_path)

        if patience_left <= 0:
            logging.info(f"Early stopping at epoch {epoch+1}. Best val_loss={best_val_loss:.4f} "
                         f"acc={best_val_acc:.2%}.")
            break

    logging.info(f"Training done. Best val_loss={best_val_loss:.4f} acc={best_val_acc:.2%}.")
    if args.wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train AlphaChess SFT v2")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_games", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true", help="Enable board mirror augmentation")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of games for validation")
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="data/GM_games.pgn")
    parser.add_argument("--output_dir", type=str, default="models/sft_v2")
    parser.add_argument("--log_interval", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
