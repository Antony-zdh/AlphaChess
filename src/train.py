import chess
import chess.pgn
import os
import sys
import torch
import numpy as np
import argparse
import logging
import wandb
import random
from torch.utils.data import IterableDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import BoardEncoder, AlphaChess


class ChessDataset(IterableDataset):
    # TODO: Adapt the Dataset to read from PGN and yield training samples
    def __init__(self, pgn_file, max_games=None, buffer_size=10000):
        self.pgn_file = pgn_file
        self.max_games = max_games
        self.encoder = BoardEncoder()
        self.buffer_size = buffer_size

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        # Buffer to shuffle incoming moves
        buffer = []
        
        # If multiple workers, we need to split the file (Hard with PGN)
        # So usually we fallback to num_workers=0 or 1 for PGN reading.
        # We will stream the file linearly.
        
        with open(self.pgn_file, 'r', encoding='utf-8') as pgn_handle:
            count = 0
            while True:
                offset = pgn_handle.tell()
                try:
                    game = chess.pgn.read_game(pgn_handle)
                except ValueError:
                    continue # Skip bad headers
                
                if game is None:
                    break # End of file
                
                # Check outcome for Value Head training
                result_map = {'1-0': 1.0, '0-1': -1.0, '1/2-1/2': 0.0}
                game_result = result_map.get(game.headers.get("Result", "*"), 0.0)

                board = game.board()
                
                for move in game.mainline_moves():
                    try:
                        # 1. State Tensor
                        # Note: encode() creates the 19x8x8 input
                        X = self.encoder.encode(board)
                        
                        # 2. Policy Target (Move Index)
                        # Flattened index: from_sq * 64 + to_sq
                        # This covers all 4096 possible from-to combinations
                        move_idx = move.from_square * 64 + move.to_square
                        
                        # 3. Value Target (Relative to current player)
                        # If White won (1.0) and it's Black's turn, value is -1.0
                        current_turn_val = 1.0 if board.turn == chess.WHITE else -1.0
                        y_val = game_result * current_turn_val

                        buffer.append((X, move_idx, torch.tensor([y_val], dtype=torch.float32)))

                        # if buffer is full, yield a random element
                        if len(buffer) >= self.buffer_size:
                            idx = random.randint(0, len(buffer)-1)
                            yield buffer[idx]
                            # Replace the yielded element with new one
                            buffer[idx] = buffer[-1]
                            buffer.pop()
                    except Exception as e:
                        logging.info(f"Exception happened during data loading: {e}")
                    
                    board.push(move)
                
                count += 1
                if self.max_games and count >= self.max_games:
                    break
        
        # Yield remaining elements in buffer
        random.shuffle(buffer)
        for item in buffer:
            yield item


def pilot_test():
    """ A quick test to ensure data flows correctly """
    pgn_path = os.path.join("data", "raw", "GM_games.pgn")

    if not os.path.exists(pgn_path):
        print(f"❌ Error: {pgn_path} not found.")
        return
    
    # Initialize the dataset
    dataset = ChessDataset(pgn_path, max_games=4)

    # Initialize dataloader
    dataloader = DataLoader(dataset, batch_size=4)

    try: 
        # Fetch the first batch
        first_batch = next(iter(dataloader))
        X, y_policy, y_value = first_batch

        print("✅ DataLoader is working!")
        print(f"Input Tensor Shape: {X.shape}")          # Expecting (batch_size, 19, 8, 8)
        print(f"Policy Target Shape: {y_policy.shape}")  # Expecting (batch_size,)
        print(f"Value Target Shape: {y_value.shape}")    # Expecting (batch_size, 1)

        assert X.shape == (4, 19, 8, 8), "Input tensor shape mismatch"
        assert y_policy.shape == (4,), "Policy target shape mismatch"
        assert y_value.shape == (4, 1), "Value target shape mismatch"
        assert y_policy.max() < 4096 and y_policy.min() >= 0, "Policy target values out of range"

        print("\nPrinting Policy Targets (Move Indices):", y_policy.tolist())
        move_from = y_policy[0] // 64
        move_to = y_policy[0] % 64
        # Print in UCI format
        uci_move = chess.Move(move_from.item(), move_to.item()).uci()
        print(f"Example UCI Move from first policy target: {uci_move}")

        move_from = y_policy[1] // 64
        move_to = y_policy[1] % 64
        # Print in UCI format
        uci_move = chess.Move(move_from.item(), move_to.item()).uci()
        print(f"Example UCI Move from second policy target: {uci_move}")


    except Exception as e:
        print("❌ DataLoader test failed:", str(e))
        import traceback
        traceback.print_exc()


def train(args):
    """ Main training loop """
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Paths
    data_path = args.data_path
    model_save_dir = args.output_dir
    os.makedirs(model_save_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"❌ Error: PGN file not found at {data_path}. Please ensure the dataset is available.")
        return

    # Initialize Components
    model = AlphaChess().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    dataset = ChessDataset(data_path, max_games=args.max_games, buffer_size=10000)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Loss Functions
    ce_loss = torch.nn.CrossEntropyLoss() # For Policy Head
    mse_loss = torch.nn.MSELoss() # For Value Head

    # Use Wandb for logging (include hyperparameters)
    if args.wandb:
        wandb.init(project="AlphaChess_Training", config=vars(args))
        wandb.watch(model, log="gradients", log_freq=args.log_interval)


    print("Starting Supervised Training Loop...")
    model.train() # Set model to training mode

    # Track global steps
    global_step = 0

    for epoch in range(args.epochs):
        total_batches = 0
        interval_loss = 0.0
        interval_ce = 0.0
        interval_mse = 0.0
        interval_acc = 0.0
        interval_batches = 0

        for batch_idx, (X, y_policy, y_value) in enumerate(dataloader):
            # Move to GPU/CPU
            X = X.to(device)
            y_policy = y_policy.to(device)
            y_value = y_value.to(device)

            # Forward Pass
            pred_policy, pred_value = model(X)

            # Compute Losses
            policy_ce_loss = ce_loss(pred_policy, y_policy)
            value_mse_loss = mse_loss(pred_value, y_value)
            loss = policy_ce_loss + value_mse_loss * 0.5 # Weighted sum

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            total_batches += 1
            policy_acc = (pred_policy.argmax(dim=1) == y_policy).sum().item()
            policy_acc = policy_acc / y_policy.size(0)

            interval_loss += loss.item()
            interval_ce += policy_ce_loss.item()
            interval_mse += value_mse_loss.item()
            interval_acc += policy_acc
            interval_batches += 1

            # Update global step
            global_step += 1
            
            # Get current LR
            current_lr = optimizer.param_groups[0]['lr']

            # Logging (per log_interval (batches))
            if total_batches % args.log_interval == 0:
                logging.info(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}] "
                             f"Loss: {interval_loss/interval_batches:.4f} "
                             f"CE Loss: {interval_ce/interval_batches:.4f} "
                             f"MSE Loss: {interval_mse/interval_batches:.4f} "
                             f"Policy Acc: {interval_acc/interval_batches:.2%}"
                             f" LR: {current_lr:.2e}")
                if args.wandb:
                    wandb.log({
                        "epoch": epoch + 1,
                        "loss": interval_loss/interval_batches,
                        "ce_loss": interval_ce/interval_batches,
                        "mse_loss": interval_mse/interval_batches,
                        "policy_accuracy": interval_acc/interval_batches,
                        "lr": current_lr
                    })

                # Reset interval metrics
                interval_loss = 0.0
                interval_ce = 0.0
                interval_mse = 0.0
                interval_acc = 0.0
                interval_batches = 0
        
        # Step the scheduler
        scheduler.step()

        # Save checkpoint (per save_interval (epochs))
        if (epoch + 1) % args.save_interval == 0:
            logging.info(f"Saving model checkpoint for epoch {epoch+1}...")
            checkpoint_path = os.path.join(model_save_dir, f"alpha_chess_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved at {checkpoint_path}")


def main():
    """ Arguments Parsing """
    parser = argparse.ArgumentParser(description="Train AlphaChess Model")
    # Properties of run
    parser.add_argument('--pilot', action='store_true', help="Run pilot test for data loading")
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=512, help="Training batch size")
    parser.add_argument('--max_games', type=int, default=None, help="Max games to use from PGN for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for optimizer")
    # Paths
    parser.add_argument('--data_path', type=str, default=os.path.join("data", "raw", "GM_games.pgn"), help="Path to PGN file for training data")
    parser.add_argument('--ckpt_dir', type=str, default=os.path.join("models"), help="Directory to continue training model checkpoints")
    parser.add_argument('--output_dir', type=str, default=os.path.join("models"), help="Path to save the trained model")
    # Logging and checkpointing
    parser.add_argument('--log_interval', type=int, default=5, help="Batches between logging training status")
    parser.add_argument('--save_interval', type=int, default=1, help="Epochs between saving model checkpoints")
    parser.add_argument('--wandb', action='store_true', help="Use Weights & Biases for experiment tracking")

    args = parser.parse_args()

    if args.pilot:
        pilot_test()
    else:
        train(args)
    

if __name__ == "__main__":
    main()