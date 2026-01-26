import chess
import torch
import numpy as np
from torch.utils.data import IterableDataset

from model import BoardEncoder


class ChessDataset(IterableDataset):
    # TODO: Adapt the Dataset to read from PGN and yield training samples
    def __init__(self, pgn_file, max_games=None):
        self.pgn_file = pgn_file
        self.max_games = max_games
        self.encoder = BoardEncoder()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
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

                    yield X, move_idx, torch.tensor([y_val], dtype=torch.float32)
                    
                    board.push(move)
                
                count += 1
                if self.max_games and count >= self.max_games:
                    break