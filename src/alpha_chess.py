import chess
import torch
import numpy as np
import os
import sys

from .model import AlphaChess, BoardEncoder


class ChessGame:
    """ 
    Class representing a chess game. 
    Players can be either human or AI.
    Parameters:
    - player0_role (white): Role of player 0 ('human' or 'AI').
    - player1_role (black): Role of player 1 ('human' or 'AI').
    """
    def __init__(self, player0_role='human', player1_role='AI', model_path=None, use_gui=True):
        self.board = chess.Board()
        if model_path is None:
            print("Please provide a valid model path for AI player.")
            raise ValueError("Model path is required for AI player.")
        self.model_path = model_path

        self.players = [
            self._create_player(player0_role, chess.WHITE, model_path),
            self._create_player(player1_role, chess.BLACK, model_path)
        ]

        # Graphical User Interface (GUI)
        self.use_gui = use_gui


    def _create_player(self, role, color, model_path=None):
        if role == 'human':
            return HumanPlayer(color)
        elif role == 'AI':
            return AIPlayer(color, model_path)
        else:
            raise ValueError("Invalid player role. Choose 'human' or 'AI'.")
    

    def play(self):
        """ 
        Main game loop. 
        If GUI is not present, it falls back to CLI mode.
        """
        print("Starting Alpha Chess!")
        # Print initial board
        if self.use_gui:
            from .ui import ChessGUI
            gui = ChessGUI(self.board)
            gui.draw_board()
        else:
            print(self.board)
        # Main Game Loop
        while not self.board.is_game_over():
            current_player = self.players[0] if self.board.turn == chess.WHITE else self.players[1]
            move = current_player.make_move(self.board, gui if self.use_gui else None)
            
            # Apply the move 
            if move in self.board.legal_moves:
                self.board.push(move)

            # Update View
            if gui:
                gui.draw_board()
            else:
                print(f"{'White' if self.board.turn == chess.WHITE else 'Black'} plays {move.uci()}")
                print(self.board)
        result = self.board.result()
        print("Game over:", result)
        winner = 'White wins' if self.board.outcome().winner == True else 'Black wins' if self.board.outcome().winner == False else 'Draw'
        print(f"{winner}!")
        if gui:
            gui.wait_for_quit()
        return result


class Player:
    """ Abstract class for a chess player."""
    def __init__(self, color):
        self.color = color  # chess.WHITE or chess.BLACK


    def make_move(self, board, gui=None):
        raise NotImplementedError("This method should be implemented by subclasses.")


class HumanPlayer(Player):
    """ Class representing a human chess player. """
    def __init__(self, color):
        super().__init__(color)
        self.role = 'human'


    def make_move(self, board, gui=None):
        if gui:
            # Wait for GUI input
            return gui.get_human_move(self.color)
        else:
            # CLI input
            return self._get_move_from_cli(board)

    
    def _get_move_from_cli(self, board):
        while True:
            try:
                move = input(f"Enter your move ({self.color}): ")
                chess_move = chess.Move.from_uci(move)
                if chess_move in board.legal_moves:
                    return chess_move
                else:
                    print("Illegal move. Try again.")
            except ValueError:
                print("Invalid move format. Use UCI format (e.g., e2e4). Try again.")


class AIPlayer(Player):
    """ Class representing an AI chess player. """
    def __init__(self, color, model_path):
        super().__init__(color)
        self.role = 'AI'
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained model
        self.model = AlphaChess().to(self.device)
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            except Exception as e:
                print(f"Error loading model from {self.model_path}: {e}")
                self.model = None
                raise RuntimeError(f"Failed to load model from {self.model_path}")
        else:
            print(f"Model file not found at {self.model_path}")
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        # Set model to evaluation mode
        self.model.eval()

        # Board encoder
        self.encoder = BoardEncoder()


    def make_move(self, board, gui=None):
        # Pump out a count down in GUI mode to indicate AI is thinking
        # if gui:
        #     gui.display_countdown(self.role)

        # Fallback to random if model is None
        if self.model is None:
            import random
            legal_moves = list(board.legal_moves)
            return random.choice(legal_moves)

        # 1. Encode Board
        X = self.encoder.encode(board).unsqueeze(0).to(self.device)  # Add batch dimension (1, 19, 8, 8)

        # 2. Forward Pass
        with torch.no_grad():
            policy_logits, value = self.model(X)  # (1, 4096)
            # Remove batch dimension
            policy_logits = policy_logits.squeeze(0)  # (4096,)
            value = value.item()  # Scalar

        # 3. Action Masking (Filter illegal moves)
        legal_moves = list(board.legal_moves)

        def move_index(move):
            return move.from_square * 64 + move.to_square
        
        legal_indices = [move_index(move) for move in legal_moves]
        mask = torch.zeros(64 * 64, device=self.device)  # 4096 possible moves
        mask[legal_indices] = 1.0
        policy_logits = policy_logits.masked_fill(mask==0, float('-inf'))

        # 4. Softmax to get probabilities and select best move
        policy_probs = torch.softmax(policy_logits, dim=0)
        best_move_index = torch.argmax(policy_probs).item()
        best_move_from = best_move_index // 64
        best_move_to = best_move_index % 64

        # 5. Make the move
        # Ensure the move is legal and in UCI format
        best_move = chess.Move(best_move_from, best_move_to)
        if best_move in legal_moves:
            return best_move
        else:
            raise RuntimeError("AI selected an illegal move.")