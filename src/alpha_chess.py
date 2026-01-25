import chess


class ChessGame:
    """ 
    Class representing a chess game. 
    Players can be either human or AI.
    Parameters:
    - player0_role (white): Role of player 0 ('human' or 'AI').
    - player1_role (black): Role of player 1 ('human' or 'AI').
    """
    def __init__(self, player0_role='human', player1_role='AI', use_gui=True):
        self.board = chess.Board()
        self.players = [
            self._create_player(player0_role, chess.WHITE),
            self._create_player(player1_role, chess.BLACK)
        ]

        # Graphical User Interface (GUI)
        self.use_gui = use_gui


    def _create_player(self, role, color):
        if role == 'human':
            return HumanPlayer(color)
        elif role == 'AI':
            return AIPlayer(color)
        else:
            raise ValueError("Invalid player role. Choose 'human' or 'AI'.")

        
    # def step(self, move = None):
    #     # This method accepts a move from the current player
    #     # and updates the game board accordingly.
    #     current_player = self.players[0] if self.board.turn == chess.WHITE else self.players[1]
    #     move = current_player.make_move(self.board, move)
    #     self.board.push(move)
    

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
        print(f"{self.board.outcome().winner} wins!")
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
    def __init__(self, color):
        super().__init__(color)
        self.role = 'AI'


    def make_move(self, board, gui=None):
        # Pump out a count down in GUI mode to indicate AI is thinking
        # if gui:
        #     gui.display_countdown(self.role)

        # Simple AI: choose a random legal move
        # TODO: Implement a better AI algorithm
        import random
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves)