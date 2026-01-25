import sys
import os

# Ensure the PATH includes the src directory
sys.path.append(os.path.dirname(__file__))
# Get the parent directory (project root, f:\alpha_chess)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src import ChessGame

def main():
    print("Welcome to Alpha Chess!")
    game = ChessGame(player0_role='human', player1_role='AI', use_gui=True)
    game.play()


if __name__ == "__main__":
    main()