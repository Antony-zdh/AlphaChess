import sys
import os
import argparse

# Ensure the PATH includes the src directory
sys.path.append(os.path.dirname(__file__))
# Get the parent directory (project root, f:\alpha_chess)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src import ChessGame

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Alpha Chess Game")
    parser.add_argument('--player0', choices=['human', 'AI'], default='human', help="Role of player 0 (white)")
    parser.add_argument('--player1', choices=['human', 'AI'], default='AI', help="Role of player 1 (black)")
    parser.add_argument('--model_path', type=str, default="models/alpha_chess.pth", help="Path to the AI model")
    parser.add_argument('--use_gui', action='store_true', help="Use graphical user interface")

    args = parser.parse_args()

    print("Welcome to Alpha Chess!")
    game = ChessGame(
        player0_role=args.player0,
        player1_role=args.player1, 
        model_path=args.model_path, 
        use_gui=args.use_gui
        )
    
    game.play()


if __name__ == "__main__":
    main()