"""Human vs AI (MCTS) chess game in the terminal. You play one color, the
AlphaChess model + MCTS plays the other. Uses the best SFT checkpoint by default.

Run (on GPU 3, interactive — use your own terminal, not Claude's Bash):
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. uv run python src/play_vs_ai.py
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. uv run python src/play_vs_ai.py --human_color black --mcts_sims 400
"""
import argparse
import chess
import torch

from eval.opponents import AlphaZeroOpponent


def main():
    p = argparse.ArgumentParser(description="Human vs AI chess game")
    p.add_argument("--model_path", default="models/sft_large/alpha_chess_best.pth",
                   help="Best model (~1150 ELO)")
    p.add_argument("--mcts_sims", type=int, default=200, help="MCTS sims (higher = stronger AI)")
    p.add_argument("--human_color", default="white", choices=["white", "black"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    az = AlphaZeroOpponent(args.model_path, device,
                           n_simulations=args.mcts_sims, greedy=True)
    board = chess.Board()
    human_white = args.human_color == "white"
    print(f"\nYou play {'White' if human_white else 'Black'}. AI: {args.model_path} (MCTS {args.mcts_sims} sims)\n")

    while not board.is_game_over():
        print(board.unicode()), print()
        is_human_turn = (board.turn == chess.WHITE) == human_white
        if is_human_turn:
            legal = sorted(m.uci() for m in board.legal_moves)
            while True:
                mv = input(f"Your move (uci, e.g. e2e4) {legal}: ").strip().lower()
                if mv in legal:
                    break
                print("  illegal, try again")
            board.push_uci(mv)
        else:
            print("AI thinking...")
            move = az.select_move(board)
            print(f"AI plays: {move.uci()}\n")
            board.push(move)

    print(board.unicode())
    print("\nResult:", board.result())


if __name__ == "__main__":
    main()
