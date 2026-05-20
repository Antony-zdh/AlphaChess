import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import chess
import torch

from eval.elo_system import EloConfig, EloGame, EloSystem
from eval.opponents import BaseOpponent, create_opponent


@dataclass
class MatchResult:
    white: str
    black: str
    result: str
    ply_count: int


def play_games(
    white_model: BaseOpponent,
    black_model: BaseOpponent,
    white_name: str,
    black_name: str,
    games: int,
    batch_size: int,
) -> List[MatchResult]:
    results: List[MatchResult] = []
    boards = [chess.Board() for _ in range(games)]
    ply_counts = [0 for _ in range(games)]

    active_indices = [i for i in range(games)]
    while active_indices:
        opponent_map = {
            chess.WHITE: white_model,
            chess.BLACK: black_model,
        }
        opponent_to_indices: Dict[BaseOpponent, List[int]] = {}
        for idx in active_indices:
            opponent = opponent_map[boards[idx].turn]
            opponent_to_indices.setdefault(opponent, []).append(idx)

        for opponent, group_indices in opponent_to_indices.items():
            if opponent.supports_batch:
                for start in range(0, len(group_indices), batch_size):
                    batch_ids = group_indices[start:start + batch_size]
                    if not batch_ids:
                        continue
                    batch_boards = [boards[i] for i in batch_ids]
                    moves = opponent.select_moves_batch(batch_boards)
                    for local_idx, global_idx in enumerate(batch_ids):
                        boards[global_idx].push(moves[local_idx])
                        ply_counts[global_idx] += 1
            else:
                for idx in group_indices:
                    move = opponent.select_move(boards[idx])
                    boards[idx].push(move)
                    ply_counts[idx] += 1

        active_indices = [i for i in active_indices if not boards[i].is_game_over()]

    for idx, board in enumerate(boards):
        results.append(MatchResult(
            white=white_name,
            black=black_name,
            result=board.result(),
            ply_count=ply_counts[idx],
        ))

    return results


def round_robin(models: Dict[str, BaseOpponent], games_per_pair: int, batch_size: int) -> List[MatchResult]:
    names = sorted(models.keys())
    results: List[MatchResult] = []

    for i, white_name in enumerate(names):
        for black_name in names[i + 1:]:
            white_model = models[white_name]
            black_model = models[black_name]
            results.extend(play_games(white_model, black_model, white_name, black_name, games_per_pair, batch_size))
            results.extend(play_games(black_model, white_model, black_name, white_name, games_per_pair, batch_size))

    return results


def write_games(path: str, games: List[MatchResult]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for game in games:
            f.write(json.dumps(game.__dict__) + "\n")


def load_games(path: str) -> List[EloGame]:
    if not os.path.exists(path):
        return []
    games = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            games.append(EloGame(white=payload["white"], black=payload["black"], result=payload["result"]))
    return games


def main() -> None:
    parser = argparse.ArgumentParser(description="Round robin evaluation with Elo ratings.")
    parser.add_argument("--model_glob", default="models/*.pth", help="Glob for models to include.")
    parser.add_argument("--models_list", default=None, help="Comma-separated list of model paths to include.")
    parser.add_argument("--games_per_pair", type=int, default=16, help="Games per pair (per color).")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--games_log", default="eval/elo_games.jsonl", help="Path to append game results.")
    parser.add_argument("--ratings_out", default="eval/elo_ratings.json", help="Path to write Elo ratings.")
    parser.add_argument("--k_factor", type=float, default=24.0, help="Elo K-factor.")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.models_list:
        model_specs = [path.strip() for path in args.models_list.split(",") if path.strip()]
    else:
        model_specs = sorted(glob.glob(args.model_glob))
    if not model_specs:
        raise FileNotFoundError(f"No models matched {args.model_glob}")

    models: Dict[str, BaseOpponent] = {}
    for spec in model_specs:
        opponent_spec = create_opponent(spec, device)
        if opponent_spec.name in models:
            raise ValueError(f"Duplicate model name: {opponent_spec.name}")
        models[opponent_spec.name] = opponent_spec.opponent

    match_results = round_robin(models, args.games_per_pair, args.batch_size)
    write_games(args.games_log, match_results)

    elo_system = EloSystem(EloConfig(k_factor=args.k_factor))
    elo_system.update_from_games(load_games(args.games_log))

    os.makedirs(os.path.dirname(args.ratings_out), exist_ok=True)
    with open(args.ratings_out, "w", encoding="utf-8") as f:
        f.write(elo_system.state.to_json())

    print("Elo ratings saved to", args.ratings_out)


if __name__ == "__main__":
    main()
