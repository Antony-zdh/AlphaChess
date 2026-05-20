"""
Evaluate AlphaChess checkpoints against Stockfish at multiple skill levels.

Finds the absolute strength bracket of each checkpoint by sweeping Stockfish
skill levels and identifying where the model's score crosses 50%.

Usage:
  python eval/evaluate_vs_stockfish.py \
      --models models/sft_models/alpha_chess_epoch_16.pth \
      --skill_levels 0,2,4,6,8,10 \
      --games 20 \
      --stockfish_path stockfish
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import chess
import torch

from eval.opponents import AlphaChessOpponent, StockfishOpponent
from eval.competition import play_games


def evaluate_model_at_level(
    model_name: str,
    alpha_opponent: AlphaChessOpponent,
    stockfish_path: str,
    skill_level: int,
    games: int,
    batch_size: int,
) -> dict:
    sf_opponent = StockfishOpponent(stockfish_path, skill_level)
    approx_elo = sf_opponent.approx_elo
    half = games // 2

    try:
        # Play half the games as white, half as black
        results_as_white = play_games(
            alpha_opponent, sf_opponent,
            model_name, f"stockfish_skill{skill_level}",
            half, batch_size,
        )
        results_as_black = play_games(
            sf_opponent, alpha_opponent,
            f"stockfish_skill{skill_level}", model_name,
            games - half, batch_size,
        )
    finally:
        sf_opponent.close()

    wins = draws = losses = 0
    for r in results_as_white:
        if r.result == "1-0":
            wins += 1
        elif r.result == "0-1":
            losses += 1
        else:
            draws += 1
    for r in results_as_black:
        if r.result == "0-1":
            wins += 1
        elif r.result == "1-0":
            losses += 1
        else:
            draws += 1

    total = wins + draws + losses
    score = (wins + 0.5 * draws) / total if total else 0.0
    return {
        "model": model_name,
        "skill_level": skill_level,
        "approx_elo": approx_elo,
        "games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total if total else 0.0,
        "draw_rate": draws / total if total else 0.0,
        "loss_rate": losses / total if total else 0.0,
        "score": score,
    }


def estimate_elo_bracket(
    results: List[dict],
) -> Optional[str]:
    """Find the skill-level range where score crosses 50%, return a label."""
    above = [r for r in results if r["score"] >= 0.5]
    below = [r for r in results if r["score"] < 0.5]

    if not above:
        lowest = min(results, key=lambda r: r["skill_level"])
        return f"below skill {lowest['skill_level']} (~{lowest['approx_elo']})"
    if not below:
        highest = max(results, key=lambda r: r["skill_level"])
        return f"above skill {highest['skill_level']} (~{highest['approx_elo']})"

    best_above = max(above, key=lambda r: r["skill_level"])
    worst_below = min(below, key=lambda r: r["skill_level"])
    lo_elo = best_above["approx_elo"]
    hi_elo = worst_below["approx_elo"]
    return f"~{lo_elo}–{hi_elo} (beats skill {best_above['skill_level']}, loses to skill {worst_below['skill_level']})"


def print_summary(all_results: List[dict], skill_levels: List[int]) -> None:
    models = sorted({r["model"] for r in all_results})
    header = f"{'Model':<40}" + "".join(f"  Sk{s:02d}" for s in skill_levels)
    print("\n" + header)
    print("-" * len(header))
    for model in models:
        row = f"{model:<40}"
        model_results = {r["skill_level"]: r for r in all_results if r["model"] == model}
        for s in skill_levels:
            r = model_results.get(s)
            row += f"  {r['score']:.2f} " if r else "   --- "
        print(row)
    print()
    print("Score = (wins + 0.5*draws) / games. Stockfish approx ELO per skill level:")
    elo_map = StockfishOpponent.SKILL_ELO
    print("  " + "  ".join(f"Sk{s}≈{elo_map.get(s,'?')}" for s in skill_levels))
    print()
    print("Estimated ELO brackets:")
    for model in models:
        model_results = [r for r in all_results if r["model"] == model]
        bracket = estimate_elo_bracket(model_results)
        print(f"  {model}: {bracket}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AlphaChess vs Stockfish skill levels.")
    parser.add_argument("--models", default=None, help="Comma-separated model paths.")
    parser.add_argument("--model_glob", default=None, help="Glob for model paths.")
    parser.add_argument("--stockfish_path", default="stockfish", help="Path to stockfish binary.")
    parser.add_argument("--skill_levels", default="0,2,4,6,8,10", help="Comma-separated skill levels.")
    parser.add_argument("--games", type=int, default=20, help="Games per model per skill level.")
    parser.add_argument("--batch_size", type=int, default=16, help="AlphaChess inference batch size.")
    parser.add_argument("--output", default="eval/stockfish_results.json", help="Output JSON path.")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    skill_levels = [int(s.strip()) for s in args.skill_levels.split(",")]

    model_paths: List[str] = []
    if args.models:
        model_paths = [p.strip() for p in args.models.split(",") if p.strip()]
    if args.model_glob:
        model_paths += sorted(glob.glob(args.model_glob))
    if not model_paths:
        # Default: SFT baseline + latest PPO checkpoint
        defaults = [
            "models/sft_models/alpha_chess_epoch_16.pth",
            *sorted(glob.glob("models/ppo_models/ppo_model_itr_*/ppo_model_itr_*.pth")),
        ]
        model_paths = [p for p in defaults if os.path.exists(p)]
    if not model_paths:
        raise FileNotFoundError("No model paths found. Pass --models or --model_glob.")

    all_results: List[dict] = []

    for model_path in model_paths:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        print(f"\nEvaluating {model_name}...")
        alpha_opponent = AlphaChessOpponent(model_path, device, greedy=True)

        for skill_level in skill_levels:
            approx_elo = StockfishOpponent.SKILL_ELO.get(skill_level, "?")
            print(f"  vs Skill {skill_level} (~{approx_elo} ELO) ... ", end="", flush=True)
            result = evaluate_model_at_level(
                model_name, alpha_opponent,
                args.stockfish_path, skill_level,
                args.games, args.batch_size,
            )
            all_results.append(result)
            print(
                f"W={result['wins']} D={result['draws']} L={result['losses']} "
                f"score={result['score']:.2f}"
            )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {args.output}")

    print_summary(all_results, skill_levels)


if __name__ == "__main__":
    main()
