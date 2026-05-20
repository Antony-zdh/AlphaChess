import argparse
import glob
import json
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import chess
import torch

from src.model import BoardEncoder, AlphaChess


@dataclass
class GameRecord:
    board: chess.Board
    ppo_color: chess.Color
    ply_count: int = 0
    result: Optional[str] = None
    ppo_values: List[float] = field(default_factory=list)
    last_ppo_value: Optional[float] = None
    sft_values: List[float] = field(default_factory=list)
    last_sft_value: Optional[float] = None


def apply_random_opening(record: GameRecord, rng: random.Random, opening_plies: int) -> None:
    for _ in range(opening_plies):
        if record.board.is_game_over():
            record.result = record.board.result()
            return
        legal_moves = list(record.board.legal_moves)
        record.board.push(rng.choice(legal_moves))
        record.ply_count += 1
    if record.board.is_game_over():
        record.result = record.board.result()


def select_moves_batch(
    model: AlphaChess,
    boards: List[chess.Board],
    device: torch.device,
    temperature: float,
) -> Tuple[List[int], List[float]]:
    encoder = BoardEncoder()
    state_tensors = [encoder.encode(board) for board in boards]
    states = torch.stack(state_tensors).to(device)

    with torch.no_grad():
        policy_logits, values = model(states)

    actions = []
    value_list = []
    for idx, board in enumerate(boards):
        legal_moves = list(board.legal_moves)
        legal_move_indices = [move.from_square * 64 + move.to_square for move in legal_moves]
        mask = torch.zeros_like(policy_logits[idx])
        mask[legal_move_indices] = 1.0
        masked_logits = policy_logits[idx].masked_fill(mask == 0, float("-inf"))
        if temperature > 0:
            scaled_logits = masked_logits / temperature
            probs = torch.softmax(scaled_logits, dim=0)
            action = torch.multinomial(probs, 1).item()
        else:
            action = torch.argmax(masked_logits).item()
        actions.append(action)
        value_list.append(values[idx].item())

    return actions, value_list


def action_to_move(board: chess.Board, action: int) -> chess.Move:
    move_from = action // 64
    move_to = action % 64
    legal_moves = list(board.legal_moves)
    candidates = [m for m in legal_moves if m.from_square == move_from and m.to_square == move_to]
    if candidates:
        queen_promo = [m for m in candidates if m.promotion == chess.QUEEN]
        return queen_promo[0] if queen_promo else candidates[0]
    return legal_moves[0]


def evaluate_checkpoint(
    ppo_path: str,
    baseline_path: str,
    games: int,
    batch_size: int,
    device: torch.device,
    opening_plies: int,
    seed: int,
    baseline_temperature: float,
) -> dict:
    ppo_model = AlphaChess().to(device)
    baseline_model = AlphaChess().to(device)

    ppo_model.load_state_dict(torch.load(ppo_path, map_location=device))
    baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))

    ppo_model.eval()
    baseline_model.eval()

    records: List[GameRecord] = []
    half = games // 2
    for i in range(games):
        ppo_color = chess.WHITE if i < half else chess.BLACK
        records.append(GameRecord(board=chess.Board(), ppo_color=ppo_color))

    for idx, record in enumerate(records):
        apply_random_opening(record, random.Random(seed + idx), opening_plies)

    active_indices = [i for i in range(games) if not records[i].board.is_game_over()]
    while active_indices:
        ppo_turn_indices = []
        baseline_turn_indices = []
        for idx in active_indices:
            record = records[idx]
            if record.board.turn == record.ppo_color:
                ppo_turn_indices.append(idx)
            else:
                baseline_turn_indices.append(idx)

        for group_indices, model, is_ppo in [
            (ppo_turn_indices, ppo_model, True),
            (baseline_turn_indices, baseline_model, False),
        ]:
            for start in range(0, len(group_indices), batch_size):
                batch_ids = group_indices[start:start + batch_size]
                if not batch_ids:
                    continue
                batch_boards = [records[i].board for i in batch_ids]
                temperature = 0.0 if is_ppo else baseline_temperature
                actions, values = select_moves_batch(model, batch_boards, device, temperature)
                for local_idx, global_idx in enumerate(batch_ids):
                    record = records[global_idx]
                    board = record.board
                    action = actions[local_idx]
                    value = values[local_idx]
                    move = action_to_move(board, action)
                    board.push(move)
                    record.ply_count += 1

                    if is_ppo:
                        record.ppo_values.append(value)
                        record.last_ppo_value = value
                    else:
                        record.sft_values.append(value)
                        record.last_sft_value = value

                    if board.is_game_over():
                        record.result = board.result()

        active_indices = [i for i in active_indices if not records[i].board.is_game_over()]

    wins = 0
    losses = 0
    draws = 0
    win_steps = []
    lose_steps = []
    draw_steps = []
    ppo_value_means = []
    sft_value_means = []
    terminal_values_win = []
    terminal_values_loss = []
    terminal_values_draw = []

    for record in records:
        if record.result == "1-0":
            ppo_wins = record.ppo_color == chess.WHITE
        elif record.result == "0-1":
            ppo_wins = record.ppo_color == chess.BLACK
        else:
            ppo_wins = None

        if ppo_wins is True:
            wins += 1
            win_steps.append(record.ply_count)
            if record.last_ppo_value is not None:
                terminal_values_win.append(record.last_ppo_value)
        elif ppo_wins is False:
            losses += 1
            lose_steps.append(record.ply_count)
            if record.last_ppo_value is not None:
                terminal_values_loss.append(record.last_ppo_value)
        else:
            draws += 1
            draw_steps.append(record.ply_count)
            if record.last_ppo_value is not None:
                terminal_values_draw.append(record.last_ppo_value)

        if record.ppo_values:
            ppo_value_means.append(sum(record.ppo_values) / len(record.ppo_values))
        if record.sft_values:
            sft_value_means.append(sum(record.sft_values) / len(record.sft_values))

    total = wins + losses + draws
    return {
        "checkpoint": ppo_path,
        "games": total,
        "win_rate": wins / total if total else 0.0,
        "lose_rate": losses / total if total else 0.0,
        "draw_rate": draws / total if total else 0.0,
        "avg_win_steps": sum(win_steps) / len(win_steps) if win_steps else 0.0,
        "avg_lose_steps": sum(lose_steps) / len(lose_steps) if lose_steps else 0.0,
        "avg_draw_steps": sum(draw_steps) / len(draw_steps) if draw_steps else 0.0,
        "avg_ppo_value": sum(ppo_value_means) / len(ppo_value_means) if ppo_value_means else 0.0,
        "avg_sft_value": sum(sft_value_means) / len(sft_value_means) if sft_value_means else 0.0,
        "avg_terminal_value_win": sum(terminal_values_win) / len(terminal_values_win) if terminal_values_win else 0.0,
        "avg_terminal_value_loss": sum(terminal_values_loss) / len(terminal_values_loss) if terminal_values_loss else 0.0,
        "avg_terminal_value_draw": sum(terminal_values_draw) / len(terminal_values_draw) if terminal_values_draw else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoints vs SFT baseline.")
    parser.add_argument("--ppo_root", default="models/ppo_models", help="Folder containing PPO checkpoints.")
    parser.add_argument("--baseline_path", default="models/alpha_chess_epoch_16.pth", help="SFT baseline model.")
    parser.add_argument("--games", type=int, default=128, help="Games per checkpoint.")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--output_json", default="eval/ppo_eval_results.json", help="Path to write results JSON.")
    parser.add_argument("--opening_plies", type=int, default=4, help="Random opening plies before evaluation.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for openings.")
    parser.add_argument("--baseline_temperature", type=float, default=0.0, help="Temperature for baseline move sampling.")

    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ckpt_glob = os.path.join(args.ppo_root, "ppo_model_itr_*", "ppo_model_itr_*.pth")
    checkpoints = sorted(glob.glob(ckpt_glob))
    if not checkpoints:
        raise FileNotFoundError(f"No PPO checkpoints found under {args.ppo_root}")

    results = []
    for ckpt in checkpoints:
        result = evaluate_checkpoint(
            ckpt,
            args.baseline_path,
            args.games,
            args.batch_size,
            device,
            args.opening_plies,
            args.seed,
            args.baseline_temperature,
        )
        results.append(result)
        print(
            f"{os.path.basename(ckpt)} | win {result['win_rate']:.3f} "
            f"lose {result['lose_rate']:.3f} draw {result['draw_rate']:.3f} "
            f"avg_win_steps {result['avg_win_steps']:.1f} "
            f"avg_lose_steps {result['avg_lose_steps']:.1f} "
            f"avg_draw_steps {result['avg_draw_steps']:.1f}"
        )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
