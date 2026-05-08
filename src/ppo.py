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
from buffer import Buffer, OnPolicyBuffer, OffPolicyBuffer

PPO_BATCH_SIZE = 32  # PPO uses batch of trajectories per update
PPO_MINI_BATCH_SIZE = 8  # PPO mini-batch size for multiple epochs of updates
PPO_GAMMA = 0.99  # Discount factor for rewards
PPO_LAMBDA = 0.95  # GAE lambda for advantage estimation
PPO_EPSILON = 0.2  # Clipping parameter for PPO loss
DIVISION_EPSILON = 1e-8  # Small constant to avoid division by zero


class PPOTrajectory:
    """ 
    Class representing a single trajectory (game episode) for PPO training.
    A trajectory is a sequence of step dictionaries.
    """
    def __init__(self):
        self.steps = []  # List of step dictionaries

    def add_step(self, state, action, log_prob, reward, value, next_value, done):
        """
        Add a step to the trajectory.

        Args:
            state: The board state at the time of the action.
            action: The action taken by the agent.
            log_prob: The log probability of the action under the current policy.
            reward: The reward received after taking the action (immediate reward).
            value: The value estimate for the current state.
            next_value: The value estimate for the next state.
            done: A boolean indicating whether the episode has ended after this step.
        """
        self.steps.append({
            "state": state,
            "action": action,
            "log_prob": log_prob,
            "reward": reward,
            "value": value,
            "next_value": next_value,
            "done": done
        })

    def compute_advantages_and_returns(self, gamma=PPO_GAMMA, lam=PPO_LAMBDA):
        """
        Compute advantage estimates for each step in the trajectory using GAE.

        Args:
            gamma: Discount factor for rewards.
            lam: GAE lambda parameter.

        Returns:
            A list of advantage estimates corresponding to each step in the trajectory.
        """
        advantages = []
        returns = []
        gae = 0
        for i in reversed(range(len(self.steps))):
            step = self.steps[i]
            delta = step["reward"] + gamma * step["next_value"] * (1 - step["done"]) - step["value"]
            gae = delta + gamma * lam * (1 - step["done"]) * gae
            advantages.insert(0, gae)  # Insert at the beginning to maintain order
            returns.insert(0, gae + step["value"])  # Return is advantage + value estimate
        
        # Normalize advantages for stable training
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + DIVISION_EPSILON)
        advantages = advantages.tolist()  # Convert back to list for consistency

        # Add to each step
        for i, step in enumerate(self.steps):
            step["advantage"] = advantages[i]
            step["return"] = returns[i]
        
        return self.steps


class PPOGroup():
    """ Group for PPO training, containing a batch of trajectories """
    def __init__(self, trajectories):
        if isinstance(trajectories, list) and len(trajectories) == PPO_BATCH_SIZE:
            self.trajectories = trajectories
        else:
            raise ValueError("PPOGroup requires a list of trajectories with capacity {}.".format(PPO_BATCH_SIZE))

    def get_data(self):
        """
        Flatten all trajectories into tensors for PPO training.

        Returns:
            (states, actions, old_log_probs, returns, advantages) as tensors
        """
        data = []
        for trajectory in self.trajectories:
            for step in trajectory.steps:
                data.append((
                    step["state"],
                    step["action"],
                    step["log_prob"],
                    step["return"],
                    step["advantage"]
                ))

        return data

        
class ActorCritic:
    """ Actor-Critic class for PPO, responsible for selecting actions and estimating values. """
    def __init__(self, model):
        self.model = model

    def select_action(self, state):
        policy_logits, value = self.model(state)
        policy_dist = torch.distributions.Categorical(logits=policy_logits)
        action = policy_dist.sample()
        log_prob = policy_dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()


def self_play_episode(actor_critic, device):
    """ Simulate a game episode by self-play and return the trajectory. """
    trajectory = PPOTrajectory()
    board = chess.Board()
    encoder = BoardEncoder()

    while not board.is_game_over():
        state = encoder.encode(board).unsqueeze(0).to(device)  # (1, 19, 8, 8)
        action, log_prob, value = actor_critic.select_action(state)
        move_from = action // 64
        move_to = action % 64

        if chess.Move(move_from, move_to) in board.legal_moves:
            move = chess.Move(move_from, move_to)
        else:
            # Handles promotion ambiguity by enforcing promotion to queen
            for promo_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promo_move = chess.Move(move_from, move_to, promotion=promo_piece)
                if promo_move in board.legal_moves:
                    move = promo_move
                    break

        # Fallback to random if still illegal
        if move not in board.legal_moves:
            print("Move illegal. Fall back to random move.")
            move = random.choice(board.legal_moves)
        
        board.push(move)
        reward = 0  # Intermediate reward can be defined based on heuristics
        next_value = None  # Will be computed in the next step
        done = board.is_game_over()
        trajectory.add_step(state.cpu(), action, log_prob, reward, value, next_value, done)

    # Final reward based on game outcome
    result = board.result()
    if result == "1-0":
        final_reward = 1  # White wins
    elif result == "0-1":
        final_reward = -1  # Black wins
    else:
        final_reward = 0  # Draw

    # Update the last step with the final reward
    if trajectory.steps:
        trajectory.steps[-1]["reward"] = final_reward

    return trajectory


def train_ppo(args):
    """ Main training loop for PPO. """

    # 1. Initialize model, optimizer, and other components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaChess().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.episodes)
    buffer = OnPolicyBuffer()  # PPO uses on-policy data
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    
    # Load SFT model
    if os.path.exists(args.model_path):
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Loaded model from {args.model_path}")
        except Exception as e:
            print(f"Error loading model from {args.model_path}: {e}")
            raise RuntimeError(f"Failed to load model from {args.model_path}")
    else:
        print(f"Model file not found at {args.model_path}")
        raise FileNotFoundError(f"Model file not found at {args.model_path}")

    # Use Wandb for logging (include hyperparameters)
    if args.wandb:
        wandb.init(project="AlphaChess_RL", config=vars(args))
        wandb.watch(model, log="gradients", log_freq=args.log_interval)

    print("Starting PPO training...")
    model.train() # Set model to training mode
    actor_critic = ActorCritic(model)

    # 2. Main training loop
    for iteration in range(args.iterations):
        
        # a. Collect trajectories by self-play
        trajectories = []
        for _ in range(PPO_BATCH_SIZE):
            # Simulate a game episode and fill the trajectory
            trajectory = self_play_episode(actor_critic, device)
            trajectories.append(trajectory)


def __main__():
    parser = argparse.ArgumentParser(description="Train PPO agent for chess.")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--model_path", type=str, default="ppo_model.pth", help="Path to save the trained model")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--wandb", action="store_true", help="Enable Wandb logging")
    parser.add_argument("--log_interval", type=int, default=1, help="Interval for logging to Wandb")
    
    args = parser.parse_args()

    train_ppo(args)