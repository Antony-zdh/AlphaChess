import chess
import os
import torch
import numpy as np
import argparse
import logging
import wandb
import random
import json
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import BoardEncoder, AlphaChess

PPO_BATCH_SIZE = 32  # PPO uses batch of trajectories per update
PPO_MINI_BATCH_SIZE = 8  # PPO mini-batch size for multiple epochs of updates
PPO_NUM_UPDATES_PER_BATCH = PPO_BATCH_SIZE // PPO_MINI_BATCH_SIZE  # Number of updates per batch of trajectories
PPO_GAMMA = 0.99  # Discount factor for rewards
PPO_LAMBDA = 0.95  # GAE lambda for advantage estimation
PPO_EPSILON = 0.2  # Clipping parameter for PPO loss
DIVISION_EPSILON = 1e-8  # Small constant to avoid division by zero
PPO_VF_LOSS_COEF = 0.5  # Coefficient for value function loss
PPO_ENTROPY_COEF = 0.01  # Coefficient for entropy bonus


class PPOTrajectory:
    """ 
    Class representing a single trajectory (game episode) for PPO training.
    A trajectory is a sequence of step dictionaries.
    """
    def __init__(self):
        self.steps = []  # List of step dictionaries

    def add_step(self, state, action, log_prob, reward, value, next_value, done, legal_mask):
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
            legal_mask: A boolean mask indicating which actions were legal at this step.
        """
        self.steps.append({
            "state": state,
            "action": action,
            "log_prob": log_prob,
            "reward": reward,
            "value": value,
            "next_value": next_value,
            "done": done,
            "legal_mask": legal_mask
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
        if isinstance(trajectories, list) and len(trajectories) > 0:
            self.trajectories = trajectories
        else:
            raise ValueError("PPOGroup requires a non-empty list of trajectories.")

    def get_data_tensors(self):
        """
        Flatten all trajectories into tensors for PPO training.

        Returns:
            (states, actions, old_log_probs, returns, advantages) as tensors
        """
        states = []
        actions = []
        log_probs = []
        returns = []
        advantages = []
        legal_masks = []
        for trajectory in self.trajectories:
            for step in trajectory.steps:
                states.append(step["state"])
                actions.append(step["action"])
                log_probs.append(step["log_prob"])
                returns.append(step["return"])
                advantages.append(step["advantage"])
                legal_masks.append(step["legal_mask"])

        # Convert lists to tensors
        states = torch.stack(states).squeeze(1)  # (total_steps, 19, 8, 8)
        actions = torch.tensor(actions, dtype=torch.long)  # (total_steps,)
        log_probs = torch.tensor(log_probs, dtype=torch.float)  # (total_steps,)
        returns = torch.tensor(returns, dtype=torch.float)  # (total_steps,)
        advantages = torch.tensor(advantages, dtype=torch.float)  # (total_steps,)
        legal_masks = torch.stack(legal_masks).to(torch.bool)  # (total_steps, 4096)

        return states, actions, log_probs, returns, advantages, legal_masks

        
class ActorCritic:
    """ Actor-Critic class for PPO, responsible for selecting actions and estimating values. """
    def __init__(self, model):
        self.model = model

    def select_action(self, state, board):
        policy_logits, value = self.model(state)
        
        # Mask out illegal moves
        legal_moves = list(board.legal_moves)
        legal_move_indices = [move.from_square * 64 + move.to_square for move in legal_moves]
        mask = torch.zeros_like(policy_logits)
        mask[0, legal_move_indices] = 1.0
        masked_logits = policy_logits.masked_fill(mask == 0, float('-inf'))  # Mask illegal moves with -inf so they have zero probability after softmax

        policy_dist = torch.distributions.Categorical(logits=masked_logits)
        action = policy_dist.sample()
        log_prob = policy_dist.log_prob(action)
        return action.item(), log_prob.item(), value.item(), mask.squeeze(0).to(torch.bool).cpu()

    def select_actions_batch(self, states, boards):
        policy_logits, values = self.model(states)
        actions = []
        log_probs = []
        legal_masks = []

        for idx, board in enumerate(boards):
            legal_moves = list(board.legal_moves)
            legal_move_indices = [move.from_square * 64 + move.to_square for move in legal_moves]
            mask = torch.zeros_like(policy_logits[idx])
            mask[legal_move_indices] = 1.0
            masked_logits = policy_logits[idx].masked_fill(mask == 0, float("-inf"))

            policy_dist = torch.distributions.Categorical(logits=masked_logits)
            action = policy_dist.sample()
            log_prob = policy_dist.log_prob(action)

            actions.append(action.item())
            log_probs.append(log_prob.item())
            legal_masks.append(mask.to(torch.bool).cpu())

        values = values.squeeze(-1).detach().cpu().tolist()
        return actions, log_probs, values, legal_masks

    def get_policy_and_value(self, state):
        policy_logits, value = self.model(state)
        return policy_logits, value

    def self_play_episodes(self, device):
        """ Simulate multiple game episodes by self-play and return the trajectories. """

        # Set model to eval mode for inference during self-play
        self.model.eval()
        with torch.no_grad():  # Disable gradient computation during self-play
            batch_size = PPO_BATCH_SIZE // 2
            boards = [chess.Board() for _ in range(batch_size)]
            trajectories_white = [PPOTrajectory() for _ in range(batch_size)]
            trajectories_black = [PPOTrajectory() for _ in range(batch_size)]
            encoder = BoardEncoder()

            while any(not board.is_game_over() for board in boards):
                active_indices = [i for i, board in enumerate(boards) if not board.is_game_over()]
                active_boards = [boards[i] for i in active_indices]
                state_tensors = [encoder.encode(board) for board in active_boards]
                states = torch.stack(state_tensors).to(device)

                actions, log_probs, values, legal_masks = self.select_actions_batch(states, active_boards)

                for offset, board_idx in enumerate(active_indices):
                    board = boards[board_idx]
                    action = actions[offset]
                    log_prob = log_probs[offset]
                    value = values[offset]
                    legal_mask = legal_masks[offset]

                    move_from = action // 64
                    move_to = action % 64
                    legal_moves = list(board.legal_moves)
                    candidates = [m for m in legal_moves if m.from_square == move_from and m.to_square == move_to]
                    if candidates:
                        queen_promo = [m for m in candidates if m.promotion == chess.QUEEN]
                        move = queen_promo[0] if queen_promo else candidates[0]
                    else:
                        logging.warning("Selected an illegal move index. Resampling once.")
                        resample_actions, resample_log_probs, resample_values, resample_masks = self.select_actions_batch(
                            states[offset:offset + 1], [board]
                        )
                        action = resample_actions[0]
                        log_prob = resample_log_probs[0]
                        value = resample_values[0]
                        legal_mask = resample_masks[0]
                        move_from = action // 64
                        move_to = action % 64
                        legal_moves = list(board.legal_moves)
                        candidates = [m for m in legal_moves if m.from_square == move_from and m.to_square == move_to]
                        if candidates:
                            queen_promo = [m for m in candidates if m.promotion == chess.QUEEN]
                            move = queen_promo[0] if queen_promo else candidates[0]
                        else:
                            logging.error("Resampled an illegal move index again. Board FEN: %s", board.fen())
                            raise RuntimeError("Failed to select a legal move after one resample.")

                    reward = 0
                    next_value = 0
                    if board.turn == chess.WHITE:
                        assert 0 <= action < 4096, "Action index out of bounds. This should not happen with proper encoding."
                        assert bool(legal_mask[action].item()), "Selected action is not legal for White. This should not happen with proper masking."
                        board.push(move)
                        done = board.is_game_over()
                        trajectories_white[board_idx].add_step(
                            state_tensors[offset],
                            action,
                            log_prob,
                            reward,
                            value,
                            next_value,
                            done,
                            legal_mask,
                        )
                    else:
                        assert 0 <= action < 4096, "Action index out of bounds. This should not happen with proper encoding."
                        assert bool(legal_mask[action].item()), "Selected action is not legal for Black. This should not happen with proper masking."
                        board.push(move)
                        done = board.is_game_over()
                        trajectories_black[board_idx].add_step(
                            state_tensors[offset],
                            action,
                            log_prob,
                            reward,
                            value,
                            next_value,
                            done,
                            legal_mask,
                        )

            for board_idx in range(batch_size):
                result = boards[board_idx].result()
                if result == "1-0":
                    trajectories_white[board_idx].steps[-1]["reward"] = 1
                    trajectories_black[board_idx].steps[-1]["reward"] = -1
                elif result == "0-1":
                    trajectories_white[board_idx].steps[-1]["reward"] = -1
                    trajectories_black[board_idx].steps[-1]["reward"] = 1
                else:
                    trajectories_white[board_idx].steps[-1]["reward"] = 0
                    trajectories_black[board_idx].steps[-1]["reward"] = 0

                n_w = len(trajectories_white[board_idx].steps)
                n_b = len(trajectories_black[board_idx].steps)
                for i in range(n_w - 1):
                    trajectories_white[board_idx].steps[i]["next_value"] = -trajectories_black[board_idx].steps[i]["value"]
                for i in range(n_b - 1):
                    trajectories_black[board_idx].steps[i]["next_value"] = -trajectories_white[board_idx].steps[i + 1]["value"]

        # After the episode, switch back to training mode
        self.model.train()

        return trajectories_white + trajectories_black


def train_ppo(args):
    """ Main training loop for PPO. """

    # Find device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU. For better performance, consider using a GPU.")

    # 1. Initialize model, optimizer, and other components
    model = AlphaChess().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iterations * PPO_NUM_UPDATES_PER_BATCH)
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
        wandb.init(project="AlphaChess_PPO", config=vars(args))
        wandb.watch(model, log="gradients", log_freq=args.log_interval)

    print("Starting PPO training...")
    model.train() # Set model to training mode
    actor_critic = ActorCritic(model)

    # 2. Main training loop
    for iteration in range(1, args.iterations + 1):

        # Metrix reset
        avg_total_loss = 0
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy_bonus = 0
        
        # a. Collect trajectories by self-play
        trajectories = actor_critic.self_play_episodes(device)

        # b. Compute advantages and returns for each trajectory
        for trajectory in trajectories:
            trajectory.compute_advantages_and_returns()

        # c. Shuffle trajectories and split into trajectory-level minibatches
        random.shuffle(trajectories)
        num_updates = 0

        # d. Perform multiple minibatches of PPO updates on the collected data
        for start in range(0, len(trajectories), PPO_MINI_BATCH_SIZE):
            mini_trajs = trajectories[start:start + PPO_MINI_BATCH_SIZE]
            ppo_group = PPOGroup(mini_trajs)
            states, actions, old_log_probs, returns, advantages, legal_masks = ppo_group.get_data_tensors()
            
            # Move tensors to device
            states = states.to(device)
            actions = actions.to(device)
            old_log_probs = old_log_probs.to(device)
            returns = returns.to(device)
            advantages = advantages.to(device)
            legal_masks = legal_masks.to(device)

            # e. Compute PPO loss
            # CLIP loss
            policy_logits, values = actor_critic.get_policy_and_value(states)
            masked_logits = policy_logits.masked_fill(~legal_masks, float('-inf'))
            policy_dist = torch.distributions.Categorical(logits=masked_logits)
            new_log_probs = policy_dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            # Value loss (MSE)
            value_loss = torch.nn.functional.mse_loss(values.squeeze(-1), returns)

            # Entropy bonus for exploration
            entropy_bonus = torch.mean(policy_dist.entropy())

            # Total loss
            total_loss = policy_loss + PPO_VF_LOSS_COEF * value_loss - PPO_ENTROPY_COEF * entropy_bonus    

            # f. Backpropagation and optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # Update metrics
            avg_total_loss += total_loss.item()
            avg_policy_loss += policy_loss.item()
            avg_value_loss += value_loss.item()
            avg_entropy_bonus += entropy_bonus.item()
            num_updates += 1

        # g. Logging
        avg_total_loss /= max(num_updates, 1)
        avg_policy_loss /= max(num_updates, 1)
        avg_value_loss /= max(num_updates, 1)
        avg_entropy_bonus /= max(num_updates, 1)
        
        # Wandb logging
        if args.wandb and iteration % args.log_interval == 0:
            wandb.log({
                "iteration": iteration,
                "policy_loss": avg_policy_loss,
                "value_loss": avg_value_loss,
                "entropy_bonus": avg_entropy_bonus,
                "total_loss": avg_total_loss,
                "learning_rate": scheduler.get_last_lr()[0]
            })
        
        # Console logging
        if iteration % args.log_interval == 0:
            logging.info(f"Iteration {iteration}: Policy Loss={avg_policy_loss:.4f}, Value Loss={avg_value_loss:.4f}, Entropy Bonus={avg_entropy_bonus:.4f}, Total Loss={avg_total_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")

        # h. Save model checkpoint
        if iteration % args.save_interval == 0:
            # We save the model, optimizer, scheduler states and training configuration
            config = {
                "iteration": iteration,
                "PPO_CONFIG": {
                    "PPO_BATCH_SIZE": PPO_BATCH_SIZE,
                    "PPO_MINI_BATCH_SIZE": PPO_MINI_BATCH_SIZE,
                    "PPO_GAMMA": PPO_GAMMA,
                    "PPO_LAMBDA": PPO_LAMBDA,
                    "PPO_EPSILON": PPO_EPSILON,
                    "PPO_VF_LOSS_COEF": PPO_VF_LOSS_COEF,
                    "PPO_ENTROPY_COEF": PPO_ENTROPY_COEF,
                },
                "ARGS": {
                    "total_iterations": args.iterations,
                    "model_path": args.model_path,
                    "lr": args.lr,
                    "wandb": args.wandb,
                    "log_interval": args.log_interval,
                    "save_interval": args.save_interval,
                    "output_path": args.output_path
                }
            }
            
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()
            scheduler_state = scheduler.state_dict()
            
            checkpoint_dir = os.path.join(args.output_path, f"ppo_model_itr_{iteration}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            config_path = os.path.join(checkpoint_dir, f"config_itr_{iteration}.json")
            checkpoint_path = os.path.join(checkpoint_dir, f"ppo_model_itr_{iteration}.pth")
            optimizer_path = os.path.join(checkpoint_dir, f"optimizer_itr_{iteration}.pth")
            scheduler_path = os.path.join(checkpoint_dir, f"scheduler_itr_{iteration}.pth")
            
            with open(config_path, "w") as f:
                json.dump(config, f)
            torch.save(model_state, checkpoint_path)
            torch.save(optimizer_state, optimizer_path)
            torch.save(scheduler_state, scheduler_path)
            logging.info(f"Saved model checkpoint to {checkpoint_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for chess.")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--model_path", type=str, default="models/alpha_chess_epoch_16.pth", help="Path to load the SFT model for PPO training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--wandb", action="store_true", help="Enable Wandb logging")
    parser.add_argument("--log_interval", type=int, default=1, help="Interval for logging to Wandb")
    parser.add_argument("--save_interval", type=int, default=100, help="Interval for saving model checkpoints")
    parser.add_argument("--output_path", type=str, default="models/ppo_models/", help="Path to save the trained model and configuration")

    args = parser.parse_args()

    train_ppo(args)