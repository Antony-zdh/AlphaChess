import chess
import os
import torch
import numpy as np
import argparse
import logging
import wandb
import random
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
        if isinstance(trajectories, list) and len(trajectories) == PPO_BATCH_SIZE:
            self.trajectories = trajectories
        else:
            raise ValueError("PPOGroup requires a list of trajectories with capacity {}.".format(PPO_BATCH_SIZE))

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
        states = torch.stack(states)  # (total_steps, 19, 8, 8)
        actions = torch.tensor(actions, dtype=torch.long)  # (total_steps,)
        log_probs = torch.tensor(log_probs, dtype=torch.float)  # (total_steps,)
        returns = torch.tensor(returns, dtype=torch.float)  # (total_steps,)
        advantages = torch.tensor(advantages, dtype=torch.float)  # (total_steps,)
        legal_masks = torch.stack(legal_masks).to(torch.bool)  # (total_steps, 4096)

        return states, actions, log_probs, returns, advantages, legal_masks

    def shuffle_and_get_dl(self):
        """
        Shuffle the data for PPO training.

        Returns:
            A DataLoader that yields shuffled batches of (state, action, log_prob, return, advantage).
        """
        states, actions, log_probs, returns, advantages, legal_masks = self.get_data_tensors()
        dataset = torch.utils.data.TensorDataset(states, actions, log_probs, returns, advantages, legal_masks)
        data_loader = DataLoader(dataset, batch_size=PPO_MINI_BATCH_SIZE, shuffle=True)
        return data_loader

        
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

    def get_policy_and_value(self, state):
        policy_logits, value = self.model(state)
        return policy_logits, value

    def self_play_episode(self, device):
        """ Simulate a game episode by self-play and return the trajectory. """

        # Set model to eval mode for inference during self-play
        self.model.eval()
        with torch.no_grad():  # Disable gradient computation during self-play
            trajectory_white = PPOTrajectory()
            trajectory_black = PPOTrajectory()
            board = chess.Board()
            encoder = BoardEncoder()

            while not board.is_game_over():
                state = encoder.encode(board).unsqueeze(0).to(device)  # (1, 19, 8, 8)
                action, log_prob, value, legal_mask = self.select_action(state, board)
                move_from = action // 64
                move_to = action % 64
                legal_moves = list(board.legal_moves)
                candidates = [m for m in legal_moves if m.from_square == move_from and m.to_square == move_to]
                if candidates:
                    queen_promo = [m for m in candidates if m.promotion == chess.QUEEN]
                    move = queen_promo[0] if queen_promo else candidates[0]
                else:
                    # Should not happen with masking; resample once and fail fast if it repeats.
                    logging.warning("Selected an illegal move index. Resampling once.")
                    action, log_prob, value, legal_mask = self.select_action(state, board)
                    move_from, move_to = action // 64, action % 64
                    legal_moves = list(board.legal_moves)
                    candidates = [m for m in legal_moves if m.from_square == move_from and m.to_square == move_to]
                    if candidates:
                        queen_promo = [m for m in candidates if m.promotion == chess.QUEEN]
                        move = queen_promo[0] if queen_promo else candidates[0]
                    else:
                        logging.error("Resampled an illegal move index again. Board FEN: %s", board.fen())
                        raise RuntimeError("Failed to select a legal move after one resample.")

                reward = 0  # Intermediate reward can be defined based on heuristics
                next_value = 0  # Will be computed later (Last step has next_value = 0)
                # Put states on CPU for trajectory storage to save GPU memory
                if board.turn == chess.WHITE:
                    assert 0 <= action < 4096, "Action index out of bounds. This should not happen with proper encoding."
                    assert bool(legal_mask[action].item()), "Selected action is not legal for White. This should not happen with proper masking."
                    board.push(move)  # Push before adding step to ensure correct done value
                    done = board.is_game_over()
                    trajectory_white.add_step(state.cpu(), action, log_prob, reward, value, next_value, done, legal_mask)
                else:
                    assert 0 <= action < 4096, "Action index out of bounds. This should not happen with proper encoding."
                    assert bool(legal_mask[action].item()), "Selected action is not legal for Black. This should not happen with proper masking."
                    board.push(move)  # Push before adding step to ensure correct done value
                    done = board.is_game_over()
                    trajectory_black.add_step(state.cpu(), action, log_prob, reward, value, next_value, done, legal_mask)


            # Final reward based on game outcome
            result = board.result()
            if result == "1-0":
                trajectory_white.steps[-1]["reward"] = 1  # White wins
                trajectory_black.steps[-1]["reward"] = -1  # Black loses
            elif result == "0-1":
                trajectory_white.steps[-1]["reward"] = -1  # White loses
                trajectory_black.steps[-1]["reward"] = 1  # Black wins
            else:
                trajectory_white.steps[-1]["reward"] = 0  # Draw
                trajectory_black.steps[-1]["reward"] = 0  # Draw

            # Compute next_value for each step (value of the next state)
            for trajectory in [trajectory_white, trajectory_black]:
                for i in range(len(trajectory.steps) - 1):
                    trajectory.steps[i]["next_value"] = trajectory.steps[i + 1]["value"]
                trajectory.steps[-1]["next_value"] = 0  # Last step has no next state

        # After the episode, switch back to training mode
        self.model.train()

        return trajectory_white, trajectory_black


def train_ppo(args):
    """ Main training loop for PPO. """

    # 1. Initialize model, optimizer, and other components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        wandb.init(project="AlphaChess_RL", config=vars(args))
        wandb.watch(model, log="gradients", log_freq=args.log_interval)

    print("Starting PPO training...")
    model.train() # Set model to training mode
    actor_critic = ActorCritic(model)

    # 2. Main training loop
    for iteration in range(args.iterations):

        # Metrix reset
        avg_total_loss = 0
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy_bonus = 0
        
        # a. Collect trajectories by self-play
        trajectories = []
        for _ in range(PPO_BATCH_SIZE // 2): # Each episode generates two trajectories
            # Simulate a game episode and fill the trajectory
            trajectory_white, trajectory_black = actor_critic.self_play_episode(device)
            # After the episode, white and black trajectories are treated as individual trajectories
            trajectories.append(trajectory_white)
            trajectories.append(trajectory_black)

        # b. Compute advantages and returns for each trajectory
        for trajectory in trajectories:
            trajectory.compute_advantages_and_returns()

        # c. Create PPOGroup for a batch and get shuffled training data
        ppo_group = PPOGroup(trajectories)
        training_dl = ppo_group.shuffle_and_get_dl()  # DataLoader of (state, action, log_prob, return, advantage)
        
        # d. Perform multiple minibatches of PPO updates on the collected data
        for mini_batch in training_dl:
            states, actions, old_log_probs, returns, advantages, legal_masks = mini_batch
            
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

        # g. Logging
        avg_total_loss /= len(training_dl)
        avg_policy_loss /= len(training_dl)
        avg_value_loss /= len(training_dl)
        avg_entropy_bonus /= len(training_dl)
        
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



def __main__():
    parser = argparse.ArgumentParser(description="Train PPO agent for chess.")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--model_path", type=str, default="ppo_model.pth", help="Path to save the trained model")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--wandb", action="store_true", help="Enable Wandb logging")
    parser.add_argument("--log_interval", type=int, default=1, help="Interval for logging to Wandb")
    
    args = parser.parse_args()

    train_ppo(args)