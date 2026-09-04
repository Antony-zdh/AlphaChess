"""GRPO (Group Relative Policy Optimization) for chess self-play RL.

No critic: the value head is NOT used for advantage. Advantage = group-normalized
Monte-Carlo discounted return. Per step: adv = gamma^(T-1-t) * terminal_reward
(terminal_reward already in that trajectory's player perspective, +1 win / -1 loss
/ draw_penalty). Then normalize across ALL steps in the group:
    adv_norm = (adv - mean(group)) / (std(group) + eps)
This group-relative baseline replaces PPO's value-based GAE, sidestepping
value-head accuracy entirely.

Trajectories store numpy arrays (state, legal_mask) so they pickle through
mp.Queue via pipe (NOT torch shared memory — avoids /dev/shm exhaustion, see
az-training memory). get_data_tensors converts back to tensors on the learner.
"""
import numpy as np
import torch
import chess

try:
    from src.model import BoardEncoder, AlphaChess
except ImportError:
    from model import BoardEncoder, AlphaChess


def _to_np(x, dtype):
    if hasattr(x, "cpu"):
        return x.detach().cpu().numpy().astype(dtype)
    return np.asarray(x, dtype=dtype)


class GRPOTrajectory:
    """One game's per-step records. reward=0 for all steps except the last,
    which gets the terminal reward (from this trajectory player's perspective)."""
    def __init__(self):
        self.steps = []  # list of dict: state, action, log_prob, color, legal_mask, reward, advantage

    def add_step(self, state, action, log_prob, color, legal_mask):
        self.steps.append({
            "state": _to_np(state, np.float32),             # (19,8,8)
            "action": int(action),                           # 0..4095
            "log_prob": float(log_prob),
            "color": bool(color),                            # chess.WHITE/BLACK (side to move)
            "legal_mask": _to_np(legal_mask, np.bool_),      # (4096,)
            "reward": 0.0,
            "advantage": 0.0,
        })

    def set_terminal_reward(self, reward):
        if self.steps:
            self.steps[-1]["reward"] = float(reward)


class GRPOGroup:
    def __init__(self, trajectories):
        if not isinstance(trajectories, list) or len(trajectories) == 0:
            raise ValueError("GRPOGroup requires a non-empty list of trajectories")
        self.trajectories = trajectories

    def get_data_tensors(self):
        """Flatten all trajectories' steps into tensors. Returns
        (states, actions, old_log_probs, advantages, legal_masks) — NO returns
        (GRPO has no value target)."""
        states, actions, log_probs, advantages, legal_masks = [], [], [], [], []
        for traj in self.trajectories:
            for s in traj.steps:
                states.append(s["state"])
                actions.append(s["action"])
                log_probs.append(s["log_prob"])
                advantages.append(s["advantage"])
                legal_masks.append(s["legal_mask"])
        states = torch.as_tensor(np.stack(states))                  # (N,19,8,8)
        actions = torch.as_tensor(np.array(actions), dtype=torch.long)
        log_probs = torch.as_tensor(np.array(log_probs), dtype=torch.float)
        advantages = torch.as_tensor(np.array(advantages), dtype=torch.float)
        legal_masks = torch.as_tensor(np.stack(legal_masks))        # (N,4096) bool
        return states, actions, log_probs, advantages, legal_masks


class GRPOAdvantageManager:
    """Compute MC-discounted returns per step, then group-normalize (the GRPO
    group-relative baseline)."""
    def __init__(self, gamma=0.99, eps=1e-8):
        self.gamma = gamma
        self.eps = eps

    def compute_advantages(self, group: GRPOGroup):
        all_adv = []
        for traj in group.trajectories:
            T = len(traj.steps)
            if T == 0:
                continue
            r = traj.steps[-1]["reward"]  # terminal reward, this player's perspective
            for t in range(T):
                adv = (self.gamma ** (T - 1 - t)) * r
                traj.steps[t]["advantage"] = adv
                all_adv.append(adv)
        if not all_adv:
            return
        mean = sum(all_adv) / len(all_adv)
        var = sum((a - mean) ** 2 for a in all_adv) / len(all_adv)
        std = var ** 0.5 + self.eps
        for traj in group.trajectories:
            for s in traj.steps:
                s["advantage"] = (s["advantage"] - mean) / std
