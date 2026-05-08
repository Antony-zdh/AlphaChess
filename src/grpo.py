from abc import abstractmethod

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

class GRPOTrajectory:
    """ 
    Class representing a single trajectory (game episode) for training. 
    Each trajectory consists of a sequence of (state, policy, log_prob, reward, value) tuples.
     In PPO, we have predicted value for each state using the Value Function.
     In GRPO, we do not have critic, so all value is 0.
     The trajectory captures the entire sequence of states, actions, rewards, and values for one complete game episode.
    """
    def __init__(self):
        self.steps = []  # List of (state, policy, log_prob, reward, value) tuples

    def add_step(self, state, policy, log_prob, reward, value):
        self.steps.append((state, policy, log_prob, reward, value))


class GRPOGroup():
    """ Group for GRPO training, containing a batch of trajectories """
    def __init__(self, trajectories, capacity=32):
        self.capacity = capacity  # GRPO uses batch of trajectories per update
        if isinstance(trajectories, list) and len(trajectories) == self.capacity:
            self.rollout = trajectories  # List of trajectories (multiple game episodes)
        else:
            raise ValueError("GRPOGroup requires a list of trajectories (each being a list of (state, policy, value) tuples) with capacity {}.".format(self.capacity))


class AdvantageManager:
    """ Abstract class for managing advantages """
    def __init__(self):
        pass

    def compute_advantages(self, group):
        pass  # To be implemented by subclasses


class PPOAdvantageManager(AdvantageManager):
    """ Reward manager for PPO, computing advantage estimates for a trajectory """
    def __init__(self, gamma=0.99):
        super().__init__()
        self.gamma = gamma

    def compute_advantages(self, group):
        # Compute advantage estimates for each step in the trajectory
        pass