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

try:  # works both as `python src/buffer.py` and `from src.buffer import ...`
    from model import BoardEncoder, AlphaChess
except ImportError:
    from src.model import BoardEncoder, AlphaChess

class Buffer:
    """ Abstract class for replay buffer (on-policy or off-policy) """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    def add(self, group):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Remove oldest entry
        self.buffer.append(group)
    
    def sample(self):
        pass  # To be implemented by subclasses


class OnPolicyBuffer(Buffer):
    """ Buffer for strictly on-policy algorithms """
    def __init__(self):
        super().__init__(capacity=1)  # Only store the most recent batch (group)

    def sample(self):
        if len(self.buffer) == 0:
            raise ValueError("OnPolicyBuffer is empty. Cannot sample.")
        return self.buffer[-1]  # Always return the most recent group (trajectory)
    

class OffPolicyBuffer(Buffer):
    """ Buffer for off-policy algorithms, allowing random sampling """
    def __init__(self, capacity=1000):
        super().__init__(capacity)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            print(f"Not enough samples in buffer ({batch_size} samples requested). Returning all {len(self.buffer)} samples.")
            return random.sample(self.buffer, len(self.buffer))  # Return all if not enough samples
        return random.sample(self.buffer, batch_size)