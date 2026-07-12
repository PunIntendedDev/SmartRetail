"""
Reinforcement Learning Agents Module
------------------------------------
Defines agents for customer recommendation:
1. Discretizer helper using KMeans state clustering.
2. Tabular Q-learning agent.
3. Deep Q-Network (DQN) agent implemented in PyTorch.
"""

import os
import random
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.cluster import KMeans
from typing import Tuple, Dict, Any, List

from src.config import get_absolute_path, config_data
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class StateDiscretizer:
    """
    Clusters continuous customer states [Recency, Frequency, Monetary, PC1, PC2]
    into discrete integers 0-7 using KMeans.
    """
    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.models_dir = get_absolute_path("models_dir")
        self.kmeans_path = os.path.join(self.models_dir, "kmeans.joblib")

    def fit(self, states: np.ndarray) -> None:
        """
        Fits KMeans on customer states and saves the model binary.
        """
        logger.info(f"Fitting KMeans model with k={self.n_clusters}...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.kmeans.fit(states)
        joblib.dump(self.kmeans, self.kmeans_path)
        logger.info(f"KMeans model successfully saved to: {self.kmeans_path}")

    def load(self) -> None:
        """
        Loads the pre-fitted KMeans model from disk.
        """
        if not os.path.exists(self.kmeans_path):
            raise FileNotFoundError(f"KMeans model not found at: {self.kmeans_path}")
        self.kmeans = joblib.load(self.kmeans_path)
        logger.info(f"KMeans model loaded from: {self.kmeans_path}")

    def discretize(self, state: np.ndarray) -> int:
        """
        Predicts the discrete cluster index for a given continuous state.
        """
        if self.kmeans is None:
            self.load()
        state_reshaped = state.reshape(1, -1)
        return int(self.kmeans.predict(state_reshaped)[0])

class QLearningAgent:
    """
    Tabular Q-learning agent managing an 8x3 Q-table.
    """
    def __init__(
        self,
        state_size: int = 8,
        action_size: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize 8x3 Q-table to zeros
        self.q_table = np.zeros((state_size, action_size))
        self.models_dir = get_absolute_path("models_dir")
        self.q_table_path = os.path.join(self.models_dir, "q_table.joblib")

    def get_action(self, state_idx: int, train: bool = True) -> int:
        """
        Chooses an action based on epsilon-greedy policy.
        """
        if train and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return int(np.argmax(self.q_table[state_idx]))

    def update(self, state_idx: int, action: int, reward: float, next_state_idx: int) -> None:
        """
        Updates Q-values based on transition reward using standard Q-learning step.
        """
        best_next_action = np.argmax(self.q_table[next_state_idx])
        td_target = reward + self.gamma * self.q_table[next_state_idx, best_next_action]
        td_error = td_target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """
        Decays exploration rate.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self) -> None:
        """
        Saves Q-table weights to models directory.
        """
        joblib.dump(self.q_table, self.q_table_path)
        logger.info(f"Q-table successfully saved to: {self.q_table_path}")

    def load(self) -> None:
        """
        Loads Q-table weights from models directory.
        """
        if not os.path.exists(self.q_table_path):
            raise FileNotFoundError(f"Q-table binary not found at: {self.q_table_path}")
        self.q_table = joblib.load(self.q_table_path)
        logger.info(f"Q-table loaded from: {self.q_table_path}")

# --- PyTorch DQN Architecture ---
class QNetwork(nn.Module):
    """
    Standard PyTorch QNetwork: Input(5) -> Dense(64) -> ReLU -> Dense(64) -> ReLU -> Output(3)
    """
    def __init__(self, state_size: int = 5, action_size: int = 3):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayBuffer:
    """
    DQN Experience Replay Buffer using a double-ended queue.
    """
    def __init__(self, capacity: int = 2000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    """
    DQN Agent managing experience replay, target cloning, and SGD updates.
    """
    def __init__(
        self,
        state_size: int = 5,
        action_size: int = 3,
        lr: float = 1e-3,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99,
        epsilon_min: float = 0.05,
        buffer_size: int = 2000,
        batch_size: int = 64
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy & Target Networks
        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        
        self.models_dir = get_absolute_path("models_dir")
        self.dqn_path = os.path.join(self.models_dir, "dqn_model.pth")

    def get_action(self, state: np.ndarray, train: bool = True) -> int:
        """
        Chooses an action based on epsilon-greedy policy.
        """
        if train and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
            
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def update(self) -> float:
        """
        Executes a single SGD optimization step over a memory minibatch.
        """
        if len(self.memory) < self.batch_size:
            return 0.0
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q-values
        curr_q = self.policy_net(states_t).gather(1, actions_t)
        
        # Max next Q-values from Target Network
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            
        target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q
        
        loss = nn.MSELoss()(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return float(loss.item())

    def update_target_network(self) -> None:
        """
        Updates Target network weights by copying policy network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """
        Decays exploration rate.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self) -> None:
        """
        Saves policy model state dict to models directory.
        """
        torch.save(self.policy_net.state_dict(), self.dqn_path)
        logger.info(f"DQN weights successfully saved to: {self.dqn_path}")

    def load(self) -> None:
        """
        Loads policy network weights from models directory.
        """
        if not os.path.exists(self.dqn_path):
            raise FileNotFoundError(f"DQN model weights not found at: {self.dqn_path}")
        self.policy_net.load_state_dict(torch.load(self.dqn_path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        logger.info(f"DQN model weights loaded from: {self.dqn_path}")
