"""
Reinforcement Learning Environment Module
------------------------------------------
Implements the RetailCustomerEnv class representing the customer recommendation
environment. States are defined as [Recency, Frequency, Monetary, PC1, PC2].
Rewards are calculated using predicted future spend from best_regressor.joblib,
action multipliers, and action costs.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

# Suppress scikit-learn feature name mismatch warnings for numpy inputs
warnings.filterwarnings("ignore", category=UserWarning)

from src.config import get_absolute_path, config_data
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class RetailCustomerEnv:
    """
    Simulates recommendations for retail customers.
    One episode iterates through all customers in the split exactly once.
    State representation: [Recency, Frequency, Monetary, PC1, PC2].
    """
    def __init__(self, split: str = "train"):
        """
        Initializes the environment by loading processed data splits and the best regressor.

        Args:
            split: 'train', 'validation', or 'test'.
        """
        self.split = split
        processed_dir = get_absolute_path("processed_data_dir")
        models_dir = get_absolute_path("models_dir")
        
        # Load the pre-trained regressor
        regressor_path = os.path.join(models_dir, "best_regressor.joblib")
        if not os.path.exists(regressor_path):
            raise FileNotFoundError(f"Best regressor model not found at: {regressor_path}")
        self.regressor = joblib.load(regressor_path)
        
        # Load PCA feature splits (holds the 5D state columns: Recency, Frequency, Monetary, PC1, PC2)
        pca_filepath = os.path.join(processed_dir, f"{split}_pca.csv")
        if not os.path.exists(pca_filepath):
            raise FileNotFoundError(f"PCA-transformed feature splits not found at: {pca_filepath}")
        self.pca_df = pd.read_csv(pca_filepath)
        if "CustomerID" in self.pca_df.columns:
            self.pca_df = self.pca_df.set_index("CustomerID")
            
        # Load Raw Standardized feature splits (holds the 13 features expected by best_regressor)
        raw_scaled_filepath = os.path.join(processed_dir, f"{split}_raw_scaled.csv")
        if not os.path.exists(raw_scaled_filepath):
            raise FileNotFoundError(f"Raw standardized feature splits not found at: {raw_scaled_filepath}")
        self.raw_scaled_df = pd.read_csv(raw_scaled_filepath)
        if "CustomerID" in self.raw_scaled_df.columns:
            self.raw_scaled_df = self.raw_scaled_df.set_index("CustomerID")
            
        # Ensure indices align perfectly
        self.pca_df = self.pca_df.loc[self.raw_scaled_df.index]
        
        # Extract customer IDs
        self.customer_ids = self.pca_df.index.values
        self.num_customers = len(self.customer_ids)
        
        # Define 5D state variables
        state_cols = ["Recency", "Frequency", "Monetary", "PC1", "PC2"]
        self.states = self.pca_df[state_cols].values
        
        # Define 13D regressor input variables
        exclude_cols = ["High_Value_Customer", "Future_Spend"]
        regressor_cols = [col for col in self.raw_scaled_df.columns if col not in exclude_cols]
        self.regressor_features = self.raw_scaled_df[regressor_cols].values
        
        # Set Q-learning and DQN Action attributes from configuration
        rl_config = config_data["reinforcement_learning"]
        self.action_multipliers = {
            0: float(rl_config["adjustments"][0]),
            1: float(rl_config["adjustments"][1]),
            2: float(rl_config["adjustments"][2])
        }
        self.action_costs = {
            0: float(rl_config["costs"][0]),
            1: float(rl_config["costs"][1]),
            2: float(rl_config["costs"][2])
        }
        
        # Pre-calculate future spend predictions for all customers to optimize loops
        self.predicted_spends = self.regressor.predict(self.regressor_features)
        
        self.current_index = 0
        logger.info(f"RetailCustomerEnv initialized on '{split}' split with {self.num_customers} customers.")

    def reset(self) -> np.ndarray:
        """
        Resets environment to the first customer in the split.

        Returns:
            State vector of the first customer.
        """
        self.current_index = 0
        return self.states[self.current_index]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Simulates taking an action for the current customer:
        - Predicts future spend.
        - Computes reward based on configuration multipliers and costs.
        - Moves to the next customer.

        Args:
            action: Action index (0: No Action, 1: 10% Discount Coupon, 2: Free Premium Trial)

        Returns:
            next_state: The state vector of the next customer (or zeros if done).
            reward: Calculated reward.
            done: Boolean flag indicating if all customers in the split have been evaluated.
            info: Diagnostics dictionary.
        """
        if self.current_index >= self.num_customers:
            raise IndexError("Environment step called after episode completed (done=True). Call reset() first.")
            
        # Predict future spend using the 13 raw scaled features
        predicted_future_spend = self.predicted_spends[self.current_index]
        
        # Calculate Reward: (predicted_future_spend * multiplier) - cost
        multiplier = self.action_multipliers[action]
        cost = self.action_costs[action]
        reward = (predicted_future_spend * multiplier) - cost
        
        # Advance index to the next customer
        self.current_index += 1
        
        if self.current_index >= self.num_customers:
            done = True
            next_state = np.zeros(5)
        else:
            done = False
            next_state = self.states[self.current_index]
            
        info = {
            "customer_id": self.customer_ids[self.current_index - 1],
            "predicted_future_spend": predicted_future_spend,
            "action_reward": reward
        }
        
        return next_state, reward, done, info
