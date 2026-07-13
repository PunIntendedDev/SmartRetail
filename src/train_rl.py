"""
Reinforcement Learning Training Pipeline
----------------------------------------
Trains the K-Means state discretizer, the Tabular Q-learning agent, and the PyTorch
DQN agent on the processed training split. Saves metrics and outputs the combined
learning curves plot to outputs/figures/rl_reward_learning_curve.png.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.config import get_absolute_path, config_data
from src.environment import RetailCustomerEnv
from src.rl_agents import StateDiscretizer, QLearningAgent, DQNAgent
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def train_reinforcement_learning() -> None:
    """
    Executes the RL training pipeline:
    - Prepares the training customer recommendation environment.
    - Fits and serializes KMeans discretizer.
    - Pre-discretizes all states to speed up Q-learning.
    - Trains Q-learning agent for 500 episodes.
    - Trains DQN agent for 300 episodes with optimized updates.
    - Plots reward history comparisons.
    """
    # 1. Initialize environment
    logger.info("Initializing training environment...")
    env = RetailCustomerEnv(split="train")
    
    # 2. Fit and save K-Means discretizer
    discretizer = StateDiscretizer(n_clusters=8, random_state=42)
    discretizer.fit(env.states)
    
    # Pre-discretize all states for the training set to prevent slow KMeans lookups inside loops
    logger.info("Pre-discretizing all customer states for speed optimization...")
    train_state_indices = discretizer.kmeans.predict(env.states)
    
    # 3. Train Tabular Q-Learning Agent
    rl_config = config_data["reinforcement_learning"]
    q_params = rl_config["tabular_q"]
    
    logger.info("Initializing Tabular Q-learning agent...")
    q_agent = QLearningAgent(
        state_size=8,
        action_size=3,
        alpha=float(q_params["alpha"]),
        gamma=float(q_params["gamma"]),
        epsilon=float(q_params["epsilon_start"]),
        epsilon_decay=float(q_params["epsilon_decay"]),
        epsilon_min=float(q_params["epsilon_min"])
    )
    
    q_episodes = int(q_params["episodes"])
    q_rewards = []
    
    logger.info(f"Training Tabular Q-learning agent for {q_episodes} episodes...")
    for ep in range(1, q_episodes + 1):
        env.reset()
        total_reward = 0.0
        
        # Single epoch pass over training customers
        for idx in range(env.num_customers):
            state_idx = train_state_indices[idx]
            action = q_agent.get_action(state_idx, train=True)
            next_state, reward, done, info = env.step(action)
            
            # Next state index is known statically from index + 1
            next_state_idx = 0 if done else train_state_indices[idx + 1]
            
            q_agent.update(state_idx, action, reward, next_state_idx)
            total_reward += reward
            
        q_agent.decay_epsilon()
        q_rewards.append(total_reward)
        
        if ep % 50 == 0 or ep == 1:
            logger.info(f"  Episode {ep:3d}/{q_episodes} | Total Reward: {total_reward:10.2f} | Epsilon: {q_agent.epsilon:.3f}")
            
    q_agent.save()
    
    # 4. Train Deep Q-Network (DQN) Agent
    dqn_params = rl_config["dqn"]
    
    logger.info("Initializing PyTorch DQN agent...")
    dqn_agent = DQNAgent(
        state_size=5,
        action_size=3,
        lr=float(dqn_params["learning_rate"]),
        gamma=float(dqn_params["gamma"]),
        epsilon=float(dqn_params["epsilon_start"]),
        epsilon_decay=float(dqn_params["epsilon_decay"]),
        epsilon_min=float(dqn_params["epsilon_min"]),
        buffer_size=int(dqn_params["buffer_size"]),
        batch_size=int(dqn_params["batch_size"])
    )
    
    dqn_episodes = int(dqn_params["episodes"])
    target_update_frequency = int(dqn_params["target_update_frequency"])
    dqn_rewards = []
    
    logger.info(f"Training PyTorch DQN agent for {dqn_episodes} episodes...")
    for ep in range(1, dqn_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        step_count = 0
        
        while not done:
            action = dqn_agent.get_action(state, train=True)
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            dqn_agent.memory.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Perform optimization step once every 16 steps to avoid CPU training bottleneck
            if step_count % 16 == 0:
                dqn_agent.update()
            
        dqn_agent.decay_epsilon()
        dqn_rewards.append(total_reward)
        
        # Periodic Target Network Updates
        if ep % target_update_frequency == 0:
            dqn_agent.update_target_network()
            
        if ep % 50 == 0 or ep == 1:
            logger.info(f"  Episode {ep:3d}/{dqn_episodes} | Total Reward: {total_reward:10.2f} | Epsilon: {dqn_agent.epsilon:.3f}")
            
    dqn_agent.save()
    
    # 5. Generate and save learning curve plot
    logger.info("Plotting reward learning curves...")
    figures_dir = get_absolute_path("figures_dir")
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, q_episodes + 1), q_rewards, label="Tabular Q-Learning", color="#10b981", alpha=0.8)
    plt.plot(range(1, dqn_episodes + 1), dqn_rewards, label="Deep Q-Network (DQN)", color="#2563eb", alpha=0.8)
    plt.xlabel("Episodes")
    plt.ylabel("Total Episode Reward")
    plt.title("Reinforcement Learning Reward Learning Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    curve_path = os.path.join(figures_dir, "rl_reward_learning_curve.png")
    plt.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Learning curve successfully saved to: {curve_path}")
    logger.info("Reinforcement Learning training pipeline completed successfully.")
    
    # Run test set evaluation
    evaluate_policies_on_test()

def evaluate_policies_on_test() -> None:
    """
    Evaluates the trained Tabular Q-learning policy, the DQN policy,
    the always-No-Action baseline, and the random-action baseline on held-out test customers.
    Saves the comparative metrics to a JSON file and a bar chart.
    """
    import json
    logger.info("Starting RL Policy evaluation on held-out test split...")
    env_test = RetailCustomerEnv(split="test")
    
    # 1. Load trained agents & discretizer
    q_agent = QLearningAgent(state_size=8, action_size=3)
    q_agent.load()
    
    discretizer = StateDiscretizer(n_clusters=8)
    discretizer.load()
    
    dqn_agent = DQNAgent(state_size=5, action_size=3)
    dqn_agent.load()
    
    # 2. Evaluate DQN Policy
    env_test.reset()
    dqn_total_profit = 0.0
    for idx in range(env_test.num_customers):
        state = env_test.states[idx]
        action = dqn_agent.get_action(state, train=False)
        _, reward, _, _ = env_test.step(action)
        dqn_total_profit += reward
        
    # 3. Evaluate Tabular Q Policy
    env_test.reset()
    q_total_profit = 0.0
    for idx in range(env_test.num_customers):
        state = env_test.states[idx]
        state_idx = discretizer.discretize(state)
        action = q_agent.get_action(state_idx, train=False)
        _, reward, _, _ = env_test.step(action)
        q_total_profit += reward
        
    # 4. Evaluate Always No-Action
    env_test.reset()
    no_action_total_profit = 0.0
    for idx in range(env_test.num_customers):
        _, reward, _, _ = env_test.step(0)
        no_action_total_profit += reward
        
    # 5. Evaluate Random-Action
    env_test.reset()
    np.random.seed(42)
    random_total_profit = 0.0
    for idx in range(env_test.num_customers):
        action = np.random.choice([0, 1, 2])
        _, reward, _, _ = env_test.step(action)
        random_total_profit += reward

    logger.info(f"DQN Profit: ${dqn_total_profit:,.2f}")
    logger.info(f"Tabular Q Profit: ${q_total_profit:,.2f}")
    logger.info(f"Always No-Action Profit: ${no_action_total_profit:,.2f}")
    logger.info(f"Random Action Profit: ${random_total_profit:,.2f}")
    
    # Save comparative metrics to JSON
    results = {
        "DQN_Policy": dqn_total_profit,
        "Tabular_Q_Policy": q_total_profit,
        "Always_No_Action": no_action_total_profit,
        "Random_Action": random_total_profit
    }
    
    processed_dir = get_absolute_path("processed_data_dir")
    eval_json_path = os.path.join(processed_dir, "rl_evaluation_results.json")
    with open(eval_json_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"RL policy comparison metrics saved to: {eval_json_path}")
    
    # Save the comparative bar chart
    figures_dir = get_absolute_path("figures_dir")
    plt.figure(figsize=(8, 5))
    policies = ["Always No-Action", "Random Action", "Tabular Q-Learning", "DQN Recommendation"]
    profits = [no_action_total_profit, random_total_profit, q_total_profit, dqn_total_profit]
    colors = ["#ef4444", "#f59e0b", "#10b981", "#2563eb"]
    
    plt.bar(policies, profits, color=colors, edgecolor="black", alpha=0.85)
    plt.ylabel("Total Profit ($)")
    plt.title("RL Policy Profit Comparison on Test Customers")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Formatting values on top of bars
    for i, v in enumerate(profits):
        plt.text(i, v + (max(profits)*0.01), f"${v:,.0f}", ha='center', fontweight='bold')
        
    bar_path = os.path.join(figures_dir, "rl_profit_comparison.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"RL comparison bar chart saved to: {bar_path}")

if __name__ == "__main__":
    train_reinforcement_learning()
