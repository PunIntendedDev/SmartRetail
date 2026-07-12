"""
Plotting Utilities
------------------
Contains functions to programmatically generate and save the 15 publication-quality
figures requested in the project specifications. All figures are styled consistently,
labeled, and saved under outputs/figures/.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

# Set styling defaults
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "savefig.dpi": 300,
    "figure.autolayout": True
})

def save_fig(fig: plt.Figure, filename: str) -> str:
    """
    Saves a matplotlib figure under outputs/figures/ and closes it.

    Args:
        fig: Matplotlib Figure object.
        filename: Name of the output image file (e.g. 'roc_curve.png').

    Returns:
        The absolute path to the saved figure.
    """
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "figures"))
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path

# 1. Data Cleaning Summary
def plot_data_cleaning_summary(before_count: int, after_count: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ["Raw Transactions", "Cleaned Transactions"]
    counts = [before_count, after_count]
    
    bars = ax.bar(categories, counts, color=["#e74c3c", "#2ecc71"], width=0.5)
    ax.set_ylabel("Number of Records")
    ax.set_title("Data Cleaning Impact on Transaction Count")
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:,}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom")
                    
    save_fig(fig, "data_cleaning_summary.png")

# 2. Feature Distributions
def plot_feature_distributions(df: pd.DataFrame, columns: List[str]) -> None:
    n_cols = len(columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if i >= len(axes):
            break
        sns.histplot(df[col], ax=axes[i], kde=True, color="#3498db", bins=30)
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        
        # Apply log scale for highly skewed distributions (e.g. Monetary, Frequency)
        if col in ["Monetary", "Frequency", "AverageSpend", "ProductDiversity"]:
            axes[i].set_yscale("log")
            axes[i].set_ylabel("Count (Log Scale)")
            
    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle("Customer-Level Feature Distributions", y=0.98)
    save_fig(fig, "feature_distributions.png")

# 3. Correlation Heatmap
def plot_correlation_heatmap(df: pd.DataFrame, columns: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[columns].corr()
    
    sns.heatmap(
        corr, 
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f", 
        linewidths=0.5, 
        ax=ax, 
        vmin=-1, 
        vmax=1,
        square=True
    )
    ax.set_title("Correlation Heatmap of Customer Features")
    save_fig(fig, "correlation_heatmap.png")

# 4. PCA Explained Variance
def plot_pca_explained_variance(explained_variance_ratio: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    cum_var = np.cumsum(explained_variance_ratio)
    components = np.arange(1, len(explained_variance_ratio) + 1)
    
    ax.bar(components, explained_variance_ratio, alpha=0.6, align="center", label="Individual Variance", color="#9b59b6")
    ax.step(components, cum_var, where="mid", label="Cumulative Variance", color="#8e44ad", linewidth=2)
    
    # 90% threshold line
    ax.axhline(y=0.9, color="r", linestyle="--", label="90% Threshold")
    
    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Cumulative Explained Variance")
    ax.set_xticks(components)
    ax.legend(loc="best")
    save_fig(fig, "pca_explained_variance.png")

# 5. PCA Scatter Plot
def plot_pca_scatter(df: pd.DataFrame, pc1_col: str, pc2_col: str, label_col: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color-code High-Value vs Low-Value
    sns.scatterplot(
        data=df, 
        x=pc1_col, 
        y=pc2_col, 
        hue=label_col, 
        palette={0: "#34495e", 1: "#f1c40f"},
        alpha=0.7, 
        ax=ax
    )
    
    ax.set_xlabel("Principal Component 1 (PC1)")
    ax.set_ylabel("Principal Component 2 (PC2)")
    ax.set_title("Customer Segments in PCA Space")
    
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ["Low Value" if l == "0" or l == "0.0" else "High Value" for l in labels]
    ax.legend(handles, new_labels, title="Customer Category")
    save_fig(fig, "pca_scatter_plot.png")

# 6. LDA Projection
def plot_lda_projection(df: pd.DataFrame, lda_col: str, label_col: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    
    sns.histplot(
        data=df, 
        x=lda_col, 
        hue=label_col, 
        kde=True, 
        palette={0: "#34495e", 1: "#2ecc71"},
        alpha=0.6,
        bins=30,
        multiple="stack",
        ax=ax
    )
    
    ax.set_xlabel("Linear Discriminant Score (LD1)")
    ax.set_ylabel("Count")
    ax.set_title("Customer Segmentation LDA Projection")
    
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ["Low Value" if l == "0" or l == "0.0" else "High Value" for l in labels]
    ax.legend(handles, new_labels, title="Customer Category")
    save_fig(fig, "lda_projection.png")

# 7. Confusion Matrix
def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["Low Value", "High Value"],
        yticklabels=["Low Value", "High Value"], 
        ax=ax, 
        cbar=False
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix - {model_name}")
    save_fig(fig, "confusion_matrix.png")

# 8. ROC Curve
def plot_roc_curve(roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]]) -> None:
    """
    Plots multiple ROC curves.
    roc_curves: Dict of model_name -> (fpr, tpr, roc_auc)
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    for model_name, (fpr, tpr, auc_val) in roc_curves.items():
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.3f})", linewidth=2)
        
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    save_fig(fig, "roc_curve.png")

# 9. Precision-Recall Curve
def plot_precision_recall_curve(pr_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]]) -> None:
    """
    Plots multiple Precision-Recall curves.
    pr_curves: Dict of model_name -> (precision, recall, avg_precision)
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    for model_name, (precision, recall, avg_prec) in pr_curves.items():
        ax.plot(recall, precision, label=f"{model_name} (AP = {avg_prec:.3f})", linewidth=2)
        
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    save_fig(fig, "precision_recall_curve.png")

# 10. Regression Predicted vs Actual
def plot_regression_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.scatter(y_true, y_pred, alpha=0.5, color="#16a085")
    
    # Perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")
    
    ax.set_xlabel("Actual Future Spend ($)")
    ax.set_ylabel("Predicted Future Spend ($)")
    ax.set_title(f"Predicted vs Actual Future Spend ({model_name})")
    ax.legend(loc="best")
    save_fig(fig, "regression_predicted_vs_actual.png")

# 11. Regression Residuals
def plot_regression_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    residuals = y_true - y_pred
    
    ax.scatter(y_pred, residuals, alpha=0.5, color="#d35400")
    ax.axhline(y=0, color="black", linestyle="--")
    
    ax.set_xlabel("Predicted Spend ($)")
    ax.set_ylabel("Residuals ($)")
    ax.set_title(f"Residual Plot - {model_name}")
    save_fig(fig, "regression_residual_plot.png")

# 12. Reward vs Episode
def plot_rl_reward_vs_episode(tabular_rewards: List[float], dqn_rewards: List[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Use rolling averages to smooth reward curves
    def rolling_avg(arr: List[float], window: int = 20) -> np.ndarray:
        return pd.Series(arr).rolling(window=window, min_periods=1).mean().values
        
    ax.plot(rolling_avg(tabular_rewards), label="Tabular Q-Learning (Smoothed)", color="#3498db")
    ax.plot(rolling_avg(dqn_rewards), label="Deep Q-Network (DQN, Smoothed)", color="#2ecc71")
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Reinforcement Learning Training Performance")
    ax.legend(loc="best")
    save_fig(fig, "rl_reward_vs_episode.png")

# 13. Q-value Comparison
def plot_rl_qvalues(q_values_matrix: np.ndarray, cluster_names: List[str], action_names: List[str]) -> None:
    """
    Plots a heatmap comparing Q-values of actions for different clusters.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        q_values_matrix, 
        annot=True, 
        cmap="YlGnBu", 
        fmt=".2f",
        xticklabels=action_names, 
        yticklabels=cluster_names, 
        ax=ax
    )
    ax.set_title("Tabular Q-Learning Q-Value Heatmap (State Clusters x Actions)")
    ax.set_xlabel("Action")
    ax.set_ylabel("Discretized State (K-Means Clusters)")
    save_fig(fig, "rl_qvalue_comparison.png")

# 14. RL Policy Comparison
def plot_rl_policy_comparison(action_distributions: Dict[str, List[int]], action_names: List[str]) -> None:
    """
    Compares how different policies allocate actions across the test set.
    action_distributions: Dict of policy_name -> list of action counts matching action_names
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    policies = list(action_distributions.keys())
    x = np.arange(len(action_names))
    width = 0.8 / len(policies)
    
    for i, policy in enumerate(policies):
        ax.bar(x + i * width, action_distributions[policy], width, label=policy)
        
    ax.set_xticks(x + width * (len(policies) - 1) / 2)
    ax.set_xticklabels(action_names)
    ax.set_xlabel("Actions")
    ax.set_ylabel("Count")
    ax.set_title("Action Recommendation Distribution by Policy")
    ax.legend(loc="best")
    save_fig(fig, "rl_policy_comparison.png")

# 15. Profit Comparison
def plot_profit_comparison(profits: Dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    policies = list(profits.keys())
    values = list(profits.values())
    
    colors = ["#95a5a6" if "Action 0" in p or "No Action" in p else "#3498db" for p in policies]
    # Highlight DQN and Tabular Q
    for i, p in enumerate(policies):
        if "DQN" in p:
            colors[i] = "#2ecc71"
        elif "Q-Learning" in p or "Tabular" in p:
            colors[i] = "#27ae60"
            
    bars = ax.bar(policies, values, color=colors, width=0.5)
    ax.set_ylabel("Expected Profit ($)")
    ax.set_title("Expected Policy Profit Comparison (Test Set)")
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"${height:,.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom")
                    
    save_fig(fig, "profit_comparison_bar_chart.png")
