# SmartRetail Intelligence Platform

> **AI-Powered Customer Segmentation & Personalised Promotion Recommendation System**  
> Built on the [UCI Online Retail II dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) using RFM feature engineering, dimensionality reduction, supervised classification & regression, and deep reinforcement learning.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Setup & Installation](#4-setup--installation)
5. [Running the Full Pipeline](#5-running-the-full-pipeline)
6. [Jupyter Notebooks](#6-jupyter-notebooks)
7. [Streamlit Dashboard](#7-streamlit-dashboard)
8. [Configuration](#8-configuration)
9. [Outputs](#9-outputs)

---

## 1. Project Overview

SmartRetail processes raw e-commerce transaction logs into a customer intelligence pipeline with five phases:

| Phase | Module | Description |
|---|---|---|
| **0 — Preprocessing** | `src/preprocessing.py` | RFM extraction, category mapping, train/val/test splits |
| **1 — Dimensionality Reduction** | `src/dimensionality_reduction.py` | StandardScaler → PCA (≥90% variance) → LDA |
| **2 — Classification** | `src/classification.py` | Logistic Regression (CV) + MLPClassifier → high-value vs. standard segment |
| **3 — Regression** | `src/regression.py` | Linear Regression + MLPRegressor → future spend prediction |
| **4 — Reinforcement Learning** | `src/train_rl.py` | Tabular Q-Learning + DQN → personalised promotion policy |

---

## 2. Architecture

```
Raw Transactions
      │
      ▼
 Preprocessing ──► RFM Features + Category Mix + Target Spend
      │
      ▼
 PCA + LDA ──► Compressed Feature Space (5D → PC1…PC5, LD1)
      │
      ├──► Classifier ──► Customer Segment (High-Value / Standard)
      │
      └──► Regressor  ──► Predicted Future Spend ($)
                │
                ▼
         RL Environment (RetailCustomerEnv)
                │
         ┌──────┴──────┐
         ▼             ▼
   Tabular Q-Learning  DQN (PyTorch)
         │             │
         └──────┬──────┘
                ▼
        Streamlit Dashboard
```

### RL Action Space

| Action | Promotion | Cost |
|---|---|---|
| 0 | No Action | $0.00 |
| 1 | 10% Discount Coupon | $1.00 |
| 2 | Free Premium Trial | $5.00 |

### Product Categories (Section 3.4)

Priority order for keyword classification: **Homeware → Stationery → Gadgets → Decorations → Kitchenware**

---

## 3. Repository Structure

```
SmartRetail/
├── app.py                          # Streamlit dashboard entry point
├── requirements.txt                # Python dependencies
├── category_map.json               # Reproducible CustomerID → category mapping
├── config/
│   └── config.yaml                 # All hyperparameters and paths
├── data/
│   ├── raw/                        # Downloaded XLSX from UCI
│   └── processed/                  # train.csv, validation.csv, test.csv, rl_evaluation_results.json
├── models/                         # Serialised models (.joblib, .npy, .pt)
│   ├── scaler.joblib
│   ├── pca.joblib
│   ├── lda.joblib
│   ├── kmeans.joblib
│   ├── logistic_regression.joblib
│   ├── mlp_classifier.joblib
│   ├── linear_regression.joblib
│   ├── mlp_regressor.joblib
│   ├── q_table.npy
│   └── dqn_model.pt
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_lda.ipynb
│   ├── 03_classification.ipynb
│   ├── 04_regression.ipynb
│   └── 05_qlearning_dqn.ipynb
├── outputs/
│   └── figures/                    # All saved charts (.png)
├── src/
│   ├── config.py                   # Path helpers and YAML loader
│   ├── preprocessing.py            # Data cleaning and RFM pipeline
│   ├── dimensionality_reduction.py # PCA/LDA training and transform
│   ├── classification.py           # Classifier training and evaluation
│   ├── regression.py               # Regressor training and evaluation
│   ├── environment.py              # RetailCustomerEnv (OpenAI Gym-style)
│   ├── rl_agents.py                # QLearningAgent + DQNAgent (PyTorch)
│   ├── train_rl.py                 # RL training + baseline evaluation
│   └── download_data.py            # UCI dataset downloader
└── utils/
    └── logging_utils.py            # Structured logger
```

---

## 4. Setup & Installation

### Prerequisites
- Python **3.10+**
- Windows / macOS / Linux
- ~2 GB free disk space (for virtual environment + dataset)

### Step 1 — Clone and Enter the Repository
```powershell
git clone <repo-url>
cd SmartRetail
```

### Step 2 — Create a Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # macOS / Linux
```

### Step 3 — Install Dependencies
```powershell
pip install -r requirements.txt
```

> **Note:** PyTorch (`torch`) is included in `requirements.txt` and installs the CPU build by default. For CUDA-enabled GPU training, replace the torch line with the appropriate wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## 5. Running the Full Pipeline

All commands must be run from the `SmartRetail/` root directory with the virtual environment active.

### Step 1 — Download Raw Data
```powershell
python -m src.download_data
```

### Step 2 — Preprocess & Generate Splits
```powershell
python -m src.preprocessing
```
*Outputs:* `data/processed/train.csv`, `validation.csv`, `test.csv`, `category_map.json`

### Step 3 — Dimensionality Reduction
```powershell
python -m src.dimensionality_reduction
```
*Outputs:* `models/scaler.joblib`, `pca.joblib`, `lda.joblib`

### Step 4 — Train Classifiers
```powershell
python -m src.classification
```
*Outputs:* `models/logistic_regression.joblib`, `mlp_classifier.joblib`

### Step 5 — Train Regressors
```powershell
python -m src.regression
```
*Outputs:* `models/linear_regression.joblib`, `mlp_regressor.joblib`

### Step 6 — Train RL Agents + Evaluate Baselines
```powershell
python -m src.train_rl
```
*Outputs:* `models/q_table.npy`, `dqn_model.pt`, `kmeans.joblib`,  
`outputs/figures/rl_reward_learning_curve.png`, `rl_profit_comparison.png`,  
`data/processed/rl_evaluation_results.json`

---

## 6. Jupyter Notebooks

Notebooks are located in the `notebooks/` directory and mirror the five pipeline phases. Run them sequentially for a guided walkthrough.

| Notebook | Phase | Description |
|---|---|---|
| [`01_data_preprocessing.ipynb`](notebooks/01_data_preprocessing.ipynb) | Phase 0 | Data cleaning, RFM extraction, category mapping, split generation |
| [`02_pca_lda.ipynb`](notebooks/02_pca_lda.ipynb) | Phase 1 | PCA cumulative variance plot, LDA discriminant projection |
| [`03_classification.ipynb`](notebooks/03_classification.ipynb) | Phase 2 | Cross-validated Logistic Regression, MLP, confusion matrices, ROC curves |
| [`04_regression.ipynb`](notebooks/04_regression.ipynb) | Phase 3 | Linear Regression, MLP regressor, predicted vs. actual scatter plots |
| [`05_qlearning_dqn.ipynb`](notebooks/05_qlearning_dqn.ipynb) | Phase 4 | Q-table training, DQN training, reward curves, baseline comparison chart |

### Running Notebooks

```powershell
pip install jupyter
jupyter notebook
```

Then navigate to the `notebooks/` folder in the Jupyter interface.

> **Important:** Notebooks import from `src/`. The `sys.path.insert(0, '..')` call at the top of each notebook handles this automatically when launched from within the `notebooks/` folder.

---

## 7. Streamlit Dashboard

```powershell
streamlit run app.py
```

The dashboard will open at **http://localhost:8501** and includes:

| Section | Features |
|---|---|
| **Sidebar** | Global Customer ID dropdown (from test split) |
| **Category Mix** | Plotly pie chart of the selected customer's purchase categories |
| **PCA Projection** | PC1 vs PC2 scatter with selected customer highlighted |
| **LDA Projection** | LD1 distribution with customer position marked |
| **Segment & Spend** | Predicted segment (Logistic / MLP) + predicted future spend with RMSE margin |
| **RL Recommendations** | Tabular Q and DQN Q-value bars + recommended action per model |
| **Policy Comparison** | Cumulative profit bar chart (DQN vs Q vs No-Action vs Random) |

---

## 8. Configuration

All hyperparameters are controlled via [`config/config.yaml`](config/config.yaml):

```yaml
data_split:
  train_size: 0.8
  val_size:   0.1
  test_size:  0.1
  random_state: 42

pca:
  min_variance_ratio: 0.90   # Retain enough components to explain ≥90% variance

classification:
  cv_folds: 5
  mlp_hidden_layers: [32, 16]

reinforcement_learning:
  state_clusters: 8           # KMeans clusters for state discretisation
  tabular_q:
    episodes: 500
    alpha: 0.1
    gamma: 0.9
  dqn:
    episodes: 300
    learning_rate: 0.001
    gamma: 0.9
    buffer_size: 2000
    batch_size: 64
    target_update_frequency: 20
```

---

## 9. Outputs

| File | Description |
|---|---|
| `outputs/figures/rl_reward_learning_curve.png` | Tabular Q and DQN episode reward curves |
| `outputs/figures/rl_profit_comparison.png` | Test-set policy baseline bar chart |
| `outputs/figures/pca_explained_variance.png` | Cumulative PCA explained variance |
| `outputs/figures/classification_*.png` | Confusion matrices, ROC/PR curves |
| `outputs/figures/regression_*.png` | Predicted vs. actual scatter plots |
| `data/processed/rl_evaluation_results.json` | JSON of cumulative profits per policy |
| `category_map.json` | Reproducible CustomerID → category mapping |

---

*SmartRetail — AI-powered retail personalisation.*
