"""
SmartRetail AI - SaaS Analytics Platform
---------------------------------------
A commercial-grade SaaS intelligence platform built on the UCI Online Retail database.
Implements modern card-based UI, prediction inference pipelines,
and diagnostic audit views. Fully offline (loads saved binaries).
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from typing import Dict, Any, Optional

from src.rl_agents import StateDiscretizer, QLearningAgent, DQNAgent

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartRetail AI Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

# -----------------------------------------------------------------------------
# LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_estimator(filename: str) -> Optional[Any]:
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_dataset(filename: str) -> Optional[pd.DataFrame]:
    path = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "CustomerID" in df.columns:
            df = df.set_index("CustomerID")
        return df
    return None

@st.cache_data
def load_json_file(path: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# Load datasets
train_df = load_dataset("train.csv")
validation_df = load_dataset("validation.csv")
test_df = load_dataset("test.csv")

train_pca_df = load_dataset("train_pca.csv")
test_pca_df = load_dataset("test_pca.csv")

train_lda_df = load_dataset("train_lda.csv")
test_lda_df = load_dataset("test_lda.csv")

test_raw_scaled_df = load_dataset("test_raw_scaled.csv")

# Load models and results
best_classifier = load_estimator("best_classifier.joblib")
best_regressor = load_estimator("best_regressor.joblib")
scaler_raw = load_estimator("scaler.joblib")
scaler_pca = load_estimator("pca_scaler.joblib")
scaler_ns = load_estimator("non_spend_scaler.joblib")
pca_model = load_estimator("pca.joblib")
lda_model = load_estimator("lda.joblib")

class_results = load_json_file(os.path.join(PROCESSED_DIR, "classification_results.json"))
reg_results = load_json_file(os.path.join(PROCESSED_DIR, "regression_results.json"))
pca_metadata = load_json_file(os.path.join(PROCESSED_DIR, "pca_metadata.json"))
metadata = load_json_file(os.path.join(PROCESSED_DIR, "metadata.json"))
rl_eval_results = load_json_file(os.path.join(PROCESSED_DIR, "rl_evaluation_results.json"))

# Load RL Agents
@st.cache_resource
def load_rl_agents():
    try:
        dqn = DQNAgent(state_size=5, action_size=3)
        dqn.load()
        q_agent = QLearningAgent(state_size=8, action_size=3)
        q_agent.load()
        discretizer = StateDiscretizer(n_clusters=8)
        discretizer.load()
        return dqn, q_agent, discretizer
    except Exception as e:
        st.warning(f"RL agent binaries loading failed: {str(e)}")
        return None, None, None

dqn_agent, q_agent, discretizer = load_rl_agents()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style='margin-top: 8px; margin-bottom: 16px; border-bottom: 1px solid #1f2937; padding-bottom: 10px;'>
        <div style='font-size: 1.3rem; font-weight: 800; color: #60a5fa;'>🛍️ SmartRetail AI</div>
        <div style='font-size: 0.72rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px;'>Customer Behavior Analytics</div>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "SELECT PAGE:",
    [
        "🏠 Dashboard",
        "👤 Customer Profile",
        "📈 Model Performance",
        "🧠 PCA & LDA",
        "📊 Dataset Explorer",
        "ℹ About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 CUSTOMER SELECTION")
if test_df is not None:
    customer_ids = sorted([int(cid) for cid in test_df.index.tolist()])
    selected_customer_id = st.sidebar.selectbox("Select Customer ID (Test Split):", customer_ids)
else:
    st.sidebar.warning("Datasets not loaded. Run preprocessing first.")
    selected_customer_id = None

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 UI Preferences")
theme_mode = st.sidebar.radio("Theme", ["Light", "Dark"], index=0, horizontal=True)
compact_mode = st.sidebar.toggle("Compact mode", value=False)

# Theme styling configuration
if theme_mode == "Dark":
    TOKENS = {
        "bg": "#0b1220",
        "surface": "#111827",
        "text": "#e5e7eb",
        "muted": "#9ca3af",
        "primary": "#3b82f6",
        "primary_dark": "#2563eb",
        "border": "#1f2937",
        "shadow": "0 8px 26px rgba(0,0,0,0.35)",
        "hero_a": "#1e3a8a",
        "hero_b": "#2563eb",
        "plot_bg": "#0f172a",
        "sidebar_bg": "#020617",
        "sidebar_border": "#0f172a",
        "badge_bg": "#1f2937",
        "badge_text": "#c7d2fe",
        "badge_border": "#374151"
    }
else:
    TOKENS = {
        "bg": "#f5f7fb",
        "surface": "#ffffff",
        "text": "#111827",
        "muted": "#6b7280",
        "primary": "#2563eb",
        "primary_dark": "#1e40af",
        "border": "#e5e7eb",
        "shadow": "0 6px 20px rgba(15, 23, 42, 0.06)",
        "hero_a": "#1d4ed8",
        "hero_b": "#2563eb",
        "plot_bg": "#f8fafc",
        "sidebar_bg": "#0f172a",
        "sidebar_border": "#1e293b",
        "badge_bg": "#eef2ff",
        "badge_text": "#3730a3",
        "badge_border": "#c7d2fe"
    }

card_pad = "12px" if compact_mode else "18px"
card_radius = "12px" if compact_mode else "16px"
hero_pad = "20px" if compact_mode else "30px"
h1_size = "1.7rem" if compact_mode else "2.1rem"
kpi_size = "1.15rem" if compact_mode else "1.45rem"
subheader_size = "0.88rem" if compact_mode else "0.95rem"

st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
    :root {{
        --bg: {TOKENS["bg"]};
        --surface: {TOKENS["surface"]};
        --text: {TOKENS["text"]};
        --muted: {TOKENS["muted"]};
        --primary: {TOKENS["primary"]};
        --primary-dark: {TOKENS["primary_dark"]};
        --border: {TOKENS["border"]};
        --shadow: {TOKENS["shadow"]};
        --radius: {card_radius};
        --plot-bg: {TOKENS["plot_bg"]};
    }}

    html, body, [class*="css"], .stMarkdown {{
        font-family: 'Inter', sans-serif !important;
        color: var(--text) !important;
    }}

    .stApp {{
        background: var(--bg);
    }}

    #MainMenu, footer {{visibility: hidden;}}
    .stAppDeployButton {{display: none;}}

    header[data-testid="stHeader"] {{
        visibility: visible !important;
        background: transparent !important;
    }}

    section[data-testid="stSidebar"] {{
        background: {TOKENS["sidebar_bg"]} !important;
        border-right: 1px solid {TOKENS["sidebar_border"]} !important;
        min-width: 300px !important;
        max-width: 300px !important;
        width: 300px !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {{
        color: #e2e8f0 !important;
    }}

    .hero-section {{
        background: linear-gradient(135deg, {TOKENS["hero_a"]} 0%, {TOKENS["hero_b"]} 100%);
        border-radius: 20px;
        padding: {hero_pad};
        color: #fff !important;
        margin-bottom: 20px;
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.22);
    }}
    .hero-section h1 {{
        color: #fff !important;
        font-size: {h1_size} !important;
        font-weight: 800 !important;
        margin: 0 0 6px 0 !important;
    }}
    .hero-section h3 {{
        color: #dbeafe !important;
        font-size: {subheader_size} !important;
        font-weight: 500 !important;
        margin: 0 0 8px 0 !important;
    }}
    .hero-section p {{
        color: #e5edff !important;
        font-size: 0.9rem !important;
        margin: 0 !important;
        line-height: 1.45;
    }}

    .saas-header {{
        font-size: 1.45rem;
        font-weight: 800;
        color: var(--text);
        margin: 4px 0 2px 0;
    }}
    .saas-subheader {{
        font-size: {subheader_size};
        color: var(--muted);
        margin: 0 0 12px 0;
    }}

    .saas-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: {card_pad};
        margin-bottom: 12px;
        box-shadow: var(--shadow);
    }}

    .kpi-blue {{ border-top: 4px solid #3b82f6; }}
    .kpi-purple {{ border-top: 4px solid #8b5cf6; }}
    .kpi-yellow {{ border-top: 4px solid #f59e0b; }}
    .kpi-green {{ border-top: 4px solid #10b981; }}
    .kpi-red {{ border-top: 4px solid #ef4444; }}

    .kpi-title {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: var(--muted);
        margin-bottom: 5px;
        font-weight: 700;
    }}
    .kpi-value {{
        font-size: {kpi_size};
        font-weight: 800;
        line-height: 1.2;
        color: var(--text);
    }}
    .kpi-desc {{
        font-size: 0.76rem;
        color: var(--muted);
        margin-top: 5px;
    }}

    div.stButton > button:first-child {{
        background: var(--primary);
        color: #fff !important;
        border: none !important;
        border-radius: 10px;
        padding: 0.55rem .9rem;
        font-weight: 700;
        box-shadow: 0 6px 14px rgba(37, 99, 235, .30);
        width: 100%;
    }}
    div.stButton > button:first-child:hover {{
        background: var(--primary-dark);
    }}

    .plot-card {{
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 10px;
        background: var(--surface);
        box-shadow: var(--shadow);
        margin-bottom: 12px;
    }}

    /* Make st.container(border=True) look identical to .saas-card */
    div[data-testid="stVerticalBlockBorderWrapper"] > div:first-child {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: {card_pad} !important;
        box-shadow: var(--shadow) !important;
    }}

    .badge {{
        background: {TOKENS["badge_bg"]};
        color: {TOKENS["badge_text"]};
        border: 1px solid {TOKENS["badge_border"]};
        padding: 4px 9px;
        border-radius: 999px;
        font-size: .75rem;
        font-weight: 600;
        display: inline-block;
        margin: 4px 6px 4px 0;
    }}

    .timeline {{
        border-left: 2px solid var(--border);
        padding-left: 14px;
        margin-left: 6px;
    }}
    .timeline-item {{
        margin-bottom: 16px;
        position: relative;
    }}
    .timeline-item::before {{
        content: '';
        position: absolute;
        left: -21px;
        top: 6px;
        background: var(--primary);
        border-radius: 50%;
        width: 10px;
        height: 10px;
    }}
    .timeline-title {{
        font-weight: 700;
        color: var(--text);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# FORCE FIX: Override ALL Streamlit text colors
# ============================================================
st.markdown(f"""
    <style>
        /* Clean text coloring using CSS variables */
        .stApp {{
            color: var(--text) !important;
        }}
        
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, 
        .stApp p, .stApp label, .stApp span, .stApp li, .stApp small {{
            color: var(--text) !important;
        }}
        
        /* Streamlit widget label overrides */
        .stApp [data-testid="stWidgetLabel"] p {{
            color: var(--text) !important;
        }}
        
        /* Expander headers & expander containers */
        div[data-testid="stExpander"] {{
            border: 1px solid var(--border) !important;
            background-color: var(--surface) !important;
            border-radius: var(--radius) !important;
        }}
        div[data-testid="stExpander"] details summary {{
            color: var(--text) !important;
            background-color: var(--surface) !important;
        }}
        div[data-testid="stExpander"] details summary:hover {{
            background-color: var(--bg) !important;
        }}
        div[data-testid="stExpander"] [data-testid="stVerticalBlock"] {{
            background-color: var(--surface) !important;
            color: var(--text) !important;
        }}
        
        /* Sidebar layout styling - always dark sidebar, high-specificity overrides */
        section[data-testid="stSidebar"] {{
            background-color: {TOKENS["sidebar_bg"]} !important;
        }}
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] *,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] h5,
        section[data-testid="stSidebar"] h6,
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] small,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        section[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
        section[data-testid="stSidebar"] [data-testid="stMarkdown"] span,
        section[data-testid="stSidebar"] [data-testid="stRadio"] label,
        section[data-testid="stSidebar"] [data-testid="stRadio"] span,
        section[data-testid="stSidebar"] [role="radiogroup"] label,
        section[data-testid="stSidebar"] [role="radiogroup"] span {{
            color: #e2e8f0 !important;
        }}
        section[data-testid="stSidebar"] div[data-baseweb="select"] {{
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
        }}
        section[data-testid="stSidebar"] div[data-baseweb="select"] * {{
            color: #e2e8f0 !important;
        }}
        section[data-testid="stSidebar"] input {{
            background-color: #1e293b !important;
            color: #e2e8f0 !important;
            border: 1px solid #334155 !important;
        }}

        /* Tabs styling */
        div[data-testid="stTabs"] button[data-baseweb="tab"] {{
            color: var(--muted) !important;
        }}
        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
            color: var(--primary) !important;
            font-weight: 700 !important;
        }}
        
        /* Metric styling */
        div[data-testid="stMetricLabel"] > div {{
            color: var(--muted) !important;
        }}
        div[data-testid="stMetricValue"] > div {{
            color: var(--text) !important;
        }}
        
        /* Selectboxes & Text inputs inside main app */
        .stApp div[data-baseweb="select"] {{
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
        }}
        .stApp div[data-baseweb="select"] * {{
            color: var(--text) !important;
        }}
        .stApp input {{
            background-color: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
        }}
        
        /* KPI cards styling */
        .kpi-title {{
            color: var(--muted) !important;
        }}
        .kpi-value {{
            color: var(--text) !important;
        }}
        .kpi-desc {{
            color: var(--muted) !important;
        }}
        
        /* Keep hero-section text white */
        .hero-section, .hero-section * {{
            color: #ffffff !important;
        }}
        .hero-section h1, .hero-section h3, .hero-section p {{
            color: #ffffff !important;
        }}
        
        /* Notification/Alert formatting preservation */
        div[data-testid="stNotification"] p {{
            color: inherit !important;
        }}
        
        /* Consistent table borders in both light and dark modes */
        div[data-testid="stTable"] table {{
            border: 1px solid var(--border) !important;
            border-collapse: collapse !important;
            border-radius: var(--radius) !important;
            overflow: hidden;
            width: 100%;
        }}
        div[data-testid="stTable"] thead tr th {{
            background-color: var(--surface) !important;
            color: var(--muted) !important;
            border-bottom: 2px solid var(--border) !important;
            border-right: 1px solid var(--border) !important;
            padding: 10px 14px !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }}
        div[data-testid="stTable"] tbody tr td {{
            border-bottom: 1px solid var(--border) !important;
            border-right: 1px solid var(--border) !important;
            padding: 9px 14px !important;
            color: var(--text) !important;
            font-size: 0.875rem !important;
        }}
        div[data-testid="stTable"] tbody tr:last-child td {{
            border-bottom: none !important;
        }}
        div[data-testid="stTable"] tbody tr:hover td {{
            background-color: var(--bg) !important;
        }}
    </style>
""", unsafe_allow_html=True)

def vspace(h: int = 8) -> None:
    st.markdown(f"<div style='height:{h}px'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE: DASHBOARD
# -----------------------------------------------------------------------------
if page == "🏠 Dashboard":
    st.markdown(
        """
        <div class='hero-section'>
            <h1>SmartRetail AI Ingestion Platform</h1>
            <h3>Customer Loyalty Engine & Marketing Optimization</h3>
            <p>An enterprise-grade SaaS intelligence platform utilizing reinforcement learning policies to identify loyalty tiers and recommend optimal marketing strategies.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    total_customers = 0
    if metadata:
        total_customers = (
            metadata.get("train_customers_count", 0)
            + metadata.get("val_customers_count", 0)
            + metadata.get("test_customers_count", 0)
        )

    best_classifier_name = class_results.get("best_model_selected", "N/A").replace("_", " ") if class_results else "N/A"
    best_regressor_name = reg_results.get("best_model_selected", "N/A").replace("_", " ") if reg_results else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='saas-card kpi-blue'><div class='kpi-title'>Total Profiles</div><div class='kpi-value'>{total_customers:,}</div><div class='kpi-desc'>Customers across all splits</div></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            "<div class='saas-card kpi-purple'><div class='kpi-title'>Features</div><div class='kpi-value'>13 Dimensions</div><div class='kpi-desc'>Behavioral + category spends</div></div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div class='saas-card kpi-yellow'><div class='kpi-title'>Best Classifier</div><div class='kpi-value'>{best_classifier_name}</div><div class='kpi-desc'>Highest validation F1-score</div></div>",
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"<div class='saas-card kpi-green'><div class='kpi-title'>Best Regressor</div><div class='kpi-value'>{best_regressor_name}</div><div class='kpi-desc'>Highest validation R²-score</div></div>",
            unsafe_allow_html=True
        )

    vspace(12)

    # RL Policy Profit Comparison Section (Section 6.4 & 7.2)
    st.markdown("### 🤖 Reinforcement Learning Recommendations Performance")
    col_l, col_r = st.columns([3, 2])
    
    with col_l:
        with st.container(border=True):
            st.markdown("#### Cumulative Policy Net Profit on Test Set")
            if rl_eval_results:
                profits_data = pd.DataFrame({
                    "Policy": ["Always No-Action", "Random Action", "Tabular Q-Learning", "DQN Policy"],
                    "Net Profit ($)": [
                        rl_eval_results.get("Always_No_Action", 0),
                        rl_eval_results.get("Random_Action", 0),
                        rl_eval_results.get("Tabular_Q_Policy", 0),
                        rl_eval_results.get("DQN_Policy", 0)
                    ]
                })
                
                fig = px.bar(
                    profits_data,
                    x="Policy",
                    y="Net Profit ($)",
                    color="Policy",
                    color_discrete_map={
                        "Always No-Action": "#ef4444",
                        "Random Action": "#f59e0b",
                        "Tabular Q-Learning": "#10b981",
                        "DQN Policy": "#2563eb"
                    },
                    text="Net Profit ($)",
                    template="plotly_white"
                )
                fig.update_layout(
                    paper_bgcolor=TOKENS["surface"],
                    plot_bgcolor=TOKENS["plot_bg"],
                    xaxis_title="",
                    yaxis_title="Net Profit ($)",
                    showlegend=False,
                    height=360,
                    margin=dict(l=20, r=20, t=20, b=80),
                    xaxis=dict(
                        tickangle=-20,
                        gridcolor=TOKENS["border"]
                    ),
                    yaxis=dict(
                        gridcolor=TOKENS["border"]
                    ),
                    font=dict(color=TOKENS["text"], family="Inter")
                )
                fig.update_traces(
                    texttemplate='$%{text:,.0f}',
                    textposition='outside',
                    textfont=dict(color=TOKENS["text"])
                )
                st.plotly_chart(fig, width='stretch', theme=None)
            else:
                st.warning("RL evaluation results not found. Run train_rl.py first.")

    with col_r:
        with st.container(border=True):
            st.markdown("#### 🧠 Marketing Segment Policy Analysis")
            st.markdown(
                """
**High-Value Loyalty Segment**  
The DQN agent prefers **Action 2 — Free Premium Trial** (cost: $5). 
This segment responds with a 2× spend multiplier, yielding strong positive net profit.

**Medium-Value Repeat Segment**  
The agent prefers **Action 1 — 10% Discount Coupon** (cost: $1). 
This stimulates purchase frequency without exhausting the marketing budget.

**At-Risk / Low-Value Segment**  
The agent selects **Action 0 — No Action** (cost: $0). 
Avoids burning budget on low-probability transactors, preserving capital for higher-value customers.
                """
            )

# -----------------------------------------------------------------------------
# PAGE: CUSTOMER PROFILE
# -----------------------------------------------------------------------------
elif page == "👤 Customer Profile":
    st.markdown("<div class='saas-header'>👤 Customer Profile & Decision Hub</div>", unsafe_allow_html=True)
    st.markdown("<div class='saas-subheader'>Look up profiles, run supervised classifications, and inspect RL action recommendations.</div>", unsafe_allow_html=True)

    if selected_customer_id is None:
        st.error("Select a Customer ID from the sidebar to inspect a profile.")
    else:
        # Load customer profile
        cust_row = test_df.loc[selected_customer_id]
        
        # Display Profile stats
        p1, p2, p3, p4 = st.columns(4)
        p1.markdown(f"<div class='saas-card kpi-blue'><div class='kpi-title'>Recency</div><div class='kpi-value'>{int(cust_row['Recency'])} days</div><div class='kpi-desc'>Since last transaction</div></div>", unsafe_allow_html=True)
        p2.markdown(f"<div class='saas-card kpi-purple'><div class='kpi-title'>Frequency</div><div class='kpi-value'>{int(cust_row['Frequency'])} orders</div><div class='kpi-desc'>Unique transaction ledger count</div></div>", unsafe_allow_html=True)
        p3.markdown(f"<div class='saas-card kpi-green'><div class='kpi-title'>Monetary (Spend)</div><div class='kpi-value'>${cust_row['Monetary']:,.2f}</div><div class='kpi-desc'>Feature window (Months 1-9)</div></div>", unsafe_allow_html=True)
        p4.markdown(f"<div class='saas-card kpi-yellow'><div class='kpi-title'>Product Diversity</div><div class='kpi-value'>{int(cust_row['ProductDiversity'])} items</div><div class='kpi-desc'>Distinct stock items bought</div></div>", unsafe_allow_html=True)

        vspace(8)

        # Columns for predictions and spend mix
        left_col, right_col = st.columns([3, 2])

        with left_col:
            with st.container(border=True):
                st.markdown("#### 📊 Category Spending Composition (%)")
                
                categories = ["Homeware", "Stationery", "Gadgets", "Decorations", "Kitchenware"]
                spend_pcts = [cust_row[f"{c}_Spend_Pct"] * 100.0 for c in categories]
                total_cat = sum(spend_pcts)
                other_pct = max(0.0, 100.0 - total_cat)
                
                categories.append("Other")
                spend_pcts.append(other_pct)
                
                spend_mix = pd.DataFrame({"Category": categories, "Percentage (%)": spend_pcts})
                fig = px.bar(
                    spend_mix,
                    y="Category",
                    x="Percentage (%)",
                    orientation="h",
                    color="Category",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    text="Percentage (%)",
                    template="plotly_white"
                )
                fig.update_layout(
                    paper_bgcolor=TOKENS["surface"],
                    plot_bgcolor=TOKENS["plot_bg"],
                    xaxis_title="Percentage (%)",
                    yaxis_title="Category",
                    showlegend=False,
                    height=220,
                    margin=dict(l=10, r=20, t=10, b=10),
                    xaxis=dict(
                        gridcolor=TOKENS["border"]
                    ),
                    yaxis=dict(
                        gridcolor=TOKENS["border"]
                    ),
                    font=dict(color=TOKENS["text"], family="Inter")
                )
                fig.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside',
                    textfont=dict(color=TOKENS["text"])
                )
                st.plotly_chart(fig, width='stretch', theme=None)

        with right_col:
            with st.container(border=True):
                st.markdown("#### 🔮 Supervised Predictions")
                
                # Predict segment
                best_rep = class_results.get("best_model_representation", "raw_scaled") if class_results else "raw_scaled"
                if best_rep == "raw_scaled" and test_raw_scaled_df is not None:
                    features = [col for col in test_raw_scaled_df.columns if col not in ["High_Value_Customer", "Future_Spend"]]
                    X = test_raw_scaled_df.loc[selected_customer_id][features].values.reshape(1, -1)
                elif best_rep == "pca" and test_pca_df is not None:
                    features = [col for col in test_pca_df.columns if col not in ["High_Value_Customer", "Future_Spend"]]
                    X = test_pca_df.loc[selected_customer_id][features].values.reshape(1, -1)
                elif best_rep == "lda" and test_lda_df is not None:
                    features = [col for col in test_lda_df.columns if col not in ["High_Value_Customer", "Future_Spend"]]
                    X = test_lda_df.loc[selected_customer_id][features].values.reshape(1, -1)
                else:
                    X = None

                if X is not None and best_classifier is not None:
                    pred_class = int(best_classifier.predict(X)[0])
                    pred_proba = best_classifier.predict_proba(X)[0][pred_class]
                    
                    segment_color = "#10b981" if pred_class == 1 else "#ef4444"
                    segment_label = "HIGH-VALUE CUSTOMER" if pred_class == 1 else "STANDARD CUSTOMER"
                    
                    st.markdown(
                        f"**Predicted Loyalty Segment:** <span style='color: {segment_color}; font-weight: 800;'>{segment_label}</span> ({pred_proba*100:.1f}% confidence)",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("**Predicted Loyalty Segment:** Binary classification model offline")

                # Predict Spend
                if test_raw_scaled_df is not None and best_regressor is not None:
                    features = [col for col in test_raw_scaled_df.columns if col not in ["High_Value_Customer", "Future_Spend"]]
                    X_reg = test_raw_scaled_df.loc[selected_customer_id][features].values.reshape(1, -1)
                    
                    pred_spend = max(0.0, float(best_regressor.predict(X_reg)[0]))
                    test_rmse = reg_results.get("test_evaluation", {}).get("rmse", 0.0) if reg_results else 0.0
                    
                    st.markdown(
                        f"**Predicted Future Spend (Months 10-12):** <span style='font-size: 1.15rem; font-weight:800; color: #2563eb;'>${pred_spend:,.2f}</span> <br><span style='font-size: 0.76rem; color:var(--muted);'>Model RMSE Margin: ± ${test_rmse:,.2f}</span>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("**Predicted Future Spend:** Regressor model offline")

        # RL Recommendations Panel
        st.markdown("### 🤖 Reinforcement Learning Marketing Action Recommendations")
        
        if dqn_agent is None or q_agent is None or discretizer is None:
            st.error("Reinforcement learning agents are offline. Ensure models are trained.")
        else:
            rl_l, rl_r = st.columns([3, 2])
            
            # Fetch continuous 5D state
            state_5d = test_pca_df.loc[selected_customer_id][["Recency", "Frequency", "Monetary", "PC1", "PC2"]].values
            
            # Tabular discretization
            state_idx = discretizer.discretize(state_5d)
            tab_q_vals = q_agent.q_table[state_idx]
            
            # DQN model Q-values
            state_t = torch.FloatTensor(state_5d).unsqueeze(0).to(dqn_agent.device)
            with torch.no_grad():
                dqn_q_vals = dqn_agent.policy_net(state_t).cpu().numpy()[0]
                
            actions = ["No Action (cost $0)", "10% Discount Coupon (cost $1)", "Free Premium Trial (cost $5)"]
            
            with rl_l:
                with st.container(border=True):
                    st.markdown(f"#### RL Policy Q-Values (State Cluster ID: **Cluster {state_idx}**)")
                    
                    q_vals_df = pd.DataFrame({
                        "Marketing Action": actions * 2,
                        "Estimated Q-Value": list(dqn_q_vals) + list(tab_q_vals),
                        "Model": ["DQN (Graded)"] * 3 + ["Tabular Q-learning"] * 3
                    })
                    
                    fig = px.bar(
                        q_vals_df,
                        x="Estimated Q-Value",
                        y="Marketing Action",
                        color="Model",
                        barmode="group",
                        color_discrete_map={
                            "DQN (Graded)": "#2563eb",
                            "Tabular Q-learning": "#10b981"
                        },
                        template="plotly_white",
                        height=200
                    )
                    fig.update_layout(
                        paper_bgcolor=TOKENS["surface"],
                        plot_bgcolor=TOKENS["plot_bg"],
                        yaxis_title="",
                        margin=dict(l=10, r=20, t=10, b=10),
                        xaxis=dict(
                            gridcolor=TOKENS["border"]
                        ),
                        yaxis=dict(
                            gridcolor=TOKENS["border"]
                        ),
                        legend=dict(
                            font=dict(color=TOKENS["text"])
                        ),
                        font=dict(color=TOKENS["text"], family="Inter")
                    )
                    st.plotly_chart(fig, width='stretch', theme=None)

            with rl_r:
                with st.container(border=True):
                    st.markdown("#### Recommended Action")
                    
                    rec_action = int(np.argmax(dqn_q_vals))
                    action_explanations = {
                        0: "The agent recommends **No Action**. Budget acquisition costs outweigh expected spend multipliers for this profile.",
                        1: "The agent recommends a **10% Discount Coupon** (cost $1). Expected repeat transactional values yield optimal return.",
                        2: "The agent recommends a **Free Premium Trial** (cost $5). This customer shows high loyalty potential, making premium retention highly profitable."
                    }
                    
                    st.markdown(
                        f"""
                        <div style='background-color: {TOKENS["badge_bg"]}; border: 1px solid {TOKENS["badge_border"]}; padding: 15px; border-radius: 12px; margin-top: 10px;'>
                            <div style='font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #3b82f6;'>DQN Decided Action</div>
                            <div style='font-size: 1.15rem; font-weight: 800; color: {TOKENS["primary"]}; margin-top: 4px;'>{actions[rec_action]}</div>
                            <div style='font-size: 0.88rem; color: var(--text); margin-top: 10px; line-height:1.45;'>{action_explanations[rec_action]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# -----------------------------------------------------------------------------
# PAGE: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif page == "📈 Model Performance":
    st.markdown("<div class='saas-header'>📈 Model Benchmarking Diagnostics</div>", unsafe_allow_html=True)
    st.markdown("<div class='saas-subheader'>Comparative metrics audited on validation and test splits.</div>", unsafe_allow_html=True)

    if class_results is None or reg_results is None:
        st.error("Audit metrics files not found in data/processed/.")
    else:
        st.markdown("### Classification Benchmarks")
        with st.container(border=True):
            comp_data = []
            for key, val in class_results.get("validation_comparisons", {}).items():
                comp_data.append({
                    "Model & Feature Space": key.replace("_", " "),
                    "Accuracy": f"{val.get('val_accuracy', 0)*100:.2f}%",
                    "Precision": f"{val.get('val_precision', 0)*100:.2f}%",
                    "Recall": f"{val.get('val_recall', 0)*100:.2f}%",
                    "F1-Score": f"{val.get('val_f1', 0):.4f}",
                    "ROC-AUC": f"{val.get('val_auc', 0):.4f}"
                })
            if comp_data:
                st.table(pd.DataFrame(comp_data))
            else:
                st.info("No classification validation comparison data available.")

        st.markdown("#### Test Set Scores (Best Classifier)")
        tm = class_results.get("test_evaluation", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f"<div class='saas-card kpi-blue'><div class='kpi-title'>Accuracy</div><div class='kpi-value'>{tm.get('accuracy', 0)*100:.1f}%</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='saas-card kpi-blue'><div class='kpi-title'>Precision</div><div class='kpi-value'>{tm.get('precision', 0)*100:.1f}%</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='saas-card kpi-blue'><div class='kpi-title'>Recall</div><div class='kpi-value'>{tm.get('recall', 0)*100:.1f}%</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='saas-card kpi-purple'><div class='kpi-title'>F1-Score</div><div class='kpi-value'>{tm.get('f1_score', 0):.3f}</div></div>", unsafe_allow_html=True)
        c5.markdown(f"<div class='saas-card kpi-green'><div class='kpi-title'>ROC AUC</div><div class='kpi-value'>{tm.get('roc_auc', 0):.3f}</div></div>", unsafe_allow_html=True)

        f1, f2, f3 = st.columns(3)
        with f1:
            p = os.path.join(FIGURES_DIR, "confusion_matrix.png")
            if os.path.exists(p):
                st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
                st.image(p, caption="Confusion Matrix")
                st.markdown("</div>", unsafe_allow_html=True)
        with f2:
            p = os.path.join(FIGURES_DIR, "roc_curve.png")
            if os.path.exists(p):
                st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
                st.image(p, caption="ROC Curves")
                st.markdown("</div>", unsafe_allow_html=True)
        with f3:
            p = os.path.join(FIGURES_DIR, "precision_recall_curve.png")
            if os.path.exists(p):
                st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
                st.image(p, caption="Precision-Recall Curves")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Regression Benchmarks")
        with st.container(border=True):
            reg_comp = []
            for key, val in reg_results.get("validation_comparisons", {}).items():
                reg_comp.append({
                    "Regressor Model": key.replace("_", " "),
                    "MSE": f"{val.get('mse', 0):.2f}",
                    "RMSE": f"{val.get('rmse', 0):.2f}",
                    "R² Score": f"{val.get('r2_score', 0):.4f}"
                })
            if reg_comp:
                st.table(pd.DataFrame(reg_comp))
            else:
                st.info("No regression validation comparison data available.")

        st.markdown("#### Test Set Scores (Best Regressor)")
        rtm = reg_results.get("test_evaluation", {})
        r1, r2, r3 = st.columns(3)
        r1.markdown(f"<div class='saas-card kpi-blue'><div class='kpi-title'>MSE</div><div class='kpi-value'>{rtm.get('mse', 0):,.1f}</div></div>", unsafe_allow_html=True)
        r2.markdown(f"<div class='saas-card kpi-purple'><div class='kpi-title'>RMSE</div><div class='kpi-value'>${rtm.get('rmse', 0):,.1f}</div></div>", unsafe_allow_html=True)
        r3.markdown(f"<div class='saas-card kpi-green'><div class='kpi-title'>R² Score</div><div class='kpi-value'>{rtm.get('r2_score', 0):.4f}</div></div>", unsafe_allow_html=True)

        rr1, rr2 = st.columns(2)
        with rr1:
            p = os.path.join(FIGURES_DIR, "regression_predicted_vs_actual.png")
            if os.path.exists(p):
                st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
                st.image(p, caption="Predicted vs Actual Spend")
                st.markdown("</div>", unsafe_allow_html=True)
        with rr2:
            p = os.path.join(FIGURES_DIR, "regression_residual_plot.png")
            if os.path.exists(p):
                st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
                st.image(p, caption="Residual Plot")
                st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE: PCA & LDA
# -----------------------------------------------------------------------------
elif page == "🧠 PCA & LDA":
    st.markdown("<div class='saas-header'>🧠 Dimensionality Reduction Mappings</div>", unsafe_allow_html=True)
    st.markdown("<div class='saas-subheader'>Orthogonal mappings and class separability diagnostics fitted on train split only.</div>", unsafe_allow_html=True)

    if pca_metadata is None:
        st.error("Missing PCA/LDA metadata file.")
    else:
        left, right = st.columns([2, 3])

        with left:
            st.markdown("### Methodology")
            with st.expander("🔬 Unsupervised PCA", expanded=True):
                st.markdown("PCA maps category spend fractions into orthogonal components, reducing redundancy while preserving principal variance directions.")
            with st.expander("🔬 Supervised LDA", expanded=True):
                st.markdown("LDA projects behavioral dimensions into discriminant axis `LD1` to maximize separation between customer classes.")

            selected_components = pca_metadata.get("selected_components_count", 0)
            cumulative_variance = pca_metadata.get("cumulative_variance_explained", [])
            cum_var = cumulative_variance[selected_components - 1] if selected_components > 0 and len(cumulative_variance) >= selected_components else 0.0

            st.markdown(
                f"<div class='saas-card kpi-blue'><div class='kpi-title'>Selected Components</div><div class='kpi-value'>{selected_components} / 5</div><div class='kpi-desc'>Configured variance ratio: 90%</div></div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='saas-card kpi-green'><div class='kpi-title'>Variance Retained</div><div class='kpi-value'>{cum_var*100:.2f}%</div><div class='kpi-desc'>Cumulative explained variance</div></div>",
                unsafe_allow_html=True
            )

            lda_coefs = pca_metadata.get("lda_discriminant_coefficients", {})
            if lda_coefs:
                top_feat = max(lda_coefs, key=lambda k: abs(lda_coefs[k]))
                st.markdown(
                    f"<div class='saas-card kpi-purple'><div class='kpi-title'>Top Discriminating Feature</div><div class='kpi-value' style='font-size:1.1rem'>{top_feat}</div><div class='kpi-desc'>LDA coefficient: {lda_coefs[top_feat]:.3f}</div></div>",
                    unsafe_allow_html=True
                )

            viva_explanation = pca_metadata.get("viva_explanation", "No explanation available.")
            st.markdown(
                f"""
                <div class='saas-card kpi-yellow'>
                    <div class='kpi-title'>💡 Multicollinearity Note</div>
                    <div style='font-size:0.88rem; color:var(--text); margin-top:5px; line-height:1.45;'>
                        "{viva_explanation}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with right:
            st.markdown("### Projection Charts & Selected Customer Location")
            
            view_mode = st.radio("Select Projection View:", ["PCA Scatter View (PC1 vs PC2)", "LDA Class View (1D Projection)"], horizontal=True)
            
            if selected_customer_id is None:
                st.info("Select a customer from the sidebar to view their position highlighted on the charts.")
                
            if view_mode == "PCA Scatter View (PC1 vs PC2)":
                st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
                if train_pca_df is not None:
                    # Plotly PC1 vs PC2
                    fig = px.scatter(
                        train_pca_df,
                        x="PC1",
                        y="PC2",
                        color="High_Value_Customer",
                        color_discrete_map={0: "#94a3b8", 1: "#3b82f6"},
                        hover_data=["Recency", "Frequency", "Monetary"],
                        template="plotly_white",
                        title="PC1 vs PC2 Train Coordinates"
                    )
                    
                    if selected_customer_id is not None and test_pca_df is not None:
                        # Append Selected Customer
                        cust_row = test_pca_df.loc[selected_customer_id]
                        fig.add_trace(
                            go.Scatter(
                                x=[cust_row["PC1"]],
                                y=[cust_row["PC2"]],
                                mode="markers",
                                marker=dict(color="#ef4444", size=16, symbol="star", line=dict(color="black", width=1.5)),
                                name=f"Customer {selected_customer_id}"
                            )
                        )
                    
                    fig.update_layout(
                        paper_bgcolor=TOKENS["surface"],
                        plot_bgcolor=TOKENS["plot_bg"],
                        legend_title_text="Segment (Train)",
                        xaxis=dict(
                            gridcolor=TOKENS["border"]
                        ),
                        yaxis=dict(
                            gridcolor=TOKENS["border"]
                        ),
                        legend=dict(
                            font=dict(color=TOKENS["text"])
                        ),
                        font=dict(color=TOKENS["text"], family="Inter")
                    )
                    st.plotly_chart(fig, width='stretch', theme=None)
                st.markdown("</div>", unsafe_allow_html=True)
                
            else:
                st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
                if train_lda_df is not None:
                    # Plotly LD1 Histogram/Density
                    fig = px.histogram(
                        train_lda_df,
                        x="LD1",
                        color="High_Value_Customer",
                        color_discrete_map={0: "#94a3b8", 1: "#3b82f6"},
                        marginal="box",
                        barmode="overlay",
                        template="plotly_white",
                        title="LDA Discriminant Axis (LD1)"
                    )
                    
                    if selected_customer_id is not None and test_lda_df is not None:
                        cust_row = test_lda_df.loc[selected_customer_id]
                        # Draw vertical line for customer position
                        fig.add_vline(
                            x=cust_row["LD1"], 
                            line_width=3, 
                            line_dash="dash", 
                            line_color="#ef4444",
                            annotation_text=f"Cust {selected_customer_id}",
                            annotation_position="top right"
                        )
                        
                    fig.update_layout(
                        paper_bgcolor=TOKENS["surface"],
                        plot_bgcolor=TOKENS["plot_bg"],
                        legend_title_text="Segment (Train)",
                        xaxis=dict(
                            gridcolor=TOKENS["border"]
                        ),
                        yaxis=dict(
                            gridcolor=TOKENS["border"]
                        ),
                        legend=dict(
                            font=dict(color=TOKENS["text"])
                        ),
                        font=dict(color=TOKENS["text"], family="Inter")
                    )
                    st.plotly_chart(fig, width='stretch', theme=None)
                st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE: DATASET EXPLORER
# -----------------------------------------------------------------------------
elif page == "📊 Dataset Explorer":
    st.markdown("<div class='saas-header'>📊 Customer Records Auditor</div>", unsafe_allow_html=True)
    st.markdown("<div class='saas-subheader'>Inspect preprocessed split files and visualize feature distributions.</div>", unsafe_allow_html=True)

    dataset_choice = st.selectbox("Choose Dataset Split:", ["Train Set", "Validation Set", "Test Set"])
    filename_map = {
        "Train Set": "train.csv",
        "Validation Set": "validation.csv",
        "Test Set": "test.csv"
    }

    df = load_dataset(filename_map[dataset_choice])
    if df is None:
        st.error(f"Missing split file: {filename_map[dataset_choice]}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='saas-card kpi-blue'><div class='kpi-title'>Rows</div><div class='kpi-value'>{df.shape[0]:,}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='saas-card kpi-purple'><div class='kpi-title'>Columns</div><div class='kpi-value'>{df.shape[1]}</div></div>", unsafe_allow_html=True)
        c3.markdown("<div class='saas-card kpi-yellow'><div class='kpi-title'>Features</div><div class='kpi-value'>13</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='saas-card kpi-green'><div class='kpi-title'>Customers</div><div class='kpi-value'>{df.shape[0]:,}</div></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### Interactive Preview (First 50 Rows)")
            st.dataframe(df.head(50), width='stretch')

        with st.container(border=True):
            st.markdown("#### Feature Distribution Visualizer")

        expected_cols = [
            "Recency", "Frequency", "Monetary", "AverageSpend",
            "ProductDiversity", "TotalOrders", "AverageBasketSize",
            "AverageQuantityPerOrder"
        ]
        available_cols = [c for c in expected_cols if c in df.columns]

        if not available_cols:
            st.warning("No expected continuous columns found in this dataset.")
        else:
            selected_col = st.selectbox("Select Continuous Feature:", available_cols)
            color_col = "High_Value_Customer" if "High_Value_Customer" in df.columns else None

            if color_col:
                fig = px.histogram(
                    df,
                    x=selected_col,
                    color=color_col,
                    color_discrete_map={0: "#94a3b8", 1: "#2563eb"},
                    marginal="box",
                    barmode="overlay",
                    template="plotly_white"
                )
                name_map = {"0": "Standard Customer", "1": "High Value"}
                fig.for_each_trace(lambda t: t.update(name=name_map.get(t.name, t.name)))
            else:
                fig = px.histogram(
                    df,
                    x=selected_col,
                    marginal="box",
                    barmode="overlay",
                    template="plotly_white"
                )

            fig.update_layout(
                bargap=0.08,
                xaxis_title=selected_col,
                yaxis_title="Count",
                paper_bgcolor=TOKENS["surface"],
                plot_bgcolor=TOKENS["plot_bg"],
                legend_title_text="Customer Segment" if color_col else "",
                xaxis=dict(
                    gridcolor=TOKENS["border"]
                ),
                yaxis=dict(
                    gridcolor=TOKENS["border"]
                ),
                legend=dict(
                    font=dict(color=TOKENS["text"])
                ),
                font=dict(color=TOKENS["text"], family="Inter")
            )
            st.plotly_chart(fig, width='stretch', theme=None)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE: ABOUT
# -----------------------------------------------------------------------------
else:
    st.markdown("<div class='saas-header'>ℹ️ About Platform Implementation</div>", unsafe_allow_html=True)
    st.markdown("<div class='saas-subheader'>Technical workflow, technologies, and leakage-control audit trail.</div>", unsafe_allow_html=True)

    st.markdown("### 🔄 Implementation Pipeline Timeline")
    st.markdown(
        """
        <div class='timeline'>
            <div class='timeline-item'>
                <div class='timeline-title'>Phase 1: Ingestion & Transaction Cleaning</div>
                <div style='font-size:0.88rem; color:var(--muted);'>Loaded UCI Online Retail data. Removed duplicates, null customer IDs, and invalid negative quantity/pricing records.</div>
            </div>
            <div class='timeline-item'>
                <div class='timeline-title'>Phase 2: Split Assignment & Preprocessing</div>
                <div style='font-size:0.88rem; color:var(--muted);'>Split customer IDs into 80/10/10. Built input variables on Months 1-9 and targets on Months 10-12 only.</div>
            </div>
            <div class='timeline-item'>
                <div class='timeline-title'>Phase 3: PCA & LDA Projections</div>
                <div style='font-size:0.88rem; color:var(--muted);'>Standardized features and fit PCA/LDA using training data only.</div>
            </div>
            <div class='timeline-item'>
                <div class='timeline-title'>Phase 4: Model Training</div>
                <div style='font-size:0.88rem; color:var(--muted);'>Trained and validated classifiers/regressors, then serialized best-performing models.</div>
            </div>
            <div class='timeline-item'>
                <div class='timeline-title'>Phase 5: Dashboard Deployment</div>
                <div style='font-size:0.88rem; color:var(--muted);'>Integrated serialized assets into a responsive offline Streamlit app.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### 🛠️ Technology Stack")
    st.markdown(
        """
        <span class='badge'>Python 3.11</span>
        <span class='badge'>Scikit-Learn</span>
        <span class='badge'>PyTorch</span>
        <span class='badge'>Streamlit</span>
        <span class='badge'>Pandas</span>
        <span class='badge'>NumPy</span>
        <span class='badge'>Plotly</span>
        <span class='badge'>Joblib</span>
        <span class='badge'>Matplotlib & Seaborn</span>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    l, r = st.columns(2)
    with l:
        st.markdown(
            """
            ### 🔒 Data Leakage Prevention
            - **Split isolation** before scaling and fitting.
            - **Feature isolation**: no Months 10–12 target spend used as input.
            - **Transform fitting** (scalers/PCA/LDA) on train split only.
            - **Threshold lock** computed from train distribution only.
            """
        )
    with r:
        st.markdown(
            """
            ### 👤 Developer Notes
            - **Project Type**: Final Year ML Product + Dashboard
            - **Role**: ML Engineer & UI Developer
            - **Mode**: Fully offline inference using serialized binaries
            """
        )