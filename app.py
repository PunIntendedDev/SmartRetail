"""
SmartRetail AI - Customer Intelligence Dashboard
------------------------------------------------
A fully offline Streamlit application that loads pre-trained artifacts
(.joblib / .pt / .json) and serves supervised predictions, dimensionality
projections, and reinforcement-learning marketing recommendations.

Run locally with a single command:
    streamlit run app.py

UI design notes (HCI principles applied):
- Match between system & real world: plain-language labels, currency/units on every metric.
- Visibility of system status: live artifact-readiness indicators are always on screen.
- Recognition over recall: the active customer is pinned in the sidebar across every view.
- Consistency & standards: one light design-token system drives every surface.
- Aesthetic & minimalist design: generous whitespace, a restrained palette, real icons.
- Error prevention & recovery: instructive empty states instead of raw tracebacks.
- Flexibility: the required PCA/LDA toggle and customer selector are first-class controls.

This module ONLY changes presentation. All model/data logic loads from saved files
and is never retrained on launch.
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
    page_title="SmartRetail AI · Customer Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

# -----------------------------------------------------------------------------
# LOADERS  (offline; artifacts are read from disk, never retrained)
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


# Datasets
train_df = load_dataset("train.csv")
validation_df = load_dataset("validation.csv")
test_df = load_dataset("test.csv")

train_pca_df = load_dataset("train_pca.csv")
test_pca_df = load_dataset("test_pca.csv")

train_lda_df = load_dataset("train_lda.csv")
test_lda_df = load_dataset("test_lda.csv")

test_raw_scaled_df = load_dataset("test_raw_scaled.csv")

# Models & serialized transforms
best_classifier = load_estimator("best_classifier.joblib")
best_regressor = load_estimator("best_regressor.joblib")
scaler_raw = load_estimator("scaler.joblib")
scaler_pca = load_estimator("pca_scaler.joblib")
scaler_ns = load_estimator("non_spend_scaler.joblib")
pca_model = load_estimator("pca.joblib")
lda_model = load_estimator("lda.joblib")

# Result metadata
class_results = load_json_file(os.path.join(PROCESSED_DIR, "classification_results.json"))
reg_results = load_json_file(os.path.join(PROCESSED_DIR, "regression_results.json"))
pca_metadata = load_json_file(os.path.join(PROCESSED_DIR, "pca_metadata.json"))
metadata = load_json_file(os.path.join(PROCESSED_DIR, "metadata.json"))
rl_eval_results = load_json_file(os.path.join(PROCESSED_DIR, "rl_evaluation_results.json"))


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

# =============================================================================
# DESIGN SYSTEM  (single light theme)
# =============================================================================
INK = "#0f172a"        # primary text
MUTED = "#64748b"      # secondary text
FAINT = "#94a3b8"      # tertiary text
BORDER = "#e6eaf0"     # hairlines
SURFACE = "#ffffff"    # cards
CANVAS = "#f5f7fb"     # app background
PRIMARY = "#2563eb"    # brand / actions
PRIMARY_DK = "#1d4ed8"
POSITIVE = "#059669"   # high-value / good
NEGATIVE = "#dc2626"   # standard / risk
AMBER = "#d97706"
GRID = "#eef1f6"

# Cohesive categorical palette for charts (no prominent purple).
CHART_COLORS = ["#2563eb", "#059669", "#d97706", "#dc2626", "#0891b2", "#64748b"]

# Marketing action reference (shared across views).
ACTIONS = ["No Action", "10% Discount Coupon", "Free Premium Trial"]
ACTION_COSTS = ["$0", "$1", "$5"]

# Icon library (Lucide-style inline SVG, stroke = currentColor). No emoji as icons.
_ICON_PATHS = {
    "gauge": '<path d="M12 14 4 6"/><path d="M20.4 14.5A9 9 0 1 0 3.6 14.5"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M6 21v-2a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v2"/>',
    "chart": '<path d="M3 3v18h18"/><rect x="7" y="10" width="3" height="7"/><rect x="12" y="6" width="3" height="11"/><rect x="17" y="13" width="3" height="4"/>',
    "brain": '<path d="M12 5a3 3 0 1 0-5.997.142M12 5a3 3 0 1 1 5.997.142M12 5v14M6.003 5.142A3 3 0 0 0 4 8c0 .5.1.9.3 1.3A3 3 0 0 0 5 15M18 8a3 3 0 0 0-.003-2.858M19 15a3 3 0 0 0 .7-5.7c.2-.4.3-.8.3-1.3"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "layers": '<path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/>',
    "trending": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "repeat": '<path d="m17 2 4 4-4 4"/><path d="M3 11v-1a4 4 0 0 1 4-4h14"/><path d="m7 22-4-4 4-4"/><path d="M21 13v1a4 4 0 0 1-4 4H3"/>',
    "dollar": '<line x1="12" y1="2" x2="12" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "grid": '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/>',
    "sparkles": '<path d="m12 3 1.9 5.8L20 10.7l-6.1 1.9L12 18l-1.9-5.4L4 10.7l6.1-1.9L12 3Z"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "alert": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
    "flag": '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1Z"/><line x1="4" y1="22" x2="4" y2="15"/>',
    "route": '<circle cx="6" cy="19" r="3"/><path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/><circle cx="18" cy="5" r="3"/>',
    "compass": '<circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>',
    "tag": '<path d="M12.586 2.586A2 2 0 0 0 11.172 2H4a2 2 0 0 0-2 2v7.172a2 2 0 0 0 .586 1.414l8 8a2 2 0 0 0 2.828 0l7.172-7.172a2 2 0 0 0 0-2.828z"/><circle cx="7.5" cy="7.5" r="1"/>',
    "book": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
}


def icon(name: str, size: int = 18) -> str:
    body = _ICON_PATHS.get(name, _ICON_PATHS["info"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round" style="display:block">{body}</svg>'
    )


# -----------------------------------------------------------------------------
# GLOBAL STYLES
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap" rel="stylesheet">

    <style>
    :root {{
        --ink:{INK}; --muted:{MUTED}; --faint:{FAINT};
        --border:{BORDER}; --surface:{SURFACE}; --canvas:{CANVAS};
        --primary:{PRIMARY}; --primary-dk:{PRIMARY_DK};
        --positive:{POSITIVE}; --negative:{NEGATIVE}; --amber:{AMBER};
        --radius:16px;
        --shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.05);
        --font-body:'Inter',system-ui,sans-serif;
        --font-head:'Plus Jakarta Sans','Inter',sans-serif;
    }}

    html, body, [class*="css"], .stMarkdown, input, textarea, button {{
        font-family: var(--font-body) !important;
    }}
    
    .stApp {{ background: var(--canvas) !important; }}
    
    .block-container {{ padding-top: 0.2rem !important; padding-bottom: 3rem; max-width: 1360px; }}
    .main .block-container {{ padding-top: 0.2rem !important; }}

    #MainMenu, footer {{ visibility: hidden; }}
    .stAppDeployButton {{ display: none; }}
    header[data-testid="stHeader"] {{ display:none !important; }}
    
    section[data-testid="stSidebar"] > div:first-child {{ padding-top: 0.2rem !important; overflow: hidden !important; }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 0.2rem !important; padding-bottom: 0.2rem !important; }}

    .stApp h1,.stApp h2,.stApp h3,.stApp h4 {{ font-family: var(--font-head) !important; color:var(--ink); }}
    .stApp p,.stApp label,.stApp span,.stApp li,.stApp small {{ color: var(--ink); }}

    /* ------------------------------------------------------ SIDEBAR */
    section[data-testid="stSidebar"] {{
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
        min-width: 300px !important; max-width: 300px !important; width: 300px !important;
        transform: none !important; visibility: visible !important;
    }}
    section[data-testid="stSidebar"] > div {{ padding: 1.2rem 1.1rem; }}
    
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarResizeHandle"] {{ display:none !important; }}

    .brand {{ display:flex; align-items:center; gap:12px; padding:16px 0 16px 0;
        margin-bottom:6px; border-bottom:1px solid var(--border); }}
    .brand-mark {{ width:42px; height:42px; border-radius:12px; flex:0 0 auto;
        display:flex; align-items:center; justify-content:center; font-size:1.15rem;
        background: linear-gradient(140deg,#2563eb,#1d4ed8); color:#fff;
        box-shadow: 0 6px 18px rgba(37,99,235,.30); }}
    .brand-name {{ font-family:var(--font-head); font-size:1.1rem; font-weight:800;
        color:var(--ink); letter-spacing:-0.02em; line-height:1.1; }}
    .brand-sub {{ font-size:0.66rem; color:var(--faint); text-transform:uppercase;
        letter-spacing:.14em; font-weight:700; margin-top:3px; }}

    .rail-label {{ font-size:0.66rem; font-weight:800; text-transform:uppercase;
        letter-spacing:.13em; color:var(--faint); margin:22px 2px 9px 2px;
        display:flex; align-items:center; gap:7px; }}
    .rail-label svg {{ color:var(--primary); }}

    /* status list */
    .status-box {{ background:var(--canvas); border:1px solid var(--border);
        border-radius:12px; padding:6px 12px; }}
    .status-row {{ display:flex; align-items:center; justify-content:space-between; padding:8px 0;
        border-bottom:1px solid var(--border); }}
    .status-row:last-child {{ border-bottom:none; }}
    .status-name {{ font-size:0.83rem; color:var(--muted); font-weight:500; }}
    .status-pill {{ display:inline-flex; align-items:center; gap:5px; font-size:0.7rem;
        font-weight:700; padding:3px 9px; border-radius:999px; }}
    .status-pill.ok {{ background:rgba(5,150,105,.10); color:var(--positive); }}
    .status-pill.ok svg {{ color:var(--positive); }}
    .status-pill.off {{ background:rgba(220,38,38,.10); color:var(--negative); }}
    .status-pill.off svg {{ color:var(--negative); }}

    /* active-customer card in sidebar */
    .cust-card {{ background:linear-gradient(140deg,#1e3a8a,#2563eb); border-radius:14px;
        padding:16px; color:#fff; box-shadow:0 8px 22px rgba(37,99,235,.28); }}
    .cust-card .cc-k {{ font-size:0.66rem; text-transform:uppercase; letter-spacing:.12em;
        font-weight:700; color:#bfdbfe; }}
    .cust-card .cc-v {{ font-family:var(--font-head); font-size:1.6rem; font-weight:800; margin-top:2px; }}
    .cust-card .cc-s {{ font-size:0.74rem; color:#dbeafe; margin-top:4px; }}

    .sb-help {{ font-size:0.78rem; color:var(--muted); line-height:1.55; }}

    /* ------------------------------------------------------ TAB NAVIGATION */
    div[data-testid="stTabs"] {{ position:relative; }}
    div[data-testid="stTabs"] div[data-baseweb="tab-list"] {{
        gap:4px; background:var(--surface); border:1px solid var(--border);
        border-radius:14px; padding:6px;
        position:sticky; top:8px; z-index:1000;
        box-shadow:0 0 0 8px var(--canvas), var(--shadow);
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"] {{
        height:auto; padding:9px 16px; border-radius:10px; background:transparent;
        color:var(--muted) !important; font-weight:600; font-size:0.9rem;
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {{ background:var(--canvas); }}
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
        background:var(--primary); color:#fff !important; box-shadow:0 4px 12px rgba(37,99,235,.28);
    }}
    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] div[data-baseweb="tab-border"] {{ display:none; }}
    div[data-testid="stTabs"] div[data-baseweb="tab-panel"] {{ padding-top:22px; }}

    /* ------------------------------------------------------ PAGE HEADER */
    .page-head {{ margin-bottom:18px; }}
    .page-title {{ font-family:var(--font-head); font-size:1.5rem; font-weight:800;
        letter-spacing:-0.02em; line-height:1.15; color:var(--ink); }}
    .page-sub {{ font-size:0.92rem; color:var(--muted); margin-top:4px; max-width:80ch; }}

    /* ------------------------------------------------------ HERO */
    .hero {{ position:relative; overflow:hidden; border-radius:20px; padding:34px 36px;
        margin-bottom:22px; color:#fff;
        background:radial-gradient(120% 140% at 0% 0%, #1e3a8a 0%, #172554 55%, #0f172a 100%);
        border:1px solid rgba(255,255,255,.06); }}
    .hero .eyebrow {{ display:inline-flex; align-items:center; gap:8px; font-size:0.72rem;
        font-weight:700; text-transform:uppercase; letter-spacing:.16em; color:#93c5fd;
        background:rgba(147,197,253,.12); padding:5px 12px; border-radius:999px; margin-bottom:14px; }}
    .hero .hero-title {{ color:#f8fafc; font-family:var(--font-head); 
        font-size:2.05rem; font-weight:800; 
        margin:0 0 10px 0; letter-spacing:-0.025em; line-height:1.12; }}
    .hero p {{ color:#cbd5e1; font-size:0.96rem; margin:0; line-height:1.65; max-width:72ch; }}

    /* ------------------------------------------------------ STAT CARDS */
    .stat {{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
        padding:18px 20px; box-shadow:var(--shadow); height:100%;
        transition:transform .16s ease, box-shadow .16s ease; }}
    .stat:hover {{ transform:translateY(-3px); box-shadow:0 12px 28px rgba(15,23,42,.10); }}
    .stat-top {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:14px; }}
    .stat-label {{ font-size:0.72rem; text-transform:uppercase; letter-spacing:.08em;
        color:var(--muted); font-weight:700; }}
    .stat-ic {{ width:36px; height:36px; border-radius:10px; flex:0 0 auto;
        display:flex; align-items:center; justify-content:center; }}
    .stat-ic.blue{{background:rgba(37,99,235,.10);color:#2563eb;}}
    .stat-ic.teal{{background:rgba(8,145,178,.10);color:#0891b2;}}
    .stat-ic.green{{background:rgba(5,150,105,.10);color:#059669;}}
    .stat-ic.amber{{background:rgba(217,119,6,.12);color:#d97706;}}
    .stat-ic.rose{{background:rgba(220,38,38,.10);color:#dc2626;}}
    .stat-ic.slate{{background:rgba(100,116,139,.12);color:#475569;}}
    .stat-value {{ font-family:var(--font-head); font-size:1.6rem; font-weight:800;
        line-height:1.1; color:var(--ink); letter-spacing:-0.02em; }}
    .stat-sub {{ font-size:0.78rem; color:var(--muted); margin-top:7px; line-height:1.45; }}

    /* ------------------------------------------------------ SECTION TITLE */
    .sec {{ display:flex; align-items:center; gap:11px; margin:6px 0 14px 0; }}
    .sec-ic {{ width:34px; height:34px; border-radius:10px; flex:0 0 auto; display:flex;
        align-items:center; justify-content:center; background:rgba(37,99,235,.10); color:var(--primary); }}
    .sec-title {{ font-family:var(--font-head); font-size:1.12rem; font-weight:700;
        letter-spacing:-0.01em; color:var(--ink); }}
    .sec-desc {{ font-size:0.84rem; color:var(--muted); margin-top:2px; }}

    /* container(border=True) -> panel language */
    div[data-testid="stVerticalBlockBorderWrapper"] > div:first-child {{
        background:var(--surface) !important; border:1px solid var(--border) !important;
        border-radius:var(--radius) !important; padding:22px !important; box-shadow:var(--shadow) !important;
    }}
    .panel-title {{ font-family:var(--font-head); font-weight:700; font-size:1rem;
        color:var(--ink); margin-bottom:14px; display:flex; align-items:center; gap:9px; }}
    .panel-title svg {{ color:var(--primary); }}

    .plot-card {{ border:1px solid var(--border); border-radius:var(--radius); padding:12px;
        background:var(--surface); box-shadow:var(--shadow); margin-bottom:14px; }}

    /* ------------------------------------------------------ RESULT PANEL */
    .result {{ border:1px solid var(--border); border-radius:14px; padding:18px; margin-bottom:14px;
        background:var(--canvas); }}
    .result .r-label {{ font-size:0.72rem; text-transform:uppercase; letter-spacing:.07em;
        color:var(--muted); font-weight:700; display:flex; align-items:center; gap:7px; }}
    .result .r-value {{ font-family:var(--font-head); font-size:1.55rem; font-weight:800;
        line-height:1.15; margin-top:8px; }}
    .result .r-meta {{ font-size:0.78rem; color:var(--muted); margin-top:6px; }}
    .conf-track {{ height:9px; border-radius:999px; background:#e5e9f0; overflow:hidden; margin-top:12px; }}
    .conf-fill {{ height:100%; border-radius:999px; transition:width .4s ease; }}
    .seg-tag {{ display:inline-flex; align-items:center; gap:6px; font-size:0.72rem; font-weight:700;
        padding:3px 10px; border-radius:999px; }}

    /* ------------------------------------------------------ ACTION ROWS (RL) */
    .act {{ display:flex; align-items:center; gap:14px; border:1px solid var(--border);
        border-radius:13px; padding:14px 16px; margin-bottom:10px; background:var(--surface);
        transition:border-color .15s ease; }}
    .act.rec {{ border:1.5px solid var(--primary); background:rgba(37,99,235,.05);
        box-shadow:0 6px 18px rgba(37,99,235,.12); }}
    .act-num {{ width:30px; height:30px; border-radius:9px; flex:0 0 auto; display:flex;
        align-items:center; justify-content:center; font-weight:800; font-size:0.85rem;
        background:var(--canvas); color:var(--muted); font-family:var(--font-head); }}
    .act.rec .act-num {{ background:var(--primary); color:#fff; }}
    .act-body {{ flex:1; min-width:0; }}
    .act-name {{ font-weight:700; font-size:0.95rem; color:var(--ink); display:flex;
        align-items:center; gap:9px; flex-wrap:wrap; }}
    .act-cost {{ font-size:0.76rem; color:var(--muted); margin-top:2px; }}
    .act-q {{ text-align:right; flex:0 0 auto; }}
    .act-q .q-val {{ font-family:var(--font-head); font-weight:800; font-size:1.1rem; color:var(--ink); }}
    .act-q .q-cap {{ font-size:0.66rem; text-transform:uppercase; letter-spacing:.06em; color:var(--faint); font-weight:700; }}
    .rec-badge {{ display:inline-flex; align-items:center; gap:5px; font-size:0.66rem; font-weight:800;
        text-transform:uppercase; letter-spacing:.06em; color:#fff; background:var(--primary);
        padding:3px 9px; border-radius:999px; }}
    .qbar-track {{ height:6px; border-radius:999px; background:#e5e9f0; overflow:hidden; margin-top:9px; }}
    .qbar-fill {{ height:100%; border-radius:999px; background:var(--faint); }}
    .act.rec .qbar-fill {{ background:var(--primary); }}

    /* ------------------------------------------------------ CALLOUT / BADGE / EMPTY */
    .callout {{ border:1px solid var(--border); border-left:4px solid var(--primary);
        background:var(--canvas); border-radius:12px; padding:16px 18px; }}
    .callout .co-label {{ font-size:0.68rem; font-weight:800; text-transform:uppercase;
        letter-spacing:.09em; color:var(--primary); }}
    .callout .co-value {{ font-family:var(--font-head); font-size:1.1rem; font-weight:800; margin-top:5px; }}
    .callout .co-body {{ font-size:0.9rem; color:var(--ink); margin-top:9px; line-height:1.55; }}

    .badge {{ background:var(--surface); color:var(--ink); border:1px solid var(--border);
        padding:7px 14px; border-radius:999px; font-size:.8rem; font-weight:600;
        display:inline-flex; align-items:center; gap:6px; margin:4px 8px 4px 0; box-shadow:var(--shadow); }}

    .empty {{ text-align:center; padding:44px 24px; background:var(--surface);
        border:1px dashed var(--border); border-radius:var(--radius); }}
    .empty-ic {{ width:52px; height:52px; border-radius:14px; margin:0 auto 14px auto; display:flex;
        align-items:center; justify-content:center; background:rgba(217,119,6,.12); color:var(--amber); }}
    .empty h4 {{ font-family:var(--font-head); font-weight:700; font-size:1.05rem; margin:0; color:var(--ink); }}
    .empty p {{ color:var(--muted); font-size:0.9rem; margin-top:8px; line-height:1.55; }}

    /* timeline */
    .timeline {{ border-left:2px solid var(--border); padding-left:22px; margin-left:8px; }}
    .timeline-item {{ margin-bottom:22px; position:relative; }}
    .timeline-item::before {{ content:''; position:absolute; left:-29px; top:3px;
        background:var(--primary); border:3px solid var(--surface); border-radius:50%;
        width:13px; height:13px; box-shadow:0 0 0 1px var(--border); }}
    .timeline-title {{ font-family:var(--font-head); font-weight:700; color:var(--ink); margin-bottom:3px; }}
    .timeline-body {{ font-size:0.88rem; color:var(--muted); line-height:1.55; }}

    /* ------------------------------------------------------ TABLES - FIXED */
    div[data-testid="stTable"] {{
        background: var(--surface) !important;
        border-radius: var(--radius) !important;
        overflow: hidden !important;
        border: 1px solid var(--border) !important;
    }}
    
    div[data-testid="stTable"] table {{
        width: 100% !important;
        border-collapse: collapse !important;
        background: var(--surface) !important;
    }}
    
    div[data-testid="stTable"] thead tr th {{
        background: var(--canvas) !important;
        color: var(--muted) !important;
        border-bottom: 2px solid var(--border) !important;
        padding: 12px 14px !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: .05em !important;
        text-align: left !important;
    }}
    
    div[data-testid="stTable"] tbody tr td {{
        border-bottom: 1px solid var(--border) !important;
        padding: 11px 14px !important;
        color: var(--ink) !important;
        font-size: 0.875rem !important;
        background: var(--surface) !important;
    }}
    
    div[data-testid="stTable"] tbody tr:last-child td {{
        border-bottom: none !important;
    }}
    
    div[data-testid="stTable"] tbody tr:hover td {{
        background: var(--canvas) !important;
    }}

    /* ------------------------------------------------------ EXPANDER */
    div[data-testid="stExpander"] {{ border:1px solid var(--border) !important;
        background:var(--surface) !important; border-radius:14px !important;
        box-shadow:var(--shadow); overflow:hidden; }}
    div[data-testid="stExpander"] details {{ background:var(--surface) !important; }}
    div[data-testid="stExpander"] summary {{ background:var(--canvas) !important;
        color:var(--ink) !important; font-weight:600; padding:12px 16px !important; }}
    div[data-testid="stExpander"] summary:hover {{ color:var(--primary) !important; }}
    div[data-testid="stExpander"] summary svg {{ fill:var(--muted) !important; }}
    div[data-testid="stExpander"] summary p {{ color:var(--ink) !important; font-weight:600 !important; }}
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
        background:var(--surface) !important; padding:16px !important; }}

    /* ------------------------------------------------------ DATAFRAME - FIXED */
    div[data-testid="stDataFrame"] {{
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        overflow: hidden !important;
        background: var(--surface) !important;
    }}
    
    div[data-testid="stDataFrame"] * {{
        color: var(--ink) !important;
    }}
    
    div[data-testid="stDataFrame"] table {{
        background: var(--surface) !important;
    }}
    
    div[data-testid="stDataFrame"] thead th {{
        background: var(--canvas) !important;
        color: var(--muted) !important;
        border-bottom: 2px solid var(--border) !important;
    }}
    
    div[data-testid="stDataFrame"] tbody td {{
        background: var(--surface) !important;
        border-bottom: 1px solid var(--border) !important;
        color: var(--ink) !important;
    }}
    
    div[data-testid="stDataFrame"] tbody tr:hover td {{
        background: var(--canvas) !important;
    }}

    /* ------------------------------------------------------ SELECT BOX */
    div[data-baseweb="select"] > div {{
        background: var(--surface) !important;
        border-color: var(--border) !important;
        border-radius: 10px !important;
        color: var(--ink) !important;
    }}
    
    div[data-baseweb="select"] input {{
        color: var(--ink) !important;
    }}
    
    div[data-baseweb="select"] * {{
        color: var(--ink) !important;
    }}
    
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background: var(--canvas) !important;
    }}
    
    /* ------------------------------------------------------ POPOVER */
    div[data-baseweb="popover"] div[role="listbox"],
    div[data-baseweb="popover"] ul[role="listbox"],
    ul[data-baseweb="menu"] {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 30px rgba(15,23,42,.14) !important;
    }}
    
    div[data-baseweb="popover"] li,
    ul[data-baseweb="menu"] li {{
        color: var(--ink) !important;
        background: var(--surface) !important;
    }}
    
    div[data-baseweb="popover"] li:hover,
    ul[data-baseweb="menu"] li:hover,
    div[data-baseweb="popover"] li[aria-selected="true"],
    ul[data-baseweb="menu"] li[aria-selected="true"] {{
        background: var(--canvas) !important;
    }}

    /* Remove the logo spacer that's creating the gap */
    [data-testid="stSidebarHeader"] {{
        display: none !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# UI HELPERS
# -----------------------------------------------------------------------------
def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='page-head'><div class='page-title'>{title}</div>"
        f"<div class='page-sub'>{subtitle}</div></div>",
        unsafe_allow_html=True,
    )


def section_title(text: str, icon_name: str = "activity", desc: str = "") -> None:
    d = f"<div class='sec-desc'>{desc}</div>" if desc else ""
    st.markdown(
        f"<div class='sec'><div class='sec-ic'>{icon(icon_name, 18)}</div>"
        f"<div><div class='sec-title'>{text}</div>{d}</div></div>",
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, sub: str = "", accent: str = "blue", icon_name: str = "activity") -> None:
    s = f"<div class='stat-sub'>{sub}</div>" if sub else ""
    st.markdown(
        f"""
        <div class="stat">
          <div class="stat-top">
            <div class="stat-label">{label}</div>
            <div class="stat-ic {accent}">{icon(icon_name, 18)}</div>
          </div>
          <div class="stat-value">{value}</div>
          {s}
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_title(text: str, icon_name: str = "activity") -> None:
    st.markdown(f"<div class='panel-title'>{icon(icon_name, 17)}{text}</div>", unsafe_allow_html=True)


def empty_state(title: str, body: str, icon_name: str = "alert") -> None:
    st.markdown(
        f"""
        <div class="empty">
          <div class="empty-ic">{icon(icon_name, 26)}</div>
          <h4>{title}</h4>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plot(fig, height: int = 340, legend_title: Optional[str] = None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=70, r=90, t=34, b=58),
        font=dict(color=INK, family="Inter", size=12),
        legend=dict(font=dict(color=INK)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER, automargin=True),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER, automargin=True),
        colorway=CHART_COLORS,
        title=None,
    )
    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    return fig


def vspace(h: int = 8) -> None:
    st.markdown(f"<div style='height:{h}px'></div>", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR — BRAND · CUSTOMER SELECTOR · SYSTEM STATUS
# =============================================================================
st.sidebar.markdown(
    """
    <div class="brand">
      <div class="brand-mark">◆</div>
      <div>
        <div class="brand-name">SmartRetail&nbsp;AI</div>
        <div class="brand-sub">Customer Intelligence</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Customer selector (required GUI component) ---
st.sidebar.markdown(
    f"<div class='rail-label'>{icon('user', 13)}Customer Selector</div>", unsafe_allow_html=True
)
if test_df is not None:
    customer_ids = sorted([int(cid) for cid in test_df.index.tolist()])
    selected_customer_id = st.sidebar.selectbox(
        "Customer ID (test split)",
        customer_ids,
        label_visibility="collapsed",
        help="Choose a Customer ID from the held-out test split to analyze.",
    )
    st.sidebar.markdown(
        f"""
        <div class="cust-card">
          <div class="cc-k">Active customer</div>
          <div class="cc-v">#{selected_customer_id}</div>
          <div class="cc-s">Source · held-out test split</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.sidebar.warning("Datasets not loaded. Run preprocessing first.")
    selected_customer_id = None

# --- System status (visibility of system status) ---
st.sidebar.markdown(
    f"<div class='rail-label'>{icon('activity', 13)}System Status</div>", unsafe_allow_html=True
)


def _status_row(label: str, ok: bool) -> str:
    cls = "ok" if ok else "off"
    dot = "check" if ok else "alert"
    state = "Ready" if ok else "Offline"
    return (
        f"<div class='status-row'><span class='status-name'>{label}</span>"
        f"<span class='status-pill {cls}'>{icon(dot, 12)}{state}</span></div>"
    )


data_ok = test_df is not None
models_ok = (best_classifier is not None) and (best_regressor is not None)
rl_ok = (dqn_agent is not None) and (q_agent is not None) and (discretizer is not None)

st.sidebar.markdown(
    "<div class='status-box'>"
    + _status_row("Datasets", data_ok)
    + _status_row("Supervised models", models_ok)
    + _status_row("RL agents", rl_ok)
    + "</div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"<div class='rail-label'>{icon('book', 13)}How to use</div>", unsafe_allow_html=True
)
st.sidebar.markdown(
    "<div class='sb-help'>Pick a Customer ID above, then open the "
    "<strong>Customer Intelligence</strong> tab to view its loyalty prediction, "
    "spend forecast, and the recommended marketing action.</div>",
    unsafe_allow_html=True,
)

# =============================================================================
# MAIN — TAB NAVIGATION
# =============================================================================
tab_overview, tab_customer, tab_proj, tab_models, tab_data, tab_about = st.tabs(
    [
        "Overview",
        "Customer Intelligence",
        "PCA / LDA Projections",
        "Model Performance",
        "Dataset Explorer",
        "About",
    ]
)

# -----------------------------------------------------------------------------
# TAB: OVERVIEW
# -----------------------------------------------------------------------------
with tab_overview:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Customer Loyalty Engine</div>
          <div class="hero-title">SmartRetail AI Intelligence Platform</div>
          <p>An offline analytics platform that scores customer loyalty, forecasts future spend,
          and applies reinforcement-learning policies to recommend the most profitable marketing
          action for every customer &mdash; all served from pre-trained model artifacts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_customers = 0
    if metadata:
        total_customers = (
            metadata.get("train_customers_count", 0)
            + metadata.get("val_customers_count", 0)
            + metadata.get("test_customers_count", 0)
        )
    best_classifier_name = (
        class_results.get("best_model_selected", "N/A").replace("_", " ") if class_results else "N/A"
    )
    best_regressor_name = (
        reg_results.get("best_model_selected", "N/A").replace("_", " ") if reg_results else "N/A"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Total Profiles", f"{total_customers:,}", "Customers across all splits", "blue", "users")
    with c2:
        stat_card("Feature Space", "13 Dimensions", "Behavioral + category spend", "teal", "layers")
    with c3:
        stat_card("Best Classifier", best_classifier_name, "Top validation F1-score", "amber", "cpu")
    with c4:
        stat_card("Best Regressor", best_regressor_name, "Top validation R²-score", "green", "trending")

    vspace(20)

    # ---- REQUIRED: cumulative-profit summary chart (RL policy vs baselines) ----
    section_title(
        "Cumulative Policy Net Profit",
        "route",
        "Net profit each policy would have earned on the held-out test set — RL policies vs. baselines.",
    )
    col_l, col_r = st.columns([3, 2])

    with col_l:
        with st.container(border=True):
            panel_title("Policy Comparison · Test Set", "chart")
            if rl_eval_results:
                profits_data = pd.DataFrame(
                    {
                        "Policy": ["Always No-Action", "Random Action", "Tabular Q-Learning", "DQN Policy"],
                        "Net Profit ($)": [
                            rl_eval_results.get("Always_No_Action", 0),
                            rl_eval_results.get("Random_Action", 0),
                            rl_eval_results.get("Tabular_Q_Policy", 0),
                            rl_eval_results.get("DQN_Policy", 0),
                        ],
                        "Kind": ["Baseline", "Baseline", "RL Policy", "RL Policy"],
                    }
                )
                profits_data = profits_data.sort_values("Net Profit ($)")
                fig = px.bar(
                    profits_data,
                    x="Net Profit ($)",
                    y="Policy",
                    orientation="h",
                    color="Policy",
                    color_discrete_map={
                        "Always No-Action": "#dc2626",
                        "Random Action": "#d97706",
                        "Tabular Q-Learning": "#0891b2",
                        "DQN Policy": "#2563eb",
                    },
                    text="Net Profit ($)",
                )
                style_plot(fig, height=360)
                fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Net Profit ($)")
                fig.update_traces(
                    texttemplate="$%{text:,.0f}", textposition="outside",
                    textfont=dict(color=INK),
                    cliponaxis=False,
                )
                st.plotly_chart(fig, width="stretch", theme=None)
            else:
                empty_state(
                    "RL evaluation results not found",
                    "Run train_rl.py to generate rl_evaluation_results.json, then reload.",
                )

    with col_r:
        with st.container(border=True):
            panel_title("Segment Policy Playbook", "compass")
            st.markdown(
                """
**High-Value Loyalty Segment**  
The DQN agent prefers **Free Premium Trial** (cost $5).
This segment responds with a 2× spend multiplier, yielding strong positive net profit.

**Medium-Value Repeat Segment**  
The agent prefers a **10% Discount Coupon** (cost $1),
stimulating purchase frequency without exhausting the marketing budget.

**At-Risk / Low-Value Segment**  
The agent selects **No Action** (cost $0),
preserving capital for higher-value customers.
                """
            )

# -----------------------------------------------------------------------------
# TAB: CUSTOMER INTELLIGENCE  (predictions + RL recommendation)
# -----------------------------------------------------------------------------
with tab_customer:
    page_header(
        "Customer Intelligence",
        "Supervised predictions and the reinforcement-learning marketing recommendation for the selected customer.",
    )

    if selected_customer_id is None:
        empty_state(
            "No customer selected",
            "Choose a Customer ID from the sidebar to inspect a profile and generate recommendations.",
        )
    else:
        cust_row = test_df.loc[selected_customer_id]

        # --- profile snapshot ---
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            stat_card("Recency", f"{int(cust_row['Recency'])} days", "Since last transaction", "blue", "clock")
        with p2:
            stat_card("Frequency", f"{int(cust_row['Frequency'])} orders", "Unique transactions", "teal", "repeat")
        with p3:
            stat_card("Monetary", f"${cust_row['Monetary']:,.2f}", "Spend · Months 1–9", "green", "dollar")
        with p4:
            stat_card("Diversity", f"{int(cust_row['ProductDiversity'])} items", "Distinct products bought", "amber", "grid")

        vspace(18)

        # ---- REQUIRED: classification + regression result panel ----
        section_title(
            "Supervised Predictions",
            "cpu",
            "Loyalty classification (label + confidence) and future-spend regression (prediction + RMSE).",
        )
        pred_l, pred_r = st.columns(2)

        # classification
        with pred_l:
            with st.container(border=True):
                panel_title("Loyalty Classification", "target")
                best_rep = class_results.get("best_model_representation", "raw_scaled") if class_results else "raw_scaled"
                if best_rep == "raw_scaled" and test_raw_scaled_df is not None:
                    features = [c for c in test_raw_scaled_df.columns if c not in ["High_Value_Customer", "Future_Spend"]]
                    X = test_raw_scaled_df.loc[selected_customer_id][features].values.reshape(1, -1)
                elif best_rep == "pca" and test_pca_df is not None:
                    features = [c for c in test_pca_df.columns if c not in ["High_Value_Customer", "Future_Spend"]]
                    X = test_pca_df.loc[selected_customer_id][features].values.reshape(1, -1)
                elif best_rep == "lda" and test_lda_df is not None:
                    features = [c for c in test_lda_df.columns if c not in ["High_Value_Customer", "Future_Spend"]]
                    X = test_lda_df.loc[selected_customer_id][features].values.reshape(1, -1)
                else:
                    X = None

                if X is not None and best_classifier is not None:
                    pred_class = int(best_classifier.predict(X)[0])
                    pred_proba = best_classifier.predict_proba(X)[0][pred_class]
                    conf = pred_proba * 100.0
                    if pred_class == 1:
                        seg_color, seg_label, seg_bg = POSITIVE, "High-Value Customer", "rgba(5,150,105,.10)"
                    else:
                        seg_color, seg_label, seg_bg = NEGATIVE, "Standard Customer", "rgba(220,38,38,.10)"
                    st.markdown(
                        f"""
                        <div class="result">
                          <div class="r-label">{icon('user', 13)}Predicted loyalty segment</div>
                          <div class="r-value" style="color:{seg_color};">{seg_label}</div>
                          <div style="margin-top:10px;">
                            <span class="seg-tag" style="background:{seg_bg}; color:{seg_color};">
                              {icon('flag', 12)}Class {pred_class}</span>
                          </div>
                          <div class="conf-track"><div class="conf-fill" style="width:{conf:.0f}%; background:{seg_color};"></div></div>
                          <div class="r-meta">{conf:.1f}% model confidence &middot; representation: {best_rep}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    empty_state("Classifier offline", "The classification artifact could not be loaded.", "alert")

        # regression
        with pred_r:
            with st.container(border=True):
                panel_title("Future Spend Forecast", "trending")
                if test_raw_scaled_df is not None and best_regressor is not None:
                    features = [c for c in test_raw_scaled_df.columns if c not in ["High_Value_Customer", "Future_Spend"]]
                    X_reg = test_raw_scaled_df.loc[selected_customer_id][features].values.reshape(1, -1)
                    pred_spend = max(0.0, float(best_regressor.predict(X_reg)[0]))
                    test_rmse = reg_results.get("test_evaluation", {}).get("rmse", 0.0) if reg_results else 0.0
                    lo = max(0.0, pred_spend - test_rmse)
                    hi = pred_spend + test_rmse
                    st.markdown(
                        f"""
                        <div class="result">
                          <div class="r-label">{icon('dollar', 13)}Predicted spend &middot; Months 10–12</div>
                          <div class="r-value" style="color:{PRIMARY};">${pred_spend:,.2f}</div>
                          <div style="margin-top:10px;">
                            <span class="seg-tag" style="background:rgba(37,99,235,.10); color:{PRIMARY};">
                              {icon('activity', 12)}RMSE ± ${test_rmse:,.2f}</span>
                          </div>
                          <div class="r-meta">Expected range ${lo:,.2f} – ${hi:,.2f} within one RMSE margin.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    empty_state("Regressor offline", "The regression artifact could not be loaded.", "alert")

        vspace(6)

        # category composition
        with st.container(border=True):
            panel_title("Category Spending Composition", "grid")
            categories = ["Homeware", "Stationery", "Gadgets", "Decorations", "Kitchenware"]
            spend_pcts = [cust_row[f"{c}_Spend_Pct"] * 100.0 for c in categories]
            total_cat = sum(spend_pcts)
            other_pct = max(0.0, 100.0 - total_cat)
            categories.append("Other")
            spend_pcts.append(other_pct)
            spend_mix = pd.DataFrame({"Category": categories, "Percentage (%)": spend_pcts})
            fig = px.bar(
                spend_mix, y="Category", x="Percentage (%)", orientation="h",
                color="Category", color_discrete_sequence=CHART_COLORS, text="Percentage (%)",
            )
            style_plot(fig, height=250)
            fig.update_layout(showlegend=False, xaxis_title="Percentage (%)", yaxis_title="")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              textfont=dict(color=INK), cliponaxis=False)
            st.plotly_chart(fig, width="stretch", theme=None)

        vspace(14)

        # ---- REQUIRED: RL Q-values for all 3 actions + highlight argmax ----
        section_title(
            "Reinforcement Learning Recommendation",
            "route",
            "Estimated Q-value for each of the 3 marketing actions; the DQN-selected (argmax) action is highlighted.",
        )

        if dqn_agent is None or q_agent is None or discretizer is None:
            empty_state(
                "Reinforcement learning agents are offline",
                "Ensure the RL models are trained and their binaries are available, then reload.",
            )
        else:
            state_5d = test_pca_df.loc[selected_customer_id][["Recency", "Frequency", "AverageSpend", "PC1", "PC2"]].values
            state_idx = discretizer.discretize(state_5d)
            tab_q_vals = q_agent.q_table[state_idx]
            state_t = torch.FloatTensor(state_5d).unsqueeze(0).to(dqn_agent.device)
            with torch.no_grad():
                dqn_q_vals = dqn_agent.policy_net(state_t).cpu().numpy()[0]
            rec_action = int(np.argmax(dqn_q_vals))

            rl_l, rl_r = st.columns([3, 2])

            with rl_l:
                with st.container(border=True):
                    panel_title(f"DQN Q-Values · State Cluster #{state_idx}", "cpu")
                    q_min = float(np.min(dqn_q_vals))
                    q_max = float(np.max(dqn_q_vals))
                    span = (q_max - q_min) or 1.0
                    rows_html = ""
                    for i, name in enumerate(ACTIONS):
                        qv = float(dqn_q_vals[i])
                        width = int(round(((qv - q_min) / span) * 100))
                        is_rec = i == rec_action
                        badge = (
                            f"<span class='rec-badge'>{icon('check', 11)}Recommended</span>"
                            if is_rec else ""
                        )
                        rows_html += f"""
                        <div class="act {'rec' if is_rec else ''}">
                          <div class="act-num">{i}</div>
                          <div class="act-body">
                            <div class="act-name">{name}{badge}</div>
                            <div class="act-cost">Acquisition cost {ACTION_COSTS[i]}</div>
                            <div class="qbar-track"><div class="qbar-fill" style="width:{width}%;"></div></div>
                          </div>
                          <div class="act-q">
                            <div class="q-val">{qv:.3f}</div>
                            <div class="q-cap">Q-value</div>
                          </div>
                        </div>
                        """
                    st.markdown(rows_html, unsafe_allow_html=True)

            with rl_r:
                with st.container(border=True):
                    panel_title("Decision & Cross-Check", "compass")
                    action_explanations = {
                        0: "Acquisition costs outweigh the expected spend multiplier for this profile, so no marketing spend is committed.",
                        1: "A 10% discount coupon (cost $1) maximizes expected repeat-transaction value for this profile.",
                        2: "A free premium trial (cost $5) is justified — high loyalty potential makes premium retention profitable.",
                    }
                    st.markdown(
                        f"""
                        <div class="callout">
                          <div class="co-label">DQN recommended action</div>
                          <div class="co-value" style="color:{PRIMARY};">Action {rec_action} · {ACTIONS[rec_action]}</div>
                          <div class="co-body">{action_explanations[rec_action]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    vspace(12)
                    st.markdown(
                        "<div style='font-size:0.78rem; color:var(--muted); font-weight:700; "
                        "text-transform:uppercase; letter-spacing:.06em; margin-bottom:8px;'>"
                        "Tabular Q-learning cross-check</div>",
                        unsafe_allow_html=True,
                    )
                    tab_rec = int(np.argmax(tab_q_vals))
                    cross = pd.DataFrame(
                        {
                            "Action": ACTIONS,
                            "DQN Q": [f"{v:.3f}" for v in dqn_q_vals],
                            "Tabular Q": [f"{v:.3f}" for v in tab_q_vals],
                        }
                    )
                    st.table(cross)
                    agree = "agree" if tab_rec == rec_action else "differ"
                    st.markdown(
                        f"<div style='font-size:0.82rem; color:var(--muted); margin-top:6px;'>"
                        f"The two agents <strong>{agree}</strong> on the optimal action "
                        f"(Tabular argmax: Action {tab_rec}).</div>",
                        unsafe_allow_html=True,
                    )

# -----------------------------------------------------------------------------
# TAB: PCA / LDA PROJECTIONS  (required PCA vs LDA toggle)
# -----------------------------------------------------------------------------
with tab_proj:
    page_header(
        "Dimensionality Reduction Projections",
        "Toggle between the unsupervised PCA scatter and the supervised LDA class view. Transforms are fit on the train split only.",
    )

    if pca_metadata is None:
        empty_state("Missing PCA/LDA metadata", "The PCA/LDA metadata file could not be found in data/processed/.")
    else:
        left, right = st.columns([2, 3])

        with left:
            section_title("Projection Summary", "sparkles")
            selected_components = pca_metadata.get("selected_components_count", 0)
            cumulative_variance = pca_metadata.get("cumulative_variance_explained", [])
            cum_var = (
                cumulative_variance[selected_components - 1]
                if selected_components > 0 and len(cumulative_variance) >= selected_components
                else 0.0
            )
            stat_card("Selected Components", f"{selected_components} / 5", "Variance target: 90%", "blue", "layers")
            vspace(10)
            stat_card("Variance Retained", f"{cum_var*100:.2f}%", "Cumulative explained variance", "green", "trending")

            lda_coefs = pca_metadata.get("lda_discriminant_coefficients", {})
            if lda_coefs:
                top_feat = max(lda_coefs, key=lambda k: abs(lda_coefs[k]))
                vspace(10)
                stat_card("Top Discriminating Feature", top_feat, f"LDA coefficient {lda_coefs[top_feat]:.3f}", "teal", "target")

            vspace(12)
            with st.expander("Unsupervised PCA", expanded=True):
                st.markdown("PCA maps category-spend fractions into orthogonal components, reducing redundancy while preserving principal variance directions.")
            with st.expander("Supervised LDA", expanded=False):
                st.markdown("LDA projects behavioral dimensions onto the discriminant axis LD1 to maximize separation between customer classes.")

            viva_explanation = pca_metadata.get("viva_explanation", "No explanation available.")
            vspace(6)
            st.markdown(
                f"<div class='callout'><div class='co-label'>Multicollinearity note</div>"
                f"<div class='co-body'>{viva_explanation}</div></div>",
                unsafe_allow_html=True,
            )

        with right:
            section_title("Projection View", "grid", "Switch the projection basis. The selected customer is marked in red.")
            # REQUIRED: PCA / LDA toggle
            view_mode = st.radio(
                "Projection view",
                ["PCA Scatter (PC1 vs PC2)", "LDA Class View (1D)"],
                horizontal=True,
                label_visibility="collapsed",
            )
            if selected_customer_id is None:
                st.info("Select a customer from the sidebar to highlight their position on the chart.")

            if view_mode == "PCA Scatter (PC1 vs PC2)":
                with st.container(border=True):
                    if train_pca_df is not None:
                        fig = px.scatter(
                            train_pca_df, x="PC1", y="PC2", color="High_Value_Customer",
                            color_discrete_map={0: "#94a3b8", 1: "#2563eb"},
                            hover_data=["Recency", "Frequency", "AverageSpend"],
                        )
                        if selected_customer_id is not None and test_pca_df is not None:
                            crow = test_pca_df.loc[selected_customer_id]
                            fig.add_trace(
                                go.Scatter(
                                    x=[crow["PC1"]], y=[crow["PC2"]], mode="markers",
                                    marker=dict(color="#dc2626", size=18, symbol="star",
                                                line=dict(color="#ffffff", width=1.5)),
                                    name=f"Customer {selected_customer_id}",
                                )
                            )
                        style_plot(fig, height=460, legend_title="Segment (Train)")
                        st.plotly_chart(fig, width="stretch", theme=None)
                    else:
                        empty_state("PCA projection unavailable", "train_pca.csv could not be loaded.")
            else:
                with st.container(border=True):
                    if train_lda_df is not None:
                        fig = px.histogram(
                            train_lda_df, x="LD1", color="High_Value_Customer",
                            color_discrete_map={0: "#94a3b8", 1: "#2563eb"},
                            marginal="box", barmode="overlay",
                        )
                        if selected_customer_id is not None and test_lda_df is not None:
                            crow = test_lda_df.loc[selected_customer_id]
                            fig.add_vline(
                                x=crow["LD1"], line_width=3, line_dash="dash", line_color="#dc2626",
                                annotation_text=f"Cust {selected_customer_id}", annotation_position="top right",
                            )
                        style_plot(fig, height=460, legend_title="Segment (Train)")
                        st.plotly_chart(fig, width="stretch", theme=None)
                    else:
                        empty_state("LDA projection unavailable", "train_lda.csv could not be loaded.")

# -----------------------------------------------------------------------------
# TAB: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
with tab_models:
    page_header(
        "Model Benchmarking Diagnostics",
        "Comparative metrics audited on the validation and test splits.",
    )

    if class_results is None or reg_results is None:
        empty_state(
            "Audit metrics not found",
            "Expected classification/regression result files were not found in data/processed/.",
        )
    else:
        section_title(
            "Classification Benchmarks — Validation Set",
            "cpu",
            "Model-selection metrics used to choose the best classifier before the one-time test evaluation.",
        )
        with st.container(border=True):
            comp_data = []
            for key, val in class_results.get("validation_comparisons", {}).items():
                comp_data.append(
                    {
                        "Model & Feature Space": key.replace("_", " "),
                        "Accuracy": f"{val.get('val_accuracy', 0)*100:.2f}%",
                        "Precision": f"{val.get('val_precision', 0)*100:.2f}%",
                        "Recall": f"{val.get('val_recall', 0)*100:.2f}%",
                        "F1-Score": f"{val.get('val_f1', 0):.4f}",
                        "ROC-AUC": f"{val.get('val_auc', 0):.4f}",
                    }
                )
            if comp_data:
                st.table(pd.DataFrame(comp_data))
            else:
                st.info("No classification validation comparison data available.")

        vspace(10)
        section_title("Test Set Scores — Best Classifier", "target")
        tm = class_results.get("test_evaluation", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            stat_card("Accuracy", f"{tm.get('accuracy', 0)*100:.1f}%", accent="blue", icon_name="target")
        with c2:
            stat_card("Precision", f"{tm.get('precision', 0)*100:.1f}%", accent="blue", icon_name="target")
        with c3:
            stat_card("Recall", f"{tm.get('recall', 0)*100:.1f}%", accent="blue", icon_name="target")
        with c4:
            stat_card("F1-Score", f"{tm.get('f1_score', 0):.3f}", accent="teal", icon_name="activity")
        with c5:
            stat_card("ROC AUC", f"{tm.get('roc_auc', 0):.3f}", accent="green", icon_name="trending")

        vspace(12)
        f1, f2, f3 = st.columns(3)
        for col, fname, cap in [
            (f1, "confusion_matrix.png", "Confusion Matrix"),
            (f2, "roc_curve.png", "ROC Curves"),
            (f3, "precision_recall_curve.png", "Precision-Recall Curves"),
        ]:
            with col:
                p = os.path.join(FIGURES_DIR, fname)
                if os.path.exists(p):
                    with st.container(border=True):
                        st.image(p, caption=cap, use_container_width=True)

        vspace(6)
        section_title(
            "Regression Benchmarks — Validation Set",
            "trending",
            "Model-selection metrics used to choose the best regressor before the one-time test evaluation.",
        )
        with st.container(border=True):
            reg_comp = []
            for key, val in reg_results.get("validation_comparisons", {}).items():
                reg_comp.append(
                    {
                        "Regressor Model": key.replace("_", " "),
                        "MSE": f"{val.get('mse', 0):.2f}",
                        "RMSE": f"{val.get('rmse', 0):.2f}",
                        "R² Score": f"{val.get('r2_score', 0):.4f}",
                    }
                )
            if reg_comp:
                st.table(pd.DataFrame(reg_comp))
            else:
                st.info("No regression validation comparison data available.")

        vspace(10)
        section_title("Test Set Scores — Best Regressor", "target")
        rtm = reg_results.get("test_evaluation", {})
        r1, r2, r3 = st.columns(3)
        with r1:
            stat_card("MSE", f"{rtm.get('mse', 0):,.1f}", accent="blue", icon_name="activity")
        with r2:
            stat_card("RMSE", f"${rtm.get('rmse', 0):,.1f}", accent="amber", icon_name="dollar")
        with r3:
            stat_card("R² Score", f"{rtm.get('r2_score', 0):.4f}", accent="green", icon_name="trending")

        vspace(12)
        rr1, rr2 = st.columns(2)
        for col, fname, cap in [
            (rr1, "regression_predicted_vs_actual.png", "Predicted vs Actual Spend"),
            (rr2, "regression_residual_plot.png", "Residual Plot"),
        ]:
            with col:
                p = os.path.join(FIGURES_DIR, fname)
                if os.path.exists(p):
                    with st.container(border=True):
                        st.image(p, caption=cap, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB: DATASET EXPLORER
# -----------------------------------------------------------------------------
with tab_data:
    page_header(
        "Dataset Explorer",
        "Inspect the preprocessed split files and visualize feature distributions.",
    )

    dataset_choice = st.selectbox("Choose dataset split", ["Train Set", "Validation Set", "Test Set"])
    filename_map = {"Train Set": "train.csv", "Validation Set": "validation.csv", "Test Set": "test.csv"}

    df = load_dataset(filename_map[dataset_choice])
    if df is None:
        empty_state("Split file missing", f"Could not find {filename_map[dataset_choice]} in the processed data directory.")
    else:
        vspace(4)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            stat_card("Rows", f"{df.shape[0]:,}", accent="blue", icon_name="database")
        with c2:
            stat_card("Columns", f"{df.shape[1]}", accent="teal", icon_name="grid")
        with c3:
            stat_card("Features", "13", accent="amber", icon_name="layers")
        with c4:
            stat_card("Customers", f"{df.shape[0]:,}", accent="green", icon_name="users")

        vspace(14)
        with st.container(border=True):
            panel_title("Interactive Preview · First 50 Rows", "database")
            st.dataframe(df.head(50), width="stretch")

        vspace(8)
        with st.container(border=True):
            panel_title("Feature Distribution Visualizer", "chart")
            expected_cols = [
                "Recency", "Frequency", "Monetary", "AverageSpend",
                "ProductDiversity", "TotalOrders", "AverageBasketSize", "AverageQuantityPerOrder",
            ]
            available_cols = [c for c in expected_cols if c in df.columns]
            if not available_cols:
                st.warning("No expected continuous columns found in this dataset.")
            else:
                selected_col = st.selectbox("Select continuous feature", available_cols)
                color_col = "High_Value_Customer" if "High_Value_Customer" in df.columns else None
                if color_col:
                    fig = px.histogram(
                        df, x=selected_col, color=color_col,
                        color_discrete_map={0: "#94a3b8", 1: "#2563eb"},
                        marginal="box", barmode="overlay",
                    )
                    name_map = {"0": "Standard Customer", "1": "High Value"}
                    fig.for_each_trace(lambda t: t.update(name=name_map.get(t.name, t.name)))
                else:
                    fig = px.histogram(df, x=selected_col, marginal="box", barmode="overlay")
                style_plot(fig, height=380, legend_title="Customer Segment" if color_col else None)
                fig.update_layout(bargap=0.08, xaxis_title=selected_col, yaxis_title="Count")
                st.plotly_chart(fig, width="stretch", theme=None)

# -----------------------------------------------------------------------------
# TAB: ABOUT
# -----------------------------------------------------------------------------
with tab_about:
    page_header(
        "About the Platform",
        "Technical workflow, technology stack, and the leakage-control audit trail.",
    )

    section_title("Implementation Pipeline", "route")
    st.markdown(
        """
        <div class="timeline">
          <div class="timeline-item">
            <div class="timeline-title">Phase 1 · Ingestion &amp; Transaction Cleaning</div>
            <div class="timeline-body">Loaded UCI Online Retail data. Removed duplicates, null customer IDs, and invalid negative quantity/pricing records.</div>
          </div>
          <div class="timeline-item">
            <div class="timeline-title">Phase 2 · Split Assignment &amp; Preprocessing</div>
            <div class="timeline-body">Split customer IDs into 80/10/10. Built input variables on Months 1–9 and targets on Months 10–12 only.</div>
          </div>
          <div class="timeline-item">
            <div class="timeline-title">Phase 3 · PCA &amp; LDA Projections</div>
            <div class="timeline-body">Standardized features and fit PCA/LDA using training data only.</div>
          </div>
          <div class="timeline-item">
            <div class="timeline-title">Phase 4 · Model Training</div>
            <div class="timeline-body">Trained and validated classifiers/regressors, then serialized the best-performing models.</div>
          </div>
          <div class="timeline-item">
            <div class="timeline-title">Phase 5 · Dashboard Deployment</div>
            <div class="timeline-body">Integrated serialized assets into a responsive offline Streamlit application.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    vspace(12)
    section_title("Technology Stack", "layers")
    st.markdown(
        """
        <div>
          <span class="badge">Python 3.11</span>
          <span class="badge">Scikit-Learn</span>
          <span class="badge">PyTorch</span>
          <span class="badge">Streamlit</span>
          <span class="badge">Pandas</span>
          <span class="badge">NumPy</span>
          <span class="badge">Plotly</span>
          <span class="badge">Joblib</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    vspace(16)
    l, r = st.columns(2)
    with l:
        with st.container(border=True):
            panel_title("Data Leakage Prevention", "shield")
            st.markdown(
                """
- **Split isolation** before scaling and fitting.
- **Feature isolation**: no Months 10–12 target spend used as input.
- **Transform fitting** (scalers/PCA/LDA) on the train split only.
- **Threshold lock** computed from the train distribution only.
                """
            )
    with r:
        with st.container(border=True):
            panel_title("Developer Notes", "info")
            st.markdown(
                """
- **Project Type**: Semester-Final ML Product + Dashboard
- **Role**: ML Engineer & UI Developer
- **GitHub**: https://github.com/PunIntendedDev/SmartRetail.git
- **Dataset**: https://archive.ics.uci.edu/dataset/502/online+retail+ii
                """
            )