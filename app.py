# """
# SmartRetail AI - SaaS Analytics Platform
# ---------------------------------------
# A commercial-grade SaaS intelligence platform built on the UCI Online Retail database.
# Implements glassmorphic cards, custom typography, prediction inference pipelines,
# and diagnostic auditories. Fully offline (loads saved binaries).
# """

# import os
# import json
# import joblib
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import streamlit as st
# from typing import Dict, Any, Tuple, Optional

# # --- Custom Page Config ---
# st.set_page_config(
#     page_title="SmartRetail AI Dashboard",
#     page_icon="🛍️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- Custom CSS injected through st.markdown ---
# st.markdown(
#     """
#     <link rel="preconnect" href="https://fonts.googleapis.com">
#     <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
#     <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    
#     <style>
#     /* Global Styles and Fonts */
#     html, body, [class*="css"], .stMarkdown {
#         font-family: 'Inter', sans-serif !important;
#         background-color: #f8fafc !important;
#         color: #111827 !important;
#     }
    
#     /* Hide default Streamlit elements */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
#     .stAppDeployButton {display: none;}
    
#     /* Custom Hero Header */
#     .hero-section {
#         background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
#         border-radius: 20px;
#         padding: 40px;
#         color: #ffffff !important;
#         margin-bottom: 30px;
#         box-shadow: 0 10px 25px rgba(37, 99, 235, 0.15);
#     }
#     .hero-section h1 {
#         color: #ffffff !important;
#         font-weight: 800 !important;
#         font-size: 3rem !important;
#         margin-bottom: 10px !important;
#         letter-spacing: -1px;
#     }
#     .hero-section h3 {
#         color: #e2e8f0 !important;
#         font-weight: 400 !important;
#         font-size: 1.25rem !important;
#         margin-bottom: 15px !important;
#     }
#     .hero-section p {
#         color: #f1f5f9 !important;
#         font-size: 0.95rem !important;
#         margin: 0 !important;
#     }
    
#     /* SaaS Premium Cards */
#     .saas-card {
#         background-color: #ffffff !important;
#         border: 1px solid #e2e8f0 !important;
#         border-radius: 20px !important;
#         padding: 24px;
#         margin-bottom: 20px;
#         box-shadow: 0 4px 15px rgba(17, 24, 39, 0.05);
#         transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
#     }
#     .saas-card:hover {
#         transform: translateY(-6px);
#         box-shadow: 0 12px 30px rgba(17, 24, 39, 0.10);
#         border-color: #2563eb !important;
#     }
    
#     /* Gradient Top Border Indicators */
#     .kpi-blue { border-top: 5px solid #2563eb; }
#     .kpi-purple { border-top: 5px solid #8b5cf6; }
#     .kpi-yellow { border-top: 5px solid #f59e0b; }
#     .kpi-green { border-top: 5px solid #10b981; }
#     .kpi-red { border-top: 5px solid #ef4444; }
    
#     .kpi-title {
#         font-size: 0.78rem;
#         text-transform: uppercase;
#         letter-spacing: 1.2px;
#         color: #6b7280;
#         margin-bottom: 8px;
#         font-weight: 600;
#     }
#     .kpi-value {
#         font-size: 1.85rem;
#         font-weight: 800;
#         color: #111827;
#         line-height: 1.1;
#     }
#     .kpi-desc {
#         font-size: 0.75rem;
#         color: #2563eb;
#         margin-top: 6px;
#     }
    
#     /* Outcomes Box Elements */
#     .outcome-box {
#         background-color: #ffffff;
#         border: 1px solid #e2e8f0;
#         border-radius: 16px;
#         padding: 20px;
#         text-align: center;
#         box-shadow: 0 4px 12px rgba(17,24,39,0.02);
#         margin-bottom: 12px;
#     }
#     .outcome-val {
#         font-size: 1.35rem;
#         font-weight: 700;
#         color: #111827;
#         margin-top: 5px;
#     }
    
#     /* Buttons Customization */
#     div.stButton > button:first-child {
#         background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
#         color: #ffffff !important;
#         border: none !important;
#         padding: 14px 28px;
#         font-weight: 700;
#         font-size: 1.05rem;
#         border-radius: 20px;
#         box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
#         transition: all 0.25s ease;
#         width: 100%;
#         margin-top: 10px;
#     }
#     div.stButton > button:first-child:hover {
#         background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
#         transform: scale(1.02);
#         box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45);
#     }
    
#     /* Sidebar Overrides */
#     section[data-testid="stSidebar"] {
#         background-color: #111827 !important;
#         border-right: 1px solid #1f2937 !important;
#     }
#     section[data-testid="stSidebar"] * {
#         color: #f3f4f6 !important;
#     }
#     section[data-testid="stSidebar"] .stRadio > label {
#         font-weight: 700;
#         color: #f3f4f6 !important;
#     }
    
#     /* Standard plots frame */
#     .plot-card {
#         border: 1px solid #e2e8f0;
#         border-radius: 16px;
#         padding: 16px;
#         background-color: #ffffff;
#         box-shadow: 0 2px 8px rgba(17,24,39,0.03);
#         margin-bottom: 16px;
#         text-align: center;
#     }
    
#     /* Tech Badges */
#     .badge {
#         background-color: #f1f5f9;
#         color: #475569;
#         border: 1px solid #e2e8f0;
#         padding: 6px 14px;
#         border-radius: 20px;
#         font-size: 0.8rem;
#         font-weight: 600;
#         display: inline-block;
#         margin-right: 8px;
#         margin-bottom: 8px;
#     }
    
#     /* Timeline styling */
#     .timeline {
#         border-left: 2px solid #e2e8f0;
#         padding-left: 20px;
#         margin-left: 10px;
#     }
#     .timeline-item {
#         margin-bottom: 25px;
#         position: relative;
#     }
#     .timeline-item::before {
#         content: '';
#         position: absolute;
#         left: -27px;
#         top: 5px;
#         background-color: #2563eb;
#         border-radius: 50%;
#         width: 12px;
#         height: 12px;
#     }
#     .timeline-title {
#         font-weight: 700;
#         color: #111827;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # --- Path Configuration Helpers ---
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
# PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
# FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

# # --- Caching Asset Loaders ---
# @st.cache_resource
# def load_model_file(filename: str) -> Optional[Any]:
#     path = os.path.join(MODELS_DIR, filename)
#     if os.path.exists(path):
#         return joblib.load(path)
#     return None

# @st.cache_data
# def load_json_file(path: str) -> Optional[Dict[str, Any]]:
#     if os.path.exists(path):
#         with open(path, "r") as f:
#             return json.load(f)
#     return None

# @st.cache_data
# def load_dataset(filename: str) -> Optional[pd.DataFrame]:
#     path = os.path.join(PROCESSED_DIR, filename)
#     if os.path.exists(path):
#         return pd.read_csv(path)
#     return None

# # Load models and assets globally
# best_classifier = load_model_file("best_classifier.joblib")
# best_regressor = load_model_file("best_regressor.joblib")
# scaler_raw = load_model_file("scaler.joblib")
# scaler_pca = load_model_file("pca_scaler.joblib")
# scaler_ns = load_model_file("non_spend_scaler.joblib")
# pca_model = load_model_file("pca.joblib")
# lda_model = load_model_file("lda.joblib")

# class_results = load_json_file(os.path.join(PROCESSED_DIR, "classification_results.json"))
# reg_results = load_json_file(os.path.join(PROCESSED_DIR, "regression_results.json"))
# pca_metadata = load_json_file(os.path.join(PROCESSED_DIR, "pca_metadata.json"))
# metadata = load_json_file(os.path.join(PROCESSED_DIR, "metadata.json"))

# # --- Sidebar Navigation ---
# st.sidebar.markdown(
#     """
#     <div style='margin-top: 10px; margin-bottom: 25px; border-bottom: 1px solid #1f2937; padding-bottom: 12px;'>
#         <div style='font-size: 1.45rem; font-weight: 800; color: #3b82f6;'>🛍️ SmartRetail AI</div>
#         <div style='font-size: 0.72rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px;'>Customer Behavior Analytics</div>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# page = st.sidebar.radio(
#     "SELECT PAGE:",
#     [
#         "🏠 Dashboard", 
#         "👤 Customer Prediction", 
#         "📈 Model Performance", 
#         "🧠 PCA & LDA", 
#         "📊 Dataset Explorer", 
#         "ℹ About"
#     ]
# )

# st.sidebar.markdown("---")
# st.sidebar.markdown("### PIPELINE CONNECTION:")

# def status_widget(name: str, obj: Any) -> None:
#     if obj is not None:
#         st.sidebar.markdown(f"🟢 **{name}**: Connected")
#     else:
#         st.sidebar.markdown(f"🔴 **{name}**: Offline")

# status_widget("Best Classifier", best_classifier)
# status_widget("Best Regressor", best_regressor)
# status_widget("PCA Model", pca_model)
# status_widget("LDA Model", lda_model)

# # ==============================================================================
# # HOME PAGE (DASHBOARD)
# # ==============================================================================
# if page == "🏠 Dashboard":
#     # Hero Section
#     st.markdown(
#         """
#         <div class='hero-section'>
#             <h1>SmartRetail AI</h1>
#             <h3>Customer Behavior Analysis & Dynamic Marketing Strategy</h3>
#             <p>A production-ready SaaS intelligence engine designed to structure customer loyalty profiles and forecast spending behavior using transactional data registers.</p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
    
#     # Calculate counts and names
#     total_customers = 0
#     if metadata:
#         total_customers = (
#             metadata.get("train_customers_count", 0) + 
#             metadata.get("val_customers_count", 0) + 
#             metadata.get("test_customers_count", 0)
#         )
        
#     best_classifier_name = "N/A"
#     if class_results:
#         best_classifier_name = class_results.get("best_model_selected", "N/A").replace("_", " ")
        
#     best_regressor_name = "N/A"
#     if reg_results:
#         best_regressor_name = reg_results.get("best_model_selected", "N/A").replace("_", " ")
        
#     # Beautiful top-border Custom Metric Cards
#     kcol1, kcol2, kcol3, kcol4 = st.columns(4)
#     with kcol1:
#         st.markdown(
#             f"""
#             <div class='saas-card kpi-blue'>
#                 <div class='kpi-title'>👥 Customers</div>
#                 <div class='kpi-value'>{total_customers:,}</div>
#                 <div class='kpi-desc'>Profiles parsed across splits</div>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
#     with kcol2:
#         st.markdown(
#             f"""
#             <div class='saas-card kpi-purple'>
#                 <div class='kpi-title'>📊 Features</div>
#                 <div class='kpi-value'>13</div>
#                 <div class='kpi-desc'>Behavioral & composition columns</div>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
#     with kcol3:
#         st.markdown(
#             f"""
#             <div class='saas-card kpi-yellow'>
#                 <div class='kpi-title'>🏆 Best Classifier</div>
#                 <div class='kpi-value'>{best_classifier_name}</div>
#                 <div class='kpi-desc'>Highest validation score F1</div>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
#     with kcol4:
#         st.markdown(
#             f"""
#             <div class='saas-card kpi-green'>
#                 <div class='kpi-title'>💰 Best Regressor</div>
#                 <div class='kpi-value'>{best_regressor_name}</div>
#                 <div class='kpi-desc'>Highest validation score R²</div>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
        
#     # Row for Pipeline overview and stats summaries
#     col_det1, col_det2 = st.columns(2)
#     with col_det1:
#         st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
#         st.markdown("### 🖥️ Pipeline Operations Overview")
#         st.markdown(
#             """
#             - **Data Cleaning**: Drops cancelled orders, duplicate records, null CustomerIDs, and negative rates.
#             - **Leakage Prevention**: Engineers variables strictly on Months 1-9; calculates forecasting targets strictly on Months 10-12.
#             - **Dimensional Projection**: Drop spend collinearity (`Other_Spend_Pct` exclusion) and runs PCA & LDA strictly on Train set features.
#             - **Estimator Verification**: Trains classification and regression configurations. Models are loaded directly from disk.
#             """
#         )
#         st.markdown("</div>", unsafe_allow_html=True)
        
#     with col_det2:
#         st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
#         st.markdown("### 📁 Dataset & Model Settings")
#         if metadata:
#             st.markdown(
#                 f"""
#                 *   **Split proportions**: 80% Train | 10% Validation | 10% Test (CustomerID level, seed 42)
#                 *   **Classification threshold**: Monetary Spend $\ge$ `${metadata['high_value_threshold']:.2f}` (80th percentile of Train)
#                 *   **PCA Features count**: `4` Principal Components (explaining $\ge$ `93.26%` cumulative category spend variance)
#                 """
#             )
#         st.markdown("</div>", unsafe_allow_html=True)

# # ==============================================================================
# # CUSTOMER PREDICTION
# # ==============================================================================
# elif page == "👤 Customer Prediction":
#     st.markdown("<div class='saas-header'>👤 Customer Profile Inference Engine</div>", unsafe_allow_html=True)
#     st.markdown("<div class='saas-subheader'>Predict Customer Segment, Confidence, Future Spend, Risk Level, and Custom Recommendation</div>", unsafe_allow_html=True)
    
#     if (
#         best_classifier is None or best_regressor is None or scaler_raw is None or 
#         pca_model is None or lda_model is None or scaler_pca is None or scaler_ns is None
#     ):
#         st.error("Pre-trained binaries missing under models/. Please run training scripts first.")
#     else:
#         # Two-column input layout
#         col_in1, col_in2 = st.columns(2)
        
#         with col_in1:
#             st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
#             st.markdown("#### 💳 Customer Behavior & Shopping Features")
#             recency = st.slider("Recency (Days since last purchase)", min_value=0, max_value=365, value=25)
#             frequency = st.number_input("Frequency (Total orders)", min_value=1, max_value=1000, value=6)
#             monetary = st.number_input("Monetary (Spent value $)", min_value=0.01, max_value=100000.0, value=750.0, step=50.0)
#             avg_spend = st.number_input("Average Spend ($ per order)", min_value=0.01, max_value=10000.0, value=125.0, step=10.0)
#             diversity = st.slider("Product Diversity (Unique StockCodes)", min_value=1, max_value=500, value=22)
#             basket_size = st.number_input("Average Basket Size (Lines per order)", min_value=1.0, max_value=100.0, value=4.8, step=0.2)
#             qty_per_order = st.number_input("Average Quantity per Order (Units per order)", min_value=1.0, max_value=1000.0, value=35.0, step=5.0)
#             st.markdown("</div>", unsafe_allow_html=True)
            
#         with col_in2:
#             st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
#             st.markdown("#### 🛍️ Category Spending Percentages")
#             homeware_pct = st.slider("Homeware Spend %", min_value=0.0, max_value=100.0, value=20.0, step=0.5)
#             stationery_pct = st.slider("Stationery Spend %", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
#             kitchenware_pct = st.slider("Kitchenware Spend %", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
#             decorations_pct = st.slider("Decorations Spend %", min_value=0.0, max_value=100.0, value=25.0, step=0.5)
#             gadgets_pct = st.slider("Gadgets Spend %", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
            
#             # Validation
#             total_pct = homeware_pct + stationery_pct + kitchenware_pct + decorations_pct + gadgets_pct
            
#             if total_pct > 100.0:
#                 st.markdown(
#                     f"""
#                     <div style='background-color: #fdf2f2; border: 1px solid #fde8e8; border-radius: 12px; padding: 16px; margin-top:15px;'>
#                         <div style='color: #ef4444; font-weight:700;'>❌ Category percentages exceed 100% (Sum: {total_pct:.1f}%)</div>
#                         <div style='color: #9b1c1c; font-size: 0.82rem; margin-top:5px;'>Please adjust composition sliders to keep the total spend within 100%.</div>
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )
#                 valid = False
#             else:
#                 other_pct = 100.0 - total_pct
#                 st.success(f"Spend validated. Remaining Other category: **{other_pct:.1f}%**")
#                 st.progress(total_pct / 100.0)
#                 valid = True
#             st.markdown("</div>", unsafe_allow_html=True)
            
#         # TotalOrders auto-assigned as Frequency for 13-feature pipeline compliance
#         total_orders = frequency
        
#         st.markdown("<br>", unsafe_allow_html=True)
        
#         col_btn, _ = st.columns([1, 1])
#         with col_btn:
#             # Predict button disabled if validations fail
#             predict_triggered = st.button("🔮 Predict Customer", disabled=not valid)
            
#         if valid and predict_triggered:
#             # Prepare inputs
#             homeware_frac = homeware_pct / 100.0
#             stationery_frac = stationery_pct / 100.0
#             gadgets_frac = gadgets_pct / 100.0
#             decorations_frac = decorations_pct / 100.0
#             kitchenware_frac = kitchenware_pct / 100.0
            
#             # Assemble raw array for scaler (13 features)
#             raw_features = [
#                 recency, frequency, monetary, avg_spend, diversity, 
#                 total_orders, basket_size, qty_per_order,
#                 homeware_frac, stationery_frac, gadgets_frac, decorations_frac, kitchenware_frac
#             ]
#             raw_array = np.array(raw_features).reshape(1, -1)
#             raw_scaled = scaler_raw.transform(raw_array)
            
#             # PCA features
#             spend_array = np.array([[homeware_frac, stationery_frac, gadgets_frac, decorations_frac, kitchenware_frac]])
#             spend_scaled = scaler_pca.transform(spend_array)
#             n_components = pca_metadata.get("selected_components_count", 4) if pca_metadata else 4
#             pca_comps = pca_model.transform(spend_scaled)[:, :n_components]
            
#             ns_array = np.array([[recency, frequency, monetary, avg_spend, diversity, total_orders, basket_size, qty_per_order]])
#             ns_scaled = scaler_ns.transform(ns_array)
#             pca_features = np.hstack([ns_scaled, pca_comps])
            
#             # LDA features
#             lda_proj = lda_model.transform(raw_scaled)
#             lda_features = np.hstack([ns_scaled, lda_proj])
            
#             # Choose classifier representation
#             best_rep = class_results.get("best_model_representation", "raw_scaled") if class_results else "raw_scaled"
#             if best_rep == "raw_scaled":
#                 X_class = raw_scaled
#             elif best_rep == "pca":
#                 X_class = pca_features
#             else:
#                 X_class = lda_features
                
#             # Perform inference
#             pred_label = best_classifier.predict(X_class)[0]
#             pred_probs = best_classifier.predict_proba(X_class)[0]
#             confidence = pred_probs[pred_label] * 100
            
#             pred_spend = best_regressor.predict(raw_scaled)[0]
#             pred_spend_val = max(0.0, float(pred_spend))
            
#             # Compute Risk Level and Recommendation
#             if pred_label == 1:
#                 risk_level = "LOW RISK"
#                 recommendation = "🌟 Fast-track loyalty programs; assign premier service representative."
#                 border_class = "kpi-green"
#             else:
#                 risk_level = "MEDIUM RISK"
#                 recommendation = "✉️ Offer 10% coupon campaigns on stationery/decorations to stimulate purchase frequency."
#                 border_class = "kpi-yellow"
                
#             st.markdown("### 📋 Prediction Diagnostic Outputs")
            
#             # Outcome grid cards
#             ocol1, ocol2, ocol3, ocol4 = st.columns(4)
#             with ocol1:
#                 st.markdown(
#                     f"""
#                     <div class='saas-card {border_class}'>
#                         <div class='kpi-title'>Prediction</div>
#                         <div class='kpi-value' style='font-size: 1.25rem; margin-top: 5px;'>
#                             {"🟢 High Value Customer" if pred_label == 1 else "🔴 Standard Customer"}
#                         </div>
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )
#             with ocol2:
#                 st.markdown(
#                     f"""
#                     <div class='saas-card {border_class}'>
#                         <div class='kpi-title'>Confidence</div>
#                         <div class='kpi-value'>{confidence:.1f}%</div>
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )
#             with ocol3:
#                 st.markdown(
#                     f"""
#                     <div class='saas-card kpi-green'>
#                         <div class='kpi-title'>Future Spend</div>
#                         <div class='kpi-value'>${pred_spend_val:,.2f}</div>
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )
#             with ocol4:
#                 st.markdown(
#                     f"""
#                     <div class='saas-card {border_class}'>
#                         <div class='kpi-title'>Risk Level</div>
#                         <div class='kpi-value' style='font-size: 1.25rem; margin-top: 5px;'>{risk_level}</div>
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )
                
#             st.markdown(
#                 f"""
#                 <div class='saas-card kpi-blue'>
#                     <div class='kpi-title'>Recommendation</div>
#                     <div style='font-size:1.05rem; font-weight:550; color:#111827; margin-top:5px;'>{recommendation}</div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )

# # ==============================================================================
# # MODEL PERFORMANCE
# # ==============================================================================
# elif page == "📈 Model Performance":
#     st.markdown("<div class='saas-header'>📈 Model Benchmarking Diagnostics</div>", unsafe_allow_html=True)
#     st.markdown("<div class='saas-subheader'>Comparative performance audit metrics evaluated strictly on test splits</div>", unsafe_allow_html=True)
    
#     if class_results is None or reg_results is None:
#         st.error("Audit metrics not found in data/processed/.")
#     else:
#         st.markdown("### Classification Benchmarks")
#         with st.container():
#             st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
#             # Comp validation table
#             comp_data = []
#             for key, val in class_results["validation_comparisons"].items():
#                 comp_data.append({
#                     "Model & Feature Space": key.replace("_", " "),
#                     "Accuracy": f"{val['val_accuracy']*100:.2f}%",
#                     "Precision": f"{val['val_precision']*100:.2f}%",
#                     "Recall": f"{val['val_recall']*100:.2f}%",
#                     "F1-Score": f"{val['val_f1']:.4f}",
#                     "ROC-AUC": f"{val['val_auc']:.4f}"
#                 })
#             st.table(pd.DataFrame(comp_data))
#             st.markdown("</div>", unsafe_allow_html=True)
            
#             # Test metrics cards
#             st.markdown("#### Test Set Scores (Best Segment Classifier)")
#             tm = class_results["test_evaluation"]
#             c1, c2, c3, c4, c5 = st.columns(5)
#             c1.markdown(f"<div class='saas-card kpi-blue'><div class='kpi-title'>Accuracy</div><div class='kpi-value'>{tm['accuracy']*100:.1f}%</div></div>", unsafe_allow_html=True)
#             c2.markdown(f"<div class='saas-card kpi-blue'><div class='kpi-title'>Precision</div><div class='kpi-value'>{tm['precision']*100:.1f}%</div></div>", unsafe_allow_html=True)
#             c3.markdown(f"<div class='saas-card kpi-blue'><div class='kpi-title'>Recall</div><div class='kpi-value'>{tm['recall']*100:.1f}%</div></div>", unsafe_allow_html=True)
#             c4.markdown(f"<div class='saas-card kpi-purple'><div class='kpi-title'>F1-Score</div><div class='kpi-value'>{tm['f1_score']:.3f}</div></div>", unsafe_allow_html=True)
#             c5.markdown(f"<div class='saas-card kpi-green'><div class='kpi-title'>ROC AUC</div><div class='kpi-value'>{tm['roc_auc']:.3f}</div></div>", unsafe_allow_html=True)
            
#             # Figure layout
#             fcol1, fcol2, fcol3 = st.columns(3)
#             with fcol1:
#                 path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
#                 if os.path.exists(path):
#                     st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
#                     st.image(path, caption="Confusion Matrix")
#                     st.markdown("</div>", unsafe_allow_html=True)
#             with fcol2:
#                 path = os.path.join(FIGURES_DIR, "roc_curve.png")
#                 if os.path.exists(path):
#                     st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
#                     st.image(path, caption="ROC Curves")
#                     st.markdown("</div>", unsafe_allow_html=True)
#             with fcol3:
#                 path = os.path.join(FIGURES_DIR, "precision_recall_curve.png")
#                 if os.path.exists(path):
#                     st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
#                     st.image(path, caption="Precision-Recall Curves")
#                     st.markdown("</div>", unsafe_allow_html=True)
                    
#         st.markdown("### Regression Benchmarks")
#         with st.container():
#             st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
#             reg_comp = []
#             for key, val in reg_results["validation_comparisons"].items():
#                 reg_comp.append({
#                     "Regressor Model": key.replace("_", " "),
#                     "MSE": f"{val['mse']:.2f}",
#                     "RMSE": f"{val['rmse']:.2f}",
#                     "R² Score": f"{val['r2_score']:.4f}"
#                 })
#             st.table(pd.DataFrame(reg_comp))
#             st.markdown("</div>", unsafe_allow_html=True)
            
#             # Test metrics cards
#             st.markdown("#### Test Set Scores (Best Spend Regressor)")
#             rtm = reg_results["test_evaluation"]
#             rcol1, rcol2, rcol3 = st.columns(3)
#             rcol1.markdown(f"<div class='saas-card kpi-blue'><div class='kpi-title'>Mean Squared Error</div><div class='kpi-value'>{rtm['mse']:,.1f}</div></div>", unsafe_allow_html=True)
#             rcol2.markdown(f"<div class='saas-card kpi-purple'><div class='kpi-title'>Root Mean Squared Error</div><div class='kpi-value'>${rtm['rmse']:,.1f}</div></div>", unsafe_allow_html=True)
#             rcol3.markdown(f"<div class='saas-card kpi-green'><div class='kpi-title'>R² Score</div><div class='kpi-value'>{rtm['r2_score']:.4f}</div></div>", unsafe_allow_html=True)
            
#             # Figure layout
#             rfcol1, rfcol2 = st.columns(2)
#             with rfcol1:
#                 path = os.path.join(FIGURES_DIR, "regression_predicted_vs_actual.png")
#                 if os.path.exists(path):
#                     st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
#                     st.image(path, caption="Predicted vs Actual Spend")
#                     st.markdown("</div>", unsafe_allow_html=True)
#             with rfcol2:
#                 path = os.path.join(FIGURES_DIR, "regression_residual_plot.png")
#                 if os.path.exists(path):
#                     st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
#                     st.image(path, caption="Residuals Plots")
#                     st.markdown("</div>", unsafe_allow_html=True)

# # ==============================================================================
# # PCA & LDA PAGE
# # ==============================================================================
# elif page == "🧠 PCA & LDA":
#     st.markdown("<div class='saas-header'>🧠 Dimensionality Reduction Layout</div>", unsafe_allow_html=True)
#     st.markdown("<div class='saas-subheader'>Orthogonal mappings and class separability boundaries fitted strictly on Train splits</div>", unsafe_allow_html=True)
    
#     if pca_metadata is None:
#         st.error("Missing PCA/LDA metadata file.")
#     else:
#         col_pca1, col_pca2 = st.columns([2, 3])
        
#         with col_pca1:
#             st.markdown("### Methodology Explanations")
            
#             with st.expander("🔬 Unsupervised PCA", expanded=True):
#                 st.markdown(
#                     """
#                     PCA maps raw category spent fractions to orthogonal vectors (components). 
#                     This drops redundant configurations while retaining the directions containing 
#                     the largest variances.
#                     """
#                 )
                
#             with st.expander("🔬 Supervised LDA", expanded=True):
#                 st.markdown(
#                     """
#                     LDA projects behavioral dimensions to compute a single discriminant index (`LD1`) 
#                     optimized strictly to separate High-Value customers from Standard profiles.
#                     """
#                 )
                
#             # Selection metric cards
#             st.markdown("#### Dimensional Settings")
#             st.markdown(
#                 f"""
#                 <div class='saas-card kpi-blue'>
#                     <div class='kpi-title'>Selected Components</div>
#                     <div class='kpi-value'>{pca_metadata['selected_components_count']} / 5</div>
#                     <div class='kpi-desc'>Preserves >= 90% cumulative variance</div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
#             cum_var = pca_metadata['cumulative_variance_explained'][pca_metadata['selected_components_count'] - 1]
#             st.markdown(
#                 f"""
#                 <div class='saas-card kpi-green'>
#                     <div class='kpi-title'>Variance Retained</div>
#                     <div class='kpi-value'>{cum_var*100:.2f}%</div>
#                     <div class='kpi-desc'>Information ratio retained</div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
            
#             # Highlight top discriminating LDA feature
#             lda_coefs = pca_metadata["lda_discriminant_coefficients"]
#             top_feat = max(lda_coefs, key=lambda k: abs(lda_coefs[k]))
#             st.markdown(
#                 f"""
#                 <div class='saas-card kpi-purple'>
#                     <div class='kpi-title'>Top Discriminating Feature</div>
#                     <div class='kpi-value' style='font-size:1.35rem; margin-top:8px;'>{top_feat}</div>
#                     <div class='kpi-desc'>LDA Coefficient: {lda_coefs[top_feat]:.3f}</div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
            
#             # Multicollinearity explanation card
#             st.markdown(
#                 f"""
#                 <div class='saas-card kpi-yellow'>
#                     <div class='kpi-title'>💡 Multicollinearity Choice</div>
#                     <div style='font-size:0.88rem; color:#111827; margin-top:5px; line-height:1.4;'>
#                         "{pca_metadata['viva_explanation']}"
#                     </div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
            
#         with col_pca2:
#             st.markdown("### Projection Charts")
#             path_var = os.path.join(FIGURES_DIR, "pca_explained_variance.png")
#             path_scat = os.path.join(FIGURES_DIR, "pca_scatter_plot.png")
#             path_proj = os.path.join(FIGURES_DIR, "lda_projection.png")
            
#             if os.path.exists(path_var):
#                 st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
#                 st.image(path_var, caption="PCA Explained Variance ratios", use_column_width=True)
#                 st.markdown("</div>", unsafe_allow_html=True)
#             if os.path.exists(path_scat):
#                 st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
#                 st.image(path_scat, caption="PC1 vs PC2 Customer Scatter Map", use_column_width=True)
#                 st.markdown("</div>", unsafe_allow_html=True)
#             if os.path.exists(path_proj):
#                 st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
#                 st.image(path_proj, caption="Supervised LDA Class Separation Projections", use_column_width=True)
#                 st.markdown("</div>", unsafe_allow_html=True)

# # ==============================================================================
# # DATASET EXPLORER
# # ==============================================================================
# elif page == "📊 Dataset Explorer":
#     st.markdown("<div class='saas-header'>📊 Customer Records Auditor</div>", unsafe_allow_html=True)
#     st.markdown("<div class='saas-subheader'>Preview preprocessed metrics and profile variables distributions</div>", unsafe_allow_html=True)
    
#     # Dataset Selector
#     dataset_choice = st.selectbox("Choose Dataset Split:", ["Train Set", "Validation Set", "Test Set"])
    
#     filename_map = {
#         "Train Set": "train.csv",
#         "Validation Set": "validation.csv",
#         "Test Set": "test.csv"
#     }
    
#     df = load_dataset(filename_map[dataset_choice])
    
#     if df is None:
#         st.error(f"Missing splits file: {filename_map[dataset_choice]}")
#     else:
#         # Top summary cards (Rows, Columns, Features, Customers)
#         dcol1, dcol2, dcol3, dcol4 = st.columns(4)
#         with dcol1:
#             st.markdown(f"<div class='saas-card kpi-blue'><div class='kpi-title'>Rows</div><div class='kpi-value'>{df.shape[0]:,}</div></div>", unsafe_allow_html=True)
#         with dcol2:
#             st.markdown(f"<div class='saas-card kpi-purple'><div class='kpi-title'>Columns</div><div class='kpi-value'>{df.shape[1]}</div></div>", unsafe_allow_html=True)
#         with dcol3:
#             st.markdown(f"<div class='saas-card kpi-yellow'><div class='kpi-title'>Features</div><div class='kpi-value'>13</div></div>", unsafe_allow_html=True)
#         with dcol4:
#             st.markdown(f"<div class='saas-card kpi-green'><div class='kpi-title'>Customers</div><div class='kpi-value'>{df.shape[0]:,}</div></div>", unsafe_allow_html=True)
            
#         st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
#         st.markdown("#### Interactive Records Preview (First 50 Rows)")
#         st.dataframe(df.head(50), use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
        
#         st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
#         st.markdown("#### Profile Feature Distributions Visualizer")
#         st.markdown("Audits values distributions colored by customer classification segment type.")
        
#         continuous_cols = [
#             "Recency", "Frequency", "Monetary", "AverageSpend", "ProductDiversity", 
#             "TotalOrders", "AverageBasketSize", "AverageQuantityPerOrder"
#         ]
#         selected_col = st.selectbox("Select Continuous Feature to Plot:", continuous_cols)
        
#         if selected_col:
#             # Plotly light-slate histogram
#             fig = px.histogram(
#                 df, 
#                 x=selected_col, 
#                 color="High_Value_Customer",
#                 color_discrete_map={0: "#afb8c1", 1: "#2563eb"},
#                 marginal="box",
#                 barmode="overlay",
#                 template="plotly_white"
#             )
#             fig.update_layout(
#                 bargap=0.08,
#                 xaxis_title=selected_col,
#                 yaxis_title="Count",
#                 paper_bgcolor="#ffffff",
#                 plot_bgcolor="#f8fafc"
#             )
#             newnames = {'0':'Standard Customer', '1':'High Value'}
#             fig.for_each_trace(lambda t: t.update(name = newnames[t.name]))
            
#             st.plotly_chart(fig, use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)

# # ==============================================================================
# # ABOUT PAGE
# # ==============================================================================
# else:
#     st.markdown("<div class='saas-header'>ℹ️ About Platform Implementation</div>", unsafe_allow_html=True)
#     st.markdown("<div class='saas-subheader'>Technical workflow, technologies, and audit trails for presentation auditing</div>", unsafe_allow_html=True)
    
#     st.markdown("### 🔄 Implementation Pipeline Timeline")
    
#     st.markdown(
#         """
#         <div class='timeline'>
#             <div class='timeline-item'>
#                 <div class='timeline-title'>Phase 1: Ingestion & Transaction Cleaning</div>
#                 <div style='font-size:0.88rem; color:#57606a;'>Downloaded UCI Online Retail xlsx ledger. Cleaned records (duplicates, missing customer identifiers, negative quantity and pricing parameters).</div>
#             </div>
#             <div class='timeline-item'>
#                 <div class='timeline-title'>Phase 2: Split Assignment & Preprocessing</div>
#                 <div style='font-size:0.88rem; color:#57606a;'>Split customer IDs 80/10/10 strictly at the CustomerID level. Engineered input variables strictly on Months 1-9 Feature Window, and targets spend strictly on Months 10-12 Target Window.</div>
#             </div>
#             <div class='timeline-item'>
#                 <div class='timeline-title'>Phase 3: PCA & LDA Projections</div>
#                 <div style='font-size:0.88rem; color:#57606a;'>Standardized and fit PCA (on spend compositions, dropping Other_Spend_Pct) and LDA (supervised class separation) strictly on training data splits.</div>
#             </div>
#             <div class='timeline-item'>
#                 <div class='timeline-title'>Phase 4: Classifiers & Regressors Training</div>
#                 <div style='font-size:0.88rem; color:#57606a;'>Fitted baseline and neural network models (GridSearchCV Logistic Regression, MLPClassifier, Linear Regression, MLPRegressor) strictly on train. Validated, evaluated on test splits, and saved.</div>
#             </div>
#             <div class='timeline-item'>
#                 <div class='timeline-title'>Phase 5: Production Deployment & Dashboard</div>
#                 <div style='font-size:0.88rem; color:#57606a;'>Wrapped saved configurations inside an interactive, light-themed, high-performance Streamlit analytics dashboard.</div>
#             </div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
    
#     st.markdown("---")
    
#     st.markdown("### 🛠️ Technology Stack Badges")
#     st.markdown(
#         """
#         <span class='badge'>Python 3.11</span>
#         <span class='badge'>Scikit-Learn</span>
#         <span class='badge'>Streamlit</span>
#         <span class='badge'>Pandas</span>
#         <span class='badge'>NumPy</span>
#         <span class='badge'>Plotly</span>
#         <span class='badge'>Joblib</span>
#         <span class='badge'>Matplotlib & Seaborn</span>
#         """,
#         unsafe_allow_html=True
#     )
    
#     st.markdown("---")
    
#     col_th1, col_th2 = st.columns(2)
#     with col_th1:
#         st.markdown(
#             """
#             ### 🔒 Strict Data Leakage Prevention
#             - **Split Isolation**: The customer level splits are determined before scaling and pipeline fitting.
#             - **Feature Isolation**: No spend metrics from Months 10–12 (`Future_Spend`) were used as inputs for feature calculation.
#             - **Scalers & PCA fitting**: Scalers and transformations are fitted strictly on training data split. Validation and Test splits are transformed only.
#             - **Boundary Lock**: The High-Value boundary ($1,629.40) is calculated strictly on the training split.
#             """
#         )
#     with col_th2:
#         st.markdown(
#             """
#             ### 👤 Developer Information
#             - **Course**: Final Year Project & Technical Viva Presentation
#             - **Role**: Senior Machine Learning System Architect & UI/UX Developer
#             - **Platform Status**: Fully Operational. All models are read from disk.
#             """
#         )


# app.py
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
import streamlit as st
from typing import Dict, Any, Optional

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
# SIDEBAR (build controls first so CSS can use values)
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
        "👤 Customer Prediction",
        "📈 Model Performance",
        "🧠 PCA & LDA",
        "📊 Dataset Explorer",
        "ℹ About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 UI Preferences")
theme_mode = st.sidebar.radio("Theme", ["Light", "Dark"], index=0, horizontal=True)
compact_mode = st.sidebar.toggle("Compact mode", value=False)

# -----------------------------------------------------------------------------
# THEME TOKENS + CSS (safe sidebar visibility fix included)
# -----------------------------------------------------------------------------
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

    /* Hide only what is safe */
    #MainMenu, footer {{visibility: hidden;}}
    .stAppDeployButton {{display: none;}}

    /* Keep Streamlit header visible so sidebar toggle behavior is not broken */
    header[data-testid="stHeader"] {{
        visibility: visible !important;
        background: transparent !important;
    }}

    /* --- Sidebar visibility fix --- */
    section[data-testid="stSidebar"] {{
        background: {TOKENS["sidebar_bg"]} !important;
        border-right: 1px solid {TOKENS["sidebar_border"]} !important;
        min-width: 300px !important;
        max-width: 300px !important;
        width: 300px !important;
        transform: none !important;
        visibility: visible !important;
        display: block !important;
        z-index: 999 !important;
    }}
    section[data-testid="stSidebar"] > div {{
        visibility: visible !important;
        display: block !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {{
        color: #e2e8f0 !important;
    }}

    @media (max-width: 992px) {{
        section[data-testid="stSidebar"] {{
            min-width: 260px !important;
            max-width: 85vw !important;
            width: 85vw !important;
        }}
    }}

    /* Hero */
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
        transition: border-color .2s ease, box-shadow .2s ease;
    }}
    .saas-card:hover {{
        border-color: #93c5fd;
        box-shadow: 0 8px 22px rgba(59, 130, 246, 0.18);
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
        word-break: break-word;
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

    div[data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def vspace(h: int = 8) -> None:
    st.markdown(f"<div style='height:{h}px'></div>", unsafe_allow_html=True)


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
def load_model_file(filename: str) -> Optional[Any]:
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_data
def load_json_file(path: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_dataset(filename: str) -> Optional[pd.DataFrame]:
    path = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# -----------------------------------------------------------------------------
# LOAD ASSETS
# -----------------------------------------------------------------------------
best_classifier = load_model_file("best_classifier.joblib")
best_regressor = load_model_file("best_regressor.joblib")
scaler_raw = load_model_file("scaler.joblib")
scaler_pca = load_model_file("pca_scaler.joblib")
scaler_ns = load_model_file("non_spend_scaler.joblib")
pca_model = load_model_file("pca.joblib")
lda_model = load_model_file("lda.joblib")

class_results = load_json_file(os.path.join(PROCESSED_DIR, "classification_results.json"))
reg_results = load_json_file(os.path.join(PROCESSED_DIR, "regression_results.json"))
pca_metadata = load_json_file(os.path.join(PROCESSED_DIR, "pca_metadata.json"))
metadata = load_json_file(os.path.join(PROCESSED_DIR, "metadata.json"))

# Sidebar status
st.sidebar.markdown("---")
st.sidebar.markdown("### PIPELINE STATUS")

def status_widget(name: str, obj: Any) -> None:
    if obj is not None:
        st.sidebar.markdown(f"🟢 **{name}**: Connected")
    else:
        st.sidebar.markdown(f"🔴 **{name}**: Offline")

status_widget("Best Classifier", best_classifier)
status_widget("Best Regressor", best_regressor)
status_widget("PCA Model", pca_model)
status_widget("LDA Model", lda_model)

# -----------------------------------------------------------------------------
# PAGE: DASHBOARD
# -----------------------------------------------------------------------------
if page == "🏠 Dashboard":
    st.markdown(
        """
        <div class='hero-section'>
            <h1>SmartRetail AI</h1>
            <h3>Customer Behavior Analysis & Dynamic Marketing Strategy</h3>
            <p>A production-ready SaaS intelligence engine designed to profile customer loyalty and forecast spending behavior using transactional records.</p>
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
            f"<div class='saas-card kpi-blue'><div class='kpi-title'>Customers</div><div class='kpi-value'>{total_customers:,}</div><div class='kpi-desc'>Profiles across splits</div></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            "<div class='saas-card kpi-purple'><div class='kpi-title'>Features</div><div class='kpi-value'>13</div><div class='kpi-desc'>Behavioral + composition</div></div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div class='saas-card kpi-yellow'><div class='kpi-title'>Best Classifier</div><div class='kpi-value'>{best_classifier_name}</div><div class='kpi-desc'>Highest validation F1</div></div>",
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"<div class='saas-card kpi-green'><div class='kpi-title'>Best Regressor</div><div class='kpi-value'>{best_regressor_name}</div><div class='kpi-desc'>Highest validation R²</div></div>",
            unsafe_allow_html=True
        )

    vspace(8)

    a, b = st.columns(2)
    with a:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("### 🖥️ Pipeline Operations Overview")
        st.markdown(
            """
            - **Data Cleaning**: Drops cancelled orders, duplicates, null CustomerIDs, and invalid negatives.
            - **Leakage Prevention**: Features from Months 1-9 only; targets from Months 10-12 only.
            - **Dimensional Projection**: Drops collinearity and applies PCA/LDA on train split.
            - **Estimator Verification**: Classification + regression models trained and loaded from disk.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("### 📁 Dataset & Model Settings")
        if metadata:
            threshold = metadata.get("high_value_threshold", 0.0)
            st.markdown(
                f"""
                - **Split proportions**: 80% Train | 10% Validation | 10% Test (CustomerID-level, seed=42)
                - **Classification threshold**: Monetary Spend >= `${threshold:.2f}` (80th percentile of train)
                - **PCA components**: 4 retained (high cumulative variance)
                """
            )
        else:
            st.info("metadata.json not found.")
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE: CUSTOMER PREDICTION
# -----------------------------------------------------------------------------
elif page == "👤 Customer Prediction":
    st.markdown("<div class='saas-header'>👤 Customer Profile Inference Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='saas-subheader'>Predict segment, confidence, future spend, risk level, and recommendation.</div>", unsafe_allow_html=True)

    required_objects = [best_classifier, best_regressor, scaler_raw, scaler_pca, scaler_ns, pca_model, lda_model]
    if any(x is None for x in required_objects):
        st.error("Required binaries are missing under models/. Please run training scripts first.")
    else:
        left, right = st.columns(2)

        with left:
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.markdown("#### 💳 Behavior & Shopping Inputs")
            recency = st.slider("Recency (days since last purchase)", 0, 365, 25)
            frequency = st.number_input("Frequency (total orders)", 1, 1000, 6)
            monetary = st.number_input("Monetary (spent value $)", 0.01, 100000.0, 750.0, step=50.0)
            avg_spend = st.number_input("Average Spend ($ per order)", 0.01, 10000.0, 125.0, step=10.0)
            diversity = st.slider("Product Diversity (unique StockCodes)", 1, 500, 22)
            basket_size = st.number_input("Average Basket Size (lines per order)", 1.0, 100.0, 4.8, step=0.2)
            qty_per_order = st.number_input("Average Quantity Per Order (units)", 1.0, 1000.0, 35.0, step=5.0)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.markdown("#### 🛍️ Category Spending Mix (%)")
            homeware_pct = st.slider("Homeware %", 0.0, 100.0, 20.0, 0.5)
            stationery_pct = st.slider("Stationery %", 0.0, 100.0, 15.0, 0.5)
            kitchenware_pct = st.slider("Kitchenware %", 0.0, 100.0, 10.0, 0.5)
            decorations_pct = st.slider("Decorations %", 0.0, 100.0, 25.0, 0.5)
            gadgets_pct = st.slider("Gadgets %", 0.0, 100.0, 15.0, 0.5)

            total_pct = homeware_pct + stationery_pct + kitchenware_pct + decorations_pct + gadgets_pct
            if total_pct > 100.0:
                st.error(f"Category percentages exceed 100% (current total: {total_pct:.1f}%).")
                valid = False
            else:
                other_pct = 100.0 - total_pct
                st.success(f"Composition valid. Remaining 'Other' category: **{other_pct:.1f}%**")
                st.progress(total_pct / 100.0)
                valid = True
            st.markdown("</div>", unsafe_allow_html=True)

        total_orders = frequency

        btn_col, _ = st.columns([1, 1])
        with btn_col:
            run_pred = st.button("🔮 Predict Customer", disabled=not valid)

        if valid and run_pred:
            # Fractions
            homeware_frac = homeware_pct / 100.0
            stationery_frac = stationery_pct / 100.0
            gadgets_frac = gadgets_pct / 100.0
            decorations_frac = decorations_pct / 100.0
            kitchenware_frac = kitchenware_pct / 100.0

            # Raw features (13)
            raw_features = [
                recency, frequency, monetary, avg_spend, diversity,
                total_orders, basket_size, qty_per_order,
                homeware_frac, stationery_frac, gadgets_frac, decorations_frac, kitchenware_frac
            ]
            raw_array = np.array(raw_features).reshape(1, -1)
            raw_scaled = scaler_raw.transform(raw_array)

            # PCA representation
            spend_array = np.array([[homeware_frac, stationery_frac, gadgets_frac, decorations_frac, kitchenware_frac]])
            spend_scaled = scaler_pca.transform(spend_array)
            n_components = pca_metadata.get("selected_components_count", 4) if pca_metadata else 4
            pca_comps = pca_model.transform(spend_scaled)[:, :n_components]

            ns_array = np.array([[recency, frequency, monetary, avg_spend, diversity, total_orders, basket_size, qty_per_order]])
            ns_scaled = scaler_ns.transform(ns_array)
            pca_features = np.hstack([ns_scaled, pca_comps])

            # LDA representation
            lda_proj = lda_model.transform(raw_scaled)
            lda_features = np.hstack([ns_scaled, lda_proj])

            # Choose representation
            best_rep = class_results.get("best_model_representation", "raw_scaled") if class_results else "raw_scaled"
            if best_rep == "raw_scaled":
                X_class = raw_scaled
            elif best_rep == "pca":
                X_class = pca_features
            else:
                X_class = lda_features

            # Predict
            pred_label = int(best_classifier.predict(X_class)[0])
            pred_probs = best_classifier.predict_proba(X_class)[0]
            confidence = float(pred_probs[pred_label] * 100.0)

            pred_spend = float(best_regressor.predict(raw_scaled)[0])
            pred_spend_val = max(0.0, pred_spend)

            if pred_label == 1:
                risk_level = "LOW RISK"
                recommendation = "🌟 Prioritize loyalty tiers, VIP nurturing, and premium retention campaigns."
                border_class = "kpi-green"
                pred_text = "🟢 High Value Customer"
            else:
                risk_level = "MEDIUM RISK"
                recommendation = "✉️ Trigger targeted offers to improve repeat purchase frequency."
                border_class = "kpi-yellow"
                pred_text = "🔴 Standard Customer"

            st.markdown("### 📋 Prediction Outputs")
            o1, o2, o3, o4 = st.columns(4)
            with o1:
                st.markdown(
                    f"<div class='saas-card {border_class}'><div class='kpi-title'>Segment</div><div class='kpi-value' style='font-size:1.1rem'>{pred_text}</div></div>",
                    unsafe_allow_html=True
                )
            with o2:
                st.markdown(
                    f"<div class='saas-card {border_class}'><div class='kpi-title'>Confidence</div><div class='kpi-value'>{confidence:.1f}%</div></div>",
                    unsafe_allow_html=True
                )
            with o3:
                st.markdown(
                    f"<div class='saas-card kpi-green'><div class='kpi-title'>Future Spend</div><div class='kpi-value'>${pred_spend_val:,.2f}</div></div>",
                    unsafe_allow_html=True
                )
            with o4:
                st.markdown(
                    f"<div class='saas-card {border_class}'><div class='kpi-title'>Risk</div><div class='kpi-value' style='font-size:1.1rem'>{risk_level}</div></div>",
                    unsafe_allow_html=True
                )

            st.markdown(
                f"""
                <div class='saas-card kpi-blue'>
                    <div class='kpi-title'>Recommendation</div>
                    <div style='font-size:1rem; font-weight:600; color:var(--text); margin-top:6px;'>{recommendation}</div>
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
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)

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
        st.markdown("</div>", unsafe_allow_html=True)

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
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)

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
        st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown("<div class='saas-header'>🧠 Dimensionality Reduction Layout</div>", unsafe_allow_html=True)
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
                f"<div class='saas-card kpi-blue'><div class='kpi-title'>Selected Components</div><div class='kpi-value'>{selected_components} / 5</div><div class='kpi-desc'>Configured dimensionality retention</div></div>",
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
                    <div style='font-size:0.9rem; color:var(--text); margin-top:5px; line-height:1.45;'>
                        "{viva_explanation}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with right:
            st.markdown("### Projection Charts")
            for fname, cap in [
                ("pca_explained_variance.png", "PCA Explained Variance"),
                ("pca_scatter_plot.png", "PC1 vs PC2 Scatter"),
                ("lda_projection.png", "LDA Class Separation")
            ]:
                p = os.path.join(FIGURES_DIR, fname)
                if os.path.exists(p):
                    st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
                    st.image(p, caption=cap, use_container_width=True)
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

        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("#### Interactive Preview (First 50 Rows)")
        st.dataframe(df.head(50), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
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
                legend_title_text="Customer Segment" if color_col else ""
            )
            st.plotly_chart(fig, use_container_width=True)

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