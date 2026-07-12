"""
Dimensionality Reduction Module
-------------------------------
Fits StandardScalers, PCA (for category spend percentages), and LDA (for supervised
dimensionality reduction using the High-Value Customer label) strictly on the training set.
Applies the fitted models to validation and test sets without data leakage.
Exports three distinct feature representations for downstream classification comparison.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import List, Dict, Any

from src.config import get_absolute_path, config_data
from utils.logging_utils import get_logger
from utils.plotting_utils import plot_pca_explained_variance, plot_pca_scatter, plot_lda_projection

logger = get_logger(__name__)

def perform_dimensionality_reduction() -> None:
    """
    Loads preprocessed customer data, performs scaling, PCA, and LDA under strict
    no-leakage constraints, saves the fitted estimators, exports transformed sets,
    and generates the requested evaluation figures.
    """
    processed_dir = get_absolute_path("processed_data_dir")
    models_dir = get_absolute_path("models_dir")
    os.makedirs(models_dir, exist_ok=True)
    
    # Load processed splits
    logger.info("Loading preprocessed customer datasets...")
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    # Check if CustomerID is set as column or index
    if "CustomerID" in train_df.columns:
        train_df = train_df.set_index("CustomerID")
    val_df = pd.read_csv(os.path.join(processed_dir, "validation.csv"))
    if "CustomerID" in val_df.columns:
        val_df = val_df.set_index("CustomerID")
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    if "CustomerID" in test_df.columns:
        test_df = test_df.set_index("CustomerID")
        
    # Define category spend features
    # NOTE: Other_Spend_Pct is intentionally excluded to prevent perfect multicollinearity, 
    # as the six spend percentages sum to 100%. This is critical for model stability and viva verification.
    spend_features = [
        "Homeware_Spend_Pct",
        "Stationery_Spend_Pct",
        "Gadgets_Spend_Pct",
        "Decorations_Spend_Pct",
        "Kitchenware_Spend_Pct"
    ]
    
    logger.info(f"Selected category spend features for PCA (excluding 'Other_Spend_Pct'): {spend_features}")
    
    # Non-spend baseline features
    non_spend_features = [
        "Recency", 
        "Frequency", 
        "Monetary", 
        "AverageSpend", 
        "ProductDiversity", 
        "TotalOrders", 
        "AverageBasketSize", 
        "AverageQuantityPerOrder"
    ]
    
    # Compile raw feature list (excluding Other_Spend_Pct)
    raw_feature_cols = non_spend_features + spend_features
    
    # Extract feature matrices and labels
    X_train_raw = train_df[raw_feature_cols]
    X_val_raw = val_df[raw_feature_cols]
    X_test_raw = test_df[raw_feature_cols]
    
    y_train = train_df["High_Value_Customer"]
    y_val = val_df["High_Value_Customer"]
    y_test = test_df["High_Value_Customer"]
    
    # 1. Scale Raw Features (StandardScaler fitted strictly on Train)
    logger.info("Fitting StandardScaler on raw training features...")
    raw_scaler = StandardScaler()
    X_train_scaled = raw_scaler.fit_transform(X_train_raw)
    X_val_scaled = raw_scaler.transform(X_val_raw)
    X_test_scaled = raw_scaler.transform(X_test_raw)
    
    # Save the raw scaler
    raw_scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(raw_scaler, raw_scaler_path)
    logger.info(f"Raw features scaler saved to: {raw_scaler_path}")
    
    # Save Feature Set 1 (Raw Standardized Features)
    train_raw_scaled_df = pd.DataFrame(X_train_scaled, index=train_df.index, columns=raw_feature_cols)
    train_raw_scaled_df["High_Value_Customer"] = train_df["High_Value_Customer"]
    train_raw_scaled_df["Future_Spend"] = train_df["Future_Spend"]
    train_raw_scaled_df.to_csv(os.path.join(processed_dir, "train_raw_scaled.csv"))
    
    val_raw_scaled_df = pd.DataFrame(X_val_scaled, index=val_df.index, columns=raw_feature_cols)
    val_raw_scaled_df["High_Value_Customer"] = val_df["High_Value_Customer"]
    val_raw_scaled_df["Future_Spend"] = val_df["Future_Spend"]
    val_raw_scaled_df.to_csv(os.path.join(processed_dir, "validation_raw_scaled.csv"))
    
    test_raw_scaled_df = pd.DataFrame(X_test_scaled, index=test_df.index, columns=raw_feature_cols)
    test_raw_scaled_df["High_Value_Customer"] = test_df["High_Value_Customer"]
    test_raw_scaled_df["Future_Spend"] = test_df["Future_Spend"]
    test_raw_scaled_df.to_csv(os.path.join(processed_dir, "test_raw_scaled.csv"))
    logger.info("Saved Raw Standardized feature sets.")
    
    # 2. PCA Pipeline (Fitted strictly on Train spend features)
    logger.info("Running PCA pipeline on category spend percentages...")
    pca_scaler = StandardScaler()
    X_train_spend_scaled = pca_scaler.fit_transform(X_train_raw[spend_features])
    X_val_spend_scaled = pca_scaler.transform(X_val_raw[spend_features])
    X_test_spend_scaled = pca_scaler.transform(X_test_raw[spend_features])
    
    # Save the pca spend scaler
    pca_scaler_path = os.path.join(models_dir, "pca_scaler.joblib")
    joblib.dump(pca_scaler, pca_scaler_path)
    
    # Fit PCA
    pca = PCA(random_state=config_data["data_split"]["random_state"])
    pca.fit(X_train_spend_scaled)
    
    # Determine components explaining >= 90% cumulative variance
    cum_variance = np.cumsum(pca.explained_variance_ratio_)
    min_variance_ratio = config_data["pca"]["min_variance_ratio"]
    n_components = int(np.argmax(cum_variance >= min_variance_ratio) + 1)
    
    logger.info(f"PCA Cumulative Variance Explained: {cum_variance}")
    logger.info(f"Selected components to explain >= {min_variance_ratio*100}% variance: {n_components}")
    
    # Save the PCA model
    pca_model_path = os.path.join(models_dir, "pca.joblib")
    joblib.dump(pca, pca_model_path)
    
    # Transform spend percentages using PCA
    pca_train = pca.transform(X_train_spend_scaled)[:, :n_components]
    pca_val = pca.transform(X_val_spend_scaled)[:, :n_components]
    pca_test = pca.transform(X_test_spend_scaled)[:, :n_components]
    
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    
    # Combine PCA columns with scaled non-spend features to form Feature Set 2 (PCA features)
    # Fit scaler on non-spend features
    non_spend_scaler = StandardScaler()
    X_train_ns_scaled = non_spend_scaler.fit_transform(X_train_raw[non_spend_features])
    X_val_ns_scaled = non_spend_scaler.transform(X_val_raw[non_spend_features])
    X_test_ns_scaled = non_spend_scaler.transform(X_test_raw[non_spend_features])
    
    # Save non-spend scaler
    joblib.dump(non_spend_scaler, os.path.join(models_dir, "non_spend_scaler.joblib"))
    
    train_pca_df = pd.DataFrame(X_train_ns_scaled, index=train_df.index, columns=non_spend_features)
    for i, col in enumerate(pc_cols):
        train_pca_df[col] = pca_train[:, i]
    train_pca_df["High_Value_Customer"] = train_df["High_Value_Customer"]
    train_pca_df["Future_Spend"] = train_df["Future_Spend"]
    train_pca_df.to_csv(os.path.join(processed_dir, "train_pca.csv"))
    
    val_pca_df = pd.DataFrame(X_val_ns_scaled, index=val_df.index, columns=non_spend_features)
    for i, col in enumerate(pc_cols):
        val_pca_df[col] = pca_val[:, i]
    val_pca_df["High_Value_Customer"] = val_df["High_Value_Customer"]
    val_pca_df["Future_Spend"] = val_df["Future_Spend"]
    val_pca_df.to_csv(os.path.join(processed_dir, "validation_pca.csv"))
    
    test_pca_df = pd.DataFrame(X_test_ns_scaled, index=test_df.index, columns=non_spend_features)
    for i, col in enumerate(pc_cols):
        test_pca_df[col] = pca_test[:, i]
    test_pca_df["High_Value_Customer"] = test_df["High_Value_Customer"]
    test_pca_df["Future_Spend"] = test_df["Future_Spend"]
    test_pca_df.to_csv(os.path.join(processed_dir, "test_pca.csv"))
    logger.info("Saved PCA-transformed feature sets.")
    
    # 3. LDA Pipeline (Supervised dimensionality reduction based on High_Value_Customer class)
    logger.info("Fitting LDA strictly on training scaled raw features...")
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_scaled, y_train)
    
    # Save LDA model
    lda_model_path = os.path.join(models_dir, "lda.joblib")
    joblib.dump(lda, lda_model_path)
    
    # Report coefficients and feature importance
    coefficients = lda.coef_[0]
    coef_df = pd.DataFrame({
        "Feature": raw_feature_cols,
        "Coefficient": coefficients,
        "Absolute_Coefficient": np.abs(coefficients)
    }).sort_values(by="Absolute_Coefficient", ascending=False)
    
    logger.info("LDA Discriminant Coefficients (sorted by absolute contribution):")
    for _, row in coef_df.iterrows():
        logger.info(f"  {row['Feature']}: {row['Coefficient']:.4f}")
        
    # Transform raw features using LDA (projects to 1D)
    lda_train = lda.transform(X_train_scaled)
    lda_val = lda.transform(X_val_scaled)
    lda_test = lda.transform(X_test_scaled)
    
    # Combine LDA 1D projection with scaled non-spend features to form Feature Set 3 (LDA features)
    train_lda_df = pd.DataFrame(X_train_ns_scaled, index=train_df.index, columns=non_spend_features)
    train_lda_df["LD1"] = lda_train[:, 0]
    train_lda_df["High_Value_Customer"] = train_df["High_Value_Customer"]
    train_lda_df["Future_Spend"] = train_df["Future_Spend"]
    train_lda_df.to_csv(os.path.join(processed_dir, "train_lda.csv"))
    
    val_lda_df = pd.DataFrame(X_val_ns_scaled, index=val_df.index, columns=non_spend_features)
    val_lda_df["LD1"] = lda_val[:, 0]
    val_lda_df["High_Value_Customer"] = val_df["High_Value_Customer"]
    val_lda_df["Future_Spend"] = val_df["Future_Spend"]
    val_lda_df.to_csv(os.path.join(processed_dir, "validation_lda.csv"))
    
    test_lda_df = pd.DataFrame(X_test_ns_scaled, index=test_df.index, columns=non_spend_features)
    test_lda_df["LD1"] = lda_test[:, 0]
    test_lda_df["High_Value_Customer"] = test_df["High_Value_Customer"]
    test_lda_df["Future_Spend"] = test_df["Future_Spend"]
    test_lda_df.to_csv(os.path.join(processed_dir, "test_lda.csv"))
    logger.info("Saved LDA-projected feature sets.")
    
    # 4. Save metadata documentation for viva explaining collinearity design choice
    pca_metadata = {
        "viva_explanation": (
            "Other_Spend_Pct was intentionally excluded from PCA to avoid perfect multicollinearity. "
            "Because category spend percentages sum to 100%, including all six would make the feature matrix "
            "singular (determinant = 0), preventing valid mathematical transformation and scaling."
        ),
        "spend_features_selected_for_pca": spend_features,
        "spend_features_excluded_from_pca": ["Other_Spend_Pct"],
        "variance_explained_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "cumulative_variance_explained": [float(c) for c in cum_variance],
        "min_variance_ratio_target": float(min_variance_ratio),
        "selected_components_count": int(n_components),
        "lda_discriminant_coefficients": {row['Feature']: float(row['Coefficient']) for _, row in coef_df.iterrows()}
    }
    
    pca_meta_path = os.path.join(processed_dir, "pca_metadata.json")
    with open(pca_meta_path, "w") as f:
        json.dump(pca_metadata, f, indent=4)
    logger.info(f"PCA & LDA metadata saved to: {pca_meta_path}")
    
    # 5. Generate and save figures
    logger.info("Generating dimensionality reduction plots...")
    
    # Figure 4: PCA Explained Variance
    plot_pca_explained_variance(pca.explained_variance_ratio_)
    
    # Prepare dataframe for PCA scatter plot (PC1 vs PC2 colored by label)
    # Using Train set PCA scores for visualization
    viz_pca_df = pd.DataFrame({
        "PC1": pca.transform(X_train_spend_scaled)[:, 0],
        "PC2": pca.transform(X_train_spend_scaled)[:, 1],
        "High_Value_Customer": train_df["High_Value_Customer"].values
    })
    
    # Figure 5: PCA Scatter Plot
    plot_pca_scatter(viz_pca_df, "PC1", "PC2", "High_Value_Customer")
    
    # Prepare dataframe for LDA projection (LD1 colored by label)
    viz_lda_df = pd.DataFrame({
        "LD1": lda_train[:, 0],
        "High_Value_Customer": train_df["High_Value_Customer"].values
    })
    
    # Figure 6: LDA Projection
    plot_lda_projection(viz_lda_df, "LD1", "High_Value_Customer")
    
    logger.info("Figures successfully generated and saved under outputs/figures/.")

if __name__ == "__main__":
    perform_dimensionality_reduction()
