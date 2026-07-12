"""
Preprocessing Module
--------------------
Cleans the raw transactional data, classifies products into fixed categories,
engineers customer-level features strictly in the feature window (Months 1-9),
calculates the target spend in the target window (Months 10-12) to prevent data leakage,
splits customers into Train/Val/Test (80/10/10), labels High-Value customers based
on the training threshold, and saves the datasets.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any

from src.config import get_absolute_path, config_data
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Fixed Category Keyword Mapping (per university specification)
CATEGORY_KEYWORDS = {
    "Homeware": [
        "cushion", "pillow", "mat", "rug", "towel", "blanket", "mirror", "clock", 
        "furniture", "chair", "table", "lamp", "shade", "curtain", "homeware", 
        "doormat", "frame", "canvas", "wall", "bed", "basket"
    ],
    "Stationery": [
        "pen", "pencil", "notebook", "journal", "pad", "paper", "envelope", 
        "card", "wrap", "ribbon", "sticker", "label", "post", "calendar", 
        "eraser", "ruler", "scissors", "tape", "glue", "book"
    ],
    "Gadgets": [
        "gadget", "phone", "charger", "cable", "usb", "battery", "light", 
        "torch", "alarm", "calculator", "device", "electronic", "plug", 
        "fan", "headphones"
    ],
    "Decorations": [
        "decoration", "ornament", "garland", "wreath", "balloon", "banner", 
        "candle", "holder", "lantern", "t-light", "christmas", "easter", 
        "halloween", "party", "festive", "flower", "vase", "heart", "star", 
        "sign", "plaque", "bauble", "tree", "gift", "baubles", "decorations"
    ],
    "Kitchenware": [
        "kitchen", "bowl", "plate", "cup", "mug", "glass", "fork", "spoon", 
        "knife", "cutlery", "dish", "pan", "pot", "tray", "teapot", "kettle", 
        "bottle", "jar", "container", "jug", "coaster", "shaker", "timer", 
        "baking", "cutter", "apron", "oven", "glove"
    ]
}

def classify_description(description: Any) -> str:
    """
    Classifies a product description into one of 6 categories using keyword matching.

    Args:
        description: Product description string.

    Returns:
        One of: 'Homeware', 'Stationery', 'Gadgets', 'Decorations', 'Kitchenware', 'Other'.
    """
    if not isinstance(description, str):
        return "Other"
    
    desc_lower = description.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in desc_lower:
                return category
    return "Other"

def clean_transaction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw transaction data by:
    - Logging every step with before/after counts.
    - Handling duplicates.
    - Removing records with missing CustomerID.
    - Removing cancelled transactions (InvoiceNo starts with 'C').
    - Removing records with non-positive Quantity or UnitPrice.
    - Creating TotalPrice = Quantity * UnitPrice.

    Args:
        df: Raw pandas DataFrame.

    Returns:
        Cleaned pandas DataFrame.
    """
    logger.info("Starting raw transaction data cleaning process...")
    current_count = df.shape[0]
    logger.info(f"Initial record count: {current_count:,}")
    
    # 1. Handle duplicates
    df_no_dup = df.drop_duplicates()
    new_count = df_no_dup.shape[0]
    logger.info(f"Duplicate removal: {current_count:,} -> {new_count:,} (Removed {current_count - new_count:,} duplicate rows)")
    df = df_no_dup
    current_count = new_count
    
    # 2. Remove CustomerID null
    df_no_null = df.dropna(subset=["CustomerID"])
    new_count = df_no_null.shape[0]
    logger.info(f"Missing CustomerID removal: {current_count:,} -> {new_count:,} (Removed {current_count - new_count:,} rows)")
    df = df_no_null.copy()
    df["CustomerID"] = df["CustomerID"].astype(int)
    current_count = new_count
    
    # 3. Remove cancelled invoices (InvoiceNo starts with 'C')
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df_no_cxl = df[~df["InvoiceNo"].str.startswith("C", na=False)]
    new_count = df_no_cxl.shape[0]
    logger.info(f"Cancelled invoices removal (InvoiceNo starts with 'C'): {current_count:,} -> {new_count:,} (Removed {current_count - new_count:,} rows)")
    df = df_no_cxl
    current_count = new_count
    
    # 4. Remove Quantity <= 0
    df_pos_qty = df[df["Quantity"] > 0]
    new_count = df_pos_qty.shape[0]
    logger.info(f"Non-positive Quantity removal: {current_count:,} -> {new_count:,} (Removed {current_count - new_count:,} rows)")
    df = df_pos_qty
    current_count = new_count
    
    # 5. Remove UnitPrice <= 0
    df_pos_price = df[df["UnitPrice"] > 0]
    new_count = df_pos_price.shape[0]
    logger.info(f"Non-positive UnitPrice removal: {current_count:,} -> {new_count:,} (Removed {current_count - new_count:,} rows)")
    df = df_pos_price
    current_count = new_count
    
    # 6. Create TotalPrice and ensure datetime format
    df = df.copy()
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    
    logger.info(f"Completed cleaning process. Final cleaned records: {df.shape[0]:,}")
    return df

def generate_customer_features(df_features: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """
    Engineers customer-level features strictly in the feature window:
    - Recency: Days since last purchase relative to snapshot date.
    - Frequency: Count of unique invoices.
    - Monetary: Sum of TotalPrice.
    - Average Spend: Monetary / Total Orders.
    - Product Diversity: Count of unique StockCodes.
    - Total Orders: Count of unique invoices (same as Frequency).
    - Average Basket Size: Average number of transaction lines per order.
    - Average Quantity per Order: Average sum of Quantity per order.
    - Category Spend Percentage: Pct of monetary spent on Homeware, Stationery, Gadgets, Decorations, Kitchenware, Other.

    Args:
        df_features: Transaction DataFrame for the feature window (Months 1-9).
        snapshot_date: Snapshot date for recency calculation.

    Returns:
        DataFrame containing customer features.
    """
    logger.info("Computing customer-level features on the Feature Window (Months 1-9)...")
    
    # Group by CustomerID
    cust_groups = df_features.groupby("CustomerID")
    
    # Basic metrics
    recency = cust_groups["InvoiceDate"].max().apply(lambda x: (snapshot_date - x).days)
    total_orders = cust_groups["InvoiceNo"].nunique()
    frequency = total_orders  # Frequency defined as unique orders
    monetary = cust_groups["TotalPrice"].sum()
    product_diversity = cust_groups["StockCode"].nunique()
    average_spend = monetary / total_orders
    
    # Aggregated quantities and order counts for order-level metrics
    total_qty = cust_groups["Quantity"].sum()
    total_lines = cust_groups["StockCode"].count()
    
    average_basket_size = total_lines / total_orders  # Average transaction lines per basket
    average_qty_per_order = total_qty / total_orders  # Average units of items per basket
    
    customer_df = pd.DataFrame({
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "AverageSpend": average_spend,
        "ProductDiversity": product_diversity,
        "TotalOrders": total_orders,
        "AverageBasketSize": average_basket_size,
        "AverageQuantityPerOrder": average_qty_per_order
    })
    
    # Category Spend Percentages
    logger.info("Categorizing product descriptions...")
    df_features = df_features.copy()
    df_features["Category"] = df_features["Description"].apply(classify_description)
    
    # Category Pivot Table (TotalPrice sum grouped by CustomerID and Category)
    category_spend = df_features.pivot_table(
        index="CustomerID",
        columns="Category",
        values="TotalPrice",
        aggfunc="sum",
        fill_value=0.0
    )
    
    # Ensure all 6 categories exist
    for cat in list(CATEGORY_KEYWORDS.keys()) + ["Other"]:
        if cat not in category_spend.columns:
            category_spend[cat] = 0.0
            
    # Calculate spend percentages
    for cat in category_spend.columns:
        customer_df[f"{cat}_Spend_Pct"] = category_spend[cat] / customer_df["Monetary"]
        # Fill any division by zero errors
        customer_df[f"{cat}_Spend_Pct"] = customer_df[f"{cat}_Spend_Pct"].fillna(0.0)
        
    return customer_df

def process_data() -> None:
    """
    Orchestrates data cleaning, feature engineering, split determination,
    High-Value classification, figures plotting, and saved datasets creation.
    """
    raw_dir = get_absolute_path("raw_data_dir")
    processed_dir = get_absolute_path("processed_data_dir")
    os.makedirs(processed_dir, exist_ok=True)
    
    excel_path = os.path.join(raw_dir, "Online Retail.xlsx")
    csv_path = os.path.join(raw_dir, "online_retail.csv")
    
    if not os.path.exists(excel_path):
        logger.error("=" * 80)
        logger.error("CRITICAL ERROR: DATASET FILE NOT FOUND.")
        logger.error(f"Please manually download 'Online Retail.xlsx' from:")
        logger.error("https://archive.ics.uci.edu/dataset/352/online+retail")
        logger.error(f"and place it at: {excel_path}")
        logger.error("=" * 80)
        raise FileNotFoundError(f"Missing original dataset 'Online Retail.xlsx' at: {excel_path}")
        
    # Read/Convert Excel to CSV for performance
    if os.path.exists(csv_path):
        logger.info(f"Loading raw transaction CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        logger.info(f"Loading raw Excel file: {excel_path}")
        df = pd.read_excel(excel_path)
        logger.info(f"Saving transaction CSV for subsequent runs: {csv_path}")
        df.to_csv(csv_path, index=False)
        
    # Clean transactions
    df_cleaned = clean_transaction_data(df)
    
    # Date splitting parameters from config
    feature_window_end = pd.to_datetime(config_data["preprocess"]["feature_window_end"])
    target_window_start = pd.to_datetime(config_data["preprocess"]["target_window_start"])
    
    # Split transactions into Feature and Target windows
    df_features = df_cleaned[df_cleaned["InvoiceDate"] <= feature_window_end]
    df_target = df_cleaned[df_cleaned["InvoiceDate"] >= target_window_start]
    
    logger.info(f"Transactions in Feature Window (Months 1-9): {df_features.shape[0]:,}")
    logger.info(f"Transactions in Target Window (Months 10-12): {df_target.shape[0]:,}")
    
    # Define Snapshot Date = max(InvoiceDate in feature window) + 1 day
    snapshot_date = df_features["InvoiceDate"].max() + pd.Timedelta(days=1)
    logger.info(f"Feature Window Snapshot Date: {snapshot_date}")
    
    # Generate Customer-Level Feature Dataset (engineered from Months 1-9 only)
    customer_features = generate_customer_features(df_features, snapshot_date)
    
    # Calculate Target variable: Future Spend (total spending in target window)
    logger.info("Computing target variable 'Future_Spend' from Target Window (Months 10-12)...")
    target_spend = df_target.groupby("CustomerID")["TotalPrice"].sum()
    customer_features["Future_Spend"] = customer_features.index.map(target_spend).fillna(0.0)
    
    # Customer Level Splitting (Train: 80%, Val: 10%, Test: 10%)
    customer_ids = customer_features.index.values
    random_state = config_data["data_split"]["random_state"]
    
    train_ids, temp_ids = train_test_split(
        customer_ids, 
        train_size=config_data["data_split"]["train_size"], 
        random_state=random_state
    )
    
    # Split the remaining 20% into validation (10% overall) and test (10% overall)
    val_ratio_of_temp = config_data["data_split"]["val_size"] / (
        config_data["data_split"]["val_size"] + config_data["data_split"]["test_size"]
    )
    val_ids, test_ids = train_test_split(
        temp_ids, 
        train_size=val_ratio_of_temp, 
        random_state=random_state
    )
    
    logger.info(f"Splits determined: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # Build Train, Val, Test customer DataFrames
    train_df = customer_features.loc[train_ids].copy()
    val_df = customer_features.loc[val_ids].copy()
    test_df = customer_features.loc[test_ids].copy()
    
    # Label High-Value customers (80th percentile of Monetary value, computed strictly on training set)
    high_value_pct = config_data["preprocess"]["high_value_percentile"]
    monetary_threshold = np.percentile(train_df["Monetary"], high_value_pct)
    logger.info(f"High-Value threshold ({high_value_pct}th percentile of Monetary on Train): ${monetary_threshold:.2f}")
    
    train_df["High_Value_Customer"] = (train_df["Monetary"] >= monetary_threshold).astype(int)
    val_df["High_Value_Customer"] = (val_df["Monetary"] >= monetary_threshold).astype(int)
    test_df["High_Value_Customer"] = (test_df["Monetary"] >= monetary_threshold).astype(int)
    
    # Save Customer mappings to category_map.json
    logger.info("Generating description-to-category mapping for category_map.json...")
    unique_descriptions = df_cleaned["Description"].dropna().unique()
    mapping_dict = {desc: classify_description(desc) for desc in unique_descriptions}
    category_map_path = get_absolute_path("category_map_path")
    with open(category_map_path, "w") as f:
        json.dump(mapping_dict, f, indent=4)
    logger.info(f"Category map saved to: {category_map_path}")
    
    # Save processed CSV files as train.csv, validation.csv, test.csv
    train_df.to_csv(os.path.join(processed_dir, "train.csv"))
    val_df.to_csv(os.path.join(processed_dir, "validation.csv"))
    test_df.to_csv(os.path.join(processed_dir, "test.csv"))
    logger.info("Processed split datasets saved under data/processed/.")
    
    # Save customer IDs split representation for exact reproducibility
    splits_json_path = os.path.join(processed_dir, "customer_splits.json")
    splits_data = {
        "train_customer_ids": [int(cid) for cid in train_ids],
        "val_customer_ids": [int(cid) for cid in val_ids],
        "test_customer_ids": [int(cid) for cid in test_ids]
    }
    with open(splits_json_path, "w") as f:
        json.dump(splits_data, f, indent=4)
    logger.info(f"Split customer IDs saved to: {splits_json_path}")
    
    # Save preprocessing metadata for downstream modeling
    metadata = {
        "snapshot_date": str(snapshot_date),
        "high_value_threshold": float(monetary_threshold),
        "train_customers_count": int(train_df.shape[0]),
        "val_customers_count": int(val_df.shape[0]),
        "test_customers_count": int(test_df.shape[0])
    }
    
    metadata_path = os.path.join(processed_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved to: {metadata_path}")

    # Generate Preprocessing Plots
    logger.info("Generating preprocessing figures...")
    from utils.plotting_utils import plot_data_cleaning_summary, plot_feature_distributions, plot_correlation_heatmap
    
    # 1. Data Cleaning Summary
    plot_data_cleaning_summary(before_count=int(df.shape[0]), after_count=int(df_cleaned.shape[0]))
    
    # 2. Feature Distributions & 3. Correlation Heatmap
    feature_cols = [
        "Recency", "Frequency", "Monetary", "AverageSpend", "ProductDiversity", 
        "TotalOrders", "AverageBasketSize", "AverageQuantityPerOrder"
    ]
    plot_feature_distributions(train_df, feature_cols)
    plot_correlation_heatmap(train_df, feature_cols)
    logger.info("Preprocessing figures successfully saved under outputs/figures/.")

if __name__ == "__main__":
    process_data()
