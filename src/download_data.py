"""
Dataset Downloader Module
--------------------------
Downloads the UCI Online Retail Dataset (Excel file) and saves it under data/raw/.
If the download fails or is blocked, it displays an error instructing the user
to manually download and place the dataset.
"""

import os
import sys
import requests
from src.config import get_absolute_path
from utils.logging_utils import get_logger

logger = get_logger(__name__)

UCI_DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

def download_dataset() -> None:
    """
    Downloads the Online Retail dataset Excel file from UCI and saves it.
    If the file exists, it skips the download. If download fails, it exits with an error.
    """
    raw_dir = get_absolute_path("raw_data_dir")
    os.makedirs(raw_dir, exist_ok=True)
    
    target_path = os.path.join(raw_dir, "Online Retail.xlsx")
    
    # Check if file already exists
    if os.path.exists(target_path):
        logger.info(f"Dataset already exists locally at: {target_path}. Skipping download.")
        return
        
    logger.info(f"Attempting to download Online Retail Dataset from: {UCI_DATASET_URL}")
    try:
        response = requests.get(UCI_DATASET_URL, timeout=60, stream=True)
        response.raise_for_status()
        
        # Write file in chunks to show progression if large
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    
        logger.info(f"Successfully downloaded and saved dataset to: {target_path}")
        
    except Exception as e:
        logger.error(f"Failed to download the Online Retail Dataset: {str(e)}")
        # Output clear instructions to the user to place the dataset manually
        logger.error("=" * 80)
        logger.error("CRITICAL ERROR: AUTOMATIC DOWNLOAD FAILED.")
        logger.error("Please download the dataset manually:")
        logger.error("1. Go to: https://archive.ics.uci.edu/dataset/352/online+retail")
        logger.error("2. Download the data file 'Online Retail.xlsx'")
        logger.error(f"3. Place the file manually inside the raw data folder at:")
        logger.error(f"   {target_path}")
        logger.error("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    download_dataset()
