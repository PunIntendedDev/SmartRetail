"""
Logging Utilities
-----------------
Sets up a standardized logging handler that writes logs to both the console
and a log file inside the logs/ directory.
"""

import logging
import os
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured Logger instance. If handlers already exist on the logger,
    returns the existing logger.

    Args:
        name: Name of the logger, typically __name__ of the calling module.

    Returns:
        A standardized logging.Logger object.
    """
    logger = logging.getLogger(name)
    
    # Check if handlers are already configured to prevent duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Formatter configuration
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler (writes to stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler (writes to logs/app.log)
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file_path = os.path.join(logs_dir, "app.log")
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
