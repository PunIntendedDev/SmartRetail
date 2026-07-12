"""
Config Loader Module
-------------------
Provides a unified utility to load project settings from config/config.yaml and
resolve relative directory paths to absolute paths dynamically.
"""

import os
import yaml
from typing import Any, Dict

# Resolve project directories
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Loads configuration settings from the specified YAML file.

    Args:
        config_path: The absolute or relative path to the YAML file.

    Returns:
        A dictionary representing the configuration structure.

    Raises:
        FileNotFoundError: If the config file cannot be resolved.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Instantiate global configuration
config_data = load_config()

def get_absolute_path(path_key: str) -> str:
    """
    Resolves a path key from the configuration file to its absolute path on disk.

    Args:
        path_key: Key inside the 'paths' section of config.yaml (e.g. 'raw_data_dir').

    Returns:
        The absolute path to the directory or file.
    """
    paths = config_data.get("paths", {})
    target_path = paths.get(path_key)
    
    if not target_path:
        raise KeyError(f"Path key '{path_key}' not found in configuration 'paths' section.")
        
    if os.path.isabs(target_path):
        return target_path
        
    return os.path.abspath(os.path.join(PROJECT_ROOT, target_path))
