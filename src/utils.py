"""STOCKER Pro Utilities Module

This module provides a comprehensive set of utility functions used throughout the STOCKER Pro application.
It consolidates functionality previously spread across multiple files for better maintainability and organization.

Sections:
    - Logging: Advanced logging configuration and utilities
    - Caching: Data caching mechanisms for performance optimization
    - Data Processing: Helpers for data manipulation and validation
    - Portfolio Utilities: Functions for portfolio calculations and validation
    - Error Handling: Decorators and utilities for consistent error handling
    - Recommendation: Utilities for generating recommendations
    - Type Validation: Helpers for validating data types and formats
"""

import os
import json
import yaml
import hashlib
import logging
import logging.config
import functools
import traceback
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pathlib import Path
from datetime import datetime, timedelta

# Configure basic logger
logger = logging.getLogger(__name__)

# ===== Logging Utilities =====

def setup_logging(level: int = logging.INFO):
    """
    Configure basic logging for the application.
    
    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def get_advanced_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Create an advanced logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def configure_logging(config_path: Optional[str] = None, default_level: int = logging.INFO) -> None:
    """
    Configure logging from a YAML configuration file.
    
    Args:
        config_path: Path to logging configuration file
        default_level: Default logging level if config file is not found
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                return
            except Exception as e:
                print(f"Error loading logging configuration: {e}")
    
    # Fallback to basic configuration
    setup_logging(default_level)

# ===== Caching Utilities =====

def _get_cache_path(cache_dir: Union[str, Path], key: str) -> Path:
    """
    Generate a cache file path from a cache key.
    
    Args:
        cache_dir: Directory for cache files
        key: Cache key (usually a JSON string of request parameters)
        
    Returns:
        Path to the cache file
    """
    # Create a hash of the key to use as filename
    filename = hashlib.md5(key.encode('utf-8')).hexdigest() + '.json'
    return Path(cache_dir) / filename

def save_to_cache(cache_dir: Union[str, Path], key: str, data: Any) -> None:
    """
    Save data to cache with timestamp.
    
    Args:
        cache_dir: Directory for cache files
        key: Cache key (usually a JSON string of request parameters)
        data: Data to cache (must be JSON serializable)
    """
    try:
        cache_path = _get_cache_path(cache_dir, key)
        
        # Create cache directory if it doesn't exist
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # Add timestamp to cached data
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        # Write to cache file
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f)
            
        logger.debug(f"Data saved to cache: {cache_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save data to cache: {e}")

def load_from_cache(cache_dir: Union[str, Path], key: str, expiry_hours: int = 24) -> Optional[Any]:
    """
    Load data from cache if it exists and is not expired.
    
    Args:
        cache_dir: Directory for cache files
        key: Cache key (usually a JSON string of request parameters)
        expiry_hours: Cache expiry time in hours
        
    Returns:
        Cached data if found and not expired, None otherwise
    """
    try:
        cache_path = _get_cache_path(cache_dir, key)
        
        # Check if cache file exists
        if not os.path.exists(cache_path):
            return None
            
        # Read cache file
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            
        # Parse cache timestamp
        timestamp = datetime.fromisoformat(cache_data['timestamp'])
        
        # Check if cache is expired
        if datetime.now() - timestamp > timedelta(hours=expiry_hours):
            logger.debug(f"Cache expired: {cache_path}")
            return None
            
        logger.debug(f"Using cached data from: {cache_path}")
        return cache_data['data']
        
    except Exception as e:
        logger.warning(f"Failed to load data from cache: {e}")
        return None

def clear_cache(cache_dir: Union[str, Path], older_than_hours: Optional[int] = None) -> int:
    """
    Clear all cache files or only those older than a specified time.
    
    Args:
        cache_dir: Directory for cache files
        older_than_hours: Only clear files older than this many hours (None for all files)
        
    Returns:
        Number of files deleted
    """
    try:
        cache_dir_path = Path(cache_dir)
        
        if not cache_dir_path.exists():
            return 0
            
        count = 0
        for cache_file in cache_dir_path.glob('*.json'):
            try:
                if older_than_hours is not None:
                    # Read cache file to check timestamp
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                        
                    # Parse cache timestamp
                    timestamp = datetime.fromisoformat(cache_data['timestamp'])
                    
                    # Skip if not old enough
                    if datetime.now() - timestamp <= timedelta(hours=older_than_hours):
                        continue
                
                # Delete the file
                os.remove(cache_file)
                count += 1
                
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")
                
        logger.info(f"Cleared {count} cache files from {cache_dir}")
        return count
        
    except Exception as e:
        logger.warning(f"Failed to clear cache: {e}")
        return 0

def get_cache_size(cache_dir: Union[str, Path]) -> int:
    """
    Get the total size of all cache files in bytes.
    
    Args:
        cache_dir: Directory for cache files
        
    Returns:
        Total size in bytes
    """
    try:
        cache_dir_path = Path(cache_dir)
        
        if not cache_dir_path.exists():
            return 0
            
        total_size = 0
        for cache_file in cache_dir_path.glob('*.json'):
            total_size += os.path.getsize(cache_file)
                
        return total_size
        
    except Exception as e:
        logger.warning(f"Failed to get cache size: {e}")
        return 0

def cache_to_file(file_path: str, ttl_seconds: int = 3600):
    """
    Decorator to cache function results to a file with time-to-live.
    
    Args:
        file_path: Path to cache file
        ttl_seconds: Time-to-live in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Check if cache file exists and is not expired
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        cache_data = json.load(f)
                    
                    # Check if cache is expired
                    if 'timestamp' in cache_data:
                        cache_time = datetime.fromisoformat(cache_data['timestamp'])
                        if (datetime.now() - cache_time).total_seconds() < ttl_seconds:
                            return cache_data['result']
                except Exception as e:
                    logger.warning(f"Error reading cache: {e}")
            
            # Call the function and cache the result
            result = func(*args, **kwargs)
            
            try:
                with open(file_path, 'w') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'result': result
                    }, f)
            except Exception as e:
                logger.warning(f"Error writing cache: {e}")
            
            return result
        return wrapper
    return decorator

# ===== Data Processing Utilities =====

def top_n_recommender(df: pd.DataFrame, score_col: str = "score", n: int = 5) -> List[str]:
    """
    Returns top-N items by score.
    
    Args:
        df: DataFrame with scores
        score_col: Column name for scores
        n: Number of items to return
        
    Returns:
        List of top-N item indices
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Input 'df' must be a pandas DataFrame, got {type(df)}")
        return []
        
    if score_col not in df.columns:
        logger.warning(f"Score column '{score_col}' not found in DataFrame columns: {df.columns.tolist()}")
        return []
        
    try:
        # Ensure the score column is numeric before sorting
        if not pd.api.types.is_numeric_dtype(df[score_col]):
            logger.warning(f"Score column '{score_col}' is not numeric. Attempting conversion.")
            # Attempt conversion, coercing errors to NaN
            df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
            # Drop rows where conversion failed
            df = df.dropna(subset=[score_col])

        if df.empty:
            logger.warning("DataFrame is empty after handling non-numeric scores.")
            return []

        # Sort and get top N
        top_items = df.sort_values(score_col, ascending=False).head(n)
        return top_items.index.tolist()
    except Exception as e:
        logger.error(f"Error in top_n_recommender while sorting/selecting top N: {e}")
        return []

# ===== Error Handling Utilities =====

def log_exceptions(func: Callable) -> Callable:
    """
    Decorator to log exceptions raised by a function.
    
    Args:
        func: Function to monitor for exceptions
        
    Returns:
        Exception-logging function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in {func.__name__}: {str(e)}")
            raise
    return wrapper

def error_message_detail(error: Exception, error_detail: Any) -> str:
    """
    Extracts detailed error information including file name, line number, and the error message.
    
    Args:
        error: The exception that occurred
        error_detail: The sys module to access traceback details
        
    Returns:
        A formatted error message string
    """
    exc_tb = error_detail.exc_info()[2]
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "<unknown>"
    line_number = exc_tb.tb_lineno if exc_tb else -1
    error_message = (
        f"Error occurred in python script: [{file_name}] at line number [{line_number}]: {str(error)}"
    )
    logging.error(error_message)
    return error_message

# ===== Portfolio Utilities =====

def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None, min_rows: int = 1) -> bool:
    """
    Validate a DataFrame against requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        True if validation passes
        
    Raises:
        TypeError: If input is not a DataFrame
        ValueError: If DataFrame doesn't meet requirements
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None")
        
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows, got {len(df)}")
        
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")
            
    return True

def validate_series(series: Union[pd.Series, List, np.ndarray], min_length: int = 1) -> pd.Series:
    """
    Validate and convert input to pandas Series.
    
    Args:
        series: Input to validate and convert
        min_length: Minimum length required
        
    Returns:
        Validated pandas Series
        
    Raises:
        TypeError: If input cannot be converted to Series
        ValueError: If Series doesn't meet requirements
    """
    if series is None:
        raise ValueError("Input Series cannot be None")
        
    # Convert to pandas Series if not already
    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series)
        except Exception as e:
            raise TypeError(f"Could not convert to pandas Series: {e}")
    
    if not isinstance(series, pd.Series):
        raise TypeError(f"Expected pandas Series, got {type(series).__name__}")
        
    if len(series) < min_length:
        raise ValueError(f"Series must have at least {min_length} elements, got {len(series)}")
        
    return series

def validate_weights(weights: Union[Dict, List, np.ndarray], assets: List[str]) -> np.ndarray:
    """
    Validate and normalize portfolio weights.
    
    Args:
        weights: Portfolio weights as dict, list, or array
        assets: List of asset names
        
    Returns:
        Normalized weights as numpy array
        
    Raises:
        TypeError: If weights cannot be converted to array
        ValueError: If weights don't match assets or contain invalid values
    """
    # Handle dictionary of weights
    if isinstance(weights, dict):
        # Check for missing assets
        missing_assets = [asset for asset in assets if asset not in weights]
        if missing_assets:
            raise ValueError(f"Missing weights for assets: {missing_assets}")
            
        # Convert to array in the same order as assets
        weights_array = np.array([weights[asset] for asset in assets])
    else:
        # Convert to numpy array if not already
        try:
            weights_array = np.array(weights)
        except Exception as e:
            raise TypeError(f"Could not convert weights to numpy array: {e}")
    
    # Check for NaN or infinite values
    if np.any(~np.isfinite(weights_array)):
        raise ValueError("Weights contain NaN or infinite values")
        
    # Check if number of weights matches number of assets
    if len(weights_array) != len(assets):
        raise ValueError(f"Number of weights ({len(weights_array)}) doesn't match number of assets ({len(assets)})")
        
    # Normalize weights to sum to 1
    weights_sum = np.sum(weights_array)
    if weights_sum > 0:
        weights_array = weights_array / weights_sum
        
    return weights_array

# Export all functions
__all__ = [
    # Logging
    'setup_logging',
    'get_advanced_logger',
    'configure_logging',
    
    # Caching
    'save_to_cache',
    'load_from_cache',
    'clear_cache',
    'get_cache_size',
    'cache_to_file',
    
    # Data Processing
    'top_n_recommender',
    
    # Error Handling
    'log_exceptions',
    'error_message_detail',
    
    # Portfolio Utilities
    'validate_dataframe',
    'validate_series',
    'validate_weights'
]