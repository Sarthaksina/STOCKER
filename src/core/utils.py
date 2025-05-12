"""
Core utility functions for STOCKER Pro.

This module provides comprehensive utility functions used throughout the STOCKER Pro application.
It consolidates general-purpose functionality for better maintainability and organization.

Sections:
    - File and Path Utilities: Functions for file operations and path management
    - Data Processing: Helpers for data manipulation and validation
    - Caching: Data caching mechanisms for performance optimization
    - Error Handling: Decorators and utilities for consistent error handling
    - Formatting: Functions for formatting numbers and text
    - Time Series: Utilities for time series data processing
"""

import os
import time
import json
import pickle
import hashlib
import logging
import functools
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar, cast
from datetime import datetime, timedelta

from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')  # Generic type for function return values

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Project root directory path
    """
    return Path(__file__).parent.parent.parent

def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Created directory: {directory_path}")

def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, 
         exceptions: Tuple = (Exception,)) -> Callable:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts: {str(e)}")
                        raise
                    
                    logger.warning(f"Attempt {attempts} failed, retrying in {current_delay:.2f}s: {str(e)}")
                    time.sleep(current_delay)
                    current_delay *= backoff
                    
        return wrapper
    return decorator

def validate_dataframe(df: pd.DataFrame, required_columns: List[str], 
                     min_rows: int = 1) -> Tuple[bool, Optional[str]]:
    """
    Validate a pandas DataFrame.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required columns
        min_rows: Minimum number of rows required
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        return False, f"Expected DataFrame, got {type(df)}"
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    # Check minimum rows
    if len(df) < min_rows:
        return False, f"DataFrame has {len(df)} rows, minimum required is {min_rows}"
    
    # Check for NaN values in required columns
    for col in required_columns:
        if df[col].isna().any():
            return False, f"Column '{col}' contains NaN values"
    
    return True, None

def sanitize_input(input_string: str, max_length: int = 1000) -> str:
    """
    Sanitize input string by removing potentially harmful characters.
    
    Args:
        input_string: Input string to sanitize
        max_length: Maximum length of the sanitized string
        
    Returns:
        Sanitized string
    """
    # Truncate to max length
    truncated = input_string[:max_length]
    
    # Remove or replace potentially harmful characters
    sanitized = "".join(c for c in truncated if c.isalnum() or c in " _-.,;:!?()[]{}'\"/\\@#$%^&*+=<>")
    
    return sanitized

def load_json(file_path: str) -> Dict:
    """
    Load JSON from file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the loaded JSON data
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {str(e)}")
        return {}

def save_json(data: Dict, file_path: str) -> bool:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to the JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {str(e)}")
        return False

def load_pickle(file_path: str) -> Any:
    """
    Load Python object from pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Loaded object
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle from {file_path}: {str(e)}")
        return None

def save_pickle(obj: Any, file_path: str) -> bool:
    """
    Save Python object to pickle file.
    
    Args:
        obj: Object to save
        file_path: Path to the pickle file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        return True
    except Exception as e:
        logger.error(f"Error saving pickle to {file_path}: {str(e)}")
        return False

def format_number(number: float, precision: int = 2) -> str:
    """
    Format a number with specified precision and add thousands separators.
    
    Args:
        number: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted number string
    """
    return f"{number:,.{precision}f}"

def format_percentage(number: float, precision: int = 2) -> str:
    """
    Format a number as a percentage with specified precision.
    
    Args:
        number: Number to format as percentage
        precision: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{number * 100:.{precision}f}%"

def get_rolling_windows(df: pd.DataFrame, window_sizes: List[int], 
                      target_column: str) -> pd.DataFrame:
    """
    Calculate rolling windows of various sizes for a target column.
    
    Args:
        df: Input DataFrame
        window_sizes: List of window sizes
        target_column: Target column name
        
    Returns:
        DataFrame with rolling window features
    """
    result_df = df.copy()
    
    for window in window_sizes:
        result_df[f"{target_column}_mean_{window}"] = df[target_column].rolling(window=window).mean()
        result_df[f"{target_column}_std_{window}"] = df[target_column].rolling(window=window).std()
        
    return result_df

def calculate_returns(df: pd.DataFrame, price_column: str, 
                    periods: List[int] = [1, 5, 10, 21, 63, 126, 252]) -> pd.DataFrame:
    """
    Calculate returns over various periods.
    
    Args:
        df: Input DataFrame
        price_column: Price column name
        periods: List of periods to calculate returns for
        
    Returns:
        DataFrame with return features
    """
    result_df = df.copy()
    
    for period in periods:
        # Calculate period return
        result_df[f"return_{period}d"] = df[price_column].pct_change(periods=period)
        
    return result_df

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