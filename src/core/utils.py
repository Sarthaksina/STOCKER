"""
Core utility functions for STOCKER Pro.

This module provides utility functions used throughout the application.
"""

import os
import time
import json
import pickle
import logging
import functools
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar, cast

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