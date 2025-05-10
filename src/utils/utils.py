"""STOCKER Pro Utilities Module

This module provides centralized utilities for the STOCKER Pro application.
It consolidates various helper functions, caching utilities, and common operations
used across the application.

Main Components:
- Caching utilities for data persistence
- Data validation and processing helpers
- Financial calculations and metrics
- Recommendation system utilities
"""

import os
import json
import hashlib
import logging
import numpy as np
import pandas as pd
from typing import Any, Optional, Union, List, Dict
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Cache Utilities

def _get_cache_path(cache_dir: Union[str, Path], key: str) -> Path:
    """Generate a cache file path from a cache key.
    
    Args:
        cache_dir: Directory for cache files
        key: Cache key (usually a JSON string of request parameters)
        
    Returns:
        Path to the cache file
    """
    filename = hashlib.md5(key.encode('utf-8')).hexdigest() + '.json'
    return Path(cache_dir) / filename

def save_to_cache(cache_dir: Union[str, Path], key: str, data: Any) -> None:
    """Save data to cache with timestamp.
    
    Args:
        cache_dir: Directory for cache files
        key: Cache key (usually a JSON string of request parameters)
        data: Data to cache (must be JSON serializable)
    """
    try:
        cache_path = _get_cache_path(cache_dir, key)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f)
            
        logger.debug(f"Data saved to cache: {cache_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save data to cache: {e}")

def load_from_cache(cache_dir: Union[str, Path], key: str, expiry_hours: int = 24) -> Optional[Any]:
    """Load data from cache if it exists and is not expired.
    
    Args:
        cache_dir: Directory for cache files
        key: Cache key (usually a JSON string of request parameters)
        expiry_hours: Cache expiry time in hours
        
    Returns:
        Cached data if found and not expired, None otherwise
    """
    try:
        cache_path = _get_cache_path(cache_dir, key)
        
        if not os.path.exists(cache_path):
            return None
            
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            
        timestamp = datetime.fromisoformat(cache_data['timestamp'])
        
        if datetime.now() - timestamp > timedelta(hours=expiry_hours):
            logger.debug(f"Cache expired: {cache_path}")
            return None
            
        logger.debug(f"Using cached data from: {cache_path}")
        return cache_data['data']
        
    except Exception as e:
        logger.warning(f"Failed to load data from cache: {e}")
        return None

def clear_cache(cache_dir: Union[str, Path], older_than_hours: Optional[int] = None) -> int:
    """Clear all cache files or only those older than a specified time.
    
    Args:
        cache_dir: Directory for cache files
        older_than_hours: Only clear files older than this many hours (None for all files)
        
    Returns:
        Number of files deleted
    """
    try:
        cache_dir = Path(cache_dir)
        if not cache_dir.exists():
            return 0
            
        files_deleted = 0
        current_time = datetime.now()
        
        for cache_file in cache_dir.glob('*.json'):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                timestamp = datetime.fromisoformat(cache_data['timestamp'])
                
                if older_than_hours is None or \
                   current_time - timestamp > timedelta(hours=older_than_hours):
                    cache_file.unlink()
                    files_deleted += 1
                    
            except Exception as e:
                logger.warning(f"Error processing cache file {cache_file}: {e}")
                continue
                
        return files_deleted
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return 0

# Data Processing and Validation Utilities

def validate_date_range(
    start_date: str,
    end_date: str,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None
) -> bool:
    """Validate a date range.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        min_date: Optional minimum allowed date
        max_date: Optional maximum allowed date
        
    Returns:
        True if date range is valid
        
    Raises:
        ValueError: If date range is invalid
    """
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if start > end:
            raise ValueError("Start date must be before end date")
            
        if min_date:
            min_dt = pd.to_datetime(min_date)
            if start < min_dt:
                raise ValueError(f"Start date must be after {min_date}")
                
        if max_date:
            max_dt = pd.to_datetime(max_date)
            if end > max_dt:
                raise ValueError(f"End date must be before {max_date}")
                
        return True
        
    except Exception as e:
        logger.error(f"Date validation error: {e}")
        raise ValueError(f"Invalid date format or range: {e}")

# Financial Calculations

def calculate_returns(
    prices: Union[pd.Series, pd.DataFrame],
    method: str = 'simple'
) -> Union[pd.Series, pd.DataFrame]:
    """Calculate returns from price data.
    
    Args:
        prices: Price data as Series or DataFrame
        method: Return calculation method ('simple' or 'log')
        
    Returns:
        Returns series or DataFrame
    """
    if method not in ['simple', 'log']:
        raise ValueError("method must be 'simple' or 'log'")
        
    if method == 'simple':
        returns = prices.pct_change()
    else:  # log returns
        returns = np.log(prices / prices.shift(1))
        
    return returns

def rolling_metrics(
    data: Union[pd.Series, pd.DataFrame],
    window: int,
    metrics: List[str]
) -> pd.DataFrame:
    """Calculate rolling metrics for time series data.
    
    Args:
        data: Input time series data
        window: Rolling window size
        metrics: List of metrics to calculate ('mean', 'std', 'min', 'max')
        
    Returns:
        DataFrame with rolling metrics
    """
    valid_metrics = ['mean', 'std', 'min', 'max']
    invalid_metrics = [m for m in metrics if m not in valid_metrics]
    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {invalid_metrics}")
        
    result = pd.DataFrame(index=data.index)
    
    for metric in metrics:
        if metric == 'mean':
            result[f'rolling_{metric}'] = data.rolling(window).mean()
        elif metric == 'std':
            result[f'rolling_{metric}'] = data.rolling(window).std()
        elif metric == 'min':
            result[f'rolling_{metric}'] = data.rolling(window).min()
        elif metric == 'max':
            result[f'rolling_{metric}'] = data.rolling(window).max()
            
    return result

# Recommendation System

def top_n_recommender(data: pd.DataFrame, metric_column: str, n: int = 5, ascending: bool = False) -> List[Dict[str, Any]]:
    """Get top N items based on a metric column.
    
    Args:
        data: DataFrame containing items and metrics
        metric_column: Column name for sorting
        n: Number of top items to return
        ascending: Sort order (False for highest first)
        
    Returns:
        List of dictionaries with top N items
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if metric_column not in data.columns:
            raise ValueError(f"Column {metric_column} not found in DataFrame")
            
        sorted_data = data.sort_values(metric_column, ascending=ascending)
        top_n = sorted_data.head(n)
        
        return top_n.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error in top_n_recommender: {e}")
        return []