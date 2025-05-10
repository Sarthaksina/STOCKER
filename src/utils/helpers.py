"""STOCKER Pro Helpers Module

This module provides helper functions used across the STOCKER Pro application.
It includes utilities for recommendations, data validation, and common operations.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

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
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if metric_column not in data.columns:
            raise ValueError(f"Column {metric_column} not found in DataFrame")
            
        # Sort and get top N
        sorted_data = data.sort_values(metric_column, ascending=ascending)
        top_n = sorted_data.head(n)
        
        # Convert to list of dictionaries
        return top_n.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error in top_n_recommender: {e}")
        return []

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