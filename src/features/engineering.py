"""
Feature engineering utilities for financial time series data.

This module provides functions and classes for creating features from financial data,
including lag features, rolling statistics, and time-based features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from src.core.exceptions import FeatureEngineeringError
from src.core.logging import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Main class for feature engineering operations on financial time series data.
    
    This class provides methods to create various features for financial analysis
    and machine learning models.
    """
    
    def __init__(self, data: pd.DataFrame = None):
        """
        Initialize the FeatureEngineer with optional data.
        
        Args:
            data (pd.DataFrame, optional): DataFrame containing financial time series data.
        """
        self.data = data
        
    def fit(self, data: pd.DataFrame) -> 'FeatureEngineer':
        """
        Set the data for feature engineering.
        
        Args:
            data (pd.DataFrame): DataFrame containing financial time series data.
            
        Returns:
            FeatureEngineer: Self for method chaining.
        """
        self.data = data
        return self
    
    def create_lag_features(self, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Create lag features for specified columns.
        
        Args:
            columns (List[str]): List of column names to create lags for.
            lags (List[int]): List of lag periods.
            
        Returns:
            pd.DataFrame: DataFrame with added lag features.
            
        Raises:
            FeatureEngineeringError: If data is not set or columns don't exist.
        """
        if self.data is None:
            raise FeatureEngineeringError("Data not set. Call fit() first or provide data in constructor.")
        
        result = self.data.copy()
        
        for col in columns:
            if col not in result.columns:
                raise FeatureEngineeringError(f"Column {col} not found in data.")
            
            for lag in lags:
                result[f"{col}_lag_{lag}"] = result[col].shift(lag)
                
        return result
    
    def create_rolling_features(
        self, 
        columns: List[str], 
        windows: List[int], 
        functions: Dict[str, callable] = None
    ) -> pd.DataFrame:
        """
        Create rolling window features for specified columns.
        
        Args:
            columns (List[str]): List of column names to create rolling features for.
            windows (List[int]): List of window sizes.
            functions (Dict[str, callable], optional): Dictionary of functions to apply to rolling windows.
                If None, defaults to mean, std, min, max.
                
        Returns:
            pd.DataFrame: DataFrame with added rolling features.
            
        Raises:
            FeatureEngineeringError: If data is not set or columns don't exist.
        """
        if self.data is None:
            raise FeatureEngineeringError("Data not set. Call fit() first or provide data in constructor.")
        
        result = self.data.copy()
        
        if functions is None:
            functions = {
                'mean': np.mean,
                'std': np.std,
                'min': np.min,
                'max': np.max
            }
        
        for col in columns:
            if col not in result.columns:
                raise FeatureEngineeringError(f"Column {col} not found in data.")
            
            for window in windows:
                rolling = result[col].rolling(window=window)
                
                for func_name, func in functions.items():
                    result[f"{col}_{func_name}_{window}"] = rolling.apply(func, raw=True)
                
        return result
    
    def create_timeframe_features(self) -> pd.DataFrame:
        """
        Create time-based features from the index of the DataFrame.
        
        Assumes the index is a DatetimeIndex.
        
        Returns:
            pd.DataFrame: DataFrame with added time-based features.
            
        Raises:
            FeatureEngineeringError: If data is not set or index is not DatetimeIndex.
        """
        if self.data is None:
            raise FeatureEngineeringError("Data not set. Call fit() first or provide data in constructor.")
        
        result = self.data.copy()
        
        if not isinstance(result.index, pd.DatetimeIndex):
            try:
                result.index = pd.to_datetime(result.index)
            except Exception as e:
                raise FeatureEngineeringError(f"Failed to convert index to DatetimeIndex: {str(e)}")
        
        # Extract date components
        result['day_of_week'] = result.index.dayofweek
        result['day_of_month'] = result.index.day
        result['week_of_year'] = result.index.isocalendar().week
        result['month'] = result.index.month
        result['quarter'] = result.index.quarter
        result['year'] = result.index.year
        
        # Add is_month_start/end and quarter_start/end
        result['is_month_start'] = result.index.is_month_start.astype(int)
        result['is_month_end'] = result.index.is_month_end.astype(int)
        result['is_quarter_start'] = result.index.is_quarter_start.astype(int)
        result['is_quarter_end'] = result.index.is_quarter_end.astype(int)
        
        return result
    
    def create_return_features(
        self, 
        price_column: str, 
        periods: List[int] = [1, 5, 10, 21, 63]
    ) -> pd.DataFrame:
        """
        Create return features over multiple periods.
        
        Args:
            price_column (str): Column name containing price data.
            periods (List[int], optional): List of periods to calculate returns for.
                Defaults to [1, 5, 10, 21, 63] (daily, weekly, biweekly, monthly, quarterly).
                
        Returns:
            pd.DataFrame: DataFrame with added return features.
            
        Raises:
            FeatureEngineeringError: If data is not set or price column doesn't exist.
        """
        if self.data is None:
            raise FeatureEngineeringError("Data not set. Call fit() first or provide data in constructor.")
        
        if price_column not in self.data.columns:
            raise FeatureEngineeringError(f"Price column {price_column} not found in data.")
        
        result = self.data.copy()
        
        for period in periods:
            # Simple returns
            result[f"return_{period}d"] = result[price_column].pct_change(period)
            
            # Log returns
            result[f"log_return_{period}d"] = np.log(result[price_column] / result[price_column].shift(period))
        
        return result
    
    def create_all_features(
        self, 
        price_column: str,
        volume_column: Optional[str] = None,
        lags: List[int] = [1, 2, 3, 5, 10],
        windows: List[int] = [5, 10, 21, 63],
        return_periods: List[int] = [1, 5, 10, 21, 63]
    ) -> pd.DataFrame:
        """
        Create a comprehensive set of features for financial analysis.
        
        Args:
            price_column (str): Column name containing price data.
            volume_column (str, optional): Column name containing volume data.
            lags (List[int], optional): List of lag periods.
            windows (List[int], optional): List of window sizes.
            return_periods (List[int], optional): List of periods to calculate returns for.
                
        Returns:
            pd.DataFrame: DataFrame with all features added.
            
        Raises:
            FeatureEngineeringError: If data is not set or required columns don't exist.
        """
        if self.data is None:
            raise FeatureEngineeringError("Data not set. Call fit() first or provide data in constructor.")
        
        if price_column not in self.data.columns:
            raise FeatureEngineeringError(f"Price column {price_column} not found in data.")
        
        features_df = self.create_timeframe_features()
        
        columns_for_lags = [price_column]
        if volume_column and volume_column in self.data.columns:
            columns_for_lags.append(volume_column)
        
        features_df = FeatureEngineer(features_df).create_lag_features(columns_for_lags, lags)
        features_df = FeatureEngineer(features_df).create_rolling_features(columns_for_lags, windows)
        features_df = FeatureEngineer(features_df).create_return_features(price_column, return_periods)
        
        return features_df


# Standalone functions for ease of use

def generate_features(
    data: pd.DataFrame,
    price_column: str,
    volume_column: Optional[str] = None,
    include_time_features: bool = True,
    include_lags: bool = True,
    include_rolling: bool = True,
    include_returns: bool = True,
    lags: List[int] = [1, 2, 3, 5, 10],
    windows: List[int] = [5, 10, 21, 63],
    return_periods: List[int] = [1, 5, 10, 21, 63]
) -> pd.DataFrame:
    """
    Generate a comprehensive set of features from financial data.
    
    Args:
        data (pd.DataFrame): DataFrame containing financial time series data.
        price_column (str): Column name containing price data.
        volume_column (str, optional): Column name containing volume data.
        include_time_features (bool, optional): Whether to include time-based features.
        include_lags (bool, optional): Whether to include lag features.
        include_rolling (bool, optional): Whether to include rolling window features.
        include_returns (bool, optional): Whether to include return features.
        lags (List[int], optional): List of lag periods.
        windows (List[int], optional): List of window sizes.
        return_periods (List[int], optional): List of periods to calculate returns for.
            
    Returns:
        pd.DataFrame: DataFrame with generated features.
        
    Raises:
        FeatureEngineeringError: If required columns don't exist.
    """
    engineer = FeatureEngineer(data)
    result = data.copy()
    
    if include_time_features:
        result = engineer.fit(result).create_timeframe_features()
    
    if include_lags:
        columns_for_lags = [price_column]
        if volume_column and volume_column in data.columns:
            columns_for_lags.append(volume_column)
        result = engineer.fit(result).create_lag_features(columns_for_lags, lags)
    
    if include_rolling:
        columns_for_rolling = [price_column]
        if volume_column and volume_column in data.columns:
            columns_for_rolling.append(volume_column)
        result = engineer.fit(result).create_rolling_features(columns_for_rolling, windows)
    
    if include_returns:
        result = engineer.fit(result).create_return_features(price_column, return_periods)
    
    return result


def create_timeframe_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from the index of the DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame with DatetimeIndex.
        
    Returns:
        pd.DataFrame: DataFrame with added time-based features.
    """
    return FeatureEngineer(data).create_timeframe_features()


def create_lag_features(
    data: pd.DataFrame,
    columns: List[str],
    lags: List[int]
) -> pd.DataFrame:
    """
    Create lag features for specified columns.
    
    Args:
        data (pd.DataFrame): DataFrame containing data.
        columns (List[str]): List of column names to create lags for.
        lags (List[int]): List of lag periods.
        
    Returns:
        pd.DataFrame: DataFrame with added lag features.
    """
    return FeatureEngineer(data).create_lag_features(columns, lags)


def create_rolling_features(
    data: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    functions: Dict[str, callable] = None
) -> pd.DataFrame:
    """
    Create rolling window features for specified columns.
    
    Args:
        data (pd.DataFrame): DataFrame containing data.
        columns (List[str]): List of column names to create rolling features for.
        windows (List[int]): List of window sizes.
        functions (Dict[str, callable], optional): Dictionary of functions to apply.
            
    Returns:
        pd.DataFrame: DataFrame with added rolling features.
    """
    return FeatureEngineer(data).create_rolling_features(columns, windows, functions) 