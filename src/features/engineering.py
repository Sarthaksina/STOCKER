"""
Feature engineering utilities for financial time series data.

This module provides functions and classes for creating features from financial data,
including lag features, rolling statistics, and time-based features. It also includes
utilities for feature transformation, selection, and outlier removal.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE

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


def remove_outliers(df: pd.DataFrame, method: str = "zscore", threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame using various methods.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        method (str): Method to use for outlier detection ('zscore' or 'iqr')
        threshold (float): Threshold for outlier detection
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if method == "zscore":
        z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))
        return df[(z_scores < threshold).all(axis=1)]
    elif method == "iqr":
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    else:
        logger.warning(f"Unknown outlier removal method: {method}. Returning original DataFrame.")
        return df


def feature_engineer(
    df: pd.DataFrame,
    config: Dict[str, Any],
    target_col: Optional[str] = None,
    logger_instance: Optional[Any] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive feature engineering pipeline with configuration.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        config (Dict[str, Any]): Configuration dictionary
        target_col (Optional[str]): Target column name for classification/regression tasks
        logger_instance (Optional[Any]): Logger instance
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Processed DataFrame and artifacts dictionary
    """
    log = logger_instance or logger
    artifact = {"steps": [], "errors": [], "feature_names": list(df.columns)}
    
    try:
        # Apply data transformation if configured
        if config.get("data_transformation", {}).get("enabled", False):
            from src.core.utils import build_transformation_pipeline
            pipeline = build_transformation_pipeline(config.get("data_transformation", {}))
            df = pd.DataFrame(pipeline.fit_transform(df), columns=df.columns)
            artifact["steps"].append("Applied data transformation pipeline.")
        
        # Remove outliers if configured
        if config.get("outlier_removal", {}).get("enabled", False):
            method = config["outlier_removal"].get("method", "zscore")
            threshold = config["outlier_removal"].get("threshold", 3.0)
            before = len(df)
            df = remove_outliers(df, method, threshold)
            after = len(df)
            artifact["steps"].append(f"Removed outliers using {method}, threshold={threshold} (rows: {before} -> {after})")
        
        # Handle class imbalance if configured
        if target_col and config.get("imbalance_handling", {}).get("enabled", False):
            smote = SMOTE(random_state=config.get("random_state", 42))
            X = df.drop(columns=[target_col])
            y = df[target_col]
            X_res, y_res = smote.fit_resample(X, y)
            df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_col)], axis=1)
            artifact["steps"].append("Applied SMOTE for class imbalance.")
        
        # Apply custom features if configured
        if config.get("custom_features"):
            for func in config["custom_features"]:
                df = func(df)
                artifact["steps"].append(f"Applied custom feature: {func.__name__}")
        
        artifact["feature_names"] = list(df.columns)
        log.info(f"Feature engineering completed. Steps: {artifact['steps']}")
    except Exception as e:
        log.error(f"Feature engineering failed: {e}")
        artifact["errors"].append(str(e))
        raise
    
    return df, artifact


# Alias for backward compatibility
FeatureEngineering = FeatureEngineer