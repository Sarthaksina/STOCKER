"""Technical indicators for financial time series analysis.

This module provides a comprehensive set of technical indicators used in financial
analysis and algorithmic trading. These indicators are widely used to analyze price
movements, identify trends, and generate trading signals.

Categories of indicators:
- Trend indicators (Moving Averages, MACD, ADX)
- Momentum indicators (RSI, Stochastic Oscillator, CCI)
- Volatility indicators (Bollinger Bands, ATR, Keltner Channels)
- Volume indicators (OBV, MFI, VWAP)
- Support/Resistance (Pivot Points, Fibonacci Retracement)

All functions use pandas DataFrames with proper error handling and type hints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE

from src.core.exceptions import FeatureEngineeringError
from src.core.logging import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """Class containing methods to calculate various technical indicators."""

    def calculate_macd(
        self,
        data: pd.DataFrame,
        price_column: str = 'close',
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        Calculate the Moving Average Convergence Divergence (MACD) indicator.
        
        Args:
            data (pd.DataFrame): DataFrame containing price data.
            price_column (str, optional): Column name containing price data. Defaults to 'close'.
            fast_period (int, optional): Fast EMA period. Defaults to 12.
            slow_period (int, optional): Slow EMA period. Defaults to 26.
            signal_period (int, optional): Signal line period. Defaults to 9.
            
        Returns:
            pd.DataFrame: DataFrame with added MACD indicators.
            
        Raises:
            FeatureEngineeringError: If price column doesn't exist.
        """
        if price_column not in data.columns:
            raise FeatureEngineeringError(f"Price column {price_column} not found in data.")
            
        result = data.copy()
        
        # Calculate fast and slow EMAs
        fast_ema = result[price_column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = result[price_column].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line and signal line
        result['macd_line'] = fast_ema - slow_ema
        result['macd_signal'] = result['macd_line'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate MACD histogram
        result['macd_histogram'] = result['macd_line'] - result['macd_signal']
        
        return result

    def calculate_rsi(
        self,
        data: pd.DataFrame,
        price_column: str = 'close',
        period: int = 14
    ) -> pd.DataFrame:
        """
        Calculate the Relative Strength Index (RSI) indicator.
        
        Args:
            data (pd.DataFrame): DataFrame containing price data.
            price_column (str, optional): Column name containing price data. Defaults to 'close'.
            period (int, optional): RSI period. Defaults to 14.
            
        Returns:
            pd.DataFrame: DataFrame with added RSI indicator.
            
        Raises:
            FeatureEngineeringError: If price column doesn't exist.
        """
        if price_column not in data.columns:
            raise FeatureEngineeringError(f"Price column {price_column} not found in data.")
            
        result = data.copy()
        
        # Calculate price changes
        delta = result[price_column].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss over the period
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        result['rsi'] = 100 - (100 / (1 + rs))
        
        return result

    def calculate_bollinger_bands(
        self,
        data: pd.DataFrame,
        price_column: str = 'close',
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands indicator.
        
        Args:
            data (pd.DataFrame): DataFrame containing price data.
            price_column (str, optional): Column name containing price data. Defaults to 'close'.
            period (int, optional): Simple moving average period. Defaults to 20.
            std_dev (float, optional): Number of standard deviations. Defaults to 2.0.
            
        Returns:
            pd.DataFrame: DataFrame with added Bollinger Bands indicators.
            
        Raises:
            FeatureEngineeringError: If price column doesn't exist.
        """
        if price_column not in data.columns:
            raise FeatureEngineeringError(f"Price column {price_column} not found in data.")
            
        result = data.copy()
        
        # Calculate SMA and standard deviation
        result['bb_middle'] = result[price_column].rolling(window=period).mean()
        result['bb_std'] = result[price_column].rolling(window=period).std()
        
        # Calculate upper and lower bands
        result['bb_upper'] = result['bb_middle'] + (result['bb_std'] * std_dev)
        result['bb_lower'] = result['bb_middle'] - (result['bb_std'] * std_dev)
        
        # Calculate %B (where price is relative to the bands)
        result['bb_pct_b'] = (result[price_column] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        # Calculate bandwidth
        result['bb_bandwidth'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        
        return result
