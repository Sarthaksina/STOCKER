"""
Technical indicators for financial time series analysis.

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


def calculate_stochastic_oscillator(
    data: pd.DataFrame,
    high_column: str = 'high',
    low_column: str = 'low',
    close_column: str = 'close',
    k_period: int = 14,
    d_period: int = 3
) -> pd.DataFrame:
    """
    Calculate the Stochastic Oscillator indicator.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data.
        high_column (str, optional): Column name containing high prices. Defaults to 'high'.
        low_column (str, optional): Column name containing low prices. Defaults to 'low'.
        close_column (str, optional): Column name containing close prices. Defaults to 'close'.
        k_period (int, optional): %K period. Defaults to 14.
        d_period (int, optional): %D period. Defaults to 3.
        
    Returns:
        pd.DataFrame: DataFrame with added Stochastic Oscillator indicators.
        
    Raises:
        FeatureEngineeringError: If required columns don't exist.
    """
    for col, name in [(high_column, 'high'), (low_column, 'low'), (close_column, 'close')]:
        if col not in data.columns:
            raise FeatureEngineeringError(f"{name.capitalize()} column {col} not found in data.")
    
    result = data.copy()
    
    # Calculate %K (fast stochastic)
    lowest_low = result[low_column].rolling(window=k_period).min()
    highest_high = result[high_column].rolling(window=k_period).max()
    
    result['stoch_k'] = 100 * ((result[close_column] - lowest_low) / (highest_high - lowest_low))
    
    # Calculate %D (slow stochastic) as simple moving average of %K
    result['stoch_d'] = result['stoch_k'].rolling(window=d_period).mean()
    
    return result


def calculate_average_directional_index(
    data: pd.DataFrame,
    high_column: str = 'high',
    low_column: str = 'low',
    close_column: str = 'close',
    period: int = 14
) -> pd.DataFrame:
    """
    Calculate the Average Directional Index (ADX) indicator.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data.
        high_column (str, optional): Column name containing high prices. Defaults to 'high'.
        low_column (str, optional): Column name containing low prices. Defaults to 'low'.
        close_column (str, optional): Column name containing close prices. Defaults to 'close'.
        period (int, optional): ADX period. Defaults to 14.
        
    Returns:
        pd.DataFrame: DataFrame with added ADX indicator.
        
    Raises:
        FeatureEngineeringError: If required columns don't exist.
    """
    for col, name in [(high_column, 'high'), (low_column, 'low'), (close_column, 'close')]:
        if col not in data.columns:
            raise FeatureEngineeringError(f"{name.capitalize()} column {col} not found in data.")
    
    result = data.copy()
    
    # Calculate True Range (TR)
    high_low = result[high_column] - result[low_column]
    high_close_prev = abs(result[high_column] - result[close_column].shift(1))
    low_close_prev = abs(result[low_column] - result[close_column].shift(1))
    
    true_range = pd.DataFrame({
        'hl': high_low,
        'hcp': high_close_prev,
        'lcp': low_close_prev
    }).max(axis=1)
    
    # Calculate directional movement
    positive_dm = result[high_column] - result[high_column].shift(1)
    negative_dm = result[low_column].shift(1) - result[low_column]
    
    positive_dm = positive_dm.where((positive_dm > 0) & (positive_dm > negative_dm), 0)
    negative_dm = negative_dm.where((negative_dm > 0) & (negative_dm > positive_dm), 0)
    
    # Calculate smoothed TR, +DM, and -DM
    atr = true_range.rolling(window=period).mean()
    plus_di = 100 * (positive_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (negative_dm.rolling(window=period).mean() / atr)
    
    # Calculate Directional Index (DX)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX
    result['adx'] = dx.rolling(window=period).mean()
    result['plus_di'] = plus_di
    result['minus_di'] = minus_di
    
    return result


def calculate_volatility(
    data: pd.DataFrame,
    price_column: str = 'close',
    periods: List[int] = [5, 10, 20, 50, 100]
) -> pd.DataFrame:
    """
    Calculate price volatility over different periods.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data.
        price_column (str, optional): Column name containing price data. Defaults to 'close'.
        periods (List[int], optional): List of periods to calculate volatility for.
            Defaults to [5, 10, 20, 50, 100].
        
    Returns:
        pd.DataFrame: DataFrame with added volatility indicators.
        
    Raises:
        FeatureEngineeringError: If price column doesn't exist.
    """
    if price_column not in data.columns:
        raise FeatureEngineeringError(f"Price column {price_column} not found in data.")
        
    result = data.copy()
    
    # Calculate log returns
    log_returns = np.log(result[price_column] / result[price_column].shift(1))
    
    # Calculate volatility for each period
    for period in periods:
        result[f'volatility_{period}d'] = log_returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
    
    return result


def calculate_trend_indicators(
    data: pd.DataFrame,
    price_column: str = 'close',
    ma_periods: List[int] = [10, 20, 50, 100, 200],
    ema_periods: List[int] = [10, 20, 50, 100, 200]
) -> pd.DataFrame:
    """
    Calculate moving averages and trend indicators.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data.
        price_column (str, optional): Column name containing price data. Defaults to 'close'.
        ma_periods (List[int], optional): List of periods for simple moving averages.
        ema_periods (List[int], optional): List of periods for exponential moving averages.
        
    Returns:
        pd.DataFrame: DataFrame with added trend indicators.
        
    Raises:
        FeatureEngineeringError: If price column doesn't exist.
    """
    if price_column not in data.columns:
        raise FeatureEngineeringError(f"Price column {price_column} not found in data.")
        
    result = data.copy()
    
    # Calculate Simple Moving Averages
    for period in ma_periods:
        result[f'sma_{period}'] = result[price_column].rolling(window=period).mean()
        # Calculate price relative to MA
        result[f'price_to_sma_{period}'] = result[price_column] / result[f'sma_{period}'] - 1
    
    # Calculate Exponential Moving Averages
    for period in ema_periods:
        result[f'ema_{period}'] = result[price_column].ewm(span=period, adjust=False).mean()
        # Calculate price relative to EMA
        result[f'price_to_ema_{period}'] = result[price_column] / result[f'ema_{period}'] - 1
    
    # Calculate moving average crossovers
    if len(ma_periods) > 1:
        ma_periods.sort()
        for i in range(len(ma_periods) - 1):
            fast_period = ma_periods[i]
            slow_period = ma_periods[i + 1]
            result[f'sma_{fast_period}_{slow_period}_cross'] = (
                result[f'sma_{fast_period}'] - result[f'sma_{slow_period}']
            )
    
    if len(ema_periods) > 1:
        ema_periods.sort()
        for i in range(len(ema_periods) - 1):
            fast_period = ema_periods[i]
            slow_period = ema_periods[i + 1]
            result[f'ema_{fast_period}_{slow_period}_cross'] = (
                result[f'ema_{fast_period}'] - result[f'ema_{slow_period}']
            )
    
    return result


def calculate_macd(
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
    indicators = TechnicalIndicators()
    return indicators.calculate_macd(
        data, price_column, fast_period, slow_period, signal_period
    )


def calculate_technical_indicators(
    data: pd.DataFrame,
    include_macd: bool = True,
    include_rsi: bool = True,
    include_bollinger: bool = True,
    include_stochastic: bool = True,
    include_adx: bool = True,
    include_volatility: bool = True,
    include_trend: bool = True,
    price_column: str = 'close',
    high_column: str = 'high',
    low_column: str = 'low'
) -> pd.DataFrame:
    """
    Calculate a comprehensive set of technical indicators.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data.
        include_macd (bool, optional): Whether to include MACD. Defaults to True.
        include_rsi (bool, optional): Whether to include RSI. Defaults to True.
        include_bollinger (bool, optional): Whether to include Bollinger Bands. Defaults to True.
        include_stochastic (bool, optional): Whether to include Stochastic Oscillator. Defaults to True.
        include_adx (bool, optional): Whether to include ADX. Defaults to True.
        include_volatility (bool, optional): Whether to include volatility measures. Defaults to True.
        include_trend (bool, optional): Whether to include trend indicators. Defaults to True.
        price_column (str, optional): Column name containing close prices. Defaults to 'close'.
        high_column (str, optional): Column name containing high prices. Defaults to 'high'.
        low_column (str, optional): Column name containing low prices. Defaults to 'low'.
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    result = data.copy()
    
    if include_macd:
        result = calculate_macd(result, price_column=price_column)
    
    if include_rsi:
        result = calculate_rsi(result, price_column=price_column)
    
    if include_bollinger:
        result = calculate_bollinger_bands(result, price_column=price_column)
    
    # Check if high and low columns exist for indicators that need them
    has_ohlc = all(col in result.columns for col in [high_column, low_column, price_column])
    
    if include_stochastic and has_ohlc:
        result = calculate_stochastic_oscillator(
            result, 
            high_column=high_column,
            low_column=low_column,
            close_column=price_column
        )
    
    if include_adx and has_ohlc:
        result = calculate_average_directional_index(
            result,
            high_column=high_column,
            low_column=low_column,
            close_column=price_column
        )
    
    if include_volatility:
        result = calculate_volatility(result, price_column=price_column)
    
    if include_trend:
        result = calculate_trend_indicators(result, price_column=price_column)
    
    return result 


# Alias for backward compatibility
def add_technical_indicators(
    data: pd.DataFrame,
    include_macd: bool = True,
    include_rsi: bool = True,
    include_bollinger: bool = True,
    include_stochastic: bool = True,
    include_adx: bool = True,
    include_volatility: bool = True,
    include_trend: bool = True,
    price_column: str = 'close',
    high_column: str = 'high',
    low_column: str = 'low'
) -> pd.DataFrame:
    """
    Alias for calculate_technical_indicators for backward compatibility.
    
    Calculates a comprehensive set of technical indicators.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data.
        include_macd (bool, optional): Whether to include MACD. Defaults to True.
        include_rsi (bool, optional): Whether to include RSI. Defaults to True.
        include_bollinger (bool, optional): Whether to include Bollinger Bands. Defaults to True.
        include_stochastic (bool, optional): Whether to include Stochastic Oscillator. Defaults to True.
        include_adx (bool, optional): Whether to include ADX. Defaults to True.
        include_volatility (bool, optional): Whether to include volatility measures. Defaults to True.
        include_trend (bool, optional): Whether to include trend indicators. Defaults to True.
        price_column (str, optional): Column name containing close prices. Defaults to 'close'.
        high_column (str, optional): Column name containing high prices. Defaults to 'high'.
        low_column (str, optional): Column name containing low prices. Defaults to 'low'.
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    return calculate_technical_indicators(
        data,
        include_macd,
        include_rsi,
        include_bollinger,
        include_stochastic,
        include_adx,
        include_volatility,
        include_trend,
        price_column,
        high_column,
        low_column
    )