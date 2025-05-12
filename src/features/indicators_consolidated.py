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

    def calculate_stochastic_oscillator(
        self,
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
        self,
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
            pd.DataFrame: DataFrame with added ADX, +DI, and -DI indicators.
            
        Raises:
            FeatureEngineeringError: If required columns don't exist.
        """
        for col, name in [(high_column, 'high'), (low_column, 'low'), (close_column, 'close')]:
            if col not in data.columns:
                raise FeatureEngineeringError(f"{name.capitalize()} column {col} not found in data.")
        
        result = data.copy()
        
        # Calculate ATR first
        atr_df = calculate_atr(data, period)
        result['atr'] = atr_df['atr']
        
        # Calculate +DM and -DM
        high_diff = result[high_column].diff()
        low_diff = result[low_column].diff() * -1  # Make positive when low decreases
        
        plus_dm = (high_diff > low_diff) & (high_diff > 0)
        minus_dm = (low_diff > high_diff) & (low_diff > 0)
        
        result['plus_dm'] = np.where(plus_dm, high_diff, 0)
        result['minus_dm'] = np.where(minus_dm, low_diff, 0)
        
        # Calculate smoothed +DM and -DM
        result['plus_dm_smoothed'] = result['plus_dm'].rolling(window=period).mean()
        result['minus_dm_smoothed'] = result['minus_dm'].rolling(window=period).mean()
        
        # Calculate +DI and -DI
        result['plus_di'] = 100 * (result['plus_dm_smoothed'] / result['atr'])
        result['minus_di'] = 100 * (result['minus_dm_smoothed'] / result['atr'])
        
        # Calculate DX
        result['dx'] = 100 * (abs(result['plus_di'] - result['minus_di']) / (result['plus_di'] + result['minus_di']))
        
        # Calculate ADX
        result['adx'] = result['dx'].rolling(window=period).mean()
        
        # Drop intermediate columns
        result = result.drop(['plus_dm', 'minus_dm', 'plus_dm_smoothed', 'minus_dm_smoothed', 'dx'], axis=1)
        
        return result

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> None:
        """
        Validate that the input DataFrame has the required columns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            missing_cols_str = ', '.join(missing_columns)
            raise ValueError(f"DataFrame is missing required columns: {missing_cols_str}")

    @staticmethod
    def sma(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for SMA calculation
            column: Column to use for calculation
            
        Returns:
            Series with SMA values
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer")
            
        return df[column].rolling(window=period).mean()

    @staticmethod
    def ema(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for EMA calculation
            column: Column to use for calculation
            
        Returns:
            Series with EMA values
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer")
            
        return df[column].ewm(span=period, adjust=False).mean()

    @staticmethod
    def macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, 
             signal_period: int = 9, column: str = 'close') -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence.
        
        Args:
            df: DataFrame with OHLCV data
            fast_period: Period for fast EMA
            slow_period: Period for slow EMA
            signal_period: Period for signal line
            column: Column to use for calculation
            
        Returns:
            DataFrame with MACD line, signal line, and histogram
        """
        if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            raise ValueError("Periods must be positive integers")
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
            
        # Calculate fast and slow EMAs
        fast_ema = TechnicalIndicators.ema(df, period=fast_period, column=column)
        slow_ema = TechnicalIndicators.ema(df, period=slow_period, column=column)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Combine into DataFrame
        macd_df = pd.DataFrame({
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }, index=df.index)
        
        return macd_df

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, 
                       column: str = 'close') -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for SMA calculation
            std_dev: Number of standard deviations for bands
            column: Column to use for calculation
            
        Returns:
            DataFrame with upper band, middle band, and lower band
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer")
        if std_dev <= 0:
            raise ValueError("Standard deviation multiplier must be positive")
            
        # Calculate middle band (SMA)
        middle_band = TechnicalIndicators.sma(df, period=period, column=column)
        
        # Calculate standard deviation
        std = df[column].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        # Calculate bandwidth and %B (optional)
        bandwidth = (upper_band - lower_band) / middle_band
        percent_b = (df[column] - lower_band) / (upper_band - lower_band)
        
        # Combine into DataFrame
        bb_df = pd.DataFrame({
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band,
            'bandwidth': bandwidth,
            'percent_b': percent_b
        }, index=df.index)
        
        return bb_df

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for RSI calculation
            column: Column to use for calculation
            
        Returns:
            Series with RSI values
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer")
            
        # Calculate price changes
        delta = df[column].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    @staticmethod
    def stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            df: DataFrame with OHLCV data
            k_period: Period for %K line
            d_period: Period for %D line (SMA of %K)
            
        Returns:
            DataFrame with %K and %D values
        """
        if k_period <= 0 or d_period <= 0:
            raise ValueError("Periods must be positive integers")
            
        # Calculate %K (fast stochastic)
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (slow stochastic) as SMA of %K
        d = k.rolling(window=d_period).mean()
        
        # Combine into DataFrame
        stoch_df = pd.DataFrame({
            'k': k,
            'd': d
        }, index=df.index)
        
        return stoch_df
        
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for ATR calculation
            
        Returns:
            Series with ATR values
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer")
            
        # Calculate True Range (TR)
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR as EMA of TR
        atr = tr.ewm(span=period, adjust=False).mean()
        
        return atr
        
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for ADX calculation
            
        Returns:
            DataFrame with ADX, +DI, and -DI values
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer")
            
        # Calculate True Range (TR)
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate Directional Movement (DM)
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']
        
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # Calculate Smoothed TR and DM over the period
        smoothed_tr = tr.rolling(window=period).sum()
        smoothed_pos_dm = pos_dm.rolling(window=period).sum()
        smoothed_neg_dm = neg_dm.rolling(window=period).sum()
        
        # Calculate Directional Indicators (DI)
        pos_di = 100 * (smoothed_pos_dm / smoothed_tr)
        neg_di = 100 * (smoothed_neg_dm / smoothed_tr)
        
        # Calculate Directional Index (DX)
        dx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di))
        
        # Calculate Average Directional Index (ADX)
        adx = dx.rolling(window=period).mean()
        
        # Combine into DataFrame
        adx_df = pd.DataFrame({
            'adx': adx,
            'pos_di': pos_di,
            'neg_di': neg_di
        }, index=df.index)
        
        return adx_df
        
    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume.
        
        Args:
            df: DataFrame with OHLCV data, must include 'volume' column
            
        Returns:
            Series with OBV values
        """
        if 'volume' not in df.columns:
            raise ValueError("DataFrame must contain a 'volume' column")
            
        # Calculate price change direction
        price_change = df['close'].diff()
        
        # Initialize OBV with first volume value
        obv = pd.Series(0, index=df.index)
        
        # Calculate OBV based on price change direction
        for i in range(1, len(df)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
        
    @staticmethod
    def ichimoku_cloud(df: pd.DataFrame, conversion_period: int = 9, 
                      base_period: int = 26, leading_span_b_period: int = 52, 
                      displacement: int = 26) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud.
        
        Args:
            df: DataFrame with OHLCV data
            conversion_period: Period for Tenkan-sen (Conversion Line)
            base_period: Period for Kijun-sen (Base Line)
            leading_span_b_period: Period for Senkou Span B (Leading Span B)
            displacement: Displacement period for Senkou Span A/B and Chikou Span
            
        Returns:
            DataFrame with Ichimoku Cloud components
        """
        if conversion_period <= 0 or base_period <= 0 or leading_span_b_period <= 0 or displacement <= 0:
            raise ValueError("Periods must be positive integers")
            
        # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the conversion period
        high_conv = df['high'].rolling(window=conversion_period).max()
        low_conv = df['low'].rolling(window=conversion_period).min()
        tenkan_sen = (high_conv + low_conv) / 2
        
        # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for the base period
        high_base = df['high'].rolling(window=base_period).max()
        low_base = df['low'].rolling(window=base_period).min()
        kijun_sen = (high_base + low_base) / 2
        
        # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, displaced forward
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the leading span B period, displaced forward
        high_lead = df['high'].rolling(window=leading_span_b_period).max()
        low_lead = df['low'].rolling(window=leading_span_b_period).min()
        senkou_span_b = ((high_lead + low_lead) / 2).shift(displacement)
        
        # Calculate Chikou Span (Lagging Span): Current closing price, displaced backwards
        chikou_span = df['close'].shift(-displacement)
        
        # Combine into DataFrame
        ichimoku_df = pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }, index=df.index)
        
        return ichimoku_df
        
    @staticmethod
    def keltner_channels(df: pd.DataFrame, ema_period: int = 20, 
                        atr_period: int = 10, multiplier: float = 2.0) -> pd.DataFrame:
        """
        Calculate Keltner Channels.
        
        Args:
            df: DataFrame with OHLCV data
            ema_period: Period for middle line (EMA of close)
            atr_period: Period for ATR calculation
            multiplier: Multiplier for ATR to set channel width
            
        Returns:
            DataFrame with upper, middle, and lower channels
        """
        if ema_period <= 0 or atr_period <= 0:
            raise ValueError("Periods must be positive integers")
        if multiplier <= 0:
            raise ValueError("Multiplier must be positive")
            
        # Calculate middle line (EMA of close)
        middle_line = TechnicalIndicators.ema(df, period=ema_period, column='close')
        
        # Calculate ATR
        atr_value = TechnicalIndicators.atr(df, period=atr_period)
        
        # Calculate upper and lower channels
        upper_channel = middle_line + (multiplier * atr_value)
        lower_channel = middle_line - (multiplier * atr_value)
        
        # Combine into DataFrame
        keltner_df = pd.DataFrame({
            'upper_channel': upper_channel,
            'middle_line': middle_line,
            'lower_channel': lower_channel
        }, index=df.index)
        
        return keltner_df
        
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators added
        """
        # Validate input DataFrame
        TechnicalIndicators.validate_dataframe(df)
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Add moving averages
        result_df['sma_20'] = TechnicalIndicators.sma(df, period=20)
        result_df['sma_50'] = TechnicalIndicators.sma(df, period=50)
        result_df['sma_200'] = TechnicalIndicators.sma(df, period=200)
        result_df['ema_12'] = TechnicalIndicators.ema(df, period=12)
        result_df['ema_26'] = TechnicalIndicators.ema(df, period=26)
        
        # Add MACD
        macd_df = TechnicalIndicators.macd(df)
        result_df['macd_line'] = macd_df['macd_line']
        result_df['macd_signal'] = macd_df['signal_line']
        result_df['macd_histogram'] = macd_df['histogram']
        
        # Add Bollinger Bands
        bb_df = TechnicalIndicators.bollinger_bands(df)
        result_df['bb_upper'] = bb_df['upper_band']
        result_df['bb_middle'] = bb_df['middle_band']
        result_df['bb_lower'] = bb_df['lower_band']
        result_df['bb_bandwidth'] = bb_df['bandwidth']
        result_df['bb_percent_b'] = bb_df['percent_b']
        
        # Add RSI
        result_df['rsi_14'] = TechnicalIndicators.rsi(df)
        
        # Add Stochastic Oscillator
        stoch_df = TechnicalIndicators.stochastic_oscillator(df)
        result_df['stoch_k'] = stoch_df['k']
        result_df['stoch_d'] = stoch_df['d']
        
        # Add ATR
        result_df['atr_14'] = TechnicalIndicators.atr(df)
        
        # Add ADX
        adx_df = TechnicalIndicators.adx(df)
        result_df['adx'] = adx_df['adx']
        result_df['pos_di'] = adx_df['pos_di']
        result_df['neg_di'] = adx_df['neg_di']
        
        # Add OBV
        if 'volume' in df.columns:
            result_df['obv'] = TechnicalIndicators.obv(df)
        
        # Add Ichimoku Cloud
        ichimoku_df = TechnicalIndicators.ichimoku_cloud(df)
        result_df['ichimoku_tenkan_sen'] = ichimoku_df['tenkan_sen']
        result_df['ichimoku_kijun_sen'] = ichimoku_df['kijun_sen']
        result_df['ichimoku_senkou_span_a'] = ichimoku_df['senkou_span_a']
        result_df['ichimoku_senkou_span_b'] = ichimoku_df['senkou_span_b']
        result_df['ichimoku_chikou_span'] = ichimoku_df['chikou_span']
        
        # Add Keltner Channels
        keltner_df = TechnicalIndicators.keltner_channels(df)
        result_df['keltner_upper'] = keltner_df['upper_channel']
        result_df['keltner_middle'] = keltner_df['middle_line']
        result_df['keltner_lower'] = keltner_df['lower_channel']
        
        return result_df


# Standalone functions from the original indicators.py

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


def calculate_atr(
    data: pd.DataFrame,
    period: int = 14
) -> pd.DataFrame:
    """
    Calculate the Average True Range (ATR) indicator.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data with high, low, and close columns.
        period (int, optional): ATR period. Defaults to 14.
        
    Returns:
        pd.DataFrame: DataFrame with added ATR indicator.
        
    Raises:
        FeatureEngineeringError: If required columns don't exist.
    """
    for col in ['high', 'low', 'close']:
        if col not in data.columns:
            raise FeatureEngineeringError(f"{col.capitalize()} column not found in data.")
    
    result = data.copy()
    
    # Calculate true range
    high_low = result['high'] - result['low']
    high_close_prev = abs(result['high'] - result['close'].shift(1))
    low_close_prev = abs(result['low'] - result['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate ATR
    result['atr'] = true_range.rolling(window=period).mean()
    
    return result


def calculate_adx(
    data: pd.DataFrame,
    period: int = 14
) -> pd.DataFrame:
    """
    Calculate the Average Directional Index (ADX) indicator.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data with high, low, and close columns.
        period (int, optional): ADX period. Defaults to 14.
        
    Returns:
        pd.DataFrame: DataFrame with added ADX, +DI, and -DI indicators.
        
    Raises:
        FeatureEngineeringError: If required columns don't exist.
    """
    for col in ['high', 'low', 'close']:
        if col not in data.columns:
            raise FeatureEngineeringError(f"{col.capitalize()} column not found in data.")
    
    result = data.copy()
    
    # Calculate ATR first
    atr_df = calculate_atr(data, period)
    result['atr'] = atr_df['atr']
    
    # Calculate +DM and -DM
    high_diff = result['high'].diff()
    low_diff = result['low'].diff() * -1  # Make positive when low decreases
    
    plus_dm = (high_diff > low_diff) & (high_diff > 0)
    minus_dm = (low_diff > high_diff) & (low_diff > 0)
    
    result['plus_dm'] = np.where(plus_dm, high_diff, 0)
    result['minus_dm'] = np.where(minus_dm, low_diff, 0)
    
    # Calculate smoothed +DM and -DM
    result['plus_dm_smoothed'] = result['plus_dm'].rolling(window=period).mean()
    result['minus_dm_smoothed'] = result['minus_dm'].rolling(window=period).mean()
    
    # Calculate +DI and -DI
    result['plus_di'] = 100 * (result['plus_dm_smoothed'] / result['atr'])
    result['minus_di'] = 100 * (result['minus_dm_smoothed'] / result['atr'])
    
    # Calculate DX
    result['dx'] = 100 * (abs(result['plus_di'] - result['minus_di']) / (result['plus_di'] + result['minus_di']))
    
    # Calculate ADX
    result['adx'] = result['dx'].rolling(window=period).mean()
    
    # Drop intermediate columns
    result = result.drop(['plus_dm', 'minus_dm', 'plus_dm_smoothed', 'minus_dm_smoothed', 'dx'], axis=1)
    
    return result


def calculate_obv(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the On-Balance Volume (OBV) indicator.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data with close and volume columns.
        
    Returns:
        pd.DataFrame: DataFrame with added OBV indicator.
        
    Raises:
        FeatureEngineeringError: If required columns don't exist.
    """
    for col in ['close', 'volume']:
        if col not in data.columns:
            raise FeatureEngineeringError(f"{col.capitalize()} column not found in data.")
    
    result = data.copy()
    
    # Calculate price changes
    price_change = result['close'].diff()
    
    # Create OBV series
    obv = pd.Series(index=result.index)
    obv.iloc[0] = 0  # Initialize first value
    
    # Calculate OBV based on price change direction
    for i in range(1, len(result)):
        if price_change.iloc[i] > 0:  # Price up, add volume
            obv.iloc[i] = obv.iloc[i-1] + result['volume'].iloc[i]
        elif price_change.iloc[i] < 0:  # Price down, subtract volume
            obv.iloc[i] = obv.iloc[i-1] - result['volume'].iloc[i]
        else:  # Price unchanged, OBV unchanged
            obv.iloc[i] = obv.iloc[i-1]
    
    result['obv'] = obv
    
    return result


def calculate_ichimoku_cloud(
    data: pd.DataFrame,
    conversion_period: int = 9,
    base_period: int = 26,
    leading_span_b_period: int = 52,
    displacement: int = 26
) -> pd.DataFrame:
    """
    Calculate the Ichimoku Cloud indicator.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data with high, low, and close columns.
        conversion_period (int, optional): Tenkan-sen (Conversion Line) period. Defaults to 9.
        base_period (int, optional): Kijun-sen (Base Line) period. Defaults to 26.
        leading_span_b_period (int, optional): Senkou Span B period. Defaults to 52.
        displacement (int, optional): Displacement period. Defaults to 26.
        
    Returns:
        pd.DataFrame: DataFrame with added Ichimoku Cloud components.
        
    Raises:
        FeatureEngineeringError: If required columns don't exist.
    """
    for col in ['high', 'low', 'close']:
        if col not in data.columns:
            raise FeatureEngineeringError(f"{col.capitalize()} column not found in data.")
    
    result = data.copy()
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past 9 periods
    high_9 = result['high'].rolling(window=conversion_period).max()
    low_9 = result['low'].rolling(window=conversion_period).min()
    result['tenkan_sen'] = (high_9 + low_9) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past 26 periods
    high_26 = result['high'].rolling(window=base_period).max()
    low_26 = result['low'].rolling(window=base_period).min()
    result['kijun_sen'] = (high_26 + low_26) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Conversion Line + Base Line) / 2, shifted forward 26 periods
    result['senkou_span_a'] = ((result['tenkan_sen'] + result['kijun_sen']) / 2).shift(displacement)
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past 52 periods, shifted forward 26 periods
    high_52 = result['high'].rolling(window=leading_span_b_period).max()
    low_52 = result['low'].rolling(window=leading_span_b_period).min()
    result['senkou_span_b'] = ((high_52 + low_52) / 2).shift(displacement)
    
    # Calculate Chikou Span (Lagging Span): Close price, shifted backwards 26 periods
    result['chikou_span'] = result['close'].shift(-displacement)
    
    return result


def calculate_keltner_channels(
    data: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Calculate Keltner Channels indicator.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data with high, low, and close columns.
        ema_period (int, optional): EMA period for middle line. Defaults to 20.
        atr_period (int, optional): ATR period. Defaults to 10.
        multiplier (float, optional): Multiplier for ATR to set channel width. Defaults to 2.0.
        
    Returns:
        pd.DataFrame: DataFrame with added Keltner Channels components.
        
    Raises:
        FeatureEngineeringError: If required columns don't exist.
    """
    for col in ['high', 'low', 'close']:
        if col not in data.columns:
            raise FeatureEngineeringError(f"{col.capitalize()} column not found in data.")
    
    result = data.copy()
    
    # Calculate ATR
    atr_df = calculate_atr(data, atr_period)
    result['atr'] = atr_df['atr']
    
    # Calculate middle line (EMA of close)
    result['keltner_middle'] = result['close'].ewm(span=ema_period, adjust=False).mean()
    
    # Calculate upper and lower channels
    result['keltner_upper'] = result['keltner_middle'] + (result['atr'] * multiplier)
    result['keltner_lower'] = result['keltner_middle'] - (result['atr'] * multiplier)
    
    return result


def calculate_volatility(
    data: pd.DataFrame,
    price_column: str = 'close',
    periods: List[int] = [5, 10, 20, 30, 60],
    use_log_returns: bool = True
) -> pd.DataFrame:
    """
    Calculate volatility indicators for different periods.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data.
        price_column (str, optional): Column name containing price data. Defaults to 'close'.
        periods (List[int], optional): List of periods to calculate volatility for. Defaults to [5, 10, 20, 30, 60].
        use_log_returns (bool, optional): Whether to use log returns instead of percentage returns. Defaults to True.
        
    Returns:
        pd.DataFrame: DataFrame with added volatility indicators.
        
    Raises:
        FeatureEngineeringError: If price column doesn't exist.
    """
    if price_column not in data.columns:
        raise FeatureEngineeringError(f"Price column {price_column} not found in data.")
        
    result = data.copy()
    
    # Calculate returns
    if use_log_returns:
        returns = np.log(result[price_column] / result[price_column].shift(1))
    else:
        returns = result[price_column].pct_change()
    
    # Calculate rolling volatility for different periods
    for period in periods:
        result[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
    
    return result


def calculate_trend_indicators(
    data: pd.DataFrame,
    price_column: str = 'close',
    ma_periods: List[int] = [20, 50, 100, 200],
    ema_periods: List[int] = [12, 26, 50, 100]
) -> pd.DataFrame:
    """
    Calculate trend indicators including moving averages and crossovers.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data.
        price_column (str, optional): Column name containing price data. Defaults to 'close'.
        ma_periods (List[int], optional): List of periods for simple moving averages. Defaults to [20, 50, 100, 200].
        ema_periods (List[int], optional): List of periods for exponential moving averages. Defaults to [12, 26, 50, 100].
        
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


def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a comprehensive set of technical indicators for a DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame containing OHLCV data.
        
    Returns:
        pd.DataFrame: DataFrame with technical indicators added.
        
    Raises:
        FeatureEngineeringError: If required columns don't exist.
    """
    # Validate required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in data.columns:
            raise FeatureEngineeringError(f"Required column {col} not found in data.")
    
    result = data.copy()
    
    # Create TechnicalIndicators instance
    ti = TechnicalIndicators()
    
    # Calculate MACD
    result = ti.calculate_macd(result)
    
    # Calculate RSI
    result = ti.calculate_rsi(result)
    
    # Calculate Bollinger Bands
    result = ti.calculate_bollinger_bands(result)
    
    # Calculate Stochastic Oscillator
    result = ti.calculate_stochastic_oscillator(result)
    
    # Calculate ATR
    result = calculate_atr(result)
    
    # Calculate ADX
    result = ti.calculate_average_directional_index(result)
    
    # Calculate OBV
    result = calculate_obv(result)
    
    # Calculate volatility indicators
    result = calculate_volatility(result)
    
    # Calculate trend indicators
    result = calculate_trend_indicators(result)
    
    return result


def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame containing OHLCV data.
        
    Returns:
        pd.DataFrame: DataFrame with all technical indicators added.
        
    Raises:
        FeatureEngineeringError: If required columns don't exist.
    """
    # Validate required columns
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        missing_cols_str = ', '.join(missing_columns)
        raise FeatureEngineeringError(f"DataFrame is missing required columns: {missing_cols_str}")
    
    result = data.copy()
    
    # Add MACD
    macd_df = calculate_macd(result)
    result['macd_line'] = macd_df['macd_line']
    result['macd_signal'] = macd_df['macd_signal']
    result['macd_histogram'] = macd_df['macd_histogram']
    
    # Add RSI
    rsi_df = calculate_rsi(result)
    result['rsi'] = rsi_df['rsi']
    
    # Add Bollinger Bands
    bb_df = calculate_bollinger_bands(result)
    result['bb_upper'] = bb_df['bb_upper']
    result['bb_middle'] = bb_df['bb_middle']
    result['bb_lower'] = bb_df['bb_lower']
    result['bb_pct_b'] = bb_df['bb_pct_b']
    result['bb_bandwidth'] = bb_df['bb_bandwidth']
    
    # Add Stochastic Oscillator
    stoch_df = calculate_stochastic_oscillator(result)
    result['stoch_k'] = stoch_df['stoch_k']
    result['stoch_d'] = stoch_df['stoch_d']
    
    # Add ATR
    atr_df = calculate_atr(result)
    result['atr'] = atr_df['atr']
    
    # Add ADX
    adx_df = calculate_adx(result)
    result['adx'] = adx_df['adx']
    result['plus_di'] = adx_df['plus_di']
    result['minus_di'] = adx_df['minus_di']
    
    # Add OBV if volume column exists
    if 'volume' in result.columns:
        obv_df = calculate_obv(result)
        result['obv'] = obv_df['obv']
    
    # Add Ichimoku Cloud
    ichimoku_df = calculate_ichimoku_cloud(result)
    result['tenkan_sen'] = ichimoku_df['tenkan_sen']
    result['kijun_sen'] = ichimoku_df['kijun_sen']
    result['senkou_span_a'] = ichimoku_df['senkou_span_a']
    result['senkou_span_b'] = ichimoku_df['senkou_span_b']
    result['chikou_span'] = ichimoku_df['chikou_span']
    
    # Add Keltner Channels
    keltner_df = calculate_keltner_channels(result)
    result['keltner_upper'] = keltner_df['keltner_upper']
    result['keltner_middle'] = keltner_df['keltner_middle']
    result['keltner_lower'] = keltner_df['keltner_lower']
    
    return result
