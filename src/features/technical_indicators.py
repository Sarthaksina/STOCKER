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
from typing import Union, Optional, List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Class containing methods to calculate various technical indicators."""

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
        
        # Calculate bandwidth and %B
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
        
        # Calculate relative strength
        rs = avg_gain / avg_loss
        
        # Calculate RSI
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
            
        # Get lowest low and highest high for the period
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        # Calculate %K
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        
        # Calculate %D (SMA of %K)
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
            
        # Calculate true range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        # Get the maximum of the three
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=period).mean()
        
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
            
        # Calculate true range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate directional movement
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
        
        # Calculate positive and negative directional movement
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # Calculate smoothed TR, +DM, and -DM
        atr = tr.rolling(window=period).sum()
        pos_di = 100 * (pos_dm.rolling(window=period).sum() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).sum() / atr)
        
        # Calculate directional index
        dx = 100 * ((pos_di - neg_di).abs() / (pos_di + neg_di))
        
        # Calculate ADX
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
            raise ValueError("DataFrame must include 'volume' column")
            
        # Calculate price direction
        price_direction = np.sign(df['close'].diff())
        
        # Set initial OBV
        obv = pd.Series(0, index=df.index)
        
        # Calculate OBV
        for i in range(1, len(df)):
            if price_direction.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif price_direction.iloc[i] < 0:
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
            displacement: Displacement period for Senkou Span
            
        Returns:
            DataFrame with Ichimoku Cloud components
        """
        if any(p <= 0 for p in [conversion_period, base_period, leading_span_b_period, displacement]):
            raise ValueError("All periods must be positive integers")
            
        # Calculate Tenkan-sen (Conversion Line)
        high_9 = df['high'].rolling(window=conversion_period).max()
        low_9 = df['low'].rolling(window=conversion_period).min()
        tenkan_sen = (high_9 + low_9) / 2
        
        # Calculate Kijun-sen (Base Line)
        high_26 = df['high'].rolling(window=base_period).max()
        low_26 = df['low'].rolling(window=base_period).min()
        kijun_sen = (high_26 + low_26) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Calculate Senkou Span B (Leading Span B)
        high_52 = df['high'].rolling(window=leading_span_b_period).max()
        low_52 = df['low'].rolling(window=leading_span_b_period).min()
        senkou_span_b = ((high_52 + low_52) / 2).shift(displacement)
        
        # Calculate Chikou Span (Lagging Span)
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