"""
Analytics module for STOCKER Pro.

This module provides analytics tools for financial data analysis, including
anomaly detection, pattern recognition, and market analysis agents.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

from src.core.logging import logger
from src.features.indicators import add_technical_indicators


class AnomalyDetection:
    """
    Anomaly detection for financial time series data.
    
    This class provides methods to detect anomalies in financial data using
    various statistical and machine learning approaches.
    """
    
    def __init__(self, config=None):
        """Initialize the anomaly detection module."""
        self.config = config or {}
        self.threshold = self.config.get('threshold', 3.0)  # Default: 3 standard deviations
        self.window_size = self.config.get('window_size', 20)
        self.methods = {
            'zscore': self.detect_zscore_anomalies,
            'iqr': self.detect_iqr_anomalies,
            'isolation_forest': self.detect_isolation_forest_anomalies,
            'local_outlier_factor': self.detect_lof_anomalies,
            'moving_average': self.detect_moving_average_anomalies
        }
    
    def detect_anomalies(self, data, method='zscore', **kwargs):
        """
        Detect anomalies in the provided data using the specified method.
        
        Args:
            data: DataFrame or Series containing the data
            method: Detection method to use ('zscore', 'iqr', 'isolation_forest', etc.)
            **kwargs: Additional parameters for the detection method
            
        Returns:
            Series or DataFrame with anomaly indicators (1 for anomaly, 0 for normal)
        """
        if method not in self.methods:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        
        return self.methods[method](data, **kwargs)
    
    def detect_zscore_anomalies(self, data, threshold=None, window=None):
        """
        Detect anomalies using Z-score method.
        
        Args:
            data: DataFrame or Series containing the data
            threshold: Z-score threshold for anomaly (default: from config)
            window: Window size for moving stats (default: from config)
            
        Returns:
            Series with anomaly indicators (1 for anomaly, 0 for normal)
        """
        threshold = threshold or self.threshold
        window = window or self.window_size
        
        if isinstance(data, pd.DataFrame):
            # Handle DataFrame by applying to each numeric column
            result = pd.DataFrame(index=data.index)
            for col in data.select_dtypes(include=np.number).columns:
                result[col] = self._zscore_anomalies_series(data[col], threshold, window)
            return result
        else:
            # Handle Series
            return self._zscore_anomalies_series(data, threshold, window)
    
    def _zscore_anomalies_series(self, series, threshold, window):
        """Helper method to detect Z-score anomalies in a Series."""
        if window:
            # Use rolling window
            mean = series.rolling(window=window).mean()
            std = series.rolling(window=window).std()
            z_scores = (series - mean) / std
        else:
            # Use entire series
            mean = series.mean()
            std = series.std()
            z_scores = (series - mean) / std
        
        return (abs(z_scores) > threshold).astype(int)
    
    def detect_iqr_anomalies(self, data, factor=1.5, window=None):
        """
        Detect anomalies using the Interquartile Range (IQR) method.
        
        Args:
            data: DataFrame or Series containing the data
            factor: IQR multiplier for determining outliers (default: 1.5)
            window: Window size for moving stats (default: from config)
            
        Returns:
            Series with anomaly indicators (1 for anomaly, 0 for normal)
        """
        window = window or self.window_size
        
        if isinstance(data, pd.DataFrame):
            # Handle DataFrame by applying to each numeric column
            result = pd.DataFrame(index=data.index)
            for col in data.select_dtypes(include=np.number).columns:
                result[col] = self._iqr_anomalies_series(data[col], factor, window)
            return result
        else:
            # Handle Series
            return self._iqr_anomalies_series(data, factor, window)
    
    def _iqr_anomalies_series(self, series, factor, window):
        """Helper method to detect IQR anomalies in a Series."""
        if window:
            # Use rolling window - more complex with quantiles
            result = pd.Series(index=series.index, data=0)
            
            for i in range(window, len(series) + 1):
                window_data = series.iloc[i-window:i]
                q1 = window_data.quantile(0.25)
                q3 = window_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - factor * iqr
                upper_bound = q3 + factor * iqr
                
                if i < len(series):
                    value = series.iloc[i]
                    result.iloc[i] = 1 if (value < lower_bound or value > upper_bound) else 0
            
            return result
        else:
            # Use entire series
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            
            return ((series < lower_bound) | (series > upper_bound)).astype(int)
    
    def detect_isolation_forest_anomalies(self, data, contamination=0.05, **kwargs):
        """
        Detect anomalies using Isolation Forest algorithm.
        
        Args:
            data: DataFrame or Series containing the data
            contamination: Expected proportion of anomalies (default: 0.05)
            **kwargs: Additional parameters for IsolationForest
            
        Returns:
            Series with anomaly indicators (1 for anomaly, 0 for normal)
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError("scikit-learn is required for Isolation Forest anomaly detection")
        
        # Prepare the data
        if isinstance(data, pd.Series):
            X = data.values.reshape(-1, 1)
        else:
            X = data.select_dtypes(include=np.number).values
        
        # Train the model
        model = IsolationForest(contamination=contamination, **kwargs)
        predictions = model.fit_predict(X)
        
        # Convert predictions (-1 for anomalies, 1 for normal points)
        anomalies = pd.Series(index=data.index, data=(predictions == -1).astype(int))
        
        return anomalies
    
    def detect_lof_anomalies(self, data, n_neighbors=20, contamination=0.05, **kwargs):
        """
        Detect anomalies using Local Outlier Factor algorithm.
        
        Args:
            data: DataFrame or Series containing the data
            n_neighbors: Number of neighbors to consider
            contamination: Expected proportion of anomalies
            **kwargs: Additional parameters for LocalOutlierFactor
            
        Returns:
            Series with anomaly indicators (1 for anomaly, 0 for normal)
        """
        try:
            from sklearn.neighbors import LocalOutlierFactor
        except ImportError:
            raise ImportError("scikit-learn is required for LOF anomaly detection")
        
        # Prepare the data
        if isinstance(data, pd.Series):
            X = data.values.reshape(-1, 1)
        else:
            X = data.select_dtypes(include=np.number).values
        
        # Train the model
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, **kwargs)
        predictions = model.fit_predict(X)
        
        # Convert predictions (-1 for anomalies, 1 for normal points)
        anomalies = pd.Series(index=data.index, data=(predictions == -1).astype(int))
        
        return anomalies
    
    def detect_moving_average_anomalies(self, data, window=None, threshold=None):
        """
        Detect anomalies by comparing to moving average.
        
        Args:
            data: DataFrame or Series containing the data
            window: Window size for moving average
            threshold: Threshold multiplier for standard deviation
            
        Returns:
            Series with anomaly indicators (1 for anomaly, 0 for normal)
        """
        window = window or self.window_size
        threshold = threshold or self.threshold
        
        if isinstance(data, pd.DataFrame):
            # Handle DataFrame by applying to each numeric column
            result = pd.DataFrame(index=data.index)
            for col in data.select_dtypes(include=np.number).columns:
                result[col] = self._ma_anomalies_series(data[col], window, threshold)
            return result
        else:
            # Handle Series
            return self._ma_anomalies_series(data, window, threshold)
    
    def _ma_anomalies_series(self, series, window, threshold):
        """Helper method to detect moving average anomalies in a Series."""
        ma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        
        # Calculate upper and lower bounds
        upper_bound = ma + threshold * std
        lower_bound = ma - threshold * std
        
        # Identify anomalies
        anomalies = ((series > upper_bound) | (series < lower_bound)).astype(int)
        
        # Fill NaN values from the rolling windows
        anomalies.iloc[:window-1] = 0
        
        return anomalies


class MarketAnalyticsAgent:
    """
    Analytics agent for market analysis.
    
    Provides methods for analyzing market trends, patterns, and conditions
    to generate insights and metrics.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the market analytics agent.
        
        Args:
            data: Optional DataFrame with market data
        """
        self.data = data
    
    def analyze_market_regime(self, data: Optional[pd.DataFrame] = None, window: int = 63) -> Dict[str, Any]:
        """
        Analyze market regime (bullish, bearish, or neutral).
        
        Args:
            data: DataFrame with market data (uses self.data if None)
            window: Window size for calculations
            
        Returns:
            Dictionary with market regime analysis
        """
        df = data if data is not None else self.data
        if df is None or df.empty:
            logger.error("No data provided for market regime analysis")
            return {}
        
        try:
            # Ensure we have all required indicators
            if 'sma_20' not in df.columns or 'sma_50' not in df.columns or 'sma_200' not in df.columns:
                df = add_technical_indicators(df)
            
            # Calculate returns
            df['daily_return'] = df['close'].pct_change()
            df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
            
            # Calculate volatility
            df['volatility'] = df['daily_return'].rolling(window=window).std() * np.sqrt(252)
            
            # Determine market trend
            trend_indicators = [
                df['close'] > df['sma_200'],  # Price above 200-day SMA
                df['sma_50'] > df['sma_200'],  # 50-day SMA above 200-day SMA (golden cross)
                df['sma_20'] > df['sma_50'],   # 20-day SMA above 50-day SMA
                df['close'].rolling(window=window).mean() > df['close'].rolling(window=window*2).mean(),  # Shorter MA > Longer MA
                df['cumulative_return'].rolling(window=window).mean() > 0  # Positive returns
            ]
            
            bullish_score = sum(trend.iloc[-1] for trend in trend_indicators) / len(trend_indicators)
            
            # Classify regime
            if bullish_score >= 0.7:
                regime = "Bullish"
            elif bullish_score <= 0.3:
                regime = "Bearish"
            else:
                regime = "Neutral"
            
            # Determine volatility regime
            current_vol = df['volatility'].iloc[-1]
            historical_vol = df['volatility'].rolling(window=window*2).mean().iloc[-1]
            
            if current_vol > historical_vol * 1.5:
                volatility_regime = "High"
            elif current_vol < historical_vol * 0.5:
                volatility_regime = "Low"
            else:
                volatility_regime = "Normal"
            
            # Calculate trend strength
            trend_strength = abs(bullish_score - 0.5) * 2  # Scale to 0-1
            
            return {
                "regime": regime,
                "bullish_score": bullish_score,
                "volatility_regime": volatility_regime,
                "current_volatility": current_vol,
                "historical_volatility": historical_vol,
                "trend_strength": trend_strength,
                "date": df.index[-1]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return {}
    
    def detect_market_events(self, data: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """
        Detect significant market events.
        
        Args:
            data: DataFrame with market data (uses self.data if None)
            
        Returns:
            List of detected market events
        """
        df = data if data is not None else self.data
        if df is None or df.empty:
            logger.error("No data provided for market event detection")
            return []
        
        try:
            # Ensure we have all required indicators
            if 'sma_20' not in df.columns or 'sma_50' not in df.columns or 'sma_200' not in df.columns:
                df = add_technical_indicators(df)
            
            events = []
            
            # Detect golden cross (50-day SMA crosses above 200-day SMA)
            golden_cross = ((df['sma_50'] > df['sma_200']) & 
                           (df['sma_50'].shift(1) <= df['sma_200'].shift(1)))
            
            # Detect death cross (50-day SMA crosses below 200-day SMA)
            death_cross = ((df['sma_50'] < df['sma_200']) & 
                          (df['sma_50'].shift(1) >= df['sma_200'].shift(1)))
            
            # Find large daily movements (>= 3%)
            large_moves = df['close'].pct_change().abs() >= 0.03
            
            # Compile events
            for date in df[golden_cross].index[golden_cross]:
                events.append({
                    "date": date,
                    "type": "Golden Cross",
                    "description": "50-day SMA crossed above 200-day SMA",
                    "significance": "Bullish"
                })
                
            for date in df[death_cross].index[death_cross]:
                events.append({
                    "date": date,
                    "type": "Death Cross",
                    "description": "50-day SMA crossed below 200-day SMA",
                    "significance": "Bearish"
                })
                
            for date in df[large_moves].index[large_moves]:
                pct_change = df.loc[date, 'close'] / df.loc[date, 'close'].shift(1) - 1
                events.append({
                    "date": date,
                    "type": "Large Daily Move",
                    "description": f"{pct_change:.2%} daily change",
                    "significance": "High Volatility"
                })
            
            return events
            
        except Exception as e:
            logger.error(f"Error detecting market events: {e}")
            return []
    
    def analyze_trend_strength(self, data: Optional[pd.DataFrame] = None, window: int = 63) -> Dict[str, float]:
        """
        Analyze trend strength using various metrics.
        
        Args:
            data: DataFrame with market data (uses self.data if None)
            window: Window size for calculations
            
        Returns:
            Dictionary with trend strength metrics
        """
        df = data if data is not None else self.data
        if df is None or df.empty:
            logger.error("No data provided for trend strength analysis")
            return {}
        
        try:
            # Ensure we have all required indicators
            if 'adx' not in df.columns or 'rsi_14' not in df.columns:
                df = add_technical_indicators(df, include_all=True)
            
            # Calculate linear regression slope
            y = df['close'].values
            x = np.arange(len(y))
            
            # Simple linear regression
            slope = np.polyfit(x[-window:], y[-window:], 1)[0]
            normalized_slope = slope / df['close'].iloc[-window] * 100
            
            # Calculate R-squared
            p = np.poly1d(np.polyfit(x[-window:], y[-window:], 1))
            trend_r2 = 1 - np.sum((y[-window:] - p(x[-window:]))**2) / np.sum((y[-window:] - np.mean(y[-window:]))**2)
            
            # Get ADX (average directional index)
            adx_value = df['adx'].iloc[-1] if 'adx' in df.columns else 0
            
            # Calculate percentage of days trending in same direction
            returns = df['close'].pct_change()
            positive_days = (returns > 0).rolling(window=window).mean().iloc[-1]
            directional_consistency = max(positive_days, 1-positive_days)
            
            # Combine metrics
            return {
                "slope": normalized_slope,
                "r_squared": trend_r2,
                "adx": adx_value,
                "directional_consistency": directional_consistency,
                "trend_strength": (normalized_slope / 10 + trend_r2 + adx_value / 100 + directional_consistency) / 4
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend strength: {e}")
            return {}


class CorrelationAnalyzer:
    """
    Analyzer for market correlations.
    
    Provides methods for analyzing correlations between different securities,
    sectors, and market factors.
    """
    
    def __init__(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize the correlation analyzer.
        
        Args:
            data_dict: Optional dictionary mapping symbols to DataFrames
        """
        self.data_dict = data_dict or {}
    
    def calculate_correlation_matrix(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None, 
                                   window: int = 63) -> pd.DataFrame:
        """
        Calculate correlation matrix between securities.
        
        Args:
            data_dict: Dictionary mapping symbols to DataFrames (uses self.data_dict if None)
            window: Window size for rolling correlation
            
        Returns:
            DataFrame with correlation matrix
        """
        data = data_dict if data_dict is not None else self.data_dict
        if not data:
            logger.error("No data provided for correlation analysis")
            return pd.DataFrame()
        
        try:
            # Extract returns
            returns_dict = {}
            for symbol, df in data.items():
                if isinstance(df, pd.DataFrame) and not df.empty and 'close' in df.columns:
                    returns_dict[symbol] = df['close'].pct_change().dropna()
            
            # Create returns DataFrame
            returns = pd.DataFrame(returns_dict)
            
            # Calculate correlation matrix
            correlation_matrix = returns.iloc[-window:].corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def find_pairs(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None, 
                 threshold: float = 0.8, window: int = 63) -> List[Dict[str, Any]]:
        """
        Find highly correlated pairs for potential pair trading.
        
        Args:
            data_dict: Dictionary mapping symbols to DataFrames (uses self.data_dict if None)
            threshold: Correlation threshold
            window: Window size for correlation calculation
            
        Returns:
            List of highly correlated pairs
        """
        try:
            # Calculate correlation matrix
            corr_matrix = self.calculate_correlation_matrix(data_dict, window)
            
            # Find pairs above threshold
            pairs = []
            
            # Get upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find high correlation pairs
            for col in upper.columns:
                for idx in upper.index:
                    value = upper.loc[idx, col]
                    if value > threshold:
                        pairs.append({
                            "symbol1": idx,
                            "symbol2": col,
                            "correlation": value,
                            "window": window
                        })
            
            return sorted(pairs, key=lambda x: x["correlation"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding pairs: {e}")
            return []
    
    def calculate_beta(self, symbol: str, market_symbol: str, data_dict: Optional[Dict[str, pd.DataFrame]] = None,
                      window: int = 252) -> float:
        """
        Calculate beta of a symbol against a market index.
        
        Args:
            symbol: Symbol to calculate beta for
            market_symbol: Market index symbol
            data_dict: Dictionary mapping symbols to DataFrames (uses self.data_dict if None)
            window: Window size for calculation
            
        Returns:
            Beta value
        """
        data = data_dict if data_dict is not None else self.data_dict
        if not data or symbol not in data or market_symbol not in data:
            logger.error(f"Missing data for beta calculation: {symbol} or {market_symbol}")
            return 0.0
        
        try:
            # Calculate returns
            symbol_returns = data[symbol]['close'].pct_change().dropna()
            market_returns = data[market_symbol]['close'].pct_change().dropna()
            
            # Align data
            common_idx = symbol_returns.index.intersection(market_returns.index)
            if len(common_idx) < 30:
                logger.warning(f"Insufficient data for beta calculation: {len(common_idx)} points")
                return 0.0
                
            symbol_returns = symbol_returns.loc[common_idx]
            market_returns = market_returns.loc[common_idx]
            
            # Use the last 'window' periods
            if len(symbol_returns) > window:
                symbol_returns = symbol_returns.iloc[-window:]
                market_returns = market_returns.iloc[-window:]
            
            # Calculate covariance and variance
            covariance = symbol_returns.cov(market_returns)
            variance = market_returns.var()
            
            if variance == 0:
                return 0.0
            
            beta = covariance / variance
            
            return beta
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 0.0 