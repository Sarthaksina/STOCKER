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


def analyze_seasonality(
    data: pd.DataFrame, 
    price_column: str = 'close',
    period: str = 'monthly',
    decompose_method: str = 'additive'
) -> Dict[str, Any]:
    """
    Analyze seasonality patterns in time series data.
    
    Args:
        data: DataFrame with datetime index and price data
        price_column: Column name for price data (default: 'close')
        period: Period for seasonality analysis ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
        decompose_method: Decomposition method ('additive' or 'multiplicative')
        
    Returns:
        Dictionary with seasonality analysis results
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        import matplotlib.pyplot as plt
        from pandas.tseries.frequencies import to_offset
        
        # Ensure we have a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, attempting to convert")
            try:
                data = data.copy()
                data.index = pd.to_datetime(data.index)
            except:
                logger.error("Failed to convert index to DatetimeIndex")
                return {"error": "Index must be a DatetimeIndex"}
        
        # Determine frequency based on period parameter
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M',
            'quarterly': 'Q',
            'yearly': 'Y'
        }
        
        freq = freq_map.get(period.lower(), 'M')  # Default to monthly
        
        # Resample data to ensure regular frequency
        resampled = data[price_column].resample(freq).mean()
        
        # Fill any missing values
        resampled = resampled.interpolate()
        
        # Perform seasonal decomposition
        result = seasonal_decompose(
            resampled, 
            model=decompose_method,
            extrapolate_trend='freq'
        )
        
        # Calculate seasonal patterns
        if period.lower() in ['monthly', 'quarterly']:
            # Get monthly or quarterly patterns
            seasonal_patterns = result.seasonal.groupby(result.seasonal.index.month).mean()
            period_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            pattern_index = seasonal_patterns.index
            pattern_labels = [period_labels[i-1] for i in pattern_index]
        elif period.lower() == 'weekly':
            # Get day-of-week patterns
            seasonal_patterns = result.seasonal.groupby(result.seasonal.index.dayofweek).mean()
            pattern_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        elif period.lower() == 'yearly':
            # Get yearly patterns (by month)
            seasonal_patterns = result.seasonal.groupby(result.seasonal.index.month).mean()
            period_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            pattern_index = seasonal_patterns.index
            pattern_labels = [period_labels[i-1] for i in pattern_index]
        else:
            # Daily patterns (by hour if available, otherwise day of week)
            if hasattr(result.seasonal.index, 'hour'):
                seasonal_patterns = result.seasonal.groupby(result.seasonal.index.hour).mean()
                pattern_labels = [str(h) for h in range(24)]
            else:
                seasonal_patterns = result.seasonal.groupby(result.seasonal.index.dayofweek).mean()
                pattern_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        
        # Calculate strength of seasonality
        # (variance of the seasonality component / variance of the trend+seasonality)
        detrended = result.observed - result.trend
        seasonality_strength = np.var(result.seasonal) / np.var(detrended)
        
        # Prepare results
        analysis_result = {
            "decomposition": {
                "trend": result.trend.to_dict(),
                "seasonal": result.seasonal.to_dict(),
                "residual": result.resid.to_dict()
            },
            "seasonal_patterns": dict(zip(pattern_labels, seasonal_patterns.values)),
            "seasonality_strength": float(seasonality_strength),
            "period": period,
            "method": decompose_method
        }
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error in seasonality analysis: {e}")
        return {"error": str(e)}


def analyze_trend(
    data: pd.DataFrame,
    price_column: str = 'close',
    window_short: int = 20,
    window_medium: int = 50,
    window_long: int = 200
) -> Dict[str, Any]:
    """
    Analyze price trends using moving averages and trend indicators.
    
    Args:
        data: DataFrame with price data
        price_column: Column name for price data (default: 'close')
        window_short: Short-term moving average window (default: 20)
        window_medium: Medium-term moving average window (default: 50)
        window_long: Long-term moving average window (default: 200)
        
    Returns:
        Dictionary with trend analysis results
    """
    try:
        result = data.copy()
        
        # Calculate moving averages
        result[f'sma_{window_short}'] = result[price_column].rolling(window=window_short).mean()
        result[f'sma_{window_medium}'] = result[price_column].rolling(window=window_medium).mean()
        result[f'sma_{window_long}'] = result[price_column].rolling(window=window_long).mean()
        
        # Calculate exponential moving averages
        result[f'ema_{window_short}'] = result[price_column].ewm(span=window_short, adjust=False).mean()
        result[f'ema_{window_medium}'] = result[price_column].ewm(span=window_medium, adjust=False).mean()
        result[f'ema_{window_long}'] = result[price_column].ewm(span=window_long, adjust=False).mean()
        
        # Calculate price relative to moving averages
        result['price_to_sma_short'] = result[price_column] / result[f'sma_{window_short}'] - 1
        result['price_to_sma_medium'] = result[price_column] / result[f'sma_{window_medium}'] - 1
        result['price_to_sma_long'] = result[price_column] / result[f'sma_{window_long}'] - 1
        
        # Calculate moving average crossovers
        result['sma_short_medium_cross'] = result[f'sma_{window_short}'] - result[f'sma_{window_medium}']
        result['sma_medium_long_cross'] = result[f'sma_{window_medium}'] - result[f'sma_{window_long}']
        result['ema_short_medium_cross'] = result[f'ema_{window_short}'] - result[f'ema_{window_medium}']
        result['ema_medium_long_cross'] = result[f'ema_{window_medium}'] - result[f'ema_{window_long}']
        
        # Calculate trend direction and strength
        # Use the slope of the long-term moving average as a trend indicator
        result['trend_slope'] = result[f'sma_{window_long}'].diff(20) / 20  # Slope over 20 periods
        
        # Calculate ADX for trend strength if we have high/low data
        has_ohlc = all(col in result.columns for col in ['high', 'low', price_column])
        if has_ohlc:
            from src.features.indicators import calculate_average_directional_index
            result = calculate_average_directional_index(
                result,
                high_column='high',
                low_column='low',
                close_column=price_column
            )
        
        # Determine current trend
        latest = result.iloc[-1]
        
        # Trend determination rules
        is_uptrend = (
            latest[price_column] > latest[f'sma_{window_short}'] > 
            latest[f'sma_{window_medium}'] > latest[f'sma_{window_long}']
        )
        
        is_downtrend = (
            latest[price_column] < latest[f'sma_{window_short}'] < 
            latest[f'sma_{window_medium}'] < latest[f'sma_{window_long}']
        )
        
        # Check if price is above/below all moving averages
        price_above_all_mas = (
            latest[price_column] > latest[f'sma_{window_short}'] and
            latest[price_column] > latest[f'sma_{window_medium}'] and
            latest[price_column] > latest[f'sma_{window_long}']
        )
        
        price_below_all_mas = (
            latest[price_column] < latest[f'sma_{window_short}'] and
            latest[price_column] < latest[f'sma_{window_medium}'] and
            latest[price_column] < latest[f'sma_{window_long}']
        )
        
        # Check for golden cross (short MA crosses above long MA)
        golden_cross = (
            result['sma_short_medium_cross'].iloc[-1] > 0 and
            result['sma_short_medium_cross'].iloc[-2] <= 0
        )
        
        # Check for death cross (short MA crosses below long MA)
        death_cross = (
            result['sma_short_medium_cross'].iloc[-1] < 0 and
            result['sma_short_medium_cross'].iloc[-2] >= 0
        )
        
        # Determine trend strength
        trend_strength = abs(latest['trend_slope']) * window_long / latest[price_column]
        adx_strength = latest.get('adx', 0)  # Use ADX if available
        
        # Determine overall trend
        if is_uptrend or (price_above_all_mas and latest['trend_slope'] > 0):
            trend = 'uptrend'
        elif is_downtrend or (price_below_all_mas and latest['trend_slope'] < 0):
            trend = 'downtrend'
        elif latest['trend_slope'] > 0:
            trend = 'weak_uptrend'
        elif latest['trend_slope'] < 0:
            trend = 'weak_downtrend'
        else:
            trend = 'sideways'
        
        # Prepare result
        analysis_result = {
            'trend': trend,
            'trend_strength': float(trend_strength),
            'adx_strength': float(adx_strength) if has_ohlc else None,
            'price_to_sma_short': float(latest['price_to_sma_short']),
            'price_to_sma_medium': float(latest['price_to_sma_medium']),
            'price_to_sma_long': float(latest['price_to_sma_long']),
            'golden_cross': bool(golden_cross),
            'death_cross': bool(death_cross),
            'indicators': {
                'sma_short': float(latest[f'sma_{window_short}']),
                'sma_medium': float(latest[f'sma_{window_medium}']),
                'sma_long': float(latest[f'sma_{window_long}']),
                'ema_short': float(latest[f'ema_{window_short}']),
                'ema_medium': float(latest[f'ema_{window_medium}']),
                'ema_long': float(latest[f'ema_{window_long}'])
            }
        }
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {e}")
        return {"error": str(e)}


def detect_outliers(
    data: pd.DataFrame,
    price_column: str = 'close',
    method: str = 'zscore',
    threshold: float = 3.0,
    window: int = 20
) -> pd.DataFrame:
    """
    Detect outliers in financial time series data.
    
    Args:
        data: DataFrame with price data
        price_column: Column name for price data (default: 'close')
        method: Detection method ('zscore', 'iqr', 'isolation_forest', 'moving_average')
        threshold: Threshold for outlier detection (interpretation depends on method)
        window: Window size for rolling calculations
        
    Returns:
        DataFrame with outlier indicators (1 for outlier, 0 for normal)
    """
    try:
        # Create anomaly detection instance
        anomaly_detector = AnomalyDetection()
        
        # Select the appropriate method
        if method == 'zscore':
            result = anomaly_detector.detect_zscore_anomalies(
                data[price_column], threshold=threshold, window=window
            )
        elif method == 'iqr':
            result = anomaly_detector.detect_iqr_anomalies(
                data[price_column], factor=threshold, window=window
            )
        elif method == 'isolation_forest':
            result = anomaly_detector.detect_isolation_forest_anomalies(
                data[price_column], contamination=threshold/100
            )
        elif method == 'moving_average':
            result = anomaly_detector.detect_moving_average_anomalies(
                data[price_column], threshold=threshold, window=window
            )
        else:
            # Default to z-score
            result = anomaly_detector.detect_zscore_anomalies(
                data[price_column], threshold=threshold, window=window
            )
        
        # Create a DataFrame with the original data and outlier indicators
        output = data.copy()
        output['is_outlier'] = result
        
        # Add outlier values
        output['outlier_value'] = output.apply(
            lambda row: row[price_column] if row['is_outlier'] == 1 else np.nan, 
            axis=1
        )
        
        return output
        
    except Exception as e:
        logger.error(f"Error detecting outliers: {e}")
        # Return original data with empty outlier columns
        output = data.copy()
        output['is_outlier'] = 0
        output['outlier_value'] = np.nan
        return output


def detect_anomalies(
    data: pd.DataFrame,
    column: str = 'close',
    method: str = 'zscore',
    **kwargs
) -> pd.DataFrame:
    """
    Detect anomalies in financial time series data.
    
    This is an alias for detect_outliers for backward compatibility.
    
    Args:
        data: DataFrame with price data
        column: Column name for price data (default: 'close')
        method: Detection method ('zscore', 'iqr', 'isolation_forest', 'moving_average')
        **kwargs: Additional parameters for the detection method
        
    Returns:
        DataFrame with anomaly indicators (1 for anomaly, 0 for normal)
    """
    return detect_outliers(
        data=data,
        price_column=column,
        method=method,
        **kwargs
    )


def calculate_correlation_matrix(
    data_dict: Dict[str, pd.DataFrame] = None,
    price_column: str = 'close',
    window: int = 63,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Calculate correlation matrix between multiple assets.
    
    Args:
        data_dict: Dictionary mapping symbols to DataFrames with price data
        price_column: Column name for price data (default: 'close')
        window: Window size for rolling correlation (default: 63 days / 3 months)
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        DataFrame with correlation matrix
    """
    try:
        if data_dict is None or not data_dict:
            logger.error("No data provided for correlation calculation")
            return pd.DataFrame()
        
        # Extract price series for each symbol
        price_series = {}
        for symbol, df in data_dict.items():
            if price_column in df.columns:
                price_series[symbol] = df[price_column]
            else:
                logger.warning(f"Price column '{price_column}' not found for {symbol}")
        
        if not price_series:
            logger.error(f"No valid price data found with column '{price_column}'")
            return pd.DataFrame()
        
        # Create a DataFrame with all price series
        prices_df = pd.DataFrame(price_series)
        
        # Calculate returns
        returns_df = prices_df.pct_change().dropna()
        
        # Calculate correlation matrix
        if window and len(returns_df) > window:
            # Use rolling window correlation (last 'window' periods)
            returns_df = returns_df.iloc[-window:]
            
        correlation_matrix = returns_df.corr(method=method)
        
        return correlation_matrix
        
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        return pd.DataFrame()


def analyze_returns(prices: Union[List[float], pd.Series, pd.DataFrame], price_column: str = 'close') -> Dict[str, float]:
    """
    Analyze returns of a price series.
    
    Args:
        prices: List, Series, or DataFrame with price data
        price_column: Column name if prices is a DataFrame
        
    Returns:
        Dictionary with return metrics
    """
    try:
        # Convert to pandas Series if needed
        if isinstance(prices, list):
            prices = pd.Series(prices)
        elif isinstance(prices, pd.DataFrame):
            if price_column not in prices.columns:
                logger.error(f"Price column '{price_column}' not found in DataFrame")
                return {}
            prices = prices[price_column]
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Calculate metrics
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1 if len(prices) > 1 else 0
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        daily_mean = returns.mean()
        daily_std = returns.std()
        
        # Calculate drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe and Sortino ratios
        sharpe_ratio = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0
        sortino_ratio = (daily_mean / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Calculate positive and negative days
        positive_days = (returns > 0).sum()
        negative_days = (returns < 0).sum()
        win_rate = positive_days / len(returns) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'daily_mean': daily_mean,
            'daily_std': daily_std,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'positive_days': positive_days,
            'negative_days': negative_days,
            'win_rate': win_rate,
            'num_observations': len(returns)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing returns: {e}")
        return {}


def analyze_volatility(data: Union[List[float], pd.Series, pd.DataFrame], price_column: str = 'close', window: int = 21) -> Dict[str, Any]:
    """
    Analyze volatility of a price series.
    
    Args:
        data: List, Series, or DataFrame with price data
        price_column: Column name if data is a DataFrame
        window: Window size for rolling volatility
        
    Returns:
        Dictionary with volatility metrics
    """
    try:
        # Convert to pandas Series if needed
        if isinstance(data, list):
            prices = pd.Series(data)
        elif isinstance(data, pd.DataFrame):
            if price_column not in data.columns:
                logger.error(f"Price column '{price_column}' not found in DataFrame")
                return {}
            prices = data[price_column]
        else:
            prices = data
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Calculate volatility metrics
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        current_volatility = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0
        
        # Calculate volatility percentile
        vol_percentile = np.percentile(rolling_vol.dropna(), [25, 50, 75, 90])
        
        # Calculate volatility of volatility
        vol_of_vol = rolling_vol.pct_change().dropna().std() * np.sqrt(252)
        
        return {
            'daily_volatility': daily_volatility,
            'annualized_volatility': annualized_volatility,
            'current_volatility': current_volatility,
            'volatility_25th': vol_percentile[0],
            'volatility_50th': vol_percentile[1],
            'volatility_75th': vol_percentile[2],
            'volatility_90th': vol_percentile[3],
            'volatility_of_volatility': vol_of_vol,
            'rolling_volatility': rolling_vol.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing volatility: {e}")
        return {}


def analyze_seasonality(data: pd.DataFrame, price_column: str = 'close', period: str = 'monthly', decompose_method: str = 'additive') -> Dict[str, Any]:
    """
    Analyze seasonality patterns in time series data.
    
    Args:
        data: DataFrame with price data and datetime index
        price_column: Column name for price data
        period: Seasonality period ('daily', 'weekly', 'monthly', 'quarterly')
        decompose_method: Decomposition method ('additive' or 'multiplicative')
        
    Returns:
        Dictionary with seasonality analysis results
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        if price_column not in data.columns:
            logger.error(f"Price column '{price_column}' not found in DataFrame")
            return {}
            
        # Ensure data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Data does not have DatetimeIndex, attempting to convert")
            try:
                data = data.copy()
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                    data.set_index('date', inplace=True)
                else:
                    logger.error("No date column found for conversion to DatetimeIndex")
                    return {}
            except Exception as e:
                logger.error(f"Failed to convert index to DatetimeIndex: {e}")
                return {}
        
        # Determine frequency based on period
        if period == 'daily':
            freq = 5  # 5 business days in a week
        elif period == 'weekly':
            freq = 52  # 52 weeks in a year
        elif period == 'monthly':
            freq = 12  # 12 months in a year
        elif period == 'quarterly':
            freq = 4  # 4 quarters in a year
        else:
            logger.error(f"Invalid period: {period}")
            return {}
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(data[price_column], model=decompose_method, period=freq)
        
        # Extract components
        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal.dropna()
        residual = decomposition.resid.dropna()
        
        # Calculate seasonality strength
        if decompose_method == 'additive':
            seasonality_strength = np.abs(seasonal).mean() / np.abs(residual).mean() if np.abs(residual).mean() > 0 else 0
        else:  # multiplicative
            seasonality_strength = (np.abs(seasonal - 1).mean() / np.abs(residual - 1).mean()) if np.abs(residual - 1).mean() > 0 else 0
        
        # Calculate seasonal patterns by period
        if period == 'monthly':
            # Group by month and calculate average
            monthly_pattern = seasonal.groupby(seasonal.index.month).mean()
            period_pattern = {str(i): float(v) for i, v in enumerate(monthly_pattern, 1)}
        elif period == 'weekly':
            # Group by day of week and calculate average
            weekly_pattern = seasonal.groupby(seasonal.index.dayofweek).mean()
            period_pattern = {str(i): float(v) for i, v in enumerate(weekly_pattern)}
        elif period == 'quarterly':
            # Group by quarter and calculate average
            quarterly_pattern = seasonal.groupby(seasonal.index.quarter).mean()
            period_pattern = {str(i): float(v) for i, v in enumerate(quarterly_pattern, 1)}
        else:  # daily
            # No grouping needed
            period_pattern = {str(i): float(v) for i, v in enumerate(seasonal[:freq])}
        
        return {
            'seasonality_strength': float(seasonality_strength),
            'period': period,
            'method': decompose_method,
            'period_pattern': period_pattern,
            'trend': trend.tolist(),
            'seasonal': seasonal.tolist(),
            'residual': residual.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing seasonality: {e}")
        return {}


def analyze_trend(data: pd.DataFrame, price_column: str = 'close', ma_periods: List[int] = [20, 50, 200]) -> Dict[str, Any]:
    """
    Analyze price trends using moving averages and trend indicators.
    
    Args:
        data: DataFrame with price data
        price_column: Column name for price data
        ma_periods: List of periods for moving averages
        
    Returns:
        Dictionary with trend analysis results
    """
    try:
        if price_column not in data.columns:
            logger.error(f"Price column '{price_column}' not found in DataFrame")
            return {}
            
        prices = data[price_column]
        
        # Calculate moving averages
        moving_averages = {}
        for period in ma_periods:
            ma = prices.rolling(window=period).mean()
            moving_averages[f'ma_{period}'] = ma
        
        # Determine current price position relative to MAs
        current_price = prices.iloc[-1]
        price_vs_ma = {}
        for period in ma_periods:
            ma_value = moving_averages[f'ma_{period}'].iloc[-1]
            price_vs_ma[f'vs_ma_{period}'] = (current_price / ma_value) - 1 if not np.isnan(ma_value) else 0
        
        # Calculate ADX (Average Directional Index) for trend strength
        adx = None
        try:
            import talib
            high = data['high'] if 'high' in data.columns else prices
            low = data['low'] if 'low' in data.columns else prices
            close = prices
            
            adx = talib.ADX(high, low, close, timeperiod=14)
        except ImportError:
            logger.warning("talib not available, skipping ADX calculation")
        
        # Determine trend direction and strength
        trend_direction = "Neutral"
        trend_strength = 0.0
        
        # Use MA crossovers for trend direction
        if len(ma_periods) >= 2:
            ma_periods.sort()
            short_ma = moving_averages[f'ma_{ma_periods[0]}'].iloc[-1]
            long_ma = moving_averages[f'ma_{ma_periods[-1]}'].iloc[-1]
            
            if not np.isnan(short_ma) and not np.isnan(long_ma):
                if short_ma > long_ma:
                    trend_direction = "Bullish"
                    trend_strength = (short_ma / long_ma) - 1
                elif short_ma < long_ma:
                    trend_direction = "Bearish"
                    trend_strength = 1 - (short_ma / long_ma)
        
        # Use ADX for trend strength if available
        if adx is not None and not np.isnan(adx.iloc[-1]):
            adx_value = adx.iloc[-1]
            
            # ADX interpretation:
            # < 20: Weak trend
            # 20-30: Moderate trend
            # 30-50: Strong trend
            # > 50: Very strong trend
            if adx_value < 20:
                trend_strength = min(trend_strength, 0.2)
            elif adx_value < 30:
                trend_strength = max(trend_strength, 0.5)
            elif adx_value < 50:
                trend_strength = max(trend_strength, 0.8)
            else:
                trend_strength = max(trend_strength, 1.0)
        
        result = {
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'price_vs_ma': price_vs_ma
        }
        
        # Add moving averages to result
        for period in ma_periods:
            result[f'ma_{period}'] = moving_averages[f'ma_{period}'].tolist()
        
        # Add ADX if available
        if adx is not None:
            result['adx'] = adx.tolist()
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing trend: {e}")
        return {}


def calculate_sharpe_ratio(returns: Union[List[float], pd.Series], risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate Sharpe ratio for a return series.
    
    Args:
        returns: List or Series of returns
        risk_free_rate: Risk-free rate (annualized)
        annualization_factor: Annualization factor (252 for daily, 12 for monthly, etc.)
        
    Returns:
        Sharpe ratio
    """
    try:
        # Convert to pandas Series if needed
        if isinstance(returns, list):
            returns = pd.Series(returns)
        
        # Calculate excess returns
        excess_returns = returns - (risk_free_rate / annualization_factor)
        
        # Calculate Sharpe ratio
        sharpe = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Annualize
        sharpe = sharpe * np.sqrt(annualization_factor)
        
        return sharpe
        
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {e}")
        return 0.0


def calculate_sortino_ratio(returns: Union[List[float], pd.Series], risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate Sortino ratio for a return series.
    
    Args:
        returns: List or Series of returns
        risk_free_rate: Risk-free rate (annualized)
        annualization_factor: Annualization factor (252 for daily, 12 for monthly, etc.)
        
    Returns:
        Sortino ratio
    """
    try:
        # Convert to pandas Series if needed
        if isinstance(returns, list):
            returns = pd.Series(returns)
        
        # Calculate excess returns
        excess_returns = returns - (risk_free_rate / annualization_factor)
        
        # Calculate downside deviation (only negative returns)
        negative_returns = excess_returns[excess_returns < 0]
        downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0
        
        # Calculate Sortino ratio
        sortino = excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0
        
        # Annualize
        sortino = sortino * np.sqrt(annualization_factor)
        
        return sortino
        
    except Exception as e:
        logger.error(f"Error calculating Sortino ratio: {e}")
        return 0.0


def calculate_max_drawdown(prices: Union[List[float], pd.Series]) -> float:
    """
    Calculate maximum drawdown for a price series.
    
    Args:
        prices: List or Series of prices
        
    Returns:
        Maximum drawdown as a positive decimal (e.g., 0.2 for 20% drawdown)
    """
    try:
        # Convert to pandas Series if needed
        if isinstance(prices, list):
            prices = pd.Series(prices)
        
        # Calculate cumulative returns
        returns = prices.pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.cummax()
        
        # Calculate drawdown
        drawdown = (cumulative_returns / running_max) - 1
        
        # Get maximum drawdown (as a positive number)
        max_drawdown = abs(drawdown.min())
        
        return max_drawdown
        
    except Exception as e:
        logger.error(f"Error calculating maximum drawdown: {e}")
        return 0.0


def calculate_beta(returns: Union[List[float], pd.Series], benchmark_returns: Union[List[float], pd.Series]) -> float:
    """
    Calculate beta (systematic risk) relative to a benchmark.
    
    Args:
        returns: List or Series of returns
        benchmark_returns: List or Series of benchmark returns
        
    Returns:
        Beta coefficient
    """
    try:
        # Convert to pandas Series if needed
        if isinstance(returns, list):
            returns = pd.Series(returns)
        if isinstance(benchmark_returns, list):
            benchmark_returns = pd.Series(benchmark_returns)
        
        # Ensure both series have the same length
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns.iloc[-min_len:]
        benchmark_returns = benchmark_returns.iloc[-min_len:]
        
        # Calculate covariance and variance
        covariance = returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        
        # Calculate beta
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        return beta
        
    except Exception as e:
        logger.error(f"Error calculating beta: {e}")
        return 0.0


def calculate_alpha(returns: Union[List[float], pd.Series], benchmark_returns: Union[List[float], pd.Series], risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate Jensen's alpha relative to a benchmark.
    
    Args:
        returns: List or Series of returns
        benchmark_returns: List or Series of benchmark returns
        risk_free_rate: Risk-free rate (annualized)
        annualization_factor: Annualization factor (252 for daily, 12 for monthly, etc.)
        
    Returns:
        Alpha value
    """
    try:
        # Convert to pandas Series if needed
        if isinstance(returns, list):
            returns = pd.Series(returns)
        if isinstance(benchmark_returns, list):
            benchmark_returns = pd.Series(benchmark_returns)
        
        # Ensure both series have the same length
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns.iloc[-min_len:]
        benchmark_returns = benchmark_returns.iloc[-min_len:]
        
        # Calculate daily risk-free rate
        daily_rf = risk_free_rate / annualization_factor
        
        # Calculate beta
        beta = calculate_beta(returns, benchmark_returns)
        
        # Calculate alpha
        alpha = returns.mean() - (daily_rf + beta * (benchmark_returns.mean() - daily_rf))
        
        # Annualize alpha
        alpha = alpha * annualization_factor
        
        return alpha
        
    except Exception as e:
        logger.error(f"Error calculating alpha: {e}")
        return 0.0


def calculate_var(returns: Union[List[float], pd.Series], confidence_level: float = 0.95, method: str = 'historical') -> float:
    """
    Calculate Value at Risk (VaR) for a return series.
    
    Args:
        returns: List or Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        method: VaR calculation method ('historical', 'parametric', 'monte_carlo')
        
    Returns:
        VaR value (as a positive decimal)
    """
    try:
        # Convert to pandas Series if needed
        if isinstance(returns, list):
            returns = pd.Series(returns)
        
        if method == 'historical':
            # Historical VaR
            var = abs(np.percentile(returns, 100 * (1 - confidence_level)))
            
        elif method == 'parametric':
            # Parametric VaR (assumes normal distribution)
            from scipy import stats
            mean = returns.mean()
            std = returns.std()
            var = abs(mean + std * stats.norm.ppf(1 - confidence_level))
            
        elif method == 'monte_carlo':
            # Monte Carlo VaR
            from scipy import stats
            mean = returns.mean()
            std = returns.std()
            
            # Generate random samples
            np.random.seed(42)  # For reproducibility
            n_samples = 10000
            simulated_returns = np.random.normal(mean, std, n_samples)
            
            # Calculate VaR from simulated returns
            var = abs(np.percentile(simulated_returns, 100 * (1 - confidence_level)))
            
        else:
            logger.error(f"Invalid VaR method: {method}")
            return 0.0
        
        return var
        
    except Exception as e:
        logger.error(f"Error calculating VaR: {e}")
        return 0.0


def calculate_cvar(returns: Union[List[float], pd.Series], confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) for a return series.
    
    Args:
        returns: List or Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        CVaR value (as a positive decimal)
    """
    try:
        # Convert to pandas Series if needed
        if isinstance(returns, list):
            returns = pd.Series(returns)
        
        # Calculate VaR
        var = calculate_var(returns, confidence_level, 'historical')
        
        # Calculate CVaR (Expected Shortfall)
        cvar = abs(returns[returns <= -var].mean())
        
        return cvar if not np.isnan(cvar) else var  # Fallback to VaR if CVaR is NaN
        
    except Exception as e:
        logger.error(f"Error calculating CVaR: {e}")
        return 0.0