"""
Utility functions for tests.
"""
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional


def generate_synthetic_price_data(
    n_samples: int = 200,
    freq: str = "1D",
    trend: float = 0.1,
    noise: float = 1.0,
    start_price: float = 100.0,
    start_date: str = "2020-01-01",
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate synthetic price data for testing.
    
    Args:
        n_samples: Number of samples
        freq: Time frequency
        trend: Trend factor
        noise: Noise factor
        start_price: Starting price
        start_date: Starting date
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic price data including Open, High, Low, Close, Volume
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate date range
    dates = pd.date_range(start=start_date, periods=n_samples, freq=freq)
    
    # Generate close prices
    close_prices = np.zeros(n_samples)
    close_prices[0] = start_price
    
    # Add trend and noise
    for i in range(1, n_samples):
        # Previous price + trend + random noise
        close_prices[i] = close_prices[i-1] * (1 + trend/100) + np.random.normal(0, noise)
        
        # Ensure prices are positive
        if close_prices[i] <= 0:
            close_prices[i] = close_prices[i-1] * 0.9  # 10% drop if would go negative
    
    # Generate open, high, low prices
    open_prices = close_prices.copy()
    high_prices = close_prices.copy()
    low_prices = close_prices.copy()
    
    # Add some variation to open, high, low
    for i in range(n_samples):
        # Open price is close of previous day with some noise
        if i > 0:
            open_prices[i] = close_prices[i-1] * (1 + np.random.normal(0, 0.01))
        
        # High and low are based on close with some noise
        high_prices[i] = max(close_prices[i], open_prices[i]) * (1 + abs(np.random.normal(0, 0.02)))
        low_prices[i] = min(close_prices[i], open_prices[i]) * (1 - abs(np.random.normal(0, 0.02)))
    
    # Generate volume
    volume = np.random.lognormal(mean=10, sigma=1, size=n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        "Open": open_prices,
        "High": high_prices,
        "Low": low_prices,
        "Close": close_prices,
        "Volume": volume
    }, index=dates)
    
    return df


def generate_synthetic_features(
    price_data: pd.DataFrame, n_features: int = 10, seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate synthetic features from price data for testing.
    
    Args:
        price_data: Input price data
        n_features: Number of features to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic features
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get price and volume data
    close = price_data["Close"].values
    volume = price_data["Volume"].values
    index = price_data.index
    n_samples = len(close)
    
    features = {}
    
    # Generate moving averages with different window sizes
    for i in range(min(3, n_features)):
        window = 5 * (i + 1)
        ma_name = f"MA_{window}"
        ma_values = np.zeros(n_samples)
        for j in range(n_samples):
            if j >= window:
                ma_values[j] = np.mean(close[j-window:j])
            else:
                ma_values[j] = close[j]
        features[ma_name] = ma_values
    
    # Generate relative strength index (RSI) - simplified version
    if n_features > 3:
        window = 14
        rsi = np.zeros(n_samples)
        for i in range(window, n_samples):
            diff = np.diff(close[i-window:i+1])
            gains = diff[diff > 0].sum()
            losses = -diff[diff < 0].sum()
            if losses == 0:
                rsi[i] = 100
            else:
                rs = gains / losses
                rsi[i] = 100 - (100 / (1 + rs))
        features["RSI"] = rsi
    
    # Generate volume-based features
    if n_features > 4:
        # Volume moving average
        vma = np.zeros(n_samples)
        for i in range(n_samples):
            if i >= 10:
                vma[i] = np.mean(volume[i-10:i])
            else:
                vma[i] = volume[i]
        features["Volume_MA_10"] = vma
    
    # Generate random features to fill remaining slots
    remaining_features = max(0, n_features - len(features))
    for i in range(remaining_features):
        # Create random features with some correlation to price
        base = close + np.random.normal(0, np.std(close) * 0.5, n_samples)
        features[f"Feature_{i+1}"] = base
    
    # Create DataFrame
    feature_df = pd.DataFrame(features, index=index)
    
    # Add some missing values to make it realistic
    for col in feature_df.columns[:2]:
        mask = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
        feature_df.loc[mask, col] = np.nan
    
    return feature_df


class TempDirectoryTestCase:
    """Mixin for test cases that need a temporary directory."""
    
    def setup_method(self):
        """Set up a temporary directory before each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up the temporary directory after each test."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def create_test_data_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create test data for pipeline tests.
    
    Returns:
        Tuple of (train_data, validation_data, test_data)
    """
    # Generate price data
    full_data = generate_synthetic_price_data(n_samples=300)
    
    # Generate features
    features = generate_synthetic_features(full_data, n_features=15)
    
    # Combine price and features
    data = pd.concat([full_data, features], axis=1)
    
    # Split into train, validation, test
    train_data = data.iloc[:200]
    validation_data = data.iloc[200:250]
    test_data = data.iloc[250:]
    
    return train_data, validation_data, test_data


def assert_predictions_valid(predictions: np.ndarray, target_shape: Optional[Tuple[int, ...]] = None):
    """
    Assert that model predictions are valid.
    
    Args:
        predictions: Model predictions
        target_shape: Expected shape if known
    """
    assert predictions is not None
    assert isinstance(predictions, np.ndarray)
    assert not np.any(np.isnan(predictions)), "Predictions contain NaN values"
    assert not np.any(np.isinf(predictions)), "Predictions contain infinite values"
    
    if target_shape is not None:
        assert predictions.shape == target_shape, f"Expected shape {target_shape}, got {predictions.shape}"


def assert_metrics_valid(metrics: Dict[str, float]):
    """
    Assert that evaluation metrics are valid.
    
    Args:
        metrics: Dictionary of metric name to value
    """
    assert metrics is not None
    assert isinstance(metrics, dict)
    assert len(metrics) > 0, "Metrics dictionary is empty"
    
    for name, value in metrics.items():
        assert isinstance(name, str), f"Metric name {name} is not a string"
        assert isinstance(value, (int, float)), f"Metric value {value} is not a number"
        assert not np.isnan(value), f"Metric {name} is NaN"
        assert not np.isinf(value), f"Metric {name} is infinite" 