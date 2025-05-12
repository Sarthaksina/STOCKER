"""
Test module for feature engineering functionality.

This module contains tests for the FeatureEngineering class and related functionality.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.engineering import FeatureEngineering, FeatureEngineeringError


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    # Create date range for the last 100 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Random data for testing
    np.random.seed(42)  # For reproducibility
    
    # Create sample OHLCV data
    data = {
        'open': np.random.normal(100, 10, size=len(date_range)),
        'high': np.random.normal(105, 10, size=len(date_range)),
        'low': np.random.normal(95, 10, size=len(date_range)),
        'close': np.random.normal(102, 10, size=len(date_range)),
        'volume': np.random.normal(1000000, 200000, size=len(date_range))
    }
    
    # Ensure high is always highest and low is always lowest
    for i in range(len(date_range)):
        values = [data['open'][i], data['close'][i]]
        data['high'][i] = max(values) + abs(np.random.normal(3, 1))
        data['low'][i] = min(values) - abs(np.random.normal(3, 1))
        data['volume'][i] = abs(data['volume'][i])  # Ensure volume is positive
    
    # Create DataFrame
    df = pd.DataFrame(data, index=date_range)
    return df


def test_feature_engineering_initialization(sample_stock_data):
    """Test initialization of FeatureEngineering class."""
    # Test successful initialization
    fe = FeatureEngineering(sample_stock_data)
    assert fe.data is not None
    assert len(fe.data) == len(sample_stock_data)
    
    # Test initialization with bad data
    with pytest.raises(FeatureEngineeringError):
        FeatureEngineering(None)
    
    with pytest.raises(FeatureEngineeringError):
        FeatureEngineering(pd.DataFrame())
    
    # Test initialization with missing columns
    bad_data = sample_stock_data.drop(columns=['volume'])
    with pytest.raises(FeatureEngineeringError):
        FeatureEngineering(bad_data)


def test_add_price_features(sample_stock_data):
    """Test adding price features."""
    fe = FeatureEngineering(sample_stock_data)
    result = fe.add_price_features()
    
    # Check that expected columns are added
    expected_cols = [
        'price_change', 'pct_change', 'log_return', 
        'high_low_range', 'high_low_range_pct',
        'open_close_range', 'open_close_range_pct',
        'upper_shadow', 'lower_shadow', 'body_size',
        'volume_change', 'volume_relative'
    ]
    
    for col in expected_cols:
        assert col in result.columns
    
    # Check calculation correctness for a few features
    np.testing.assert_allclose(
        result['high_low_range'].values,
        (sample_stock_data['high'] - sample_stock_data['low']).values
    )
    
    np.testing.assert_allclose(
        result['open_close_range'].values,
        (sample_stock_data['close'] - sample_stock_data['open']).values
    )


def test_add_moving_averages(sample_stock_data):
    """Test adding moving averages."""
    fe = FeatureEngineering(sample_stock_data)
    result = fe.add_moving_averages(windows=[5, 10, 20])
    
    # Check that expected columns are added
    for window in [5, 10, 20]:
        assert f'sma_{window}' in result.columns
        assert f'ema_{window}' in result.columns
    
    # Check calculation correctness for SMA
    np.testing.assert_allclose(
        result['sma_5'].iloc[4:].values,
        sample_stock_data['close'].rolling(window=5).mean().iloc[4:].values,
        rtol=1e-10, atol=1e-10
    )


def test_add_technical_indicators(sample_stock_data):
    """Test adding technical indicators."""
    fe = FeatureEngineering(sample_stock_data)
    result = fe.add_technical_indicators()
    
    # Check that expected columns are added
    expected_cols = [
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower'
    ]
    
    for col in expected_cols:
        assert col in result.columns


def test_add_date_features(sample_stock_data):
    """Test adding date features."""
    fe = FeatureEngineering(sample_stock_data)
    result = fe.add_date_features()
    
    # Check that expected columns are added
    expected_cols = [
        'day_of_week', 'day_of_month', 'week_of_year',
        'month', 'quarter', 'year', 'is_month_start',
        'is_month_end', 'is_quarter_start', 'is_quarter_end',
        'is_year_start', 'is_year_end', 'is_weekend',
        'month_sin', 'month_cos'
    ]
    
    for col in expected_cols:
        assert col in result.columns
    
    # Check calculation correctness for a few features
    assert result['day_of_week'].equals(sample_stock_data.index.dayofweek)
    assert result['month'].equals(sample_stock_data.index.month)


def test_add_target_variables(sample_stock_data):
    """Test adding target variables."""
    fe = FeatureEngineering(sample_stock_data)
    result = fe.add_target_variables(horizons=[1, 5, 10])
    
    # Check that expected columns are added
    for horizon in [1, 5, 10]:
        assert f'future_price_{horizon}d' in result.columns
        assert f'target_{horizon}d_return' in result.columns
        assert f'target_{horizon}d_direction' in result.columns
    
    # Check that future price is correctly calculated for horizon=1
    # Skip the last row since it will have NaN for target
    np.testing.assert_allclose(
        result['future_price_1d'].iloc[:-1].values,
        sample_stock_data['close'].shift(-1).iloc[:-1].values
    )


def test_handle_missing_values(sample_stock_data):
    """Test handling missing values."""
    # Create some missing values
    data_with_missing = sample_stock_data.copy()
    data_with_missing.iloc[10:15, 0] = np.nan  # Create NaNs in 'open' column
    
    fe = FeatureEngineering(data_with_missing)
    result = fe.handle_missing_values()
    
    # Check that there are no missing values
    assert result.isna().sum().sum() == 0


def test_generate_all_features(sample_stock_data):
    """Test generating all features."""
    fe = FeatureEngineering(sample_stock_data)
    result = fe.generate_all_features()
    
    # Check that the result has more columns than the original data
    assert len(result.columns) > len(sample_stock_data.columns)
    
    # Check that there are no missing values
    assert result.isna().sum().sum() == 0 