"""Tests for portfolio utility functions.

This module contains tests for the utility functions in the portfolio_utils module,
including validation, performance optimization, and numerical operations.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.features.portfolio.portfolio_utils import (
    # Validation functions
    validate_dataframe,
    validate_series,
    validate_weights,
    validate_portfolio_data,
    safe_divide,
    
    # Performance optimization
    cached_calculation,
    vectorized,
    
    # Numerical operations
    rolling_window,
    exponential_decay_weights,
    calculate_ewma,
    calculate_portfolio_moments,
    
    # Data conversion
    convert_to_returns,
    ensure_datetime_index,
    resample_returns
)


# ===== Test Data Setup =====

@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    prices = pd.DataFrame({
        'AAPL': np.linspace(100, 150, 100) + np.random.normal(0, 5, 100),
        'MSFT': np.linspace(200, 250, 100) + np.random.normal(0, 8, 100),
        'GOOG': np.linspace(1000, 1200, 100) + np.random.normal(0, 20, 100)
    }, index=dates)
    return prices


@pytest.fixture
def sample_returns_data(sample_price_data):
    """Create sample returns data for testing."""
    return sample_price_data.pct_change().dropna()


# ===== Validation Function Tests =====

def test_validate_dataframe():
    """Test validate_dataframe function."""
    # Valid DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    validated = validate_dataframe(df, required_columns=['A', 'B'], min_rows=2)
    assert validated is df
    
    # Test with None
    with pytest.raises(ValueError):
        validate_dataframe(None)
    
    # Test with None but allow_none=True
    assert validate_dataframe(None, allow_none=True) is None
    
    # Test with non-DataFrame
    with pytest.raises(TypeError):
        validate_dataframe([1, 2, 3])
    
    # Test with too few rows
    with pytest.raises(ValueError):
        validate_dataframe(pd.DataFrame({'A': [1]}), min_rows=2)
    
    # Test with missing columns
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns=['A', 'B', 'C'])


def test_validate_series():
    """Test validate_series function."""
    # Valid Series
    s = pd.Series([1, 2, 3])
    validated = validate_series(s, min_length=2)
    assert validated is s
    
    # Test with None
    with pytest.raises(ValueError):
        validate_series(None)
    
    # Test with None but allow_none=True
    assert validate_series(None, allow_none=True) is None
    
    # Test with non-Series but allow_convert=True
    validated = validate_series([1, 2, 3], allow_convert=True)
    assert isinstance(validated, pd.Series)
    assert list(validated.values) == [1, 2, 3]
    
    # Test with non-Series and allow_convert=False
    with pytest.raises(TypeError):
        validate_series([1, 2, 3], allow_convert=False)
    
    # Test with too few elements
    with pytest.raises(ValueError):
        validate_series(pd.Series([1]), min_length=2)


def test_validate_weights():
    """Test validate_weights function."""
    # Test with array
    weights = np.array([0.3, 0.3, 0.4])
    validated = validate_weights(weights)
    assert np.array_equal(validated, weights)
    
    # Test with list
    weights_list = [0.3, 0.3, 0.4]
    validated = validate_weights(weights_list)
    assert np.array_equal(validated, np.array(weights_list))
    
    # Test with dict
    weights_dict = {'AAPL': 0.3, 'MSFT': 0.3, 'GOOG': 0.4}
    assets = ['AAPL', 'MSFT', 'GOOG']
    validated = validate_weights(weights_dict, assets)
    assert np.array_equal(validated, np.array([0.3, 0.3, 0.4]))
    
    # Test with Series
    weights_series = pd.Series(weights_dict)
    validated = validate_weights(weights_series, assets)
    assert np.array_equal(validated, np.array([0.3, 0.3, 0.4]))
    
    # Test normalization
    weights = np.array([3, 3, 4])
    validated = validate_weights(weights, normalize=True)
    assert np.array_equal(validated, np.array([0.3, 0.3, 0.4]))
    
    # Test with NaN
    with pytest.raises(ValueError):
        validate_weights([0.3, 0.3, np.nan])
    
    # Test with mismatched assets
    with pytest.raises(ValueError):
        validate_weights([0.3, 0.3, 0.4], assets=['AAPL', 'MSFT'])


def test_validate_portfolio_data(sample_returns_data):
    """Test validate_portfolio_data function."""
    # Test with DataFrame returns
    result = validate_portfolio_data(
        returns=sample_returns_data,
        weights=[0.3, 0.3, 0.4],
        benchmark_returns=sample_returns_data['AAPL'],
        risk_free_rate=0.02
    )
    
    assert 'returns' in result
    assert 'weights' in result
    assert 'benchmark_returns' in result
    assert 'risk_free_rate' in result
    assert isinstance(result['returns'], pd.DataFrame)
    assert isinstance(result['weights'], np.ndarray)
    assert isinstance(result['benchmark_returns'], pd.Series)
    assert isinstance(result['risk_free_rate'], float)
    
    # Test with Series returns
    result = validate_portfolio_data(
        returns=sample_returns_data['AAPL'],
        risk_free_rate=pd.Series([0.02] * len(sample_returns_data))
    )
    
    assert 'returns' in result
    assert 'risk_free_rate' in result
    assert isinstance(result['returns'], pd.Series)
    assert isinstance(result['risk_free_rate'], pd.Series)


def test_safe_divide():
    """Test safe_divide function."""
    # Normal division
    numerator = np.array([1.0, 2.0, 3.0])
    denominator = np.array([2.0, 2.0, 2.0])
    result = safe_divide(numerator, denominator)
    assert np.array_equal(result, np.array([0.5, 1.0, 1.5]))
    
    # Division by zero
    denominator = np.array([2.0, 0.0, 2.0])
    result = safe_divide(numerator, denominator)
    assert np.array_equal(result, np.array([0.5, 0.0, 1.5]))
    
    # Custom default value
    result = safe_divide(numerator, denominator, default=np.nan)
    assert np.isnan(result[1])
    assert result[0] == 0.5
    assert result[2] == 1.5


# ===== Numerical Operations Tests =====

def test_rolling_window():
    """Test rolling_window function."""
    array = np.array([1, 2, 3, 4, 5])
    result = rolling_window(array, 3)
    
    expected = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]
    ])
    
    assert np.array_equal(result, expected)


def test_exponential_decay_weights():
    """Test exponential_decay_weights function."""
    weights = exponential_decay_weights(5, decay_factor=0.5)
    
    # Check that weights sum to 1
    assert np.isclose(np.sum(weights), 1.0)
    
    # Check that weights are in descending order
    assert np.all(np.diff(weights) < 0)


def test_calculate_ewma():
    """Test calculate_ewma function."""
    # Test with constant data
    data = np.array([10.0] * 10)
    result = calculate_ewma(data, span=5)
    assert np.allclose(result, 10.0)
    
    # Test with increasing data
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = calculate_ewma(data, span=2)
    # First value is the mean of first span values or first value
    assert result[0] == 1.0
    # Subsequent values follow EWMA formula
    alpha = 2 / (2 + 1)
    expected = np.zeros_like(data)
    expected[0] = 1.0
    for i in range(1, len(data)):
        expected[i] = alpha * data[i] + (1 - alpha) * expected[i-1]
    assert np.allclose(result, expected)


def test_calculate_portfolio_moments(sample_returns_data):
    """Test calculate_portfolio_moments function."""
    weights = np.array([0.3, 0.3, 0.4])
    result = calculate_portfolio_moments(sample_returns_data, weights)
    
    assert 'mean' in result
    assert 'variance' in result
    assert 'volatility' in result
    assert 'skewness' in result
    assert 'kurtosis' in result
    assert 'covariance_matrix' in result
    
    # Check that volatility is sqrt of variance
    assert np.isclose(result['volatility'], np.sqrt(result['variance']))
    
    # Check covariance matrix dimensions
    assert result['covariance_matrix'].shape == (3, 3)


# ===== Data Conversion Tests =====

def test_convert_to_returns(sample_price_data):
    """Test convert_to_returns function."""
    # Test pct_change method
    returns_pct = convert_to_returns(sample_price_data, method='pct_change')
    expected_pct = sample_price_data.pct_change().dropna()
    assert returns_pct.equals(expected_pct)
    
    # Test log_returns method
    returns_log = convert_to_returns(sample_price_data, method='log_returns')
    expected_log = np.log(sample_price_data / sample_price_data.shift(1)).dropna()
    assert returns_log.equals(expected_log)
    
    # Test simple method
    returns_simple = convert_to_returns(sample_price_data, method='simple')
    expected_simple = (sample_price_data / sample_price_data.shift(1) - 1).dropna()
    assert returns_simple.equals(expected_simple)
    
    # Test invalid method
    with pytest.raises(ValueError):
        convert_to_returns(sample_price_data, method='invalid')


def test_ensure_datetime_index():
    """Test ensure_datetime_index function."""
    # DataFrame with string dates
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }, index=['2020-01-01', '2020-01-02', '2020-01-03'])
    
    result = ensure_datetime_index(df)
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index[0] == pd.Timestamp('2020-01-01')
    
    # DataFrame with datetime index already
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }, index=pd.date_range('2020-01-01', periods=3))
    
    result = ensure_datetime_index(df)
    assert result is df  # Should return the same object
    
    # DataFrame with non-convertible index
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }, index=['not-a-date', 'also-not-a-date', 'still-not-a-date'])
    
    with pytest.raises(ValueError):
        ensure_datetime_index(df)


def test_resample_returns(sample_returns_data):
    """Test resample_returns function."""
    # Test compound method
    monthly_returns = resample_returns(sample_returns_data, freq='M', method='compound')
    assert isinstance(monthly_returns, pd.DataFrame)
    assert len(monthly_returns) < len(sample_returns_data)  # Should have fewer rows
    
    # Test sum method
    monthly_returns_sum = resample_returns(sample_returns_data, freq='M', method='sum')
    assert isinstance(monthly_returns_sum, pd.DataFrame)
    
    # Test with Series
    series_returns = resample_returns(sample_returns_data['AAPL'], freq='M', method='compound')
    assert isinstance(series_returns, pd.Series)
    
    # Test invalid method
    with pytest.raises(ValueError):
        resample_returns(sample_returns_data, method='invalid')


# ===== Decorator Tests =====

def test_cached_calculation():
    """Test cached_calculation decorator."""
    call_count = 0
    
    @cached_calculation
    def expensive_function(x):
        nonlocal call_count
        call_count += 1
        return x * 2
    
    # First call should execute the function
    result1 = expensive_function(10)
    assert result1 == 20
    assert call_count == 1
    
    # Second call with same args should use cache
    result2 = expensive_function(10)
    assert result2 == 20
    assert call_count == 1  # Call count should not increase
    
    # Call with different args should execute the function again
    result3 = expensive_function(20)
    assert result3 == 40
    assert call_count == 2
    
    # Check cache info
    cache_info = expensive_function.cache_info()
    assert cache_info.hits == 1  # One cache hit
    assert cache_info.misses == 2  # Two cache misses
    
    # Clear cache
    expensive_function.cache_clear()
    result4 = expensive_function(10)
    assert result4 == 20
    assert call_count == 3  # Call count should increase after cache clear


def test_vectorized():
    """Test vectorized decorator."""
    @vectorized
    def add_one(x):
        return x + 1
    
    # Test with list
    result = add_one([1, 2, 3])
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([2, 3, 4]))
    
    # Test with numpy array
    result = add_one(np.array([1, 2, 3]))
    assert np.array_equal(result, np.array([2, 3, 4]))
    
    # Test with pandas Series
    result = add_one(pd.Series([1, 2, 3]))
    assert np.array_equal(result, np.array([2, 3, 4]))
    
    # Test with scalar
    result = add_one(1)
    assert result == 2