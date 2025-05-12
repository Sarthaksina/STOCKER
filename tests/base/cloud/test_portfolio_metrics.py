import pytest
import pandas as pd
import numpy as np
from stocker.cloud.portfolio_metrics_consolidated import calculate_portfolio_metrics, calculate_rolling_metrics
from stocker.cloud.portfolio_config import PortfolioConfig

@pytest.fixture
def sample_returns():
    """Create sample returns for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=252, freq='B')
    assets = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    
    # Create returns with some correlation
    returns_data = np.random.normal(0.0005, 0.01, (252, len(assets)))
    # Add some correlation
    common_factor = np.random.normal(0, 0.005, 252)
    for i in range(len(assets)):
        returns_data[:, i] += common_factor
    
    returns = pd.DataFrame(returns_data, index=dates, columns=assets)
    return returns

@pytest.fixture
def sample_weights():
    """Create sample portfolio weights"""
    return np.array([0.25, 0.25, 0.25, 0.25])  # Equal weights

@pytest.fixture
def config():
    """Create a portfolio configuration for testing"""
    config = PortfolioConfig()
    config.risk_free_rate = 0.02
    return config

class TestCalculatePortfolioMetrics:
    """Test portfolio metrics calculation functions"""
    
    def test_basic_metrics(self, sample_returns, sample_weights, config):
        """Test basic portfolio metrics calculation"""
        metrics = calculate_portfolio_metrics(
            sample_returns, 
            sample_weights,
            config=config
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'expected_return', 'volatility', 'sharpe_ratio', 
            'sortino_ratio', 'max_drawdown', 'var_95', 'cvar_95'
        ]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
        
        # Basic sanity checks
        assert metrics['volatility'] > 0
        assert metrics['max_drawdown'] <= 0
    
    def test_with_benchmark(self, sample_returns, sample_weights, config):
        """Test metrics calculation with benchmark"""
        # Create a benchmark return series
        benchmark_returns = pd.Series(
            np.random.normal(0.0004, 0.012, len(sample_returns)),
            index=sample_returns.index
        )
        
        metrics = calculate_portfolio_metrics(
            sample_returns, 
            sample_weights,
            config=config,
            benchmark_returns=benchmark_returns
        )
        
        # Check benchmark-specific metrics
        assert 'information_ratio' in metrics
        assert 'beta' in metrics
        assert 'treynor_ratio' in metrics
    
    def test_weight_normalization(self, sample_returns, config):
        """Test that weights are properly normalized"""
        # Create unnormalized weights
        unnormalized_weights = np.array([1.0, 2.0, 3.0, 4.0])
        
        metrics = calculate_portfolio_metrics(
            sample_returns, 
            unnormalized_weights,
            config=config
        )
        
        # Metrics should still be calculated correctly
        assert 'expected_return' in metrics
        assert 'volatility' in metrics

class TestCalculateRollingMetrics:
    """Test rolling metrics calculation functions"""
    
    def test_rolling_metrics(self, sample_returns, sample_weights, config):
        """Test rolling metrics calculation"""
        window = 60  # 60-day rolling window
        
        rolling_metrics = calculate_rolling_metrics(
            sample_returns,
            sample_weights,
            window=window,
            config=config
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'volatility', 'returns', 'sharpe_ratio', 
            'max_drawdown', 'downside_risk', 'sortino_ratio'
        ]
        for metric in expected_metrics:
            assert metric in rolling_metrics
            assert isinstance(rolling_metrics[metric], pd.Series)
            
            # Series should have the right length
            # First (window-1) values will be NaN
            assert len(rolling_metrics[metric]) == len(sample_returns)
            assert rolling_metrics[metric].iloc[:window-1].isna().all()
            assert not rolling_metrics[metric].iloc[window:].isna().all()