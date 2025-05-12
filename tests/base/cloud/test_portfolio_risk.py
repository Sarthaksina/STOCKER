import pytest
import pandas as pd
import numpy as np
from base.cloud.portfolio_risk import (
    calculate_var, calculate_cvar, calculate_drawdown, PortfolioRiskAnalyzer
)
from base.cloud.portfolio_config import PortfolioConfig

@pytest.fixture
def sample_returns():
    """Create sample returns for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=252, freq='B')
    returns = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
    return returns

@pytest.fixture
def risk_analyzer():
    """Create a risk analyzer instance for testing"""
    config = PortfolioConfig()
    # Set specific configuration for testing
    config.risk_free_rate = 0.02
    config.target_volatility = 0.15
    return PortfolioRiskAnalyzer(config=config)

class TestCalculateVaR:
    """Test Value at Risk calculation functions"""
    
    def test_historical_var(self, sample_returns):
        """Test historical VaR calculation"""
        var = calculate_var(sample_returns, confidence_level=0.95, method='historical')
        assert isinstance(var, float)
        assert var < 0  # VaR should be negative (representing loss)
    
    def test_parametric_var(self, sample_returns):
        """Test parametric VaR calculation"""
        var = calculate_var(sample_returns, confidence_level=0.95, method='parametric_norm')
        assert isinstance(var, float)
        assert var < 0
    
    def test_empty_returns(self):
        """Test VaR with empty returns"""
        empty_returns = pd.Series()
        var = calculate_var(empty_returns)
        assert np.isnan(var)
    
    def test_different_confidence_levels(self, sample_returns):
        """Test VaR with different confidence levels"""
        var_90 = calculate_var(sample_returns, confidence_level=0.90)
        var_95 = calculate_var(sample_returns, confidence_level=0.95)
        var_99 = calculate_var(sample_returns, confidence_level=0.99)
        
        # Higher confidence level should result in more extreme (more negative) VaR
        assert var_90 > var_95 > var_99

class TestCalculateCVaR:
    """Test Conditional Value at Risk calculation functions"""
    
    def test_historical_cvar(self, sample_returns):
        """Test historical CVaR calculation"""
        cvar = calculate_cvar(sample_returns, confidence_level=0.95, method='historical')
        assert isinstance(cvar, float)
        assert cvar < 0  # CVaR should be negative (representing loss)
    
    def test_cvar_vs_var(self, sample_returns):
        """Test that CVaR is more conservative than VaR"""
        var = calculate_var(sample_returns, confidence_level=0.95)
        cvar = calculate_cvar(sample_returns, confidence_level=0.95)
        assert cvar < var  # CVaR should be more negative than VaR

class TestCalculateDrawdown:
    """Test drawdown calculation functions"""
    
    def test_drawdown_calculation(self, sample_returns):
        """Test drawdown calculation"""
        drawdown_series, max_dd, avg_dd = calculate_drawdown(sample_returns)
        
        assert isinstance(drawdown_series, pd.Series)
        assert isinstance(max_dd, float)
        assert isinstance(avg_dd, float)
        assert max_dd <= 0  # Max drawdown should be negative or zero
        assert len(drawdown_series) == len(sample_returns)
        
        # Max drawdown should be the minimum value in the drawdown series
        assert np.isclose(max_dd, drawdown_series.min())

class TestPortfolioRiskAnalyzer:
    """Test PortfolioRiskAnalyzer class"""
    
    def test_calculate_risk_metrics(self, risk_analyzer, sample_returns):
        """Test risk metrics calculation"""
        metrics = risk_analyzer.calculate_risk_metrics(sample_returns)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'volatility', 'downside_deviation', 
            'var_historical_95', 'cvar_historical_95',
            'var_parametric_norm_95', 'cvar_parametric_norm_95'
        ]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
        
        # Volatility should be positive
        assert metrics['volatility'] > 0
        
        # VaR and CVaR should be negative
        assert metrics['var_historical_95'] < 0
        assert metrics['cvar_historical_95'] < 0
    
    def test_monte_carlo_simulation(self, risk_analyzer, sample_returns):
        """Test Monte Carlo simulation"""
        num_simulations = 100
        num_periods = 50
        
        simulated_paths = risk_analyzer.run_monte_carlo_simulation(
            sample_returns, 
            num_simulations=num_simulations,
            num_periods=num_periods
        )
        
        assert isinstance(simulated_paths, pd.DataFrame)
        assert simulated_paths.shape == (num_periods + 1, num_simulations)
        
        # First row should be all ones (starting values)
        assert np.allclose(simulated_paths.iloc[0], 1.0)