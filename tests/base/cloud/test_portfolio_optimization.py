import pytest
import pandas as pd
import numpy as np
from base.cloud.portfolio_optimization import PerformanceOptimizer
from base.cloud.portfolio_config import PortfolioConfig

@pytest.fixture
def sample_returns():
    """Create sample returns for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
    assets = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    
    # Create returns with some correlation
    returns_data = np.random.normal(0.001, 0.02, (100, len(assets)))
    # Add some correlation
    common_factor = np.random.normal(0, 0.01, 100)
    for i in range(len(assets)):
        returns_data[:, i] += common_factor
    
    returns = pd.DataFrame(returns_data, index=dates, columns=assets)
    return returns

@pytest.fixture
def performance_optimizer():
    """Create a performance optimizer instance for testing"""
    config = PortfolioConfig()
    return PerformanceOptimizer(config=config)

class TestPerformanceOptimizer:
    """Test PerformanceOptimizer class"""
    
    def test_initialization(self):
        """Test optimizer initialization"""
        optimizer = PerformanceOptimizer()
        
        assert optimizer.num_cores > 0
        assert isinstance(optimizer.config, PortfolioConfig)
    
    def test_cached_covariance(self, performance_optimizer, sample_returns):
        """Test cached covariance calculation"""
        # Convert returns to a string key for caching
        returns_key = sample_returns.to_json()
        
        # First call should compute and cache
        cov1 = performance_optimizer.cached_covariance(returns_key)
        
        # Second call should use cached value
        cov2 = performance_optimizer.cached_covariance(returns_key)
        
        # Results should be identical
        assert np.array_equal(cov1, cov2)
        
        # Result should match direct calculation
        expected_cov = sample_returns.cov().values
        assert np.allclose(cov1, expected_cov)
    
    def test_parallel_monte_carlo(self, performance_optimizer, sample_returns):
        """Test parallel Monte Carlo simulation"""
        weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weights
        initial_investment = 10000.0
        simulation_length = 50
        num_simulations = 100
        
        results = performance_optimizer.parallel_monte_carlo(
            returns=sample_returns,
            weights=weights,
            initial_investment=initial_investment,
            simulation_length=simulation_length,
            num_simulations=num_simulations
        )
        
        assert isinstance(results, np.ndarray)
        assert results.shape[0] == simulation_length + 1  # +1 for initial value
        assert results.shape[1] == num_simulations
        
        # First row should be all equal to initial investment
        assert np.allclose(results[0], initial_investment)
        
        # Values should be positive
        assert np.all(results > 0)
    
    def test_run_simulation_chunk(self, performance_optimizer):
        """Test running a simulation chunk"""
        mean_returns = np.array([0.001, 0.002, 0.001, 0.0015])
        cov_matrix = np.array([
            [0.0004, 0.0002, 0.0001, 0.0001],
            [0.0002, 0.0005, 0.0001, 0.0001],
            [0.0001, 0.0001, 0.0004, 0.0002],
            [0.0001, 0.0001, 0.0002, 0.0004]
        ])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        initial_investment = 10000.0
        simulation_length = 20
        chunk_size = 10
        
        args = (mean_returns, cov_matrix, weights, initial_investment, simulation_length, chunk_size)
        
        results = performance_optimizer._run_simulation_chunk(args)
        
        assert isinstance(results, np.ndarray)
        assert results.shape == (simulation_length + 1, chunk_size)
        assert np.allclose(results[0], initial_investment)
    
    @pytest.mark.skipif("'SKIP_SLOW_TESTS' in os.environ")
    def test_parallel_efficient_frontier(self, performance_optimizer, sample_returns):
        """Test parallel efficient frontier calculation (slow test)"""
        num_portfolios = 100  # Use fewer portfolios for testing
        risk_free_rate = 0.02
        
        ef_results = performance_optimizer.parallel_efficient_frontier(
            returns=sample_returns,
            num_portfolios=num_portfolios,
            risk_free_rate=risk_free_rate
        )
        
        assert isinstance(ef_results, dict)
        assert 'portfolios' in ef_results
        assert 'efficient_frontier' in ef_results
        assert 'optimal_portfolio' in ef_results
        
        # Check portfolios
        portfolios = ef_results['portfolios']
        assert len(portfolios) == num_portfolios
        
        # Check efficient frontier
        ef = ef_results['efficient_frontier']
        assert 'returns' in ef
        assert 'volatility' in ef
        assert 'sharpe_ratio' in ef
        assert len(ef['returns']) > 0
        
        # Check optimal portfolio
        optimal = ef_results['optimal_portfolio']
        assert 'weights' in optimal
        assert 'return' in optimal
        assert 'volatility' in optimal
        assert 'sharpe_ratio' in optimal
        assert len(optimal['weights']) == sample_returns.shape[1]
        assert np.isclose(np.sum(optimal['weights']), 1.0)