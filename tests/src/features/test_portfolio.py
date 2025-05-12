import pytest
import pandas as pd
import numpy as np
from stocker.src.features.portfolio import (
    mean_variance_portfolio, exposure_analysis, recommend_portfolio
)
from stocker.src.features.portfolio.portfolio_metrics_consolidated import (
    peer_compare, chart_performance
)

@pytest.fixture
def sample_returns():
    """Create sample returns for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
    assets = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    
    # Create returns with some correlation
    returns_data = np.random.normal(0.001, 0.02, (100, len(assets)))
    
    returns = pd.DataFrame(returns_data, index=dates, columns=assets)
    return returns

@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing"""
    return {
        'AAPL': 0.3,
        'MSFT': 0.3,
        'GOOG': 0.2,
        'AMZN': 0.2
    }

@pytest.fixture
def sample_sector_map():
    """Create a sample sector mapping"""
    return {
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'GOOG': 'Technology',
        'AMZN': 'Consumer'
    }

@pytest.fixture
def sample_asset_class_map():
    """Create a sample asset class mapping"""
    return {
        'AAPL': 'Equity',
        'MSFT': 'Equity',
        'GOOG': 'Equity',
        'AMZN': 'Equity'
    }

class TestMeanVariancePortfolio:
    """Test mean-variance portfolio optimization"""
    
    def test_mean_variance_portfolio(self, sample_returns):
        """Test mean-variance portfolio optimization"""
        result = mean_variance_portfolio(sample_returns)
        
        assert 'weights' in result
        assert 'assets' in result
        assert len(result['weights']) == len(result['assets'])
        assert len(result['weights']) == sample_returns.shape[1]
        
        # Weights should sum to approximately 1
        assert np.isclose(sum(result['weights']), 1.0, atol=1e-10)

class TestExposureAnalysis:
    """Test portfolio exposure analysis"""
    
    def test_exposure_analysis(self, sample_portfolio, sample_sector_map, sample_asset_class_map):
        """Test exposure analysis"""
        result = exposure_analysis(sample_portfolio, sample_sector_map, sample_asset_class_map)
        
        assert 'sector_exposure' in result
        assert 'asset_exposure' in result
        assert 'over_exposed' in result
        assert 'under_exposed' in result
        
        # Check sector exposure
        sector_exp = result['sector_exposure']
        assert 'Technology' in sector_exp
        assert 'Consumer' in sector_exp
        assert np.isclose(sector_exp['Technology'], 0.8)
        assert np.isclose(sector_exp['Consumer'], 0.2)
        
        # Technology should be over-exposed (>40%)
        assert 'Technology' in result['over_exposed']
        
        # Asset exposure should sum to 1
        assert np.isclose(sum(result['asset_exposure'].values()), 1.0)

class TestRecommendPortfolio:
    """Test portfolio recommendation"""
    
    def test_recommend_portfolio(self, sample_returns):
        """Test portfolio recommendation"""
        # Create price history map from returns
        prices = (1 + sample_returns).cumprod()
        price_history_map = {col: prices[col].tolist() for col in prices.columns}
        
        user_info = {'price_history_map': price_history_map}
        result = recommend_portfolio(user_info, None)
        
        assert 'weights' in result
        assert 'assets' in result
        assert len(result['weights']) == len(result['assets'])
        
        # Weights should sum to approximately 1
        assert np.isclose(sum(result['weights']), 1.0, atol=1e-10)
    
    def test_empty_price_history(self):
        """Test with empty price history"""
        user_info = {'price_history_map': {}}
        result = recommend_portfolio(user_info, None)
        
        assert 'error' in result

class TestPeerComparison:
    """Test peer comparison functionality"""
    
    def test_peer_compare(self, sample_returns):
        """Test peer comparison"""
        # Create price history map from returns
        prices = (1 + sample_returns).cumprod()
        price_history_map = {col: prices[col].tolist() for col in prices.columns}
        
        result = peer_compare(price_history_map, 'AAPL', n=3)
        
        assert 'target' in result
        assert 'peers' in result
        assert result['target'] == 'AAPL'
        assert len(result['peers']) == 3
        
        # Each peer should have symbol and correlation
        for peer in result['peers']:
            assert 'symbol' in peer
            assert 'correlation' in peer
            assert isinstance(peer['correlation'], float)
            assert -1 <= peer['correlation'] <= 1
    
    def test_invalid_target(self, sample_returns):
        """Test with invalid target"""
        # Create price history map from returns
        prices = (1 + sample_returns).cumprod()
        price_history_map = {col: prices[col].tolist() for col in prices.columns}
        
        result = peer_compare(price_history_map, 'INVALID')
        
        assert 'error' in result

class TestChartPerformance:
    """Test chart performance functionality"""
    
    def test_chart_performance(self, sample_returns):
        """Test chart performance calculation"""
        # Create price history map from returns
        prices = (1 + sample_returns).cumprod()
        price_history_map = {col: prices[col].tolist() for col in prices.columns}
        dates = [d.strftime('%Y-%m-%d') for d in sample_returns.index]
        
        result = chart_performance(dates, price_history_map)
        
        assert 'quarterly' in result
        assert 'yearly' in result
        
        # Check that quarterly and yearly results have the expected structure
        assert isinstance(result['quarterly'], dict)
        assert isinstance(result['yearly'], dict)