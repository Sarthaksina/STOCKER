import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def test_data_dir():
    """Path to the test data directory"""
    return os.path.join(os.path.dirname(__file__), 'data')

@pytest.fixture
def create_test_data_dir(test_data_dir):
    """Create test data directory if it doesn't exist"""
    os.makedirs(test_data_dir, exist_ok=True)
    return test_data_dir

@pytest.fixture
def sample_price_data():
    """Create sample price data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=252, freq='B')
    assets = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'SPY']
    
    # Create price data with trends and some correlation
    price_data = {}
    for i, asset in enumerate(assets):
        # Start with a base price between 50 and 200
        base_price = 50 + i * 50
        # Add a trend component
        trend = np.linspace(0, 0.2 + i * 0.05, 252)
        # Add a random component
        random = np.random.normal(0, 0.01, 252).cumsum()
        # Add a common market component (stronger for SPY)
        market = np.random.normal(0, 0.005, 252).cumsum()
        
        # Combine components
        if asset == 'SPY':
            # SPY is the benchmark, more influenced by market
            asset_prices = base_price * (1 + trend + random + market * 1.5)
        else:
            asset_prices = base_price * (1 + trend + random + market * (0.5 + i * 0.2))
            
        price_data[asset] = asset_prices
    
    return pd.DataFrame(price_data, index=dates)

@pytest.fixture
def sample_returns_data(sample_price_data):
    """Create sample returns data from price data"""
    return sample_price_data.pct_change().dropna()