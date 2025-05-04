import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the portfolio optimizer
from src.features.portfolio.portfolio_optimization import PerformanceOptimizer
from src.features.portfolio.portfolio_config import PortfolioConfig

def test_portfolio_optimizer():
    """Test the portfolio optimizer with sample data."""
    # Load environment variables
    load_dotenv()
    
    # Create sample price data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=252, freq='B')
    assets = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    # Generate random price data
    price_data = pd.DataFrame(index=dates)
    for asset in assets:
        # Start with price of 100
        start_price = 100
        # Generate random returns
        returns = np.random.normal(0.0005, 0.015, len(dates))
        # Calculate price series
        prices = start_price * (1 + returns).cumprod()
        price_data[asset] = prices
    
    # Initialize the optimizer
    optimizer = PerformanceOptimizer()
    
    # Calculate returns
    returns = price_data.pct_change().dropna()
    
    # Test parallel monte carlo simulation
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weights
    simulation_results = optimizer.parallel_monte_carlo(
        returns=returns,
        weights=weights,
        initial_investment=10000.0,
        simulation_length=252,
        num_simulations=1000
    )
    
    # Verify simulation results shape
    assert simulation_results.shape[0] == 252
    assert simulation_results.shape[1] >= 900  # Should be close to 1000
    
    # Test efficient frontier calculation
    ef_results = optimizer.parallel_efficient_frontier(
        returns=returns,
        num_portfolios=500,
        risk_free_rate=0.02
    )
    
    # Verify efficient frontier results
    assert 'efficient_frontier' in ef_results
    assert 'max_sharpe_portfolio' in ef_results
    assert 'min_volatility_portfolio' in ef_results
    
    # Print results
    print("Efficient Frontier Results:")
    print(f"Max Sharpe Ratio Portfolio:")
    for asset, weight in zip(assets, ef_results['max_sharpe_portfolio']['weights']):
        print(f"{asset}: {weight:.4f}")
    
    print("\nMin Volatility Portfolio:")
    for asset, weight in zip(assets, ef_results['min_volatility_portfolio']['weights']):
        print(f"{asset}: {weight:.4f}")
    
    # Test that weights sum to 1
    assert np.isclose(np.sum(ef_results['max_sharpe_portfolio']['weights']), 1.0)
    assert np.isclose(np.sum(ef_results['min_volatility_portfolio']['weights']), 1.0)

if __name__ == "__main__":
    test_portfolio_optimizer()