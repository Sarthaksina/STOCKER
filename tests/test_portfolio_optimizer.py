import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the portfolio optimizer
from stocker.cloud.portfolio_optimizer import ThunderComputePortfolioOptimizer

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
    optimizer = ThunderComputePortfolioOptimizer()
    
    # Optimize portfolio
    result = optimizer.optimize_portfolio(price_data, use_cloud=False)
    
    # Print results
    print("Optimized Portfolio Weights:")
    for asset, weight in zip(assets, result['weights']):
        print(f"{asset}: {weight:.4f}")
    
    print("\nPortfolio Metrics:")
    for metric, value in result['metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    # Generate efficient frontier
    ef_data = optimizer.generate_efficient_frontier(price_data, num_portfolios=500)
    
    # Plot efficient frontier
    optimizer.plot_efficient_frontier(ef_data, save_path='efficient_frontier.png')
    
    # Backtest portfolio
    backtest_results = optimizer.backtest_portfolio(price_data, result['weights'])
    
    # Plot backtest results
    optimizer.plot_backtest_results(backtest_results, save_path='backtest_results.png')
    
    print("\nBacktest Metrics:")
    for metric, value in backtest_results['metrics'].items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    test_portfolio_optimizer()