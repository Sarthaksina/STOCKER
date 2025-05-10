# Features Module

This module provides feature engineering and analysis tools for the STOCKER Pro platform.

## Overview

The Features module includes three core components:

1. **Feature Engineering** - Tools for creating features from financial time series data
2. **Technical Indicators** - A comprehensive set of technical indicators for market analysis
3. **Portfolio Optimization** - Modern portfolio theory implementation and performance analytics

## Structure

```
features/
├── __init__.py             # Package exports
├── engineering.py          # Feature engineering utilities
├── indicators.py           # Technical indicators implementation
├── portfolio/              # Portfolio optimization subpackage
│   ├── __init__.py         # Subpackage exports
│   ├── optimizer.py        # Portfolio optimization algorithms
│   └── analysis.py         # Portfolio performance analytics
└── README.md               # This file
```

## Usage Examples

### Feature Engineering

```python
import pandas as pd
from src.features.engineering import FeatureEngineer, generate_features

# Load stock data
stock_data = pd.read_csv("aapl_data.csv", index_col="date", parse_dates=True)

# Generate all features
features_df = generate_features(
    data=stock_data,
    price_column="close",
    volume_column="volume",
    include_time_features=True,
    include_lags=True,
    include_rolling=True,
    include_returns=True
)

# Or use the FeatureEngineer class for more control
engineer = FeatureEngineer(stock_data)
features_df = engineer.create_timeframe_features()
features_df = engineer.create_lag_features(["close", "volume"], [1, 2, 3, 5, 10])
features_df = engineer.create_rolling_features(["close"], [5, 10, 21])
features_df = engineer.create_return_features("close", [1, 5, 10, 21])
```

### Technical Indicators

```python
import pandas as pd
from src.features.indicators import calculate_technical_indicators

# Load stock data
stock_data = pd.read_csv("aapl_data.csv", index_col="date", parse_dates=True)

# Calculate all technical indicators
indicators_df = calculate_technical_indicators(
    data=stock_data,
    price_column="close",
    high_column="high",
    low_column="low"
)

# Or calculate specific indicators
from src.features.indicators import calculate_macd, calculate_rsi, calculate_bollinger_bands

macd_df = calculate_macd(stock_data, price_column="close")
rsi_df = calculate_rsi(stock_data, price_column="close", period=14)
bb_df = calculate_bollinger_bands(stock_data, price_column="close", period=20, std_dev=2.0)
```

### Portfolio Optimization

```python
import pandas as pd
from src.features.portfolio import optimize_portfolio, calculate_portfolio_statistics

# Load stock returns data
returns_data = pd.read_csv("returns.csv", index_col="date", parse_dates=True)

# Optimize portfolio for maximum Sharpe ratio
optimal_portfolio = optimize_portfolio(
    returns=returns_data,
    objective="sharpe",
    risk_free_rate=0.02/252  # Daily risk-free rate
)

# Print optimal weights
print("Optimal portfolio weights:")
for asset, weight in optimal_portfolio["weights"].items():
    print(f"{asset}: {weight:.4f}")

print(f"Expected Return: {optimal_portfolio['expected_return']:.4f}")
print(f"Volatility: {optimal_portfolio['volatility']:.4f}")
print(f"Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.4f}")

# Calculate portfolio statistics with benchmark comparison
sp500_returns = pd.read_csv("sp500_returns.csv", index_col="date", parse_dates=True)["return"]
portfolio_returns = pd.Series(data=[...], index=[...])  # Your portfolio returns

stats = calculate_portfolio_statistics(
    returns=portfolio_returns,
    benchmark_returns=sp500_returns,
    risk_free_rate=0.02/252,
    period="daily"
)

print(f"Portfolio Beta: {stats['beta']:.4f}")
print(f"Portfolio Alpha: {stats['alpha']:.4f}")
print(f"Max Drawdown: {stats['max_drawdown']:.4f}")
``` 