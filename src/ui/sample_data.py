"""Sample data provider for STOCKER Pro dashboard.

This module provides sample data for the dashboard when real data is not available.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

def generate_stock_data(
    ticker: str = "AAPL",
    days: int = 180,
    include_indicators: bool = True
) -> pd.DataFrame:
    """
    Generate sample stock price data for demonstration purposes.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days of historical data to generate
        include_indicators: Whether to include technical indicators
        
    Returns:
        DataFrame with OHLCV data and optional indicators
    """
    # Generate dates
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate price data
    base_price = {
        "AAPL": 150.0,
        "MSFT": 250.0,
        "GOOGL": 120.0,
        "AMZN": 100.0,
        "TSLA": 200.0
    }.get(ticker, 100.0)
    
    # Generate random walk
    returns = np.random.normal(0.0005, 0.015, len(dates))
    cumulative_returns = np.cumprod(1 + returns)
    close_prices = base_price * cumulative_returns
    
    # Generate OHLC data
    daily_volatility = 0.015
    high_prices = close_prices * (1 + np.random.uniform(0, daily_volatility, len(dates)))
    low_prices = close_prices * (1 - np.random.uniform(0, daily_volatility, len(dates)))
    open_prices = low_prices + np.random.uniform(0, 1, len(dates)) * (high_prices - low_prices)
    
    # Generate volume data
    volume = np.random.uniform(1000000, 10000000, len(dates))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    # Add indicators if requested
    if include_indicators:
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Bollinger Bands
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma_20'] + (df['std_20'] * 2)
        df['lower_band'] = df['sma_20'] - (df['std_20'] * 2)
        
        # RSI (simplified calculation)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def generate_portfolio_data(tickers: List[str] = None) -> Dict[str, Any]:
    """
    Generate sample portfolio data for demonstration purposes.
    
    Args:
        tickers: List of stock ticker symbols in the portfolio
        
    Returns:
        Dictionary with portfolio data
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    # Generate portfolio weights
    weights = np.random.dirichlet(np.ones(len(tickers)), size=1)[0]
    weights_dict = {ticker: weight for ticker, weight in zip(tickers, weights)}
    
    # Generate sector information
    sectors = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Communication Services",
        "AMZN": "Consumer Discretionary",
        "TSLA": "Consumer Discretionary"
    }
    
    # Generate returns data
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Generate random returns for each stock
    returns_data = {}
    for ticker in tickers:
        np.random.seed(hash(ticker) % 2**32)  # Different seed for each ticker
        returns = np.random.normal(0.0005, 0.015, len(dates))
        returns_data[ticker] = returns
    
    # Create returns DataFrame
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    # Calculate portfolio returns
    portfolio_returns = returns_df.dot(np.array(list(weights_dict.values())))
    
    # Generate benchmark returns (S&P 500 like)
    np.random.seed(123)
    benchmark_returns = pd.Series(
        np.random.normal(0.0004, 0.01, len(dates)),
        index=dates,
        name="S&P 500"
    )
    
    # Calculate portfolio metrics
    annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    max_drawdown = (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min()
    
    metrics = {
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Beta": 0.85  # Simplified
    }
    
    # Risk metrics for risk chart
    risk_metrics = {
        "VaR (95%)": portfolio_returns.quantile(0.05) * np.sqrt(252),
        "CVaR (95%)": portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean() * np.sqrt(252),
        "Sortino Ratio": 1.2,  # Simplified
        "Information Ratio": 0.75  # Simplified
    }
    
    return {
        "weights": weights_dict,
        "sector_info": sectors,
        "returns": returns_df,
        "portfolio_returns": portfolio_returns,
        "benchmark_returns": benchmark_returns,
        "metrics": metrics,
        "risk_metrics": risk_metrics
    }

def get_dashboard_data(selected_ticker: str = "AAPL") -> Dict[str, Any]:
    """
    Get sample data for the dashboard.
    
    Args:
        selected_ticker: Currently selected ticker symbol
        
    Returns:
        Dictionary with all data needed for the dashboard
    """
    # Get portfolio data
    portfolio_data = generate_portfolio_data()
    
    # Get price data for selected ticker
    price_data = generate_stock_data(ticker=selected_ticker)
    
    # Generate correlation matrix data
    tickers = list(portfolio_data["weights"].keys())
    correlation_matrix = pd.DataFrame(
        np.random.uniform(0.3, 0.9, size=(len(tickers), len(tickers))),
        index=tickers,
        columns=tickers
    )
    # Make it symmetric
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    # Set diagonal to 1
    np.fill_diagonal(correlation_matrix.values, 1)
    
    # Generate forecast data
    last_price = price_data["close"].iloc[-1]
    forecast_dates = pd.date_range(
        start=price_data.index[-1] + timedelta(days=1),
        periods=30,
        freq='B'
    )
    
    # Simple forecast (random walk with drift)
    np.random.seed(42)
    forecast_returns = np.random.normal(0.001, 0.012, len(forecast_dates))
    forecast_prices = last_price * np.cumprod(1 + forecast_returns)
    
    forecast_df = pd.DataFrame({
        "forecast": forecast_prices,
        "upper_bound": forecast_prices * (1 + np.linspace(0.01, 0.05, len(forecast_dates))),
        "lower_bound": forecast_prices * (1 - np.linspace(0.01, 0.05, len(forecast_dates)))
    }, index=forecast_dates)
    
    # Forecast metrics
    forecast_metrics = {
        "Expected Return (30d)": (forecast_df["forecast"].iloc[-1] / last_price - 1),
        "Prediction Interval": "±5%",
        "Model Confidence": 0.85,
        "MAPE": 0.032
    }
    
    # Model comparison data
    model_comparison = {
        "LSTM": 0.85,
        "XGBoost": 0.82,
        "ARIMA": 0.78,
        "Ensemble": 0.89
    }
    
    # Combine all data
    dashboard_data = {
        **portfolio_data,
        "price_data": price_data,
        "correlation_matrix": correlation_matrix,
        "forecast_data": forecast_df,
        "forecast_metrics": forecast_metrics,
        "model_comparison": model_comparison,
        "selected_ticker": selected_ticker
    }
    
    return dashboard_data
