"""
Portfolio Backtesting Module for STOCKER Pro

This module provides functionality for backtesting portfolio strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta

from stocker.cloud.portfolio_config import PortfolioConfig
from stocker.cloud.portfolio_metrics import calculate_portfolio_metrics

# Configure logging
logger = logging.getLogger(__name__)

def backtest_portfolio(price_data: pd.DataFrame, 
                      weights: np.ndarray, 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None,
                      benchmark_ticker: Optional[str] = None,
                      initial_investment: float = 10000.0,
                      rebalance_frequency: Optional[str] = None,
                      config: Optional[PortfolioConfig] = None) -> Dict[str, Any]:
    """
    Backtest portfolio performance with advanced metrics
    
    Args:
        price_data: DataFrame of asset prices
        weights: Array of asset weights
        start_date: Start date for backtest (if None, use first date in price_data)
        end_date: End date for backtest (if None, use last date in price_data)
        benchmark_ticker: Ticker symbol for benchmark (if None, use config default)
        initial_investment: Initial investment amount
        rebalance_frequency: Frequency to rebalance portfolio ('daily', 'weekly', 'monthly', 'quarterly', 'yearly', None)
        config: Portfolio configuration
        
    Returns:
        Dictionary with backtest results
    """
    # Use default config if none provided
    if config is None:
        config = PortfolioConfig()
    
    # Ensure weights sum to 1
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Filter data by date range
    if start_date is not None:
        price_data = price_data[price_data.index >= start_date]
    if end_date is not None:
        price_data = price_data[price_data.index <= end_date]
    
    # Calculate returns
    returns_data = price_data.pct_change().dropna()
    
    # Get benchmark if provided
    benchmark_returns = None
    if benchmark_ticker is not None and benchmark_ticker in returns_data.columns:
        benchmark_returns = returns_data[benchmark_ticker]
    elif config.benchmark_ticker in returns_data.columns:
        benchmark_returns = returns_data[config.benchmark_ticker]
    
    # Initialize portfolio values
    portfolio_values = pd.Series(index=returns_data.index, dtype=float)
    portfolio_weights = pd.DataFrame(index=returns_data.index, columns=price_data.columns, dtype=float)
    
    # Set initial weights and value
    current_weights = weights.copy()
    current_value = initial_investment
    
    # Set up rebalancing dates if requested
    rebalance_dates = []
    if rebalance_frequency is not None:
        if rebalance_frequency == 'daily':
            rebalance_dates = returns_data.index
        elif rebalance_frequency == 'weekly':
            rebalance_dates = returns_data.index[returns_data.index.weekday == 0]
        elif rebalance_frequency == 'monthly':
            rebalance_dates = returns_data.index[returns_data.index.day == 1]
        elif rebalance_frequency == 'quarterly':
            rebalance_dates = returns_data.index[
                (returns_data.index.month == 1) | 
                (returns_data.index.month == 4) | 
                (returns_data.index.month == 7) | 
                (returns_data.index.month == 10)
            ]
        elif rebalance_frequency == 'yearly':
            rebalance_dates = returns_data.index[
                (returns_data.index.month == 1) & 
                (returns_data.index.day == 1)
            ]
    
    # Run backtest
    for date, returns in returns_data.iterrows():
        # Update portfolio value based on returns
        asset_returns = returns[price_data.columns].values
        daily_return = np.sum(current_weights * asset_returns)
        current_value *= (1 + daily_return)
        
        # Store portfolio value
        portfolio_values[date] = current_value
        
        # Update weights due to price changes
        current_weights = current_weights * (1 + asset_returns)
        current_weights = current_weights / np.sum(current_weights)
        
        # Store current weights
        portfolio_weights.loc[date] = current_weights
        
        # Rebalance if needed
        if rebalance_frequency is not None and date in rebalance_dates:
            current_weights = weights.copy()
    
    # Calculate portfolio returns
    portfolio_returns = portfolio_values.pct_change().dropna()
    
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Calculate drawdowns
    previous_peaks = portfolio_values.cummax()
    drawdowns = (portfolio_values - previous_peaks) / previous_peaks
    
    # Calculate performance metrics
    metrics = calculate_portfolio_metrics(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=config.risk_free_rate
    )
    
    return {
        'portfolio_values': portfolio_values,
        'portfolio_returns': portfolio_returns,
        'portfolio_weights': portfolio_weights,
        'cumulative_returns': cumulative_returns,
        'drawdowns': drawdowns,
        'benchmark_returns': benchmark_returns,
        'benchmark_cumulative': (1 + benchmark_returns).cumprod() if benchmark_returns is not None else None,
        'metrics': metrics,
        'initial_investment': initial_investment,
        'final_value': portfolio_values.iloc[-1] if not portfolio_values.empty else initial_investment
    }

def plot_backtest_results(backtest_results: Dict[str, Any], 
                         title: str = "Portfolio Backtest Results",
                         figsize: Tuple[int, int] = (12, 8),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot backtest results with portfolio value, drawdowns, and benchmark comparison
    
    Args:
        backtest_results: Dictionary with backtest results from backtest_portfolio
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Plot portfolio value
    ax1 = plt.subplot(3, 1, 1)
    backtest_results['portfolio_values'].plot(ax=ax1, label='Portfolio Value')
    ax1.set_title(f"{title} - Portfolio Value")
    ax1.set_ylabel('Value ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot drawdowns
    ax2 = plt.subplot(3, 1, 2)
    backtest_results['drawdowns'].plot(ax=ax2, label='Drawdowns', color='red')
    ax2.set_title('Portfolio Drawdowns')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot cumulative returns (portfolio vs benchmark)
    ax3 = plt.subplot(3, 1, 3)
    backtest_results['cumulative_returns'].plot(ax=ax3, label='Portfolio')
    if backtest_results['benchmark_cumulative'] is not None:
        backtest_results['benchmark_cumulative'].plot(ax=ax3, label='Benchmark')
    ax3.set_title('Cumulative Returns')
    ax3.set_ylabel('Cumulative Return')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def compare_strategies(price_data: pd.DataFrame,
                      strategy_weights: Dict[str, np.ndarray],
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      benchmark_ticker: Optional[str] = None,
                      initial_investment: float = 10000.0,
                      rebalance_frequency: str = 'monthly',
                      config: Optional[PortfolioConfig] = None) -> Dict[str, Any]:
    """
    Compare multiple portfolio strategies
    
    Args:
        price_data: DataFrame of asset prices
        strategy_weights: Dictionary mapping strategy names to weight arrays
        start_date: Start date for backtest
        end_date: End date for backtest
        benchmark_ticker: Ticker symbol for benchmark
        initial_investment: Initial investment amount
        rebalance_frequency: Frequency to rebalance portfolio
        config: Portfolio configuration
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    # Run backtest for each strategy
    for strategy_name, weights in strategy_weights.items():
        logger.info(f"Backtesting strategy: {strategy_name}")
        results[strategy_name] = backtest_portfolio(
            price_data=price_data,
            weights=weights,
            start_date=start_date,
            end_date=end_date,
            benchmark_ticker=benchmark_ticker,
            initial_investment=initial_investment,
            rebalance_frequency=rebalance_frequency,
            config=config
        )
    
    return results

def plot_strategy_comparison(comparison_results: Dict[str, Dict[str, Any]],
                            metric: str = 'cumulative_returns',
                            title: str = "Strategy Comparison",
                            figsize: Tuple[int, int] = (12, 6),
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of multiple strategies
    
    Args:
        comparison_results: Dictionary with results from compare_strategies
        metric: Metric to plot ('cumulative_returns', 'portfolio_values', 'drawdowns')
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for strategy_name, results in comparison_results.items():
        results[metric].plot(ax=ax, label=strategy_name)
    
    # Add benchmark if available
    benchmark_data = next((results['benchmark_cumulative'] for results in comparison_results.values() 
                          if results['benchmark_cumulative'] is not None), None)
    if benchmark_data is not None and metric == 'cumulative_returns':
        benchmark_data.plot(ax=ax, label='Benchmark', linestyle='--')
    
    ax.set_title(f"{title} - {metric.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig