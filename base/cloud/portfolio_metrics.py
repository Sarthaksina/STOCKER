"""
Portfolio Metrics Module for STOCKER Pro

This module provides functions for calculating portfolio performance metrics.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional

from stocker.cloud.portfolio_config import PortfolioConfig

# Configure logging
logger = logging.getLogger(__name__)

def calculate_portfolio_metrics(returns: pd.DataFrame, 
                               weights: np.ndarray,
                               config: Optional[PortfolioConfig] = None,
                               benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio metrics
    
    Args:
        returns: DataFrame of asset returns
        weights: Array of asset weights
        config: Portfolio configuration
        benchmark_returns: Optional benchmark returns for relative metrics
        
    Returns:
        Dictionary of portfolio metrics
    """
    # Use default config if none provided
    if config is None:
        config = PortfolioConfig()
    
    # Ensure weights sum to 1
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Calculate expected return (annualized)
    expected_return = np.sum(returns.mean() * weights) * 252
    
    # Calculate portfolio volatility (annualized)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    # Calculate Sharpe ratio
    sharpe_ratio = (expected_return - config.risk_free_rate) / portfolio_volatility
    
    # Calculate Sortino ratio (downside risk)
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0
    downside_risk = np.sqrt(np.dot(weights.T, np.dot(downside_returns.cov() * 252, weights)))
    sortino_ratio = (expected_return - config.risk_free_rate) / downside_risk if downside_risk != 0 else 0
    
    # Calculate maximum drawdown
    portfolio_returns = returns.dot(weights)
    wealth_index = (1 + portfolio_returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    max_drawdown = drawdowns.min()
    
    # Calculate Value at Risk (VaR)
    var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
    
    # Calculate Conditional VaR (CVaR) / Expected Shortfall
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean() * np.sqrt(252)
    
    # Calculate information ratio (if benchmark available)
    information_ratio = 0
    if benchmark_returns is not None:
        active_returns = portfolio_returns - benchmark_returns
        information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252)
    
    # Calculate beta (if benchmark available)
    beta = 0
    if benchmark_returns is not None:
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
    
    # Calculate Treynor ratio (if beta available)
    treynor_ratio = (expected_return - config.risk_free_rate) / beta if beta != 0 else 0
    
    return {
        'expected_return': expected_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'information_ratio': information_ratio,
        'beta': beta,
        'treynor_ratio': treynor_ratio
    }

def calculate_rolling_metrics(returns: pd.DataFrame,
                             weights: np.ndarray,
                             window: int = 252,
                             config: Optional[PortfolioConfig] = None) -> Dict[str, pd.Series]:
    """
    Calculate rolling portfolio metrics over a specified window
    
    Args:
        returns: DataFrame of asset returns
        weights: Array of asset weights
        window: Rolling window size in days
        config: Portfolio configuration
        
    Returns:
        Dictionary of rolling metrics as pandas Series
    """
    # Use default config if none provided
    if config is None:
        config = PortfolioConfig()
    
    # Calculate portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Initialize result dictionary
    rolling_metrics = {}
    
    # Calculate rolling volatility (annualized)
    rolling_metrics['volatility'] = portfolio_returns.rolling(window=window).std() * np.sqrt(252)
    
    # Calculate rolling returns (annualized)
    rolling_metrics['returns'] = portfolio_returns.rolling(window=window).mean() * 252
    
    # Calculate rolling Sharpe ratio
    rolling_metrics['sharpe_ratio'] = (rolling_metrics['returns'] - config.risk_free_rate) / rolling_metrics['volatility']
    
    # Calculate rolling maximum drawdown
    rolling_wealth = (1 + portfolio_returns).rolling(window=window).apply(
        lambda x: (1 + x).cumprod().min() / (1 + x).cumprod().max() - 1,
        raw=True
    )
    rolling_metrics['max_drawdown'] = rolling_wealth
    
    # Calculate rolling downside deviation
    downside_returns = portfolio_returns.copy()
    downside_returns[downside_returns > 0] = 0
    rolling_metrics['downside_risk'] = downside_returns.rolling(window=window).std() * np.sqrt(252)
    
    # Calculate rolling Sortino ratio
    rolling_metrics['sortino_ratio'] = (rolling_metrics['returns'] - config.risk_free_rate) / rolling_metrics['downside_risk']
    
    return rolling_metrics