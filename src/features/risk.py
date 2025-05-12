"""Risk analysis module for STOCKER Pro.

This module provides risk assessment functions for financial analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple

from src.core.logging import get_logger
from src.features.analytics import (
    calculate_var,
    calculate_cvar,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    analyze_volatility
)

logger = get_logger(__name__)


def var_historical(returns: List[float], confidence: float = 0.05) -> Dict[str, float]:
    """
    Calculate Value at Risk (VaR) using historical method.
    
    Args:
        returns: List of historical returns
        confidence: Confidence level (default: 0.05 for 95% confidence)
        
    Returns:
        Dictionary with VaR and related metrics
    """
    var_value = calculate_var(returns, confidence_level=1-confidence)
    cvar_value = calculate_cvar(returns, confidence_level=1-confidence)
    
    return {
        "var": var_value,
        "cvar": cvar_value,
        "confidence": 1-confidence,
        "period": "daily"
    }


def max_drawdown(prices: List[float]) -> Dict[str, float]:
    """
    Calculate maximum drawdown for a price series.
    
    Args:
        prices: List of historical prices
        
    Returns:
        Dictionary with drawdown metrics
    """
    drawdown_result = calculate_max_drawdown(prices)
    
    # If calculate_max_drawdown returns a dictionary, use it directly
    if isinstance(drawdown_result, dict):
        return drawdown_result
    
    # Otherwise, create a dictionary with the result
    return {
        "max_drawdown": drawdown_result,
        "max_drawdown_pct": drawdown_result * 100
    }


def rolling_sharpe(returns: List[float], window: int = 12, risk_free_rate: float = 0.0) -> Dict[str, List[float]]:
    """
    Calculate rolling Sharpe ratio for a return series.
    
    Args:
        returns: List of historical returns
        window: Rolling window size
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary with rolling Sharpe ratios
    """
    # Convert to pandas Series for easier calculation
    returns_series = pd.Series(returns)
    
    # Calculate rolling mean and standard deviation
    rolling_mean = returns_series.rolling(window=window).mean()
    rolling_std = returns_series.rolling(window=window).std()
    
    # Calculate rolling Sharpe ratio
    rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std
    
    # Replace NaN values with 0
    rolling_sharpe = rolling_sharpe.fillna(0).tolist()
    
    return {
        "rolling_sharpe": rolling_sharpe,
        "window": window
    }


def stress_test(returns: List[float], scenarios: Dict[str, float]) -> Dict[str, float]:
    """
    Perform stress testing on a portfolio.
    
    Args:
        returns: Historical returns
        scenarios: Dictionary mapping scenario names to stress factors
        
    Returns:
        Dictionary with stress test results
    """
    results = {}
    
    for scenario_name, stress_factor in scenarios.items():
        # Apply stress factor to returns
        stressed_returns = [r * stress_factor for r in returns]
        
        # Calculate key metrics under stress
        var = calculate_var(stressed_returns)
        sharpe = calculate_sharpe_ratio(stressed_returns)
        volatility = analyze_volatility(stressed_returns)
        
        results[scenario_name] = {
            "var": var,
            "sharpe": sharpe,
            "volatility": volatility
        }
    
    return results


def risk_contribution(weights: List[float], cov_matrix: List[List[float]]) -> List[float]:
    """
    Calculate risk contribution of each asset in a portfolio.
    
    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix
        
    Returns:
        List of risk contributions
    """
    # Convert inputs to numpy arrays
    weights_array = np.array(weights)
    cov_matrix_array = np.array(cov_matrix)
    
    # Calculate portfolio volatility
    portfolio_vol = np.sqrt(weights_array.T @ cov_matrix_array @ weights_array)
    
    # Calculate marginal risk contribution
    marginal_contrib = cov_matrix_array @ weights_array
    
    # Calculate risk contribution
    risk_contrib = weights_array * marginal_contrib / portfolio_vol
    
    return risk_contrib.tolist()
