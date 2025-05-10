"""
Portfolio performance analysis.

This module provides functions for calculating portfolio performance metrics
such as Sharpe ratio, maximum drawdown, and Value at Risk.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

from src.core.exceptions import PortfolioAnalysisError
from src.core.logging import get_logger

logger = get_logger(__name__)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    period: str = 'daily'
) -> float:
    """
    Calculate the Sharpe ratio of a return series.
    
    Args:
        returns (pd.Series): Series of returns.
        risk_free_rate (float, optional): Risk-free rate. Defaults to 0.0.
        period (str, optional): Return period. Defaults to 'daily'.
            Options: 'daily', 'weekly', 'monthly', 'annual'.
            
    Returns:
        float: Sharpe ratio.
        
    Raises:
        PortfolioAnalysisError: If period is invalid.
    """
    # Set annualization factor based on period
    if period == 'daily':
        factor = 252
    elif period == 'weekly':
        factor = 52
    elif period == 'monthly':
        factor = 12
    elif period == 'annual':
        factor = 1
    else:
        raise PortfolioAnalysisError(f"Invalid period: {period}")
    
    excess_returns = returns - risk_free_rate
    
    # Annualize the Sharpe ratio
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(factor)
    
    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    period: str = 'daily',
    target_return: float = 0.0
) -> float:
    """
    Calculate the Sortino ratio of a return series.
    
    The Sortino ratio is similar to Sharpe but uses only downside risk.
    
    Args:
        returns (pd.Series): Series of returns.
        risk_free_rate (float, optional): Risk-free rate. Defaults to 0.0.
        period (str, optional): Return period. Defaults to 'daily'.
            Options: 'daily', 'weekly', 'monthly', 'annual'.
        target_return (float, optional): Minimum acceptable return. Defaults to 0.0.
            
    Returns:
        float: Sortino ratio.
        
    Raises:
        PortfolioAnalysisError: If period is invalid.
    """
    # Set annualization factor based on period
    if period == 'daily':
        factor = 252
    elif period == 'weekly':
        factor = 52
    elif period == 'monthly':
        factor = 12
    elif period == 'annual':
        factor = 1
    else:
        raise PortfolioAnalysisError(f"Invalid period: {period}")
    
    excess_returns = returns - risk_free_rate
    
    # Calculate downside deviation (only negative excess returns)
    downside_returns = excess_returns[excess_returns < target_return]
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0
    
    # Avoid division by zero
    if downside_deviation == 0:
        return 0
    
    # Annualize the Sortino ratio
    sortino = excess_returns.mean() / downside_deviation * np.sqrt(factor)
    
    return sortino


def calculate_max_drawdown(returns: pd.Series) -> Dict:
    """
    Calculate the maximum drawdown of a return series.
    
    Args:
        returns (pd.Series): Series of returns.
        
    Returns:
        Dict: Dictionary containing drawdown metrics.
            max_drawdown: Maximum drawdown as a positive percentage
            drawdown_start: Start date of the maximum drawdown
            drawdown_end: End date of the maximum drawdown
            drawdown_recovery: Recovery date or None if not recovered
            drawdown_duration: Duration of the drawdown in days
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cum_returns / running_max - 1)
    
    # Find maximum drawdown point
    max_drawdown = drawdown.min()
    max_drawdown_idx = drawdown.idxmin()
    
    # Find when the drawdown began
    temp = running_max.loc[:max_drawdown_idx]
    drawdown_start = temp.idxmax()
    
    # Find when the drawdown ended (first instance of recovery to previous peak)
    post_drawdown = cum_returns.loc[max_drawdown_idx:]
    
    try:
        recovery_idx = post_drawdown[post_drawdown >= running_max.loc[drawdown_start]].index[0]
        recovered = True
        recovery_date = recovery_idx
        drawdown_duration = (recovery_idx - drawdown_start).days
    except (IndexError, KeyError):
        recovered = False
        recovery_date = None
        drawdown_duration = (post_drawdown.index[-1] - drawdown_start).days
    
    result = {
        'max_drawdown': abs(max_drawdown),
        'drawdown_start': drawdown_start,
        'drawdown_end': max_drawdown_idx,
        'drawdown_recovery': recovery_date,
        'drawdown_duration': drawdown_duration,
        'recovered': recovered
    }
    
    return result


def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate Value at Risk (VaR) for a return series.
    
    Args:
        returns (pd.Series): Series of returns.
        confidence_level (float, optional): Confidence level for VaR. Defaults to 0.95.
        method (str, optional): Method to calculate VaR. Defaults to 'historical'.
            Options: 'historical', 'gaussian', 'cornish_fisher'.
            
    Returns:
        float: Value at Risk (as a positive percentage).
        
    Raises:
        PortfolioAnalysisError: If method is invalid.
    """
    if method == 'historical':
        # Historical VaR is simply the quantile of the return distribution
        var = -returns.quantile(1 - confidence_level)
        
    elif method == 'gaussian':
        # Parametric VaR assuming Gaussian distribution
        z_score = np.abs(stats.norm.ppf(1 - confidence_level))
        var = -(returns.mean() - z_score * returns.std())
        
    elif method == 'cornish_fisher':
        # Cornish-Fisher VaR that adjusts for skewness and kurtosis
        z_score = np.abs(stats.norm.ppf(1 - confidence_level))
        s = stats.skew(returns)
        k = stats.kurtosis(returns)
        
        # Cornish-Fisher adjustment to z-score
        z_cf = (z_score +
                (z_score**2 - 1) * s / 6 +
                (z_score**3 - 3 * z_score) * k / 24 -
                (2 * z_score**3 - 5 * z_score) * s**2 / 36)
        
        var = -(returns.mean() - z_cf * returns.std())
        
    else:
        raise PortfolioAnalysisError(f"Invalid method: {method}")
    
    return var


def calculate_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) for a return series.
    
    CVaR (also known as Expected Shortfall) represents the expected loss
    given that the loss exceeds VaR.
    
    Args:
        returns (pd.Series): Series of returns.
        confidence_level (float, optional): Confidence level for CVaR. Defaults to 0.95.
            
    Returns:
        float: Conditional Value at Risk (as a positive percentage).
    """
    # Calculate VaR
    var = calculate_var(returns, confidence_level, method='historical')
    
    # Calculate CVaR as the mean of returns below VaR
    cvar = -returns[returns <= -var].mean()
    
    return cvar


def calculate_portfolio_statistics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    period: str = 'daily'
) -> Dict:
    """
    Calculate comprehensive portfolio statistics.
    
    Args:
        returns (pd.Series): Series of portfolio returns.
        benchmark_returns (pd.Series, optional): Series of benchmark returns.
        risk_free_rate (float, optional): Risk-free rate. Defaults to 0.0.
        period (str, optional): Return period. Defaults to 'daily'.
            Options: 'daily', 'weekly', 'monthly', 'annual'.
            
    Returns:
        Dict: Dictionary containing portfolio statistics.
    """
    # Set annualization factor based on period
    if period == 'daily':
        factor = 252
    elif period == 'weekly':
        factor = 52
    elif period == 'monthly':
        factor = 12
    elif period == 'annual':
        factor = 1
    else:
        raise PortfolioAnalysisError(f"Invalid period: {period}")
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod() - 1
    
    # Annualized return
    total_return = cum_returns.iloc[-1]
    n_periods = len(returns)
    ann_return = (1 + total_return)**(factor / n_periods) - 1
    
    # Annualized volatility
    ann_vol = returns.std() * np.sqrt(factor)
    
    # Drawdown metrics
    drawdown_stats = calculate_max_drawdown(returns)
    
    # Sharpe and Sortino ratios
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, period)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, period)
    
    # Downside risk
    downside_returns = returns[returns < 0]
    downside_risk = downside_returns.std() * np.sqrt(factor)
    
    # Risk metrics
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    cvar_95 = calculate_cvar(returns, 0.95)
    
    # Positive/negative returns
    positive_periods = (returns > 0).sum() / len(returns)
    negative_periods = (returns < 0).sum() / len(returns)
    
    # Return to risk ratio
    return_risk_ratio = ann_return / ann_vol if ann_vol != 0 else 0
    
    # Best/worst periods
    best_period = returns.max()
    worst_period = returns.min()
    
    # Skewness and kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # Benchmark comparison
    benchmark_stats = {}
    if benchmark_returns is not None:
        # Make sure returns are aligned
        aligned_returns = returns.copy()
        aligned_benchmark = benchmark_returns.copy()
        
        # Filter for common dates
        common_idx = aligned_returns.index.intersection(aligned_benchmark.index)
        aligned_returns = aligned_returns.loc[common_idx]
        aligned_benchmark = aligned_benchmark.loc[common_idx]
        
        # Calculate beta
        cov = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_var = np.var(aligned_benchmark)
        beta = cov / benchmark_var if benchmark_var != 0 else 0
        
        # Calculate alpha (Jensen's alpha)
        benchmark_cum_returns = (1 + aligned_benchmark).cumprod() - 1
        benchmark_total_return = benchmark_cum_returns.iloc[-1]
        benchmark_ann_return = (1 + benchmark_total_return)**(factor / len(aligned_benchmark)) - 1
        
        alpha = ann_return - (risk_free_rate + beta * (benchmark_ann_return - risk_free_rate))
        
        # Calculate information ratio
        active_returns = aligned_returns - aligned_benchmark
        information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(factor)
        
        # Calculate tracking error
        tracking_error = active_returns.std() * np.sqrt(factor)
        
        # Calculate R-squared
        correlation = np.corrcoef(aligned_returns, aligned_benchmark)[0, 1]
        r_squared = correlation**2
        
        # Calculate Treynor ratio
        treynor = (ann_return - risk_free_rate) / beta if beta != 0 else 0
        
        benchmark_stats = {
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'r_squared': r_squared,
            'treynor_ratio': treynor,
            'correlation': correlation
        }
    
    # Combine all statistics
    stats = {
        'total_return': total_return,
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': drawdown_stats['max_drawdown'],
        'drawdown_start': drawdown_stats['drawdown_start'],
        'drawdown_end': drawdown_stats['drawdown_end'],
        'drawdown_recovery': drawdown_stats['drawdown_recovery'],
        'drawdown_duration': drawdown_stats['drawdown_duration'],
        'downside_risk': downside_risk,
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'positive_periods_pct': positive_periods,
        'negative_periods_pct': negative_periods,
        'return_risk_ratio': return_risk_ratio,
        'best_period': best_period,
        'worst_period': worst_period,
        'skewness': skewness,
        'kurtosis': kurtosis
    }
    
    # Add benchmark stats if available
    if benchmark_stats:
        stats.update(benchmark_stats)
    
    return stats 