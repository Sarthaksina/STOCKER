"""Analysis module for STOCKER Pro.

This module provides financial analysis functions for stocks and portfolios.
It serves as a compatibility layer that imports from the analytics module.
"""

# Import functions from analytics.py for backward compatibility
from src.features.analytics import (
    analyze_returns,
    analyze_volatility,
    analyze_seasonality,
    analyze_trend,
    detect_outliers,
    calculate_correlation_matrix,
    detect_anomalies,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_beta,
    calculate_alpha,
    calculate_var,
    calculate_cvar
)

# Aliases for backward compatibility
def performance_analysis(prices):
    """Analyze performance of a price series."""
    return analyze_returns(prices)

def sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio for a return series."""
    return calculate_sharpe_ratio(returns, risk_free_rate)

def valuation_metrics(stock_data):
    """Calculate valuation metrics for a stock."""
    # This is a placeholder implementation
    pe_ratio = stock_data.get('price', 0) / max(stock_data.get('eps', 1), 0.01)
    pb_ratio = stock_data.get('price', 0) / max(stock_data.get('book_value', 1), 0.01)
    
    return {
        'pe_ratio': pe_ratio,
        'pb_ratio': pb_ratio,
        'dividend_yield': stock_data.get('dividend_yield', 0),
        'roe': stock_data.get('roe', 0),
        'roa': stock_data.get('roa', 0)
    }

def alpha_beta(prices, benchmark):
    """Calculate alpha and beta for a stock relative to a benchmark."""
    alpha = calculate_alpha(prices, benchmark)
    beta = calculate_beta(prices, benchmark)
    
    return {
        'alpha': alpha,
        'beta': beta
    }

def attribution_analysis(portfolio, returns):
    """Perform attribution analysis on a portfolio."""
    # This is a placeholder implementation
    total_return = sum(portfolio[symbol] * returns.get(symbol, 0) for symbol in portfolio)
    
    attribution = {}
    for symbol in portfolio:
        weight = portfolio[symbol]
        symbol_return = returns.get(symbol, 0)
        contribution = weight * symbol_return
        attribution[symbol] = {
            'weight': weight,
            'return': symbol_return,
            'contribution': contribution,
            'attribution': contribution / total_return if total_return else 0
        }
    
    return {
        'total_return': total_return,
        'attribution': attribution
    }

def momentum_analysis(prices):
    """Calculate momentum for a price series."""
    # Simple momentum calculation (current price / price n periods ago - 1)
    if len(prices) < 20:
        return 0
    
    current = prices[-1]
    previous = prices[-20]  # 20-day momentum
    
    return (current / previous - 1) if previous > 0 else 0
