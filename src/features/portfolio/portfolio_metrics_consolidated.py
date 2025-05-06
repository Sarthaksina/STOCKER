"""Portfolio Metrics Consolidated Module for STOCKER Pro

This module provides a consolidated set of functions for calculating portfolio performance metrics.
It combines functionality previously spread across multiple files including:
- portfolio_metrics.py
- analytics_agents.py
- financial_analysis.py

Functions:
    calculate_portfolio_metrics: Calculate comprehensive portfolio metrics
    calculate_rolling_metrics: Calculate rolling portfolio metrics
    peer_compare: Compare target symbol to peers by return correlation
    sharpe_ratio: Calculate Sharpe ratio for a returns series
    valuation_metrics: Extract valuation metrics from stock data
    sentiment_agg: Aggregate sentiment from news sources
    alpha_beta: Calculate alpha and beta relative to benchmark
    attribution_analysis: Decompose portfolio return into asset contributions
    momentum_analysis: Calculate momentum score based on price history
    performance_analysis: Calculate performance metrics from price series
    chart_performance: Compute quarterly and yearly returns
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, List, Any
import feedparser

from src.constant.constants import GOOGLE_NEWS_RSS
from src.llm_utils import analyze_sentiment
from src.db import get_collection
from src.features.portfolio.portfolio_config import PortfolioConfig

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

def peer_compare(price_history_map: Dict[str, List[float]], target: str, n: int = 5) -> Dict[str, Any]:
    """
    Compare the target symbol to peers by return correlation.
    Returns top-n peers with correlation values.
    
    Args:
        price_history_map: Dictionary mapping symbols to price histories
        target: Target symbol
        n: Number of peers to return
        
    Returns:
        Dictionary with target and peers
    """
    df = pd.DataFrame(price_history_map).dropna()
    if target not in df.columns:
        logger.warning(f"Target '{target}' not found in price history map for peer comparison.")
        return {"error": f"Target '{target}' not in price history map"}
    
    returns = df.pct_change().dropna()
    if returns.empty or target not in returns.columns:
        logger.warning(f"Could not calculate returns or target '{target}' has no returns.")
        return {"error": "Could not calculate returns for peer comparison"}
        
    target_ret = returns[target]
    corrs = returns.corrwith(target_ret).drop(target, errors='ignore') # Use errors='ignore' if target might not be in index
    
    # Handle cases where correlation calculation might fail or result in NaNs
    corrs = corrs.dropna()
    
    if corrs.empty:
        logger.warning(f"No valid correlations found for target '{target}'.")
        return {"target": target, "peers": []}
        
    top = corrs.nlargest(n)
    
    return {
        "target": target,
        "peers": [{"symbol": sym, "correlation": float(corr)} for sym, corr in top.items()]
    }


def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.04) -> float:
    """
    Returns the Sharpe ratio of returns series.
    Moved from analytics_agents.py
    """
    if not returns or len(returns) < 2:
        return 0.0
    excess = np.array(returns) - risk_free_rate / 12 # Assuming monthly returns if list
    std_dev = np.std(excess)
    if std_dev < 1e-8:
        return 0.0 # Avoid division by zero
    # Assuming monthly returns, annualize
    return float(np.mean(excess) / std_dev * np.sqrt(12))

def valuation_metrics(stock_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Returns PE, PB, dividend yield, and market cap from fundamentals.
    Moved from analytics_agents.py
    """
    return {
        "pe": stock_data.get("pe", np.nan),
        "pb": stock_data.get("pb", np.nan),
        "div_yield": stock_data.get("div_yield", np.nan),
        "market_cap": stock_data.get("market_cap", np.nan)
    }

def sentiment_agg(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregates sentiment from Google News RSS feed for a given query.
    Moved from financial_analysis.py
    """
    query = params.get('query', 'stock market')
    num_articles = params.get('num_articles', 10)
    feed_url = f"{GOOGLE_NEWS_RSS}?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(feed_url)
    sentiments = []
    news_collection = get_collection('news_sentiment')

    for entry in feed.entries[:num_articles]:
        title = entry.title
        link = entry.link
        # Check if sentiment already exists
        existing = news_collection.find_one({"link": link})
        if existing:
            sentiment = existing['sentiment']
        else:
            try:
                sentiment = analyze_sentiment(title)
                # Store new sentiment
                news_collection.update_one(
                    {"link": link},
                    {"$set": {"title": title, "sentiment": sentiment, "query": query}},
                    upsert=True
                )
            except Exception as e:
                logger.error(f"Error analyzing sentiment for '{title}': {e}")
                sentiment = None # Handle potential errors during sentiment analysis
        
        if sentiment: # Only append if sentiment analysis was successful
            sentiments.append(sentiment)

    # Basic aggregation: average score
    avg_sentiment = np.mean([s['score'] for s in sentiments if s]) if sentiments else 0
    # Count positive/negative/neutral based on label
    pos_count = sum(1 for s in sentiments if s and s['label'] == 'POSITIVE')
    neg_count = sum(1 for s in sentiments if s and s['label'] == 'NEGATIVE')
    neu_count = sum(1 for s in sentiments if s and s['label'] == 'NEUTRAL')

    return {
        "average_sentiment_score": float(avg_sentiment),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "total_articles_analyzed": len(sentiments)
    }

def alpha_beta(prices: List[float], benchmark: List[float]) -> Dict[str, float]:
    """
    Computes alpha and beta relative to benchmark using price lists.
    Moved from analytics_agents.py
    """
    if len(prices) < 2 or len(benchmark) < 2 or len(prices) != len(benchmark):
        return {"alpha": np.nan, "beta": np.nan}
    
    returns = np.diff(prices) / prices[:-1]
    bench_ret = np.diff(benchmark) / benchmark[:-1]
    
    # Ensure lengths match after diff
    min_len = min(len(returns), len(bench_ret))
    returns = returns[:min_len]
    bench_ret = bench_ret[:min_len]
    
    if min_len < 2:
         return {"alpha": np.nan, "beta": np.nan}
         
    cov = np.cov(returns, bench_ret)[0][1]
    denom = np.var(bench_ret)
    
    if denom < 1e-8:
        beta = 1.0 # Or np.nan, depending on desired handling
        alpha = np.mean(returns) - beta * np.mean(bench_ret) if min_len > 0 else 0.0
    else:
        beta = cov / denom
        alpha = np.mean(returns) - beta * np.mean(bench_ret)
        
    # Assuming monthly data, annualize alpha
    alpha = alpha * 12
    
    return {"alpha": float(alpha), "beta": float(beta)}

def attribution_analysis(portfolio: Dict[str, float], returns: Dict[str, float]) -> Dict[str, float]:
    """
    Decomposes portfolio return into asset contributions.
    Moved from analytics_agents.py
    Args:
        portfolio: Dict of asset weights {symbol: weight}
        returns: Dict of asset returns {symbol: return_value}
    Returns:
        Dict of asset contributions {symbol: contribution}
    """
    contrib = {symbol: portfolio.get(symbol, 0) * returns.get(symbol, 0) 
               for symbol in set(portfolio.keys()) | set(returns.keys())}
    total_return = sum(contrib.values())
    # Return contribution as percentage of total portfolio return
    return {symbol: value / total_return if total_return else 0 
            for symbol, value in contrib.items()}

def momentum_analysis(prices: List[float], window: int = 6) -> float:
    """
    Simple momentum: last price / n-period average.
    Moved from analytics_agents.py
    Args:
        prices: List of prices.
        window: Lookback window (default 6 periods).
    Returns:
        Momentum score.
    """
    if len(prices) < window or window <= 0:
        return np.nan
    avg_price = np.mean(prices[-window:])
    if avg_price < 1e-8:
        return np.nan # Avoid division by zero
    return float(prices[-1] / avg_price)


def performance_analysis(prices: List[float]) -> Dict[str, float]:
    """
    Returns total return, annualized return, and volatility from price series.
    Moved from financial_analysis.py
    """
    if not prices or len(prices) < 2:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0
        }
    returns = np.diff(prices) / prices[:-1]
    total_return = (prices[-1] / prices[0]) - 1
    # Avoid division by zero or invalid power for annualized return
    num_periods = len(returns)
    if num_periods > 0:
        # Assuming monthly data if periods < 252, else daily
        ann_factor = 12 if num_periods < 252 else 252
        ann_factor = ann_factor / num_periods # Adjust factor based on actual periods
        # Ensure base is positive before exponentiation
        base = 1 + total_return
        if base > 0:
             ann_return = base ** ann_factor - 1
        else:
             # Handle negative base case, perhaps return NaN or specific error value
             ann_return = np.nan # Or some other indicator of invalid calculation
    else:
        ann_return = 0.0

    # Assuming daily returns for volatility calculation if enough data
    vol_ann_factor = 252 if num_periods >= 252 else 12
    volatility = np.std(returns) * np.sqrt(vol_ann_factor) if num_periods > 0 else 0.0
    return {
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "volatility": float(volatility)
    }

def chart_performance(dates: List[str], price_history_map: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Compute quarterly and yearly returns for given price series.
    
    Args:
        dates: List of date strings (ISO format)
        price_history_map: Dictionary mapping symbols to price histories
        
    Returns:
        Dictionary with quarterly and yearly returns
    """
    try:
        df = pd.DataFrame(price_history_map)
        df.index = pd.to_datetime(dates)
        df = df.sort_index()
    except Exception as e:
        logger.error(f"Error creating DataFrame for chart_performance: {e}")
        return {'error': 'Failed to process input data'}
        
    if df.empty:
        logger.warning("Input DataFrame for chart_performance is empty.")
        return {'quarterly': {}, 'yearly': {}}

    # Ensure no duplicate indices before resampling
    df = df[~df.index.duplicated(keep='first')]

    # Quarterly returns
    try:
        first_q = df.resample('Q').first()
        last_q = df.resample('Q').last()
        # Avoid division by zero or NaN results
        q_ret = (last_q.div(first_q.replace(0, np.nan)) - 1).dropna(how='all')
        quarterly = {date.strftime('%Y-%m-%d'): returns.to_dict() for date, returns in q_ret.iterrows()}
    except Exception as e:
        logger.error(f"Error calculating quarterly returns: {e}")
        quarterly = {}
    
    # Yearly returns
    try:
        first_y = df.resample('A').first()
        last_y = df.resample('A').last()
        # Avoid division by zero or NaN results
        y_ret = (last_y.div(first_y.replace(0, np.nan)) - 1).dropna(how='all')
        yearly = {date.strftime('%Y-%m-%d'): returns.to_dict() for date, returns in y_ret.iterrows()}
    except Exception as e:
        logger.error(f"Error calculating yearly returns: {e}")
        yearly = {}
    
    return {'quarterly': quarterly, 'yearly': yearly}