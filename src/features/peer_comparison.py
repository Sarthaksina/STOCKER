"""Peer comparison module for STOCKER Pro.

This module provides functions for comparing stocks to their peers based on
various metrics such as correlation, performance, and financial ratios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import pearsonr

from src.core.logging import get_logger

logger = get_logger(__name__)


def peer_compare(price_history_map: Dict[str, List[float]], target: str, n: int = 5) -> Dict[str, Any]:
    """
    Compare a target symbol to peers by return correlation and other metrics.
    
    Args:
        price_history_map: Dictionary mapping symbols to price histories
        target: Target symbol to compare against peers
        n: Number of top peers to return
        
    Returns:
        Dictionary with peer comparison results
    """
    logger.info(f"Performing peer comparison for {target} with {len(price_history_map)} potential peers")
    
    if target not in price_history_map:
        logger.warning(f"Target symbol {target} not found in price history map")
        return {"error": f"Target symbol {target} not found"}
    
    # Calculate returns for all symbols
    returns_map = {}
    for symbol, prices in price_history_map.items():
        if len(prices) < 2:
            continue
            
        # Calculate daily returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] / prices[i-1]) - 1)
            else:
                returns.append(0)
                
        returns_map[symbol] = returns
    
    # Get target returns
    target_returns = returns_map.get(target, [])
    if not target_returns:
        logger.warning(f"No valid returns calculated for target {target}")
        return {"error": f"No valid returns for target {target}"}
    
    # Calculate correlation with target
    correlations = []
    for symbol, returns in returns_map.items():
        if symbol == target or len(returns) < len(target_returns):
            continue
            
        # Ensure returns are of same length for correlation calculation
        symbol_returns = returns[:len(target_returns)]
        
        try:
            corr, p_value = pearsonr(target_returns, symbol_returns)
            correlations.append((symbol, corr, p_value))
        except Exception as e:
            logger.warning(f"Error calculating correlation for {symbol}: {e}")
    
    # Sort by correlation (absolute value, descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Take top n peers
    top_peers = correlations[:n]
    
    # Calculate additional metrics for top peers
    peer_metrics = []
    for symbol, corr, p_value in top_peers:
        # Calculate basic metrics
        symbol_returns = returns_map[symbol]
        mean_return = np.mean(symbol_returns)
        volatility = np.std(symbol_returns)
        
        # Calculate beta (simplified)
        cov = np.cov(symbol_returns[:len(target_returns)], target_returns)[0, 1]
        target_var = np.var(target_returns)
        beta = cov / target_var if target_var > 0 else 0
        
        peer_metrics.append({
            "symbol": symbol,
            "correlation": corr,
            "p_value": p_value,
            "mean_return": mean_return,
            "volatility": volatility,
            "beta": beta,
            "significance": "High" if p_value < 0.05 else "Medium" if p_value < 0.1 else "Low"
        })
    
    # Calculate target metrics
    target_mean_return = np.mean(target_returns)
    target_volatility = np.std(target_returns)
    
    result = {
        "target": {
            "symbol": target,
            "mean_return": target_mean_return,
            "volatility": target_volatility
        },
        "peers": peer_metrics,
        "peer_count": len(peer_metrics)
    }
    
    logger.info(f"Completed peer comparison for {target}, found {len(peer_metrics)} peers")
    return result


def sector_performance_comparison(returns_by_sector: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Compare performance across different sectors.
    
    Args:
        returns_by_sector: Dictionary mapping sectors to return histories
        
    Returns:
        Dictionary with sector performance comparison
    """
    sector_metrics = {}
    
    for sector, returns in returns_by_sector.items():
        if not returns:
            continue
            
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        sharpe = mean_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = np.cumprod(np.array(returns) + 1) - 1
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (running_max + 1)
        max_drawdown = np.min(drawdown)
        
        sector_metrics[sector] = {
            "mean_return": mean_return,
            "volatility": volatility,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown
        }
    
    # Rank sectors by different metrics
    sectors = list(sector_metrics.keys())
    
    return_rank = sorted(sectors, key=lambda s: sector_metrics[s]["mean_return"], reverse=True)
    sharpe_rank = sorted(sectors, key=lambda s: sector_metrics[s]["sharpe"], reverse=True)
    volatility_rank = sorted(sectors, key=lambda s: sector_metrics[s]["volatility"])
    
    return {
        "sector_metrics": sector_metrics,
        "rankings": {
            "by_return": return_rank,
            "by_sharpe": sharpe_rank,
            "by_volatility": volatility_rank
        }
    }


def find_top_peers(symbol: str, universe: List[str], data_map: Dict[str, pd.DataFrame], n: int = 5) -> List[Dict[str, Any]]:
    """
    Find top peers for a given symbol based on return correlation and other metrics.
    
    Args:
        symbol: Target symbol to find peers for
        universe: List of potential peer symbols to consider
        data_map: Dictionary mapping symbols to DataFrames with price data
        n: Number of top peers to return
        
    Returns:
        List of dictionaries with peer information
    """
    logger.info(f"Finding top {n} peers for {symbol} from universe of {len(universe)} symbols")
    
    if symbol not in data_map:
        logger.warning(f"Target symbol {symbol} not found in data map")
        return []
    
    # Filter universe to symbols that have data
    valid_universe = [s for s in universe if s in data_map and s != symbol]
    
    if not valid_universe:
        logger.warning(f"No valid symbols found in universe for peer comparison")
        return []
    
    # Extract target returns
    target_data = data_map[symbol]
    if 'close' not in target_data.columns or len(target_data) < 2:
        logger.warning(f"Insufficient price data for target {symbol}")
        return []
        
    target_returns = target_data['close'].pct_change().dropna()
    
    # Calculate correlations and other metrics
    peer_metrics = []
    for peer in valid_universe:
        peer_data = data_map[peer]
        
        if 'close' not in peer_data.columns or len(peer_data) < 2:
            continue
            
        peer_returns = peer_data['close'].pct_change().dropna()
        
        # Align return series
        common_index = target_returns.index.intersection(peer_returns.index)
        if len(common_index) < 20:  # Require at least 20 common data points
            continue
            
        aligned_target = target_returns.loc[common_index]
        aligned_peer = peer_returns.loc[common_index]
        
        try:
            # Calculate correlation
            corr, p_value = pearsonr(aligned_target, aligned_peer)
            
            # Calculate beta
            cov = np.cov(aligned_target, aligned_peer)[0, 1]
            target_var = np.var(aligned_target)
            beta = cov / target_var if target_var > 0 else 0
            
            # Calculate return and volatility metrics
            peer_mean_return = np.mean(aligned_peer)
            peer_volatility = np.std(aligned_peer)
            target_mean_return = np.mean(aligned_target)
            target_volatility = np.std(aligned_target)
            
            # Calculate relative size if market cap is available
            relative_size = 1.0  # Default to 1.0 if not available
            if 'market_cap' in peer_data.columns and 'market_cap' in target_data.columns:
                peer_mcap = peer_data['market_cap'].iloc[-1]
                target_mcap = target_data['market_cap'].iloc[-1]
                if target_mcap > 0:
                    relative_size = peer_mcap / target_mcap
            
            # Calculate similarity score (weighted combination of metrics)
            # Higher score means more similar
            corr_weight = 0.5
            beta_weight = 0.3
            size_weight = 0.2
            
            # Transform correlation to 0-1 scale (1 is perfect correlation)
            corr_score = (abs(corr) + 1) / 2
            
            # Transform beta to similarity score (1 when beta = 1, decreasing as beta moves away from 1)
            beta_score = 1 - min(abs(beta - 1), 1)
            
            # Transform size to similarity score (1 when same size, decreasing as size differs)
            size_score = 1 - min(abs(np.log10(relative_size)), 1)
            
            similarity_score = (corr_weight * corr_score) + (beta_weight * beta_score) + (size_weight * size_score)
            
            peer_metrics.append({
                "symbol": peer,
                "correlation": corr,
                "p_value": p_value,
                "beta": beta,
                "mean_return": peer_mean_return,
                "volatility": peer_volatility,
                "relative_size": relative_size,
                "similarity_score": similarity_score,
                "significance": "High" if p_value < 0.05 else "Medium" if p_value < 0.1 else "Low"
            })
            
        except Exception as e:
            logger.warning(f"Error calculating metrics for peer {peer}: {e}")
    
    # Sort by similarity score (descending)
    peer_metrics.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Take top n peers
    top_peers = peer_metrics[:n]
    
    logger.info(f"Found {len(top_peers)} peers for {symbol}")
    return top_peers
