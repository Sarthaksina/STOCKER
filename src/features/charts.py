"""Charts module for STOCKER Pro.

This module provides functions for generating chart data for financial analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from src.core.logging import get_logger

logger = get_logger(__name__)


def chart_performance(dates: List[str], price_history_map: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Compute quarterly and yearly returns for price history.
    
    Args:
        dates: List of date strings corresponding to price data points
        price_history_map: Dictionary mapping symbols to price histories
        
    Returns:
        Dictionary with performance chart data
    """
    logger.info(f"Generating performance chart data for {len(price_history_map)} symbols")
    
    # Convert dates to datetime objects
    try:
        date_objects = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in dates]
    except ValueError:
        # Try alternative format
        try:
            date_objects = [datetime.strptime(date_str, "%m/%d/%Y") for date_str in dates]
        except ValueError:
            logger.error("Could not parse date strings")
            return {"error": "Invalid date format"}
    
    # Create DataFrame with dates and prices
    df_dict = {"date": date_objects}
    for symbol, prices in price_history_map.items():
        # Ensure prices list matches dates list length
        if len(prices) != len(dates):
            logger.warning(f"Price history length mismatch for {symbol}")
            continue
            
        df_dict[symbol] = prices
    
    df = pd.DataFrame(df_dict)
    df.set_index("date", inplace=True)
    
    # Calculate quarterly returns
    quarterly_returns = {}
    for symbol in price_history_map.keys():
        if symbol not in df.columns:
            continue
            
        # Resample to quarterly and calculate returns
        quarterly_prices = df[symbol].resample("Q").last()
        symbol_quarterly_returns = quarterly_prices.pct_change().dropna()
        
        # Convert to dictionary with quarter labels
        quarterly_dict = {}
        for date, value in symbol_quarterly_returns.items():
            quarter_label = f"Q{(date.month-1)//3+1} {date.year}"
            quarterly_dict[quarter_label] = value
            
        quarterly_returns[symbol] = quarterly_dict
    
    # Calculate yearly returns
    yearly_returns = {}
    for symbol in price_history_map.keys():
        if symbol not in df.columns:
            continue
            
        # Resample to yearly and calculate returns
        yearly_prices = df[symbol].resample("Y").last()
        symbol_yearly_returns = yearly_prices.pct_change().dropna()
        
        # Convert to dictionary with year labels
        yearly_dict = {}
        for date, value in symbol_yearly_returns.items():
            yearly_dict[str(date.year)] = value
            
        yearly_returns[symbol] = yearly_dict
    
    # Calculate cumulative returns
    cumulative_returns = {}
    for symbol in price_history_map.keys():
        if symbol not in df.columns:
            continue
            
        # Calculate daily returns
        daily_returns = df[symbol].pct_change().fillna(0)
        
        # Calculate cumulative returns
        cumulative = (1 + daily_returns).cumprod() - 1
        
        # Resample to monthly for chart readability
        monthly_cumulative = cumulative.resample("M").last()
        
        # Convert to dictionary with month labels
        cumulative_dict = {}
        for date, value in monthly_cumulative.items():
            month_label = f"{date.year}-{date.month:02d}"
            cumulative_dict[month_label] = value
            
        cumulative_returns[symbol] = cumulative_dict
    
    return {
        "quarterly_returns": quarterly_returns,
        "yearly_returns": yearly_returns,
        "cumulative_returns": cumulative_returns
    }


def chart_correlation_matrix(returns_map: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Generate correlation matrix chart data.
    
    Args:
        returns_map: Dictionary mapping symbols to return histories
        
    Returns:
        Dictionary with correlation matrix data
    """
    # Create DataFrame from returns
    df = pd.DataFrame(returns_map)
    
    # Calculate correlation matrix
    corr_matrix = df.corr().round(2)
    
    # Convert to nested dictionary format for chart rendering
    symbols = list(returns_map.keys())
    matrix_data = []
    
    for i, symbol1 in enumerate(symbols):
        for j, symbol2 in enumerate(symbols):
            matrix_data.append({
                "x": symbol1,
                "y": symbol2,
                "value": corr_matrix.iloc[i, j]
            })
    
    return {
        "symbols": symbols,
        "matrix": matrix_data
    }


def chart_drawdown(prices: List[float], dates: List[str] = None) -> Dict[str, Any]:
    """
    Generate drawdown chart data.
    
    Args:
        prices: List of historical prices
        dates: Optional list of date strings corresponding to prices
        
    Returns:
        Dictionary with drawdown chart data
    """
    # Calculate returns
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0:
            returns.append((prices[i] / prices[i-1]) - 1)
        else:
            returns.append(0)
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(np.array(returns) + 1) - 1
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_returns)
    
    # Calculate drawdown
    drawdown = (cumulative_returns - running_max) / (running_max + 1)
    
    # Prepare result
    result = {
        "max_drawdown": float(np.min(drawdown)),
        "drawdown_series": drawdown.tolist(),
        "cumulative_returns": cumulative_returns.tolist()
    }
    
    # Add dates if provided
    if dates and len(dates) > len(returns):
        result["dates"] = dates[1:len(returns)+1]  # Skip first date and match length
    
    return result
