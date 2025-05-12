"""
Portfolio Core Module for STOCKER Pro

This module serves as the central coordination point for all portfolio functionality,
providing a unified interface for portfolio management, analysis, and optimization.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime
import functools
import time
import os
import json
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import from other modules as needed
from src.core.logging import get_logger
from src.core.exceptions import PortfolioError

logger = get_logger(__name__)

# Cache decorator for expensive calculations
def cache_result(max_age_seconds=3600, cache_dir=None):
    """
    Decorator to cache function results to disk
    
    Args:
        max_age_seconds: Maximum age of cache in seconds
        cache_dir: Directory to store cache files
    """
    cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '..', '..', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a unique key based on function name, args, and kwargs
            key_dict = {
                'func': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            key_str = json.dumps(key_dict, sort_keys=True, default=str)
            key_hash = hashlib.md5(key_str.encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{func.__name__}_{key_hash}.pkl")
            
            # Check if cache file exists and is not too old
            if os.path.exists(cache_file):
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < max_age_seconds:
                    try:
                        with open(cache_file, 'rb') as f:
                            result, exec_time = pickle.load(f)
                            logger.debug(f"Cache hit for {func.__name__}, saved {exec_time:.2f}s")
                            return result
                    except Exception as e:
                        logger.warning(f"Error loading cache: {e}")
            
            # Cache miss or expired, call the function
            start_time = time.time()
            result = func(*args, **kwargs)
            exec_time = time.time() - start_time
            
            # Save result to cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump((result, exec_time), f)
            except Exception as e:
                logger.warning(f"Error saving cache: {e}")
            
            return result
        return wrapper
    return decorator

class PortfolioManager:
    """
    Unified portfolio management system that coordinates all portfolio functionality.
    
    This class serves as the main entry point for portfolio operations, providing
    a simplified interface to the various portfolio modules.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the portfolio manager with configuration.
        
        Args:
            config: Portfolio configuration dictionary
        """
        self.config = config or {}
        
        # Initialize component modules
        self.risk_analyzer = None  # Will be initialized on demand
        self.visualizer = None  # Will be initialized on demand
        self.backtester = None  # Will be initialized on demand
        
        # Portfolio state
        self.price_data = None
        self.returns_data = None
        self.current_weights = None
        self.optimization_results = None
        self.backtest_results = None
        
        # Cache for expensive calculations
        self.cache = {}
        
        # Cache settings
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_enabled = True
        self.cache_ttl = 86400  # 24 hours in seconds
        
        # Performance settings
        self.use_parallel = True
        self.max_workers = os.cpu_count() or 4
        
        logger.info("Portfolio Manager initialized with performance optimizations")
    
    def _cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a unique cache key based on function arguments"""
        key_dict = {
            'args': args,
            'kwargs': kwargs
        }
        if hasattr(self, 'price_data') and self.price_data is not None:
            # Include a hash of the data shape and last few values
            key_dict['data_shape'] = self.price_data.shape
            key_dict['data_hash'] = hashlib.md5(
                pd.util.hash_pandas_object(self.price_data.iloc[-5:]).values
            ).hexdigest()
        
        key_str = pickle.dumps(key_dict)
        return f"{prefix}_{hashlib.md5(key_str).hexdigest()}"
    
    def _cache_get(self, key: str) -> Optional[Any]:
        """Retrieve an item from cache if it exists and is not expired"""
        if not self.cache_enabled:
            return None
            
        cache_path = os.path.join(self.cache_dir, key)
        if not os.path.exists(cache_path):
            return None
            
        # Check if cache is expired
        mod_time = os.path.getmtime(cache_path)
        if time.time() - mod_time > self.cache_ttl:
            os.remove(cache_path)
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def _cache_set(self, key: str, value: Any) -> None:
        """Store an item in the cache"""
        if not self.cache_enabled:
            return
            
        cache_path = os.path.join(self.cache_dir, key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def analyze_exposures(self, weights: Dict[str, float], sector_map: Dict[str, str], asset_class_map: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyzes sector and asset class exposures of the portfolio.

        Args:
            weights: Dictionary mapping symbols to weights.
            sector_map: Dictionary mapping symbols to sectors.
            asset_class_map: Dictionary mapping symbols to asset classes.

        Returns:
            Dictionary with exposure analysis results (sector and asset class breakdowns).
        """
        sector_exposure = {}
        asset_class_exposure = {}
        total_weight = sum(weights.values())

        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Portfolio weights do not sum to 1 (sum={total_weight}). Normalizing.")
            if total_weight == 0: 
                return {'sector_exposure': {}, 'asset_class_exposure': {}}  # Avoid division by zero
            weights = {symbol: weight / total_weight for symbol, weight in weights.items()}

        for symbol, weight in weights.items():
            # Sector Exposure
            sector = sector_map.get(symbol, 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight

            # Asset Class Exposure
            asset_class = asset_class_map.get(symbol, 'Unknown')
            asset_class_exposure[asset_class] = asset_class_exposure.get(asset_class, 0) + weight

        # Sort for consistent output
        sorted_sector_exposure = dict(sorted(sector_exposure.items(), key=lambda item: item[1], reverse=True))
        sorted_asset_class_exposure = dict(sorted(asset_class_exposure.items(), key=lambda item: item[1], reverse=True))

        return {
            'sector_exposure': sorted_sector_exposure,
            'asset_class_exposure': sorted_asset_class_exposure
        }
    
    def load_data(self, price_data: pd.DataFrame) -> None:
        """
        Load price data into the portfolio manager.
        
        Args:
            price_data: DataFrame with asset prices
        """
        self.price_data = price_data
        
        # Calculate returns
        self.returns_data = price_data.pct_change().dropna()
        
        # Reset cache
        self.cache = {}
        
        logger.info(f"Loaded price data for {len(price_data.columns)} assets over {len(price_data)} periods")
    
    def calculate_metrics(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate portfolio metrics based on returns and weights.
        
        Args:
            returns: DataFrame of asset returns
            weights: Dictionary of asset weights
            
        Returns:
            Dictionary of portfolio metrics
        """
        # Convert weights dict to array aligned with returns columns
        weight_array = np.array([weights.get(col, 0) for col in returns.columns])
        
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weight_array)
        
        # Calculate annualized metrics (assuming daily returns)
        ann_factor = 252  # Trading days in a year
        
        mean_return = portfolio_returns.mean() * ann_factor
        volatility = portfolio_returns.std() * np.sqrt(ann_factor)
        sharpe = mean_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns / peak - 1)
        max_drawdown = drawdown.min()
        
        metrics = {
            "mean_return": mean_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "total_return": cum_returns.iloc[-1] - 1
        }
        
        return metrics
    
    def recommend_portfolio(self, returns: pd.DataFrame, risk_tolerance: str = "moderate") -> Dict[str, Any]:
        """
        Recommend an optimal portfolio based on risk tolerance.
        
        Args:
            returns: DataFrame of asset returns
            risk_tolerance: Risk tolerance level ("low", "moderate", "high")
            
        Returns:
            Dictionary with recommended portfolio
        """
        # This is a placeholder for the actual implementation
        # In a real implementation, we would use optimization algorithms
        
        # Example logic based on risk tolerance
        if risk_tolerance == "low":
            target_volatility = 0.10  # 10% annual volatility
        elif risk_tolerance == "moderate":
            target_volatility = 0.15  # 15% annual volatility
        else:  # high
            target_volatility = 0.20  # 20% annual volatility
        
        # For demo, generate simple weights (would use optimizer in practice)
        n_assets = len(returns.columns)
        equal_weight = 1.0 / n_assets
        weights = {col: equal_weight for col in returns.columns}
        
        # Calculate metrics for the recommended portfolio
        metrics = self.calculate_metrics(returns, weights)
        
        return {
            "weights": weights,
            "metrics": metrics,
            "risk_tolerance": risk_tolerance
        }
    
    def self_assess_portfolio(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Self-assess portfolio quality and highlight potential issues.
        
        Args:
            weights: Dictionary of asset weights
            
        Returns:
            Dictionary with assessment results
        """
        assessment = {
            "diversification_score": 0,
            "risk_score": 0,
            "liquidity_score": 0,
            "overall_score": 0,
            "recommendations": []
        }
        
        # Check if returns data is available
        if self.returns_data is None:
            assessment["recommendations"].append("Load price data to enable full portfolio assessment")
            return assessment
        
        # Calculate metrics
        metrics = self.calculate_metrics(self.returns_data, weights)
        
        # Assess diversification
        asset_count = sum(1 for w in weights.values() if w > 0.01)
        max_weight = max(weights.values()) if weights else 0
        
        if asset_count < 5:
            assessment["recommendations"].append("Consider adding more assets to improve diversification")
            assessment["diversification_score"] = 3
        elif max_weight > 0.25:
            assessment["recommendations"].append("Portfolio has concentrated positions, consider rebalancing")
            assessment["diversification_score"] = 5
        else:
            assessment["diversification_score"] = 8
        
        # Assess risk
        if metrics["volatility"] > 0.25:
            assessment["recommendations"].append("Portfolio volatility is high, consider reducing risk")
            assessment["risk_score"] = 3
        elif metrics["max_drawdown"] < -0.25:
            assessment["recommendations"].append("Portfolio has high drawdown risk, consider adding hedges")
            assessment["risk_score"] = 4
        else:
            assessment["risk_score"] = 7
        
        # Calculate overall score (simplistic average for demo)
        assessment["overall_score"] = (assessment["diversification_score"] + 
                                       assessment["risk_score"] + 
                                       assessment["liquidity_score"]) / 3
        
        # Add metrics to the assessment
        assessment["metrics"] = metrics
        
        return assessment
    
    def advanced_rebalance_portfolio(self, target_weights: Dict[str, float], current_holdings: Dict[str, float], 
                                    total_value: float, min_trade_size: float = 100) -> Dict[str, Any]:
        """
        Generate a rebalancing plan with optimized trading to minimize costs.
        
        Args:
            target_weights: Target portfolio weights
            current_holdings: Current holdings in dollar value
            total_value: Total portfolio value
            min_trade_size: Minimum trade size to execute
            
        Returns:
            Dictionary with rebalancing plan
        """
        trades = {}
        current_weights = {symbol: value / total_value for symbol, value in current_holdings.items()}
        
        # Calculate ideal dollar amounts
        target_amounts = {symbol: total_value * weight for symbol, weight in target_weights.items()}
        current_amounts = current_holdings.copy()
        
        # Fill in missing values with zeros
        for symbol in set(target_amounts) | set(current_amounts):
            if symbol not in target_amounts:
                target_amounts[symbol] = 0
            if symbol not in current_amounts:
                current_amounts[symbol] = 0
        
        # Calculate trades
        for symbol in target_amounts:
            trade_amount = target_amounts[symbol] - current_amounts[symbol]
            
            # Skip small trades
            if abs(trade_amount) < min_trade_size:
                continue
                
            trades[symbol] = trade_amount
        
        # Calculate metrics before and after
        before_metrics = self.calculate_metrics(self.returns_data, current_weights) if self.returns_data is not None else {}
        after_metrics = self.calculate_metrics(self.returns_data, target_weights) if self.returns_data is not None else {}
        
        return {
            "trades": trades,
            "before_weights": current_weights,
            "after_weights": target_weights,
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "total_value": total_value
        }


def get_portfolio_manager(config: Optional[Dict] = None) -> PortfolioManager:
    """
    Factory function to create a portfolio manager.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PortfolioManager instance
    """
    return PortfolioManager(config)


def recommend_portfolio(returns: pd.DataFrame, risk_tolerance: str = "moderate") -> Dict[str, Any]:
    """
    Standalone function to recommend an optimal portfolio based on risk tolerance.
    
    Args:
        returns: DataFrame of asset returns
        risk_tolerance: Risk tolerance level ("low", "moderate", "high")
        
    Returns:
        Dictionary with recommended portfolio
    """
    manager = get_portfolio_manager()
    return manager.recommend_portfolio(returns, risk_tolerance)


def self_assess_portfolio(returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Standalone function to self-assess portfolio quality.
    
    Args:
        returns: DataFrame of asset returns
        weights: Dictionary of asset weights
        
    Returns:
        Dictionary with assessment results
    """
    manager = get_portfolio_manager()
    manager.load_data(returns)
    return manager.self_assess_portfolio(weights)


def advanced_rebalance_portfolio(returns: pd.DataFrame, target_weights: Dict[str, float], 
                               current_holdings: Dict[str, float], total_value: float, 
                               min_trade_size: float = 100) -> Dict[str, Any]:
    """
    Standalone function to generate a rebalancing plan.
    
    Args:
        returns: DataFrame of asset returns
        target_weights: Target portfolio weights
        current_holdings: Current holdings in dollar value
        total_value: Total portfolio value
        min_trade_size: Minimum trade size to execute
        
    Returns:
        Dictionary with rebalancing plan
    """
    manager = get_portfolio_manager()
    manager.load_data(returns)
    return manager.advanced_rebalance_portfolio(target_weights, current_holdings, total_value, min_trade_size) 


# ===== Portfolio Validation Utilities =====

def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None, min_rows: int = 1) -> bool:
    """
    Validate a DataFrame against requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        True if validation passes
        
    Raises:
        TypeError: If input is not a DataFrame
        ValueError: If DataFrame doesn't meet requirements
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None")
        
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows, got {len(df)}")
        
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")
            
    return True

def validate_series(series: Union[pd.Series, List, np.ndarray], min_length: int = 1) -> pd.Series:
    """
    Validate and convert input to pandas Series.
    
    Args:
        series: Input to validate and convert
        min_length: Minimum length required
        
    Returns:
        Validated pandas Series
        
    Raises:
        TypeError: If input cannot be converted to Series
        ValueError: If Series doesn't meet requirements
    """
    if series is None:
        raise ValueError("Input Series cannot be None")
        
    # Convert to pandas Series if not already
    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series)
        except Exception as e:
            raise TypeError(f"Could not convert to pandas Series: {e}")
    
    if not isinstance(series, pd.Series):
        raise TypeError(f"Expected pandas Series, got {type(series).__name__}")
        
    if len(series) < min_length:
        raise ValueError(f"Series must have at least {min_length} elements, got {len(series)}")
        
    return series

def validate_weights(weights: Union[Dict, List, np.ndarray], assets: List[str]) -> np.ndarray:
    """
    Validate and normalize portfolio weights.
    
    Args:
        weights: Portfolio weights as dict, list, or array
        assets: List of asset names
        
    Returns:
        Normalized weights as numpy array
        
    Raises:
        TypeError: If weights cannot be converted to array
        ValueError: If weights don't match assets or contain invalid values
    """
    # Handle dictionary of weights
    if isinstance(weights, dict):
        # Check for missing assets
        missing_assets = [asset for asset in assets if asset not in weights]
        if missing_assets:
            raise ValueError(f"Missing weights for assets: {missing_assets}")
            
        # Convert to array in the same order as assets
        weights_array = np.array([weights[asset] for asset in assets])
    else:
        # Convert to numpy array if not already
        try:
            weights_array = np.array(weights)
        except Exception as e:
            raise TypeError(f"Could not convert weights to numpy array: {e}")
    
    # Check for NaN or infinite values
    if np.any(~np.isfinite(weights_array)):
        raise ValueError("Weights contain NaN or infinite values")
        
    # Check if number of weights matches number of assets
    if len(weights_array) != len(assets):
        raise ValueError(f"Number of weights ({len(weights_array)}) doesn't match number of assets ({len(assets)})")
        
    # Normalize weights to sum to 1
    weights_sum = np.sum(weights_array)
    if weights_sum > 0:
        weights_array = weights_array / weights_sum
        
    return weights_array


@cache_result(max_age_seconds=86400)  # Cache for 24 hours
def suggest_high_quality_stocks(config: Any, stock_data_map: Dict[str, Any] = None, sector_map: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """
    Suggest high-quality stocks based on fundamental and technical analysis.
    
    This function identifies stocks with strong fundamentals, positive technical indicators,
    and good growth prospects. It's useful for portfolio construction and investment ideas.
    
    Args:
        config: Configuration object with settings and symbols list
        stock_data_map: Optional dictionary mapping symbols to stock data
        sector_map: Optional dictionary mapping symbols to sectors
        
    Returns:
        List of dictionaries containing stock information and quality metrics
    """
    logger.info("Generating high-quality stock suggestions")
    
    # Use provided data or fetch new data
    if not stock_data_map:
        stock_data_map = {}
        
    if not sector_map:
        sector_map = {}
    
    # Get symbols from config if available, otherwise use a default list
    symbols = getattr(config, 'symbols', [])
    if not symbols:
        # Default to a list of stable blue-chip stocks if no symbols provided
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG', 'UNH']
    
    # Define quality metrics and thresholds
    quality_metrics = {
        'min_market_cap': 1e9,  # $1 billion minimum
        'min_dividend_yield': 0.01,  # 1% minimum dividend yield
        'max_pe_ratio': 30,  # Maximum P/E ratio
        'min_roe': 0.15,  # 15% minimum return on equity
        'max_debt_to_equity': 1.5,  # Maximum debt-to-equity ratio
        'min_current_ratio': 1.2,  # Minimum current ratio
    }
    
    # Placeholder for results
    high_quality_stocks = []
    
    # In a real implementation, we would:
    # 1. Fetch fundamental data for each symbol
    # 2. Calculate quality scores based on metrics
    # 3. Sort and filter stocks by quality score
    
    # For demonstration purposes, we'll create simulated results
    for i, symbol in enumerate(symbols[:10]):  # Limit to 10 stocks for demo
        # In a real implementation, these would be actual calculated values
        quality_score = 70 + (i % 3) * 10  # Scores between 70-90
        
        # Create a stock entry with quality metrics
        stock_entry = {
            'symbol': symbol,
            'name': f"Company {symbol}",
            'sector': sector_map.get(symbol, 'Technology'),
            'quality_score': quality_score,
            'market_cap': (1 + i) * 1e9,  # Simulated market cap
            'pe_ratio': 15 + i,
            'dividend_yield': 0.01 + (i % 5) * 0.005,
            'roe': 0.15 + (i % 5) * 0.03,
            'recommendation': 'Strong Buy' if quality_score > 85 else 'Buy' if quality_score > 75 else 'Hold'
        }
        
        high_quality_stocks.append(stock_entry)
    
    # Sort by quality score (descending)
    high_quality_stocks.sort(key=lambda x: x['quality_score'], reverse=True)
    
    logger.info(f"Found {len(high_quality_stocks)} high-quality stocks")
    return high_quality_stocks