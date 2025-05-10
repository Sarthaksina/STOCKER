"""
Portfolio Facade Module for STOCKER Pro

This module implements the Facade pattern to provide a simplified interface
to the portfolio functionality while still allowing direct access to individual 
components when needed.
"""

import pandas as pd
from typing import Dict, List, Optional, Union, Any

from .core import PortfolioManager
from .analysis import calculate_portfolio_statistics
from .optimizer import optimize_portfolio, EfficientFrontier
from src.core.logging import get_logger

logger = get_logger(__name__)


class PortfolioFacade:
    """
    Facade class that provides a simplified interface to portfolio functionality.
    
    This class follows the Facade design pattern, offering a unified high-level
    interface to the various portfolio subsystems while still allowing direct
    access to those subsystems when needed.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the portfolio facade.
        
        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to None.
        """
        self.portfolio_manager = PortfolioManager(config)
        self.config = config or {}
        logger.info("Portfolio Facade initialized")
    
    def load_data(self, price_data: pd.DataFrame) -> None:
        """
        Load price data into the portfolio system.
        
        Args:
            price_data (pd.DataFrame): DataFrame with asset prices
        """
        self.portfolio_manager.load_data(price_data)
    
    def calculate_metrics(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate portfolio metrics based on returns and weights.
        
        Args:
            returns (pd.DataFrame): DataFrame of asset returns
            weights (Dict[str, float]): Dictionary of asset weights
            
        Returns:
            Dict[str, Any]: Dictionary of portfolio metrics
        """
        return self.portfolio_manager.calculate_metrics(returns, weights)
    
    def optimize_portfolio(self, 
                          returns: pd.DataFrame, 
                          objective: str = 'sharpe',
                          risk_free_rate: float = 0.0,
                          constraints: Optional[List[Dict]] = None,
                          target_return: Optional[float] = None,
                          target_volatility: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize a portfolio based on the specified objective.
        
        Args:
            returns (pd.DataFrame): DataFrame of asset returns with assets as columns
            objective (str, optional): Optimization objective.
                Options: 'sharpe', 'min_volatility', 'max_return', 'target_return', 'target_risk'.
                Defaults to 'sharpe'.
            risk_free_rate (float, optional): Risk-free rate. Defaults to 0.0.
            constraints (List[Dict], optional): List of constraints for optimization.
            target_return (float, optional): Target return for 'target_return' objective.
            target_volatility (float, optional): Target volatility for 'target_risk' objective.
                
        Returns:
            Dict[str, Any]: Optimization results
        """
        return optimize_portfolio(
            returns=returns,
            objective=objective,
            risk_free_rate=risk_free_rate,
            constraints=constraints,
            target_return=target_return,
            target_volatility=target_volatility
        )
    
    def calculate_statistics(self, 
                            returns: pd.DataFrame, 
                            benchmark_returns: Optional[pd.Series] = None,
                            risk_free_rate: float = 0.0,
                            period: str = 'daily') -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio statistics.
        
        Args:
            returns (pd.DataFrame): DataFrame of portfolio returns
            benchmark_returns (pd.Series, optional): Series of benchmark returns.
            risk_free_rate (float, optional): Risk-free rate. Defaults to 0.0.
            period (str, optional): Return period. Defaults to 'daily'.
                Options: 'daily', 'weekly', 'monthly', 'annual'.
                
        Returns:
            Dict[str, Any]: Dictionary containing portfolio statistics
        """
        return calculate_portfolio_statistics(
            returns=returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
            period=period
        )
    
    def recommend_portfolio(self, 
                           returns: pd.DataFrame, 
                           risk_tolerance: str = "moderate") -> Dict[str, Any]:
        """
        Recommend an optimal portfolio based on risk tolerance.
        
        Args:
            returns (pd.DataFrame): DataFrame of asset returns
            risk_tolerance (str, optional): Risk tolerance level ("low", "moderate", "high").
                Defaults to "moderate".
                
        Returns:
            Dict[str, Any]: Dictionary with recommended portfolio
        """
        return self.portfolio_manager.recommend_portfolio(returns, risk_tolerance)
    
    def analyze_exposures(self, 
                         weights: Dict[str, float], 
                         sector_map: Dict[str, str], 
                         asset_class_map: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyzes sector and asset class exposures of the portfolio.
        
        Args:
            weights (Dict[str, float]): Dictionary mapping symbols to weights
            sector_map (Dict[str, str]): Dictionary mapping symbols to sectors
            asset_class_map (Dict[str, str]): Dictionary mapping symbols to asset classes
            
        Returns:
            Dict[str, Any]: Dictionary with exposure analysis results
        """
        return self.portfolio_manager.analyze_exposures(weights, sector_map, asset_class_map)
    
    def get_efficient_frontier(self, 
                             returns: pd.DataFrame, 
                             risk_free_rate: float = 0.0,
                             n_points: int = 50) -> pd.DataFrame:
        """
        Generate the efficient frontier.
        
        Args:
            returns (pd.DataFrame): DataFrame of asset returns with assets as columns
            risk_free_rate (float, optional): Risk-free rate. Defaults to 0.0.
            n_points (int, optional): Number of points on the frontier. Defaults to 50.
                
        Returns:
            pd.DataFrame: DataFrame containing volatility, return, and sharpe ratio
        """
        optimizer = EfficientFrontier(returns=returns, risk_free_rate=risk_free_rate)
        return optimizer.get_efficient_frontier(n_points=n_points)
    
    def self_assess_portfolio(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Self-assess portfolio quality and highlight potential issues.
        
        Args:
            weights (Dict[str, float]): Dictionary of asset weights
            
        Returns:
            Dict[str, Any]: Dictionary with assessment results
        """
        return self.portfolio_manager.self_assess_portfolio(weights)
    
    def rebalance_portfolio(self, 
                           target_weights: Dict[str, float], 
                           current_holdings: Dict[str, float], 
                           total_value: float, 
                           min_trade_size: float = 100) -> Dict[str, Any]:
        """
        Generate a rebalancing plan with optimized trading to minimize costs.
        
        Args:
            target_weights (Dict[str, float]): Target portfolio weights
            current_holdings (Dict[str, float]): Current holdings in dollar value
            total_value (float): Total portfolio value
            min_trade_size (float, optional): Minimum trade size to execute. Defaults to 100.
            
        Returns:
            Dict[str, Any]: Dictionary with rebalancing plan
        """
        return self.portfolio_manager.advanced_rebalance_portfolio(
            target_weights, current_holdings, total_value, min_trade_size
        )