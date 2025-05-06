"""Portfolio Facade Module

This module implements the Facade pattern to provide a simplified interface
to the portfolio module's functionality. It serves as a single entry point
for accessing the most commonly used portfolio functions.

Classes:
    PortfolioFacade: Main facade class that provides access to portfolio functionality

Usage:
    from src.features.portfolio.portfolio_facade import PortfolioFacade
    
    # Create a facade instance
    portfolio = PortfolioFacade()
    
    # Use the facade to access portfolio functionality
    metrics = portfolio.calculate_metrics(returns, weights)
    optimized_weights = portfolio.optimize(returns)
    backtest_results = portfolio.backtest(price_data, strategy_func)
"""

from typing import Dict, List, Optional, Any, Callable, Union
import numpy as np
import pandas as pd

from .portfolio_config import PortfolioConfig
from .portfolio_metrics_consolidated import (
    calculate_portfolio_metrics,
    calculate_rolling_metrics,
    peer_compare,
    sharpe_ratio,
    alpha_beta,
    attribution_analysis,
    momentum_analysis,
    performance_analysis,
    chart_performance,
    valuation_metrics,
    sentiment_agg
)
from .portfolio_risk import PortfolioRiskAnalyzer
from .portfolio_backtester import PortfolioBacktester
from .portfolio_optimization import optimize_portfolio


class PortfolioFacade:
    """Facade class for portfolio functionality
    
    This class provides a simplified interface to the portfolio module's functionality.
    It encapsulates the complexity of the underlying components and provides a clean,
    high-level API for common portfolio operations.
    """
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        """Initialize the portfolio facade
        
        Args:
            config: Optional portfolio configuration
        """
        self.config = config or PortfolioConfig()
        self.risk_analyzer = PortfolioRiskAnalyzer(config=self.config)
        self.backtester = PortfolioBacktester(config=self.config)
        
    # Metrics methods
    def calculate_metrics(self, returns: pd.DataFrame, weights: np.ndarray, 
                         benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate portfolio metrics
        
        Args:
            returns: DataFrame of asset returns
            weights: Array of asset weights
            benchmark_returns: Optional benchmark returns
            
        Returns:
            Dictionary of portfolio metrics
        """
        return calculate_portfolio_metrics(returns, weights, self.config, benchmark_returns)
    
    def calculate_rolling_metrics(self, returns: pd.DataFrame, weights: np.ndarray, 
                                window: int = 252) -> Dict[str, pd.Series]:
        """Calculate rolling portfolio metrics
        
        Args:
            returns: DataFrame of asset returns
            weights: Array of asset weights
            window: Rolling window size
            
        Returns:
            Dictionary of rolling metrics
        """
        return calculate_rolling_metrics(returns, weights, window, self.config)
    
    def compare_peers(self, price_history_map: Dict[str, List[float]], target: str, n: int = 5) -> Dict[str, Any]:
        """Compare target to peers
        
        Args:
            price_history_map: Dictionary mapping symbols to price histories
            target: Target symbol
            n: Number of peers to return
            
        Returns:
            Dictionary with peer comparison results
        """
        return peer_compare(price_history_map, target, n)
    
    # Risk methods
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics
        
        Args:
            returns: Series of portfolio returns
            
        Returns:
            Dictionary of risk metrics
        """
        return self.risk_analyzer.calculate_risk_metrics(returns)
    
    def run_stress_test(self, returns: pd.Series, weights: np.ndarray, 
                       scenario: Optional[str] = None) -> Dict[str, Any]:
        """Run stress test
        
        Args:
            returns: Series of portfolio returns
            weights: Array of asset weights
            scenario: Stress scenario name
            
        Returns:
            Dictionary with stress test results
        """
        return self.backtester.run_stress_test(returns, weights, scenario)
    
    # Optimization methods
    def optimize(self, returns: pd.DataFrame, method: str = 'efficient_frontier', 
                target_return: Optional[float] = None, risk_aversion: float = 1.0) -> np.ndarray:
        """Optimize portfolio weights
        
        Args:
            returns: DataFrame of asset returns
            method: Optimization method
            target_return: Target portfolio return
            risk_aversion: Risk aversion parameter
            
        Returns:
            Array of optimized weights
        """
        return optimize_portfolio(returns, method, target_return, risk_aversion, self.config)
    
    # Backtesting methods
    def backtest(self, price_data: pd.DataFrame, strategy_func: Callable, 
                initial_capital: float = 10000.0, start_date: Optional[str] = None, 
                end_date: Optional[str] = None, benchmark_ticker: Optional[str] = None, 
                rebalance_frequency: str = 'monthly') -> Dict[str, Any]:
        """Backtest a portfolio strategy
        
        Args:
            price_data: DataFrame of asset prices
            strategy_func: Function that takes price_data and returns weights
            initial_capital: Initial capital
            start_date: Start date for backtest
            end_date: End date for backtest
            benchmark_ticker: Ticker for benchmark comparison
            rebalance_frequency: Frequency to rebalance
            
        Returns:
            Dictionary with backtest results
        """
        return self.backtester.backtest_strategy(
            price_data=price_data,
            strategy_func=strategy_func,
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
            benchmark_ticker=benchmark_ticker,
            rebalance_frequency=rebalance_frequency
        )
    
    def compare_strategies(self, price_data: pd.DataFrame, strategy_funcs: Dict[str, Callable], 
                          initial_capital: float = 10000.0, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None, benchmark_ticker: Optional[str] = None, 
                          rebalance_frequency: str = 'monthly') -> Dict[str, Any]:
        """Compare multiple portfolio strategies
        
        Args:
            price_data: DataFrame of asset prices
            strategy_funcs: Dictionary mapping strategy names to strategy functions
            initial_capital: Initial capital
            start_date: Start date for backtest
            end_date: End date for backtest
            benchmark_ticker: Ticker for benchmark comparison
            rebalance_frequency: Frequency to rebalance
            
        Returns:
            Dictionary with comparison results
        """
        return self.backtester.compare_strategies(
            price_data=price_data,
            strategy_funcs=strategy_funcs,
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
            benchmark_ticker=benchmark_ticker,
            rebalance_frequency=rebalance_frequency
        )
    
    # Analysis methods
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """Calculate Sharpe ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.config.risk_free_rate
        return sharpe_ratio(returns, risk_free_rate)
    
    def calculate_alpha_beta(self, returns: pd.Series, benchmark_returns: pd.Series, 
                           risk_free_rate: Optional[float] = None) -> Dict[str, float]:
        """Calculate alpha and beta
        
        Args:
            returns: Series of returns
            benchmark_returns: Series of benchmark returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary with alpha and beta
        """
        if risk_free_rate is None:
            risk_free_rate = self.config.risk_free_rate
        return alpha_beta(returns, benchmark_returns, risk_free_rate)
    
    def analyze_performance(self, price_series: pd.Series) -> Dict[str, Any]:
        """Analyze performance of a price series
        
        Args:
            price_series: Series of prices
            
        Returns:
            Dictionary with performance metrics
        """
        return performance_analysis(price_series)
    
    def chart_performance(self, returns: pd.Series) -> Dict[str, pd.Series]:
        """Chart performance by period
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with performance by period
        """
        return chart_performance(returns)