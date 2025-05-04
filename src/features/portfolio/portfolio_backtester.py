"""
Portfolio Backtesting Module for STOCKER Pro

This module provides comprehensive backtesting capabilities for portfolio strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime, timedelta

from stocker.cloud.portfolio_config import PortfolioConfig
from stocker.cloud.portfolio_metrics import calculate_portfolio_metrics
from stocker.cloud.portfolio_risk import PortfolioRiskAnalyzer
from stocker.cloud.portfolio_optimization import optimize_portfolio

# Configure logging
logger = logging.getLogger(__name__)

class PortfolioBacktester:
    """
    Comprehensive portfolio backtesting framework
    """
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        Initialize portfolio backtester
        
        Args:
            config: Portfolio configuration
        """
        self.config = config or PortfolioConfig()
        self.risk_analyzer = PortfolioRiskAnalyzer(config=self.config)
        
    def backtest_strategy(self, 
                         price_data: pd.DataFrame,
                         strategy_func: Callable,
                         initial_capital: float = 10000.0,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         benchmark_ticker: Optional[str] = None,
                         rebalance_frequency: str = 'monthly',
                         transaction_cost: float = 0.001) -> Dict[str, Any]:
        """
        Backtest a portfolio strategy
        
        Args:
            price_data: DataFrame of asset prices
            strategy_func: Function that takes price_data and returns weights
            initial_capital: Initial capital
            start_date: Start date for backtest
            end_date: End date for backtest
            benchmark_ticker: Ticker for benchmark comparison
            rebalance_frequency: Frequency to rebalance ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
            transaction_cost: Transaction cost as percentage
            
        Returns:
            Dictionary with backtest results
        """
        # Filter data by date range
        if start_date:
            price_data = price_data[price_data.index >= start_date]
        if end_date:
            price_data = price_data[price_data.index <= end_date]
            
        # Get benchmark if provided
        benchmark_returns = None
        if benchmark_ticker and benchmark_ticker in price_data.columns:
            benchmark_prices = price_data[benchmark_ticker]
            benchmark_returns = benchmark_prices.pct_change().dropna()
            
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Set up rebalance dates
        if rebalance_frequency == 'daily':
            rebalance_dates = returns.index
        elif rebalance_frequency == 'weekly':
            rebalance_dates = [date for date in returns.index if date.weekday() == 0]  # Monday
        elif rebalance_frequency == 'monthly':
            rebalance_dates = [date for date in returns.index if date.day == 1]
        elif rebalance_frequency == 'quarterly':
            rebalance_dates = [date for date in returns.index if date.month in [1, 4, 7, 10] and date.day == 1]
        elif rebalance_frequency == 'yearly':
            rebalance_dates = [date for date in returns.index if date.month == 1 and date.day == 1]
        else:
            raise ValueError(f"Invalid rebalance frequency: {rebalance_frequency}")
            
        # Add first date as a rebalance date if not already included
        if returns.index[0] not in rebalance_dates:
            rebalance_dates = [returns.index[0]] + rebalance_dates
            
        # Initialize portfolio tracking
        portfolio_values = []
        portfolio_weights = []
        portfolio_returns = []
        transaction_costs = []
        current_weights = None
        current_shares = None
        
        # Run backtest
        for i, date in enumerate(returns.index):
            # Get current prices
            current_prices = price_data.loc[date]
            
            # Rebalance on rebalance dates
            if date in rebalance_dates:
                # Get historical data up to this point for strategy
                historical_data = price_data.loc[:date]
                
                # Get new weights from strategy
                new_weights = strategy_func(historical_data)
                
                # Calculate transaction costs if not first rebalance
                cost = 0
                if current_weights is not None:
                    # Calculate cost based on weight changes
                    weight_changes = np.abs(new_weights - current_weights)
                    cost = np.sum(weight_changes) * transaction_cost
                    transaction_costs.append(cost)
                else:
                    transaction_costs.append(0)
                    
                # Update weights and calculate shares
                current_weights = new_weights
                
                # Calculate portfolio value (or use initial capital on first date)
                if len(portfolio_values) > 0:
                    current_value = portfolio_values[-1] * (1 - cost)
                else:
                    current_value = initial_capital
                    
                # Calculate shares based on new weights
                current_shares = current_value * current_weights / current_prices
                
                # Store weights
                portfolio_weights.append(current_weights)
            else:
                # No rebalancing, just track costs
                transaction_costs.append(0)
                
            # Calculate current portfolio value
            if current_shares is not None:
                current_value = np.sum(current_shares * current_prices)
                portfolio_values.append(current_value)
                
                # Calculate return
                if i > 0:
                    daily_return = (current_value / portfolio_values[-2]) - 1
                    portfolio_returns.append(daily_return)
                else:
                    portfolio_returns.append(0)
                    
        # Convert to Series/DataFrame
        portfolio_values = pd.Series(portfolio_values, index=returns.index)
        portfolio_returns = pd.Series(portfolio_returns, index=returns.index)
        transaction_costs = pd.Series(transaction_costs, index=returns.index)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate drawdowns
        previous_peaks = portfolio_values.cummax()
        drawdowns = (portfolio_values - previous_peaks) / previous_peaks
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(portfolio_returns, benchmark_returns)
        
        # Calculate risk metrics
        risk_metrics = self.risk_analyzer.calculate_risk_metrics(portfolio_returns)
        
        # Run stress tests
        stress_test_results = self.run_stress_tests(portfolio_returns, current_weights)
        
        return {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'drawdowns': drawdowns,
            'transaction_costs': transaction_costs,
            'final_weights': current_weights,
            'metrics': metrics,
            'risk_metrics': risk_metrics,
            'stress_tests': stress_test_results,
            'benchmark_returns': benchmark_returns
        }
    
    def _calculate_performance_metrics(self, 
                                     portfolio_returns: pd.Series,
                                     benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            portfolio_returns: Series of portfolio returns
            benchmark_returns: Optional series of benchmark returns
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate basic metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility != 0 else 0
        
        # Calculate downside deviation
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
        
        # Calculate maximum drawdown
        wealth_index = (1 + portfolio_returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        max_drawdown = drawdowns.min()
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Calculate benchmark-related metrics if benchmark available
        benchmark_metrics = {}
        if benchmark_returns is not None:
            # Calculate benchmark metrics
            benchmark_total_return = (1 + benchmark_returns).prod() - 1
            benchmark_annualized = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1
            benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
            
            # Calculate tracking error
            tracking_diff = portfolio_returns - benchmark_returns
            tracking_error = tracking_diff.std() * np.sqrt(252)
            
            # Calculate information ratio
            information_ratio = tracking_diff.mean() / tracking_diff.std() * np.sqrt(252) if tracking_diff.std() != 0 else 0
            
            # Calculate beta
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            
            # Calculate alpha (Jensen's alpha)
            alpha = annualized_return - (self.config.risk_free_rate + beta * (benchmark_annualized - self.config.risk_free_rate))
            
            benchmark_metrics = {
                'benchmark_return': benchmark_total_return,
                'benchmark_annualized': benchmark_annualized,
                'benchmark_volatility': benchmark_volatility,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha
            }
        
        # Combine all metrics
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio
        }
        
        # Add benchmark metrics if available
        if benchmark_metrics:
            metrics.update(benchmark_metrics)
            
        return metrics
    
    def run_stress_tests(self,
                        returns: pd.Series,
                        weights: np.ndarray) -> Dict[str, Any]:
        """
        Run stress tests on portfolio
        
        Args:
            returns: Series of portfolio returns
            weights: Array of asset weights
            
        Returns:
            Dictionary with stress test results
        """
        # Run predefined stress scenarios
        results = {}
        for scenario, params in self.config.stress_scenarios.items():
            # Run stress test for this scenario
            scenario_result = self.run_stress_test(returns, weights, scenario=scenario)
            results[scenario] = scenario_result
            
        return results
    
    def run_stress_test(self,
                       returns: pd.Series,
                       weights: np.ndarray,
                       scenario: Optional[str] = None,
                       custom_scenario: Optional[Dict[str, float]] = None,
                       num_simulations: int = 1000,
                       confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Run stress test on portfolio
        
        Args:
            returns: Series of portfolio returns
            weights: Array of asset weights
            scenario: Predefined scenario name (from config)
            custom_scenario: Custom scenario definition
            num_simulations: Number of Monte Carlo simulations
            confidence_level: Confidence level for VaR and CVaR
            
        Returns:
            Dictionary with stress test results
        """
        # Get scenario parameters
        scenario_params = None
        if scenario and scenario in self.config.stress_scenarios:
            scenario_params = self.config.stress_scenarios[scenario]
        elif custom_scenario:
            scenario_params = custom_scenario
        else:
            # Default to market crash scenario
            scenario_params = self.config.stress_scenarios.get('market_crash', {
                'market': -0.30,
                'volatility': 2.0
            })
        
        # Calculate normal metrics
        normal_metrics = calculate_portfolio_metrics(pd.DataFrame(returns), weights, self.config)
        
        # Apply scenario adjustments
        adjusted_returns = returns.copy()
        
        # Apply market shock if specified
        if 'market' in scenario_params:
            market_shock = scenario_params['market']
            # Apply market shock to all returns
            adjusted_returns = adjusted_returns + market_shock / 252  # Daily shock
        
        # Apply volatility adjustment if specified
        if 'volatility' in scenario_params:
            volatility_multiplier = scenario_params['volatility']
            # Calculate mean returns
            mean_returns = adjusted_returns.mean()
            # Center returns around zero
            centered_returns = adjusted_returns - mean_returns
            # Scale volatility
            scaled_returns = centered_returns * volatility_multiplier
            # Add back mean
            adjusted_returns = scaled_returns + mean_returns
        
        # Calculate stressed metrics
        stressed_metrics = calculate_portfolio_metrics(pd.DataFrame(adjusted_returns), weights, self.config)
        
        # Run Monte Carlo simulation
        np.random.seed(42)  # For reproducibility
        
        # Calculate portfolio mean and covariance
        portfolio_mean = adjusted_returns.mean() * 252
        portfolio_std = adjusted_returns.std() * np.sqrt(252)
        
        # Generate random portfolio returns
        simulated_returns = np.random.normal(
            portfolio_mean,
            portfolio_std,
            num_simulations
        )
        
        # Calculate VaR and CVaR
        var_threshold = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        cvar = simulated_returns[simulated_returns <= var_threshold].mean()
        
        # Calculate probability of negative return
        prob_negative = np.mean(simulated_returns < 0)
        
        # Calculate probability of return below -10%
        prob_below_10pct = np.mean(simulated_returns < -0.10)
        
        return {
            'scenario': scenario or 'custom',
            'scenario_params': scenario_params,
            'normal_metrics': normal_metrics,
            'stressed_metrics': stressed_metrics,
            'impact': {
                'return_impact': stressed_metrics['expected_return'] - normal_metrics['expected_return'],
                'volatility_impact': stressed_metrics['volatility'] - normal_metrics['volatility'],
                'var_impact': stressed_metrics['var_95'] - normal_metrics['var_95'],
                'sharpe_impact': stressed_metrics['sharpe_ratio'] - normal_metrics['sharpe_ratio']
            }
        }
    
    def compare_strategies(self,
                          price_data: pd.DataFrame,
                          strategy_funcs: Dict[str, Callable],
                          initial_capital: float = 10000.0,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          benchmark_ticker: Optional[str] = None,
                          rebalance_frequency: str = 'monthly') -> Dict[str, Any]:
        """
        Compare multiple portfolio strategies
        
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
        # Run backtest for each strategy
        results = {}
        for name, strategy_func in strategy_funcs.items():
            results[name] = self.backtest_strategy(
                price_data=price_data,
                strategy_func=strategy_func,
                initial_capital=initial_capital,
                start_date=start_date,
                end_date=end_date,
                benchmark_ticker=benchmark_ticker,
                rebalance_frequency=rebalance_frequency
            )
        
        # Calculate comparison metrics
        comparison = {
            'strategies': list(strategy_funcs.keys()),
            'metrics': {},
            'returns': {},
            'values': {},
            'drawdowns': {}
        }
        
        # Extract key metrics for comparison
        for name, result in results.items():
            comparison['metrics'][name] = result['metrics']
            comparison['returns'][name] = result['portfolio_returns']
            comparison['values'][name] = result['portfolio_values']
            comparison['drawdowns'][name] = result['drawdowns']
        
        # Add benchmark if available
        if benchmark_ticker and benchmark_ticker in price_data.columns:
            benchmark_prices = price_data[benchmark_ticker]
            benchmark_returns = benchmark_prices.pct_change().dropna()
            benchmark_values = initial_capital * (1 + benchmark_returns).cumprod()
            
            comparison['benchmark_returns'] = benchmark_returns
            comparison['benchmark_values'] = benchmark_values
        
        return comparison
    
    def plot_backtest_results(self,
                             backtest_results: Dict[str, Any],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot backtest results
        
        Args:
            backtest_results: Dictionary with backtest results
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot portfolio value
        portfolio_values = backtest_results['portfolio_values']
        axs[0].plot(portfolio_values.index, portfolio_values, label='Portfolio')
        
        # Plot benchmark if available
        if 'benchmark_returns' in backtest_results and backtest_results['benchmark_returns'] is not None:
            benchmark_returns = backtest_results['benchmark_returns']
            benchmark_values = portfolio_values.iloc[0] * (1 + benchmark_returns).cumprod()
            axs[0].plot(benchmark_values.index, benchmark_values, label='Benchmark', linestyle='--')
        
        axs[0].set_title('Portfolio Value')
        axs[0].set_ylabel('Value')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot drawdowns
        drawdowns = backtest_results['drawdowns']
        axs[1].fill_between(drawdowns.index, 0, drawdowns, color='red', alpha=0.3)
        axs[1].set_title('Drawdowns')
        axs[1].set_ylabel('Drawdown')
        axs[1].grid(True)
        
        # Plot returns
        returns = backtest_results['portfolio_returns']
        axs[2].bar(returns.index, returns, color='blue', alpha=0.5)
        axs[2].set_title('Daily Returns')
        axs[2].set_ylabel('Return')
        axs[2].set_xlabel('Date')
        axs[2].grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_strategy_comparison(self,
                               comparison_results: Dict[str, Any],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot strategy comparison
        
        Args:
            comparison_results: Dictionary with comparison results
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Create figure with subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot portfolio values
        for strategy, values in comparison_results['values'].items():
            axs[0].plot(values.index, values, label=strategy)
        
        # Plot benchmark if available
        if 'benchmark_values' in comparison_results:
            benchmark_values = comparison_results['benchmark_values']
            axs[0].plot(benchmark_values.index, benchmark_values, label='Benchmark', linestyle='--', color='black')
        
        axs[0].set_title('Portfolio Value Comparison')
        axs[0].set_ylabel('Value')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot drawdowns
        for strategy, drawdowns in comparison_results['drawdowns'].items():
            axs[1].plot(drawdowns.index, drawdowns, label=strategy)
        
        axs[1].set_title('Drawdown Comparison')
        axs[1].set_ylabel('Drawdown')
        axs[1].set_xlabel('Date')
        axs[1].legend()
        axs[1].grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig


# --- Deprecated functions from portfolio_backtest.py for backward compatibility ---

# Issue deprecation warning
warnings.warn(
    "The portfolio_backtest module is deprecated. Please use PortfolioBacktester from portfolio_backtester instead.",
    DeprecationWarning,
    stacklevel=2
)

def backtest_portfolio(price_data: pd.DataFrame, 
                      weights: np.ndarray, 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None,
                      benchmark_ticker: Optional[str] = None,
                      initial_investment: float = 10000.0,
                      rebalance_frequency: Optional[str] = None,
                      config: Optional[PortfolioConfig] = None) -> Dict[str, Any]:
    """
    DEPRECATED: Use PortfolioBacktester.backtest_strategy instead.
    
    Backtest portfolio performance with advanced metrics
    
    Args:
        price_data: DataFrame of asset prices
        weights: Array of asset weights
        start_date: Start date for backtest (if None, use first date in price_data)
        end_date: End date for backtest (if None, use last date in price_data)
        benchmark_ticker: Ticker symbol for benchmark (if None, use config default)
        initial_investment: Initial investment amount
        rebalance_frequency: Frequency to rebalance portfolio ('daily', 'weekly', 'monthly', 'quarterly', 'yearly', None)
        config: Portfolio configuration
        
    Returns:
        Dictionary with backtest results
    """
    warnings.warn(
        "This function is deprecated. Please use PortfolioBacktester.backtest_strategy instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create a strategy function that always returns the same weights
    def constant_weight_strategy(historical_data):
        return weights
    
    # Use the PortfolioBacktester class instead
    backtester = PortfolioBacktester(config=config)
    results = backtester.backtest_strategy(
        price_data=price_data,
        strategy_func=constant_weight_strategy,
        initial_capital=initial_investment,
        start_date=start_date,
        end_date=end_date,
        benchmark_ticker=benchmark_ticker,
        rebalance_frequency=rebalance_frequency or 'monthly'
    )
    
    return results

def plot_backtest_results(backtest_results: Dict[str, Any], 
                         title: str = "Portfolio Backtest Results",
                         figsize: Tuple[int, int] = (12, 8),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    DEPRECATED: Use PortfolioBacktester.plot_backtest_results instead.
    
    Plot backtest results with portfolio value, drawdowns, and benchmark comparison
    
    Args:
        backtest_results: Dictionary with backtest results from backtest_portfolio
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    warnings.warn(
        "This function is deprecated. Please use PortfolioBacktester.plot_backtest_results instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create a backtester instance and use its plotting method
    backtester = PortfolioBacktester()
    return backtester.plot_backtest_results(
        backtest_results=backtest_results,
        save_path=save_path
    )

def compare_strategies(price_data: pd.DataFrame,
                      strategy_weights: Dict[str, np.ndarray],
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      benchmark_ticker: Optional[str] = None,
                      initial_investment: float = 10000.0,
                      rebalance_frequency: str = 'monthly',
                      config: Optional[PortfolioConfig] = None) -> Dict[str, Any]:
    """
    DEPRECATED: Use PortfolioBacktester.compare_strategies instead.
    
    Compare multiple portfolio strategies
    
    Args:
        price_data: DataFrame of asset prices
        strategy_weights: Dictionary mapping strategy names to weight arrays
        start_date: Start date for backtest
        end_date: End date for backtest
        benchmark_ticker: Ticker symbol for benchmark
        initial_investment: Initial investment amount
        rebalance_frequency: Frequency to rebalance portfolio
        config: Portfolio configuration
        
    Returns:
        Dictionary with comparison results
    """
    warnings.warn(
        "This function is deprecated. Please use PortfolioBacktester.compare_strategies instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create a backtester instance
    backtester = PortfolioBacktester(config=config)
    
    # Convert strategy weights to strategy functions
    strategy_funcs = {}
    for name, weights in strategy_weights.items():
        strategy_funcs[name] = lambda data, w=weights: w
    
    # Use the backtester's compare_strategies method
    return backtester.compare_strategies(
        price_data=price_data,
        strategy_funcs=strategy_funcs,
        initial_capital=initial_investment,
        start_date=start_date,
        end_date=end_date,
        benchmark_ticker=benchmark_ticker,
        rebalance_frequency=rebalance_frequency
    )