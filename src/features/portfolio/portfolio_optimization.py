"""
Portfolio Optimization Module for STOCKER Pro

This module provides optimized implementations of computationally intensive portfolio operations.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import requests
import json
import hashlib
import pickle

# Update import to use relative path
from .portfolio_config import PortfolioConfig

# Configure logging
logger = logging.getLogger(__name__)

# --- RL Optimizer Dependencies ---
try:
    import gym
    from gym import spaces
    from stable_baselines3 import PPO, A2C, SAC
    HAS_RL_DEPS = True
except ImportError:
    logger.warning("RL dependencies not found. RL optimization will not be available.")
    HAS_RL_DEPS = False

# --- AWS Dependencies ---
try:
    import boto3
    HAS_AWS_DEPS = True
except ImportError:
    logger.warning("AWS dependencies not found. Cloud optimization will not be available.")
    HAS_AWS_DEPS = False

class PerformanceOptimizer:
    """
    Performance optimization for computationally intensive portfolio operations
    """
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        Initialize performance optimizer
        
        Args:
            config: Portfolio configuration
        """
        self.config = config or PortfolioConfig()
        self.num_cores = mp.cpu_count()
        logger.info(f"Performance optimizer initialized with {self.num_cores} cores available")
    
    @lru_cache(maxsize=32)
    def cached_covariance(self, returns_key: str) -> np.ndarray:
        """
        Cached computation of covariance matrix
        
        Args:
            returns_key: String representation of returns data for caching
            
        Returns:
            Covariance matrix
        """
        try:
            # Convert string key back to returns dataframe
            # This is a simplified example - in practice, you'd use a more robust approach
            returns = pd.read_json(returns_key)
            return returns.cov().values
        except Exception as e:
            logger.error(f"Error computing covariance matrix: {str(e)}")
            # Return identity matrix as fallback
            size = len(pd.read_json(returns_key).columns)
            return np.eye(size)
    
    def parallel_monte_carlo(self, 
                           returns: pd.DataFrame,
                           weights: np.ndarray,
                           initial_investment: float,
                           simulation_length: int,
                           num_simulations: int) -> np.ndarray:
        """
        Run Monte Carlo simulations in parallel
        
        Args:
            returns: DataFrame of asset returns
            weights: Array of portfolio weights
            initial_investment: Initial investment amount
            simulation_length: Length of simulation in days
            num_simulations: Number of simulations to run
            
        Returns:
            Array of simulation results
        """
        try:
            # Calculate mean and covariance
            mean_returns = returns.mean().values
            cov_matrix = returns.cov().values
            
            # Determine optimal chunk size
            chunk_size = max(1, num_simulations // (self.num_cores * 2))
            chunks = [(mean_returns, cov_matrix, weights, initial_investment, 
                      simulation_length, chunk_size) for _ in range(0, num_simulations, chunk_size)]
            
            # Run simulations in parallel
            with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
                results = list(executor.map(self._run_simulation_chunk, chunks))
            
            # Combine results
            return np.vstack(results)
        except Exception as e:
            logger.error(f"Error in parallel Monte Carlo simulation: {str(e)}")
            # Fallback to simple simulation
            return self._fallback_monte_carlo(
                returns, weights, initial_investment, simulation_length, min(100, num_simulations)
            )
    
    def _fallback_monte_carlo(self, 
                            returns: pd.DataFrame,
                            weights: np.ndarray,
                            initial_investment: float,
                            simulation_length: int,
                            num_simulations: int) -> np.ndarray:
        """
        Fallback method for Monte Carlo simulation when parallel execution fails
        
        Args:
            returns: DataFrame of asset returns
            weights: Array of portfolio weights
            initial_investment: Initial investment amount
            simulation_length: Length of simulation in days
            num_simulations: Number of simulations to run
            
        Returns:
            Array of simulation results
        """
        logger.warning("Using fallback Monte Carlo simulation method")
        
        # Calculate mean and standard deviation of returns
        mean_returns = returns.mean().values
        std_returns = returns.std().values
        
        # Initialize results array
        results = np.zeros((simulation_length, num_simulations))
        
        # Run simulations
        for i in range(num_simulations):
            # Generate random returns
            random_returns = np.random.normal(
                mean_returns.reshape(-1, 1), 
                std_returns.reshape(-1, 1), 
                size=(len(mean_returns), simulation_length)
            )
            
            # Calculate portfolio returns
            portfolio_returns = np.sum(random_returns * weights.reshape(-1, 1), axis=0)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            
            # Calculate portfolio values
            results[:, i] = initial_investment * cumulative_returns
        
        return results
    
    def _run_simulation_chunk(self, args: Tuple) -> np.ndarray:
        """
        Run a chunk of Monte Carlo simulations
        
        Args:
            args: Tuple of (mean_returns, cov_matrix, weights, initial_investment, 
                  simulation_length, chunk_size)
            
        Returns:
            Array of simulation results for this chunk
        """
        try:
            mean_returns, cov_matrix, weights, initial_investment, simulation_length, chunk_size = args
            
            # Generate random returns
            np.random.seed()  # Use different seed for each process
            Z = np.random.normal(size=(simulation_length, chunk_size, len(weights)))
            L = np.linalg.cholesky(cov_matrix)
            daily_returns = mean_returns.reshape(-1, 1) + np.tensordot(L, Z, axes=([1], [2])).T
            
            # Calculate portfolio returns
            portfolio_returns = np.sum(daily_returns * weights.reshape(-1, 1, 1), axis=0).T
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + portfolio_returns, axis=0)
            
            # Calculate portfolio values
            portfolio_values = initial_investment * cumulative_returns
            
            return portfolio_values
        except np.linalg.LinAlgError:
            # Handle case where covariance matrix is not positive definite
            logger.warning("Covariance matrix is not positive definite. Using diagonal approximation.")
            
            # Use diagonal approximation
            diag_cov = np.diag(np.diag(cov_matrix))
            L = np.linalg.cholesky(diag_cov)
            
            # Generate random returns
            Z = np.random.normal(size=(simulation_length, chunk_size, len(weights)))
            daily_returns = mean_returns.reshape(-1, 1) + np.tensordot(L, Z, axes=([1], [2])).T
            
            # Calculate portfolio returns
            portfolio_returns = np.sum(daily_returns * weights.reshape(-1, 1, 1), axis=0).T
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + portfolio_returns, axis=0)
            
            # Calculate portfolio values
            portfolio_values = initial_investment * cumulative_returns
            
            return portfolio_values
        except Exception as e:
            logger.error(f"Error in simulation chunk: {str(e)}")
            # Return zeros as fallback
            return np.zeros((simulation_length, chunk_size))
    
    def parallel_efficient_frontier(self,
                                  returns: pd.DataFrame,
                                  num_portfolios: int = 10000,
                                  risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Calculate efficient frontier using parallel processing
        
        Args:
            returns: DataFrame of asset returns
            num_portfolios: Number of portfolios to simulate
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary with efficient frontier results
        """
        try:
            num_assets = len(returns.columns)
            
            # Split the work into chunks
            chunk_size = max(100, num_portfolios // (self.num_cores * 2))
            chunks = [(returns, num_assets, chunk_size, risk_free_rate, i) 
                     for i in range(0, num_portfolios, chunk_size)]
            
            # Run in parallel
            with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
                results = list(executor.map(self._calculate_portfolios_chunk, chunks))
            
            # Combine results
            all_weights = np.vstack([r[0] for r in results])
            all_returns = np.concatenate([r[1] for r in results])
            all_volatilities = np.concatenate([r[2] for r in results])
            all_sharpe_ratios = np.concatenate([r[3] for r in results])
            
            # Find optimal portfolios
            max_sharpe_idx = np.argmax(all_sharpe_ratios)
            min_vol_idx = np.argmin(all_volatilities)
            
            return {
                'efficient_frontier': {
                    'returns': all_returns,
                    'volatilities': all_volatilities,
                    'sharpe_ratios': all_sharpe_ratios,
                    'weights': all_weights
                },
                'max_sharpe_portfolio': {
                    'return': all_returns[max_sharpe_idx],
                    'volatility': all_volatilities[max_sharpe_idx],
                    'sharpe_ratio': all_sharpe_ratios[max_sharpe_idx],
                    'weights': all_weights[max_sharpe_idx]
                },
                'min_volatility_portfolio': {
                    'return': all_returns[min_vol_idx],
                    'volatility': all_volatilities[min_vol_idx],
                    'sharpe_ratio': all_sharpe_ratios[min_vol_idx],
                    'weights': all_weights[min_vol_idx]
                }
            }
        except Exception as e:
            logger.error(f"Error calculating efficient frontier: {str(e)}")
            # Fallback to simplified calculation
            return self._fallback_efficient_frontier(returns, min(500, num_portfolios), risk_free_rate)
    
    def _fallback_efficient_frontier(self,
                                   returns: pd.DataFrame,
                                   num_portfolios: int = 500,
                                   risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Fallback method for efficient frontier calculation
        
        Args:
            returns: DataFrame of asset returns
            num_portfolios: Number of portfolios to simulate
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary with efficient frontier results
        """
        logger.warning("Using fallback efficient frontier calculation")
        
        num_assets = len(returns.columns)
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Generate random portfolios
        all_weights = np.zeros((num_portfolios, num_assets))
        all_returns = np.zeros(num_portfolios)
        all_volatilities = np.zeros(num_portfolios)
        all_sharpe_ratios = np.zeros(num_portfolios)
        
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights = weights / np.sum(weights)
            all_weights[i, :] = weights
            
            # Calculate portfolio return and volatility
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Store results
            all_returns[i] = portfolio_return
            all_volatilities[i] = portfolio_volatility
            all_sharpe_ratios[i] = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        # Find optimal portfolios
        max_sharpe_idx = np.argmax(all_sharpe_ratios)
        min_vol_idx = np.argmin(all_volatilities)
        
        return {
            'efficient_frontier': {
                'returns': all_returns,
                'volatilities': all_volatilities,
                'sharpe_ratios': all_sharpe_ratios,
                'weights': all_weights
            },
            'max_sharpe_portfolio': {
                'return': all_returns[max_sharpe_idx],
                'volatility': all_volatilities[max_sharpe_idx],
                'sharpe_ratio': all_sharpe_ratios[max_sharpe_idx],
                'weights': all_weights[max_sharpe_idx]
            },
            'min_volatility_portfolio': {
                'return': all_returns[min_vol_idx],
                'volatility': all_volatilities[min_vol_idx],
                'sharpe_ratio': all_sharpe_ratios[min_vol_idx],
                'weights': all_weights[min_vol_idx]
            }
        }
    
    def _calculate_portfolios_chunk(self, args: Tuple) -> Tuple:
        """
        Calculate a chunk of random portfolios for efficient frontier
        
        Args:
            args: Tuple of (returns, num_assets, chunk_size, risk_free_rate, chunk_id)
            
        Returns:
            Tuple of (weights, returns, volatilities, sharpe_ratios)
        """
        try:
            returns_df, num_assets, chunk_size, risk_free_rate, chunk_id = args
            
            # Calculate mean returns and covariance matrix
            mean_returns = returns_df.mean() * 252
            cov_matrix = returns_df.cov() * 252
            
            # Initialize arrays
            weights = np.zeros((chunk_size, num_assets))
            portfolio_returns = np.zeros(chunk_size)
            portfolio_volatilities = np.zeros(chunk_size)
            portfolio_sharpe_ratios = np.zeros(chunk_size)
            
            # Set random seed based on chunk_id for reproducibility
            np.random.seed(42 + chunk_id)
            
            # Generate random portfolios
            for i in range(chunk_size):
                # Generate random weights
                w = np.random.random(num_assets)
                w = w / np.sum(w)
                weights[i, :] = w
                
                # Calculate portfolio return and volatility
                portfolio_return = np.sum(mean_returns * w)
                portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                
                # Store results
                portfolio_returns[i] = portfolio_return
                portfolio_volatilities[i] = portfolio_volatility
                portfolio_sharpe_ratios[i] = (portfolio_return - risk_free_rate) / portfolio_volatility
            
            return weights, portfolio_returns, portfolio_volatilities, portfolio_sharpe_ratios
        except Exception as e:
            logger.error(f"Error in portfolio calculation chunk: {str(e)}")
            # Return empty arrays as fallback
            return (
                np.zeros((chunk_size, num_assets)),
                np.zeros(chunk_size),
                np.zeros(chunk_size),
                np.zeros(chunk_size)
            )
    
    def plot_efficient_frontier(self,
                              ef_data: Dict[str, Any],
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot efficient frontier
        
        Args:
            ef_data: Efficient frontier data from parallel_efficient_frontier
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        try:
            # Extract data
            returns = ef_data['efficient_frontier']['returns']
            volatilities = ef_data['efficient_frontier']['volatilities']
            sharpe_ratios = ef_data['efficient_frontier']['sharpe_ratios']
            
            max_sharpe_return = ef_data['max_sharpe_portfolio']['return']
            max_sharpe_volatility = ef_data['max_sharpe_portfolio']['volatility']
            
            min_vol_return = ef_data['min_volatility_portfolio']['return']
            min_vol_volatility = ef_data['min_volatility_portfolio']['volatility']
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot efficient frontier
            scatter = ax.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', alpha=0.5)
            
            # Plot optimal portfolios
            ax.scatter(max_sharpe_volatility, max_sharpe_return, marker='*', color='r', s=200, label='Max Sharpe')
            ax.scatter(min_vol_volatility, min_vol_return, marker='*', color='g', s=200, label='Min Volatility')
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Sharpe Ratio')
            
            # Set labels and title
            ax.set_xlabel('Volatility (Standard Deviation)')
            ax.set_ylabel('Expected Return')
            ax.set_title('Efficient Frontier')
            ax.legend()
            ax.grid(True)
            
            # Save figure if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
        except Exception as e:
            logger.error(f"Error plotting efficient frontier: {str(e)}")
            # Return empty figure as fallback
            return plt.figure()


class MonteCarloSimulator:
    """
    Monte Carlo simulator for portfolio analysis
    """
    
    def __init__(self, 
                 returns_data: pd.DataFrame,
                 weights: Optional[np.ndarray] = None,
                 risk_free_rate: float = 0.02,
                 simulation_years: int = 5,
                 trading_days: int = 252,
                 num_simulations: int = 1000):
        """
        Initialize the Monte Carlo simulator
        
        Args:
            returns_data: DataFrame with asset returns (each column is an asset)
            weights: Portfolio weights (if None, equal weights will be used)
            risk_free_rate: Annual risk-free rate
            simulation_years: Number of years to simulate
            trading_days: Number of trading days per year
            num_simulations: Number of Monte Carlo simulations to run
        """
        self.returns_data = returns_data
        self.num_assets = returns_data.shape[1]
        
        # Use equal weights if not provided
        if weights is None:
            self.weights = np.ones(self.num_assets) / self.num_assets
        else:
            # Normalize weights to sum to 1
            self.weights = weights / np.sum(weights)
        
        self.risk_free_rate = risk_free_rate
        self.simulation_years = simulation_years
        self.trading_days = trading_days
        self.num_simulations = num_simulations
        
        # Calculate mean returns and covariance matrix
        self.mean_returns = returns_data.mean().values
        self.cov_matrix = returns_data.cov().values
        
        # Simulation results
        self.simulation_results = None
    
    def _calculate_portfolio_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio metrics based on historical data
        
        Returns:
            Dictionary with portfolio metrics
        """
        # Calculate portfolio expected return
        portfolio_return = np.sum(self.mean_returns * self.weights) * self.trading_days
        
        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(
            np.dot(self.weights.T, np.dot(self.cov_matrix * self.trading_days, self.weights))
        )
        
        # Calculate Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation
        
        Returns:
            Dictionary with simulation results
        """
        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics()
        
        # Calculate simulation parameters
        simulation_length = self.simulation_years * self.trading_days
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        Z = np.random.normal(size=(simulation_length, self.num_simulations, self.num_assets))
        L = np.linalg.cholesky(self.cov_matrix)
        daily_returns = self.mean_returns.reshape(-1, 1) + np.tensordot(L, Z, axes=([1], [2])).T
        
        # Calculate portfolio returns
        portfolio_returns = np.sum(daily_returns * self.weights.reshape(-1, 1, 1), axis=0).T
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + portfolio_returns, axis=0)
        
        # Calculate percentiles
        percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        final_percentiles = np.percentile(cumulative_returns[-1, :], [p * 100 for p in percentiles])
        
        # Store results
        self.simulation_results = {
            'cumulative_returns': cumulative_returns,
            'metrics': metrics,
            'final_percentiles': {f'p{int(p*100)}': val for p, val in zip(percentiles, final_percentiles)},
            'simulation_parameters': {
                'simulation_years': self.simulation_years,
                'trading_days': self.trading_days,
                'num_simulations': self.num_simulations
            }
        }
        
        return self.simulation_results
    
    def plot_simulation_results(self, 
                              num_paths: int = 100,
                              figsize: Tuple[int, int] = (12, 8),
                              initial_investment: float = 10000.0) -> plt.Figure:
        """
        Plot simulation results
        
        Args:
            num_paths: Number of paths to plot
            figsize: Figure size
            initial_investment: Initial investment amount
            
        Returns:
            Matplotlib figure
        """
        if self.simulation_results is None:
            self.run_simulation()
            
        cumulative_returns = self.simulation_results['cumulative_returns']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot a subset of paths
        indices = np.random.choice(cumulative_returns.shape[1], min(num_paths, cumulative_returns.shape[1]), replace=False)
        for i in indices:
            ax.plot(cumulative_returns[:, i] * initial_investment, 'b-', alpha=0.1)
            
        # Plot percentiles
        percentiles = [0.05, 0.5, 0.95]
        percentile_values = np.percentile(cumulative_returns, [p * 100 for p in percentiles], axis=1)
        
        for i, p in enumerate(percentiles):
            ax.plot(percentile_values[i] * initial_investment, 'r-', linewidth=2, 
                   label=f'{int(p*100)}th Percentile')
            
        # Add labels and title
        ax.set_xlabel('Trading Days')
        ax.set_ylabel(f'Portfolio Value (Initial: ${initial_investment:,.0f})')
        ax.set_title('Monte Carlo Simulation of Portfolio Value')
        ax.legend()
        ax.grid(True)
        
        return fig


# ThunderComputePortfolioOptimizer class has been moved to a more appropriate location

def mean_variance_portfolio(returns: pd.DataFrame, risk_free_rate: float = 0.0) -> Dict[str, Any]:
    """
    Computes mean-variance optimal weights (no constraints).
    
    Args:
        returns: DataFrame of asset returns
        risk_free_rate: Risk-free rate for calculations
        
    Returns:
        Dictionary with optimal weights and asset names
    """
    cov = returns.cov()
    mean = returns.mean()
    inv_cov = np.linalg.pinv(cov)
    weights = inv_cov @ mean
    weights = weights / np.sum(weights)
    weights_list = weights.tolist()
    return {"weights": weights_list, "assets": list(returns.columns)}

def calculate_efficient_frontier(returns: pd.DataFrame, 
                               num_portfolios: int = 10000,
                               risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Calculate the efficient frontier using Modern Portfolio Theory
    
    Args:
        returns: DataFrame of asset returns
        num_portfolios: Number of random portfolios to generate
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        
    Returns:
        Dictionary with efficient frontier data
    """
    # Get number of assets
    num_assets = len(returns.columns)
    
    # Generate random portfolios
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        # Calculate portfolio return and volatility
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        # Calculate Sharpe Ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        # Store results
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio
    
    # Convert to DataFrame
    columns = ['Return', 'Volatility', 'Sharpe']
    portfolios = pd.DataFrame(results.T, columns=columns)
    
    # Find optimal portfolio (maximum Sharpe ratio)
    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_record[max_sharpe_idx]
    
    # Find minimum volatility portfolio
    min_vol_idx = np.argmin(results[1])
    min_vol_weights = weights_record[min_vol_idx]
    
    # Calculate efficient frontier
    target_returns = np.linspace(portfolios['Return'].min(), portfolios['Return'].max(), 100)
    efficient_volatilities = []
    
    for target in target_returns:
        # Minimize volatility for each target return
        efficient_volatilities.append(minimize_volatility(returns, target))
    
    return {
        'portfolios': portfolios,
        'optimal_weights': dict(zip(returns.columns, optimal_weights)),
        'optimal_return': results[0, max_sharpe_idx],
        'optimal_volatility': results[1, max_sharpe_idx],
        'optimal_sharpe': results[2, max_sharpe_idx],
        'min_vol_weights': dict(zip(returns.columns, min_vol_weights)),
        'min_vol_return': results[0, min_vol_idx],
        'min_vol_volatility': results[1, min_vol_idx],
        'min_vol_sharpe': results[2, min_vol_idx],
        'efficient_frontier': {
            'returns': target_returns,
            'volatilities': efficient_volatilities
        }
    }

def minimize_volatility(returns: pd.DataFrame, target_return: float) -> float:
    """
    Find the minimum volatility for a given target return
    
    Args:
        returns: DataFrame of asset returns
        target_return: Target portfolio return
        
    Returns:
        Minimum volatility
    """
    from scipy.optimize import minimize
    
    num_assets = len(returns.columns)
    args = (returns,)
    
    # Initial guess (equal weights)
    initial_weights = np.array([1.0 / num_assets] * num_assets)
    
    # Constraints
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        {'type': 'eq', 'fun': lambda x: np.sum(returns.mean() * x) * 252 - target_return}  # Target return
    )
    
    # Bounds (no short selling)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Minimize volatility
    result = minimize(
        portfolio_volatility,
        initial_weights,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result['fun']

def portfolio_volatility(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """
    Calculate portfolio volatility
    
    Args:
        weights: Array of asset weights
        returns: DataFrame of asset returns
        
    Returns:
        Portfolio volatility
    """
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))