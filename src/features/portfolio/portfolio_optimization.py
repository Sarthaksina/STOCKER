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


class ThunderComputePortfolioOptimizer:
    """
    Portfolio optimization using ThunderCompute cloud infrastructure
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 s3_bucket: Optional[str] = None,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 aws_region: str = 'eu-north-1'):
        """
        Initialize the ThunderCompute portfolio optimizer
        
        Args:
            api_key: ThunderCompute API key (if None, load from env)
            s3_bucket: S3 bucket name for data storage (if None, load from env)
            aws_access_key_id: AWS access key ID (if None, load from env)
            aws_secret_access_key: AWS secret access key (if None, load from env)
            aws_region: AWS region
        """
        # Issue deprecation warning
        warnings.warn(
            "This class is now part of portfolio_optimization.py. Import it directly from there.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Load credentials from environment variables if not provided
        self.api_key = api_key or os.environ.get('THUNDERCOMPUTE_API_KEY')
        self.s3_bucket = s3_bucket or os.environ.get('S3_BUCKET_NAME')
        self.aws_access_key_id = aws_access_key_id or os.environ.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = aws_secret_access_key or os.environ.get('AWS_SECRET_ACCESS_KEY')
        self.aws_region = aws_region or os.environ.get('AWS_REGION', 'eu-north-1')
        
        # Validate credentials
        if not self.api_key:
            raise ValueError("Missing ThunderCompute API key. Please provide it as a parameter or set THUNDERCOMPUTE_API_KEY environment variable.")
        if not self.s3_bucket:
            raise ValueError("Missing S3 bucket name. Please provide it as a parameter or set S3_BUCKET_NAME environment variable.")
        if not self.aws_access_key_id:
            raise ValueError("Missing AWS access key ID. Please provide it as a parameter or set AWS_ACCESS_KEY_ID environment variable.")
        if not self.aws_secret_access_key:
            raise ValueError("Missing AWS secret access key. Please provide it as a parameter or set AWS_SECRET_ACCESS_KEY environment variable.")
        
        # Initialize S3 client if AWS dependencies are available
        if HAS_AWS_DEPS:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
        else:
            logger.warning("AWS dependencies not found. Cloud optimization will not be available.")
            self.s3_client = None
        
        # ThunderCompute API base URL
        self.api_base_url = "https://api.thundercompute.com/v1"
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _upload_data_to_s3(self, 
                          data: pd.DataFrame, 
                          file_name: str) -> str:
        """
        Upload data to S3 bucket
        
        Args:
            data: DataFrame to upload
            file_name: Name of the file in S3
            
        Returns:
            S3 URI of the uploaded file
        """
        if not HAS_AWS_DEPS:
            raise ImportError("AWS dependencies not found. Cannot upload data to S3.")
            
        # Convert DataFrame to CSV
        csv_buffer = data.to_csv(index=True).encode()
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=file_name,
            Body=csv_buffer
        )
        
        # Return S3 URI
        return f"s3://{self.s3_bucket}/{file_name}"
    
    def _download_data_from_s3(self, 
                              file_name: str) -> pd.DataFrame:
        """
        Download data from S3 bucket
        
        Args:
            file_name: Name of the file in S3
            
        Returns:
            Downloaded DataFrame
        """
        if not HAS_AWS_DEPS:
            raise ImportError("AWS dependencies not found. Cannot download data from S3.")
            
        # Download from S3
        response = self.s3_client.get_object(
            Bucket=self.s3_bucket,
            Key=file_name
        )
        
        # Convert to DataFrame
        return pd.read_csv(response['Body'])
    
    def _submit_job(self, 
                   job_config: Dict[str, Any]) -> str:
        """
        Submit a job to ThunderCompute
        
        Args:
            job_config: Job configuration
            
        Returns:
            Job ID
        """
        # Submit job
        response = requests.post(
            f"{self.api_base_url}/jobs",
            headers=self.headers,
            json=job_config
        )
        
        # Check response
        if response.status_code == 200:
            return response.json().get('job_id', '')
        else:
            raise Exception(f"Failed to submit job: {response.text}")
    
    def _get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status from ThunderCompute
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status
        """
        # Get job status
        response = requests.get(
            f"{self.api_base_url}/jobs/{job_id}",
            headers=self.headers
        )
        
        # Check response
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get job status: {response.text}")
    
    def calculate_portfolio_metrics(self, 
                                  returns: pd.DataFrame, 
                                  weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate portfolio metrics (return, volatility, Sharpe ratio)
        
        Args:
            returns: DataFrame of asset returns
            weights: Array of asset weights
            
        Returns:
            Dictionary of portfolio metrics
        """
        # Calculate portfolio return
        portfolio_return = np.sum(returns.mean() * weights) * 252
        
        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0.02)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _negative_sharpe_ratio(self, 
                             weights: np.ndarray, 
                             returns: pd.DataFrame) -> float:
        """
        Calculate negative Sharpe ratio (for minimization)
        
        Args:
            weights: Array of asset weights
            returns: DataFrame of asset returns
            
        Returns:
            Negative Sharpe ratio
        """
        portfolio_metrics = self.calculate_portfolio_metrics(returns, weights)
        return -portfolio_metrics['sharpe_ratio']
    
    def optimize_portfolio(self, 
                         price_data: pd.DataFrame, 
                         risk_free_rate: float = 0.02,
                         use_cloud: bool = False) -> Dict[str, Any]:
        """
        Optimize portfolio weights using Modern Portfolio Theory
        
        Args:
            price_data: DataFrame of asset prices
            risk_free_rate: Risk-free rate
            use_cloud: Whether to use cloud computing
            
        Returns:
            Dictionary with optimized weights and metrics
        """
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # If using cloud computing, submit job to ThunderCompute
        if use_cloud:
            if not HAS_AWS_DEPS:
                raise ImportError("AWS dependencies not found. Cannot use cloud optimization.")
                
            # Upload data to S3
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            s3_path = self._upload_data_to_s3(
                price_data, 
                f"portfolio_optimization/{timestamp}/price_data.csv"
            )
            
            # Submit job
            job_config = {
                "job_type": "portfolio_optimization",
                "data_path": s3_path,
                "parameters": {
                    "risk_free_rate": risk_free_rate
                }
            }
            
            job_id = self._submit_job(job_config)
            
            # Wait for job to complete
            while True:
                job_status = self._get_job_status(job_id)
                if job_status['status'] == 'COMPLETED':
                    # Download results
                    result_path = job_status['result_path']
                    result_file = result_path.split('/')[-1]
                    results = self._download_data_from_s3(result_file)
                    
                    # Parse results
                    weights = results['weights'].values
                    metrics = {
                        'return': results['return'].iloc[0],
                        'volatility': results['volatility'].iloc[0],
                        'sharpe_ratio': results['sharpe_ratio'].iloc[0]
                    }
                    
                    return {
                        'weights': weights,
                        'metrics': metrics,
                        'assets': price_data.columns.tolist()
                    }
                
                elif job_status['status'] == 'FAILED':
                    raise Exception(f"Job failed: {job_status.get('error', 'Unknown error')}")
                
                # Sleep for 5 seconds
                time.sleep(5)
        
        # Otherwise, optimize locally
        else:
            try:
                from scipy.optimize import minimize
                
                num_assets = len(returns.columns)
                
                # Initial guess (equal weights)
                initial_weights = np.array([1.0 / num_assets] * num_assets)
                
                # Constraints (weights sum to 1)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                
                # Bounds (no short selling)
                bounds = tuple((0, 1) for _ in range(num_assets))
                
                # Optimize
                result = minimize(
                    self._negative_sharpe_ratio,
                    initial_weights,
                    args=(returns,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                # Get optimized weights
                weights = result['x']
                
                # Calculate metrics
                metrics = self.calculate_portfolio_metrics(returns, weights)
                
                return {
                    'weights': weights,
                    'metrics': metrics,
                    'assets': price_data.columns.tolist()
                }
            except ImportError:
                raise ImportError("SciPy not found. Cannot perform local optimization.")
    
    def backtest_portfolio(self, 
                         price_data: pd.DataFrame, 
                         weights: np.ndarray, 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Backtest portfolio performance
        
        Args:
            price_data: DataFrame of asset prices
            weights: Array of asset weights
            start_date: Start date for backtest (if None, use first date in price_data)
            end_date: End date for backtest (if None, use last date in price_data)
            
        Returns:
            Dictionary with backtest results
        """
        # Filter data by date range
        if start_date:
            price_data = price_data[price_data.index >= start_date]
        if end_date:
            price_data = price_data[price_data.index <= end_date]
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility
        
        # Calculate drawdowns
        wealth_index = (1 + portfolio_returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        max_drawdown = drawdowns.min()
        
        return {
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'metrics': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
        }