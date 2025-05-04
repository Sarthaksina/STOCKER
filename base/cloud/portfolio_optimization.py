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

from stocker.cloud.portfolio_config import PortfolioConfig

# Configure logging
logger = logging.getLogger(__name__)

# --- RL Optimizer Dependencies (Added from portfolio/rl_optimizer.py) ---
try:
    import gym
    from gym import spaces
    from stable_baselines3 import PPO, A2C, SAC
    HAS_RL_DEPS = True
except ImportError:
    logger.warning("RL dependencies not found. RL optimization will not be available.")
    HAS_RL_DEPS = False

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
        # Convert string key back to returns dataframe
        # This is a simplified example - in practice, you'd use a more robust approach
        returns = pd.read_json(returns_key)
        return returns.cov().values
    
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
    
    def _run_simulation_chunk(self, args: Tuple) -> np.ndarray:
        """
        Run a chunk of Monte Carlo simulations
        
        Args:
            args: Tuple of (mean_returns, cov_matrix, weights, initial_investment, 
                  simulation_length, chunk_size)
            
        Returns:
            Array of simulation results for this chunk
        """
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
                'weights': all_weights,
                'returns': all_returns,
                'volatilities': all_volatilities,
                'sharpe_ratios': all_sharpe_ratios
            },
            'max_sharpe': {
                'weights': all_weights[max_sharpe_idx],
                'metrics': {
                    'expected_return': all_returns[max_sharpe_idx],
                    'volatility': all_volatilities[max_sharpe_idx],
                    'sharpe_ratio': all_sharpe_ratios[max_sharpe_idx]
                }
            },
            'min_volatility': {
                'weights': all_weights[min_vol_idx],
                'metrics': {
                    'expected_return': all_returns[min_vol_idx],
                    'volatility': all_volatilities[min_vol_idx],
                    'sharpe_ratio': all_sharpe_ratios[min_vol_idx]
                }
            },
            'assets': list(returns.columns),
            'risk_free_rate': risk_free_rate
        }
    
    def _calculate_portfolios_chunk(self, args: Tuple) -> Tuple:
        """
        Calculate a chunk of random portfolios for efficient frontier
        
        Args:
            args: Tuple of (returns, num_assets, chunk_size, risk_free_rate, chunk_id)
            
        Returns:
            Tuple of (weights, returns, volatilities, sharpe_ratios)
        """
        returns, num_assets, chunk_size, risk_free_rate, chunk_id = args
        
        # Set different seed for each chunk
        np.random.seed(chunk_id)
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        # Generate random weights
        weights = np.random.random((chunk_size, num_assets))
        weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
        
        # Calculate portfolio metrics
        portfolio_returns = np.dot(weights, mean_returns) * 252
        portfolio_volatilities = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix, weights)) * np.sqrt(252)
        sharpe_ratios = (portfolio_returns - risk_free_rate) / portfolio_volatilities
        
        return weights, portfolio_returns, portfolio_volatilities, sharpe_ratios


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
            np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights))
        ) * np.sqrt(self.trading_days)
        
        # Calculate Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return {
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float(sharpe_ratio)
        }
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for the portfolio
        
        Returns:
            Dictionary with simulation results
        """
        # Calculate time steps
        total_steps = self.simulation_years * self.trading_days
        
        # Initialize array for simulation results
        simulation_results = np.zeros((self.num_simulations, total_steps + 1))
        simulation_results[:, 0] = 1  # Start with $1
        
        # Generate random returns using multivariate normal distribution
        for sim in range(self.num_simulations):
            # Generate random returns
            Z = np.random.multivariate_normal(
                self.mean_returns, 
                self.cov_matrix, 
                total_steps
            )
            
            # Calculate portfolio returns
            portfolio_returns = np.sum(Z * self.weights, axis=1)
            
            # Calculate cumulative returns
            for step in range(total_steps):
                simulation_results[sim, step + 1] = simulation_results[sim, step] * (1 + portfolio_returns[step])
        
        self.simulation_results = simulation_results
        
        # Calculate statistics
        final_values = simulation_results[:, -1]
        
        # Calculate percentiles
        percentiles = {
            "worst_case": float(np.percentile(final_values, 5)),
            "best_case": float(np.percentile(final_values, 95)),
            "median_case": float(np.percentile(final_values, 50)),
            "mean_final_value": float(np.mean(final_values))
        }
        
        # Calculate probability of loss
        prob_loss = np.mean(final_values < 1.0)
        
        # Calculate expected shortfall (CVaR) at 5%
        cvar_5 = np.mean(final_values[final_values <= np.percentile(final_values, 5)])
        
        # Calculate maximum drawdown across all simulations
        max_drawdowns = []
        for sim in range(self.num_simulations):
            cumulative_returns = simulation_results[sim, :]
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns) / peak
            max_drawdowns.append(np.max(drawdown))
        
        avg_max_drawdown = np.mean(max_drawdowns)
        
        # Portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics()
        
        return {
            "portfolio_metrics": portfolio_metrics,
            "simulation_statistics": {
                "percentiles": percentiles,
                "probability_of_loss": float(prob_loss),
                "expected_shortfall_5": float(cvar_5),
                "avg_max_drawdown": float(avg_max_drawdown)
            },
            "simulation_years": self.simulation_years,
            "num_simulations": self.num_simulations
        }
    
    def plot_simulations(self, 
                         num_paths_to_plot: int = 100, 
                         figsize: Tuple[int, int] = (12, 8),
                         title: str = "Monte Carlo Simulation of Portfolio Performance",
                         save_path: Optional[str] = None) -> None:
        """
        Plot the Monte Carlo simulation results
        
        Args:
            num_paths_to_plot: Number of random paths to plot
            figsize: Figure size
            title: Plot title
            save_path: Path to save the figure (if None, the figure will be displayed)
        """
        if self.simulation_results is None:
            raise ValueError("No simulation results available. Run run_simulation() first.")
        
        plt.figure(figsize=figsize)
        
        # Plot a subset of simulation paths
        indices = np.random.choice(self.num_simulations, min(num_paths_to_plot, self.num_simulations), replace=False)
        for idx in indices:
            plt.plot(self.simulation_results[idx], 'b-', alpha=0.1)
        
        # Plot percentiles
        for percentile in [5, 50, 95]:
            percentile_values = np.percentile(self.simulation_results, percentile, axis=0)
            plt.plot(percentile_values, 'r-', linewidth=2, label=f"{percentile}th Percentile")
        
        plt.title(title)
        plt.xlabel("Trading Days")
        plt.ylabel("Portfolio Value (Starting at $1)")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_distribution(self,
                          figsize: Tuple[int, int] = (12, 8),
                          title: str = "Distribution of Final Portfolio Values",
                          save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of final portfolio values
        
        Args:
            figsize: Figure size
            title: Plot title
            save_path: Path to save the figure (if None, the figure will be displayed)
        """
        if self.simulation_results is None:
            raise ValueError("No simulation results available. Run run_simulation() first.")
        
        final_values = self.simulation_results[:, -1]
        
        plt.figure(figsize=figsize)
        
        # Plot histogram
        sns.histplot(final_values, kde=True)
        
        # Add vertical lines for percentiles
        for percentile, color, label in zip([5, 50, 95], ['r', 'g', 'b'], 
                                           ['5th Percentile', 'Median', '95th Percentile']):
            value = np.percentile(final_values, percentile)
            plt.axvline(x=value, color=color, linestyle='--', label=f"{label}: ${value:.2f}")
        
        # Add initial investment line
        plt.axvline(x=1.0, color='k', linestyle='-', label="Initial Investment: $1.00")
        
        plt.title(title)
        plt.xlabel("Final Portfolio Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def get_extreme_scenarios(self, confidence_level: float = 0.05) -> Dict[str, Any]:
        """
        Get best and worst case scenarios with given confidence level
        """
        if self.simulation_results is None:
            self.run_simulation()
        
        final_values = self.simulation_results[:, -1]
        
        return {
            "best_case": float(np.percentile(final_values, 100*(1-confidence_level))),
            "worst_case": float(np.percentile(final_values, 100*confidence_level)),
            "confidence_level": confidence_level
        }

class StockTradingEnv(gym.Env):
    """Custom Environment for stock trading portfolio optimization"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 stock_data: pd.DataFrame, 
                 initial_balance: float = 10000.0,
                 transaction_cost_pct: float = 0.001,
                 reward_scaling: float = 1e-4,
                 window_size: int = 30):
        """
        Initialize the trading environment.
        
        Args:
            stock_data: DataFrame with stock price data (must include 'Close' column)
            initial_balance: Starting portfolio value
            transaction_cost_pct: Cost of transactions as percentage
            reward_scaling: Scaling factor for rewards
            window_size: Size of observation window
        """
        super(StockTradingEnv, self).__init__()
        
        self.stock_data = stock_data
        self.stock_dim = len(stock_data.columns.levels[0]) if isinstance(stock_data.columns, pd.MultiIndex) else 1
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.window_size = window_size
        self.terminal = False
        
        # Action space: portfolio weights for each stock (continuous from 0 to 1)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.stock_dim,), dtype=np.float32
        )
        
        # Observation space: includes price history, portfolio weights, balance
        obs_dim = self.window_size * self.stock_dim + self.stock_dim + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        self.time_step = self.window_size
        self.terminal = False
        self.portfolio_value = self.initial_balance
        self.portfolio_weights = np.zeros(self.stock_dim)
        self.portfolio_return = 0
        self.cost = 0
        self.trades = 0
        self.episode_returns = []
        
        return self._get_observation()
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Portfolio weights for each asset
        
        Returns:
            observation, reward, done, info
        """
        self.time_step += 1
        
        # Normalize action to ensure weights sum to 1
        action_sum = np.sum(action)
        if action_sum > 0:
            normalized_action = action / action_sum
        else:
            normalized_action = np.ones(self.stock_dim) / self.stock_dim
        
        # Get current prices and next prices
        current_prices = self._get_prices(self.time_step - 1)
        next_prices = self._get_prices(self.time_step)
        
        # Calculate transaction costs
        prev_portfolio_value = self.portfolio_value
        costs = self._calculate_transaction_costs(normalized_action)
        
        # Update portfolio value based on price changes and costs
        price_change_pct = next_prices / current_prices - 1
        portfolio_return = np.sum(normalized_action * price_change_pct)
        self.portfolio_value = prev_portfolio_value * (1 + portfolio_return) - costs
        
        # Calculate reward (Sharpe ratio or returns-based)
        self.episode_returns.append(portfolio_return)
        if len(self.episode_returns) > 1:
            reward = self._calculate_reward()
        else:
            reward = 0
        
        # Check if episode is done
        done = self.time_step >= len(self.stock_data) - 1
        
        # Update state
        self.portfolio_weights = normalized_action
        self.cost += costs
        
        # Get observation
        observation = self._get_observation()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return,
            'transaction_costs': costs,
            'weights': normalized_action
        }
        
        return observation, reward * self.reward_scaling, done, info
    
    def _get_prices(self, time_idx):
        """Get stock prices at a specific time index."""
        if isinstance(self.stock_data.columns, pd.MultiIndex):
            return np.array([self.stock_data.iloc[time_idx][stock]['Close'] 
                            for stock in self.stock_data.columns.levels[0]])
        else:
            return np.array([self.stock_data.iloc[time_idx]['Close']])
    
    def _get_observation(self):
        """Construct the observation from current state."""
        # Get price history for window_size
        price_history = []
        for i in range(self.time_step - self.window_size, self.time_step):
            price_history.extend(self._get_prices(i) / self._get_prices(self.time_step - self.window_size))
        
        # Combine price history with portfolio weights and value
        observation = np.concatenate((
            price_history,
            self.portfolio_weights,
            [self.portfolio_value / self.initial_balance]  # Normalized portfolio value
        ))
        
        return observation
    
    def _calculate_transaction_costs(self, new_weights):
        """Calculate transaction costs for portfolio rebalancing."""
        costs = self.transaction_cost_pct * np.sum(np.abs(new_weights - self.portfolio_weights)) * self.portfolio_value
        return costs
    
    def _calculate_reward(self):
        """Calculate reward based on recent returns."""
        # Option 1: Simple return
        # return self.episode_returns[-1]
        
        # Option 2: Sharpe ratio (if we have enough history)
        if len(self.episode_returns) > 1:
            returns = np.array(self.episode_returns[-20:])  # Use last 20 returns
            sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)  # Annualized
            return sharpe
        else:
            return 0
    
    def render(self, mode='human'):
        """Render the environment."""
        print(f"Time: {self.time_step}, Portfolio Value: {self.portfolio_value:.2f}")
        print(f"Weights: {self.portfolio_weights}")
        print(f"Return: {self.portfolio_return:.4f}, Cost: {self.cost:.4f}")

class RLPortfolioOptimizer:
    """Portfolio optimizer using Reinforcement Learning."""
    
    def __init__(self, 
                 algorithm: str = 'ppo',
                 policy: str = 'MlpPolicy',
                 train_timesteps: int = 100000,
                 model_dir: str = './models/rl'):
        """
        Initialize the RL portfolio optimizer.
        
        Args:
            algorithm: RL algorithm to use ('ppo', 'a2c', or 'sac')
            policy: Policy network architecture
            train_timesteps: Number of timesteps to train for
            model_dir: Directory to save trained models
        """
        if not HAS_RL_DEPS:
            raise ImportError("RL dependencies not installed. Please install gym and stable-baselines3.")
            
        self.algorithm = algorithm
        self.policy = policy
        self.train_timesteps = train_timesteps
        self.model_dir = model_dir
        self.model = None
        self.env = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def create_env(self, stock_data: pd.DataFrame, **kwargs):
        """Create the trading environment."""
        self.env = StockTradingEnv(stock_data=stock_data, **kwargs)
        return self.env
    
    def train(self, stock_data: pd.DataFrame = None, **kwargs):
        """
        Train the RL model for portfolio optimization.
        
        Args:
            stock_data: Stock price data (if not provided, uses existing env)
            **kwargs: Additional arguments for environment creation
        
        Returns:
            Trained model
        """
        if stock_data is not None or self.env is None:
            self.create_env(stock_data, **kwargs)
        
        # Initialize the appropriate algorithm
        if self.algorithm.lower() == 'ppo':
            self.model = PPO(self.policy, self.env, verbose=1)
        elif self.algorithm.lower() == 'a2c':
            self.model = A2C(self.policy, self.env, verbose=1)
        elif self.algorithm.lower() == 'sac':
            self.model = SAC(self.policy, self.env, verbose=1)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Train the model
        self.model.learn(total_timesteps=self.train_timesteps)
        
        # Save the trained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f"{self.algorithm}_{timestamp}")
        self.model.save(model_path)
        
        return self.model
    
    def optimize(self, stock_data: pd.DataFrame, test_days: int = 252, deterministic: bool = True):
        """
        Generate optimal portfolio weights using the trained model.
        
        Args:
            stock_data: Stock price data for testing
            test_days: Number of days to test
            deterministic: Whether to use deterministic actions
        
        Returns:
            DataFrame with portfolio weights, values, and returns
        """
        if self.model is None:
            raise ValueError("Model must be trained before optimization")
        
        # Create test environment
        test_env = StockTradingEnv(stock_data=stock_data)
        
        # Run the model
        observation = test_env.reset()
        done = False
        results = []
        
        while not done:
            action, _states = self.model.predict(observation, deterministic=deterministic)
            observation, reward, done, info = test_env.step(action)
            
            results.append({
                'timestamp': stock_data.index[test_env.time_step - 1],
                'portfolio_value': info['portfolio_value'],
                'portfolio_return': info['portfolio_return'],
                'transaction_costs': info['transaction_costs'],
                'weights': info['weights']
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        results_df.set_index('timestamp', inplace=True)
        
        # Extract weights into separate columns
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_names = stock_data.columns.levels[0]
        else:
            stock_names = ['Stock']
        
        for i, stock in enumerate(stock_names):
            results_df[f'weight_{stock}'] = results_df['weights'].apply(lambda x: x[i])
        
        # Drop the weights column (list of arrays)
        results_df.drop('weights', axis=1, inplace=True)
        
        return results_df
    
    def load_model(self, model_path: str):
        """Load a trained model from disk."""
        if self.algorithm.lower() == 'ppo':
            self.model = PPO.load(model_path)
        elif self.algorithm.lower() == 'a2c':
            self.model = A2C.load(model_path)
        elif self.algorithm.lower() == 'sac':
            self.model = SAC.load(model_path)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        return self.model
    
    def get_current_weights(self):
        """Get the current portfolio weights."""
        if self.env is None:
            raise ValueError("Environment not created. Call create_env first.")
        return self.env.portfolio_weights

def train_rl_portfolio_optimizer(
    stock_data: pd.DataFrame,
    algorithm: str = 'ppo',
    train_timesteps: int = 100000,
    initial_balance: float = 10000.0,
    transaction_cost_pct: float = 0.001,
    window_size: int = 30,
    model_dir: str = './models/rl'
):
    """
    Train a reinforcement learning portfolio optimizer.
    
    Args:
        stock_data: DataFrame with stock price data
        algorithm: RL algorithm to use ('ppo', 'a2c', or 'sac')
        train_timesteps: Number of timesteps to train for
        initial_balance: Starting portfolio value
        transaction_cost_pct: Cost of transactions as percentage
        window_size: Size of observation window
        model_dir: Directory to save trained models
    
    Returns:
        Trained RLPortfolioOptimizer
    """
    optimizer = RLPortfolioOptimizer(
        algorithm=algorithm,
        train_timesteps=train_timesteps,
        model_dir=model_dir
    )
    
    optimizer.train(
        stock_data=stock_data,
        initial_balance=initial_balance,
        transaction_cost_pct=transaction_cost_pct,
        window_size=window_size
    )
    
    return optimizer