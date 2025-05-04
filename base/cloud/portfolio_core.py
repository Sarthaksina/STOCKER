"""
Portfolio Core Module for STOCKER Pro

This module serves as the central coordination point for all portfolio functionality,
providing a unified interface for portfolio management, analysis, and optimization.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import functools
import time
import os
import json
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import portfolio modules
from stocker.cloud.portfolio_config import PortfolioConfig
from stocker.cloud.portfolio_risk import PortfolioRiskAnalyzer
from stocker.cloud.portfolio_backtest import backtest_portfolio, compare_strategies
from stocker.cloud.portfolio_backtester import PortfolioBacktester
from stocker.cloud.portfolio_visualization import PortfolioVisualizer
from stocker.portfolio.monte_carlo import MonteCarloSimulator
from stocker.portfolio.rl_optimizer import RLPortfolioOptimizer

# Configure logging
logger = logging.getLogger(__name__)

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
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        Initialize the portfolio manager with configuration.
        
        Args:
            config: Portfolio configuration object
        """
        self.config = config or PortfolioConfig()
        
        # Initialize component modules
        self.risk_analyzer = PortfolioRiskAnalyzer(config=self.config)
        self.visualizer = PortfolioVisualizer(config=self.config)
        self.backtester = PortfolioBacktester(config=self.config)
        self.monte_carlo = MonteCarloSimulator()
        
        # Initialize cache
        self.cache = PortfolioCache()
        
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
    
    def cached(func):
        """Decorator to cache function results"""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.cache_enabled:
                return func(self, *args, **kwargs)
                
            # Generate cache key
            cache_key = self._cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = self._cache_get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
                
            # Execute function and time it
            start_time = time.time()
            result = func(self, *args, **kwargs)
            exec_time = time.time() - start_time
            
            # Store in cache
            self._cache_set(cache_key, result)
            
            logger.debug(f"Cache miss for {func.__name__}, execution time: {exec_time:.2f}s")
            return result
        return wrapper
    
    def load_data(self, price_data: pd.DataFrame) -> None:
        """
        Load price data for portfolio analysis.
        
        Args:
            price_data: DataFrame of asset prices with DatetimeIndex
        """
        self.price_data = price_data
        
        # Use parallel processing for large datasets
        if self.use_parallel and len(price_data) * len(price_data.columns) > 100000:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Calculate returns in parallel for large datasets
                self.returns_data = executor.submit(
                    lambda df: df.pct_change().dropna(), 
                    price_data
                ).result()
        else:
            self.returns_data = price_data.pct_change().dropna()
            
        logger.info(f"Loaded price data with {len(price_data)} rows and {len(price_data.columns)} assets")
    
    @cached
    def analyze_risk(self) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis on the current portfolio.
        
        Returns:
            Dictionary with risk metrics
        """
        if self.returns_data is None or self.current_weights is None:
            raise ValueError("Price data and weights must be set before analyzing risk")
            
        risk_metrics = self.risk_analyzer.analyze_portfolio_risk(
            returns=self.returns_data,
            weights=self.current_weights
        )
        
        logger.info(f"Completed risk analysis: VaR={risk_metrics['var_95']:.4f}, CVaR={risk_metrics['cvar_95']:.4f}")
        return risk_metrics
    
    def run_parallel_monte_carlo(self, num_simulations: int = 1000, num_workers: int = 4) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation with parallel processing for better performance
        
        Args:
            num_simulations: Number of simulations to run
            num_workers: Number of parallel workers
            
        Returns:
            Monte Carlo simulation results
        """
        if self.returns_data is None or self.current_weights is None:
            raise ValueError("Price data and weights must be set before running Monte Carlo simulation")
        
        # Split simulations among workers
        simulations_per_worker = num_simulations // num_workers
        
        # Function to run a batch of simulations
        def run_simulation_batch(batch_size):
            return self.monte_carlo.run_simulation(
                returns=self.returns_data,
                weights=self.current_weights,
                initial_investment=10000.0,
                simulation_length=252,  # 1 year
                num_simulations=batch_size
            )
        
        # Run simulations in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            batch_results = list(executor.map(
                run_simulation_batch, 
                [simulations_per_worker] * num_workers
            ))
        
        # Combine results
        combined_results = self._combine_monte_carlo_results(batch_results)
        
        return combined_results
    
    def _combine_monte_carlo_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine results from parallel Monte Carlo simulations
        
        Args:
            batch_results: List of simulation batch results
            
        Returns:
            Combined simulation results
        """
        # Implementation depends on the structure of Monte Carlo results
        # This is a simplified example
        combined = {
            'simulation_paths': [],
            'statistics': {
                'final_values': [],
                'max_drawdowns': []
            }
        }
        
        for batch in batch_results:
            combined['simulation_paths'].extend(batch.get('simulation_paths', []))
            combined['statistics']['final_values'].extend(batch.get('statistics', {}).get('final_values', []))
            combined['statistics']['max_drawdowns'].extend(batch.get('statistics', {}).get('max_drawdowns', []))
        
        # Calculate combined statistics
        combined['statistics']['mean_final_value'] = np.mean(combined['statistics']['final_values'])
        combined['statistics']['median_final_value'] = np.median(combined['statistics']['final_values'])
        combined['statistics']['percentile_5'] = np.percentile(combined['statistics']['final_values'], 5)
        combined['statistics']['percentile_95'] = np.percentile(combined['statistics']['final_values'], 95)
        combined['statistics']['mean_max_drawdown'] = np.mean(combined['statistics']['max_drawdowns'])
        
        return combined
    
    # User presets functionality
    def save_preset(self, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """
        Save current portfolio configuration as a preset
        
        Args:
            name: Name of the preset
            description: Optional description
            
        Returns:
            Dictionary with preset information
        """
        if self.current_weights is None:
            raise ValueError("Portfolio weights must be set before saving preset")
            
        # Create preset data
        preset = {
            'name': name,
            'description': description or f"Portfolio preset created on {datetime.now().strftime('%Y-%m-%d')}",
            'created_at': datetime.now().isoformat(),
            'weights': {
                asset: weight for asset, weight in zip(self.price_data.columns, self.current_weights)
            },
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
        }
        
        # Create presets directory if it doesn't exist
        presets_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'presets')
        os.makedirs(presets_dir, exist_ok=True)
        
        # Save preset to file
        preset_path = os.path.join(presets_dir, f"{name.replace(' ', '_')}.json")
        with open(preset_path, 'w') as f:
            json.dump(preset, f, indent=2, default=str)
            
        logger.info(f"Saved portfolio preset '{name}' to {preset_path}")
        return preset
    
    def load_preset(self, name: str) -> Dict[str, Any]:
        """
        Load a saved portfolio preset
        
        Args:
            name: Name of the preset
            
        Returns:
            Dictionary with preset information
        """
        # Find preset file
        presets_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'presets')
        preset_path = os.path.join(presets_dir, f"{name.replace(' ', '_')}.json")
        
        if not os.path.exists(preset_path):
            raise ValueError(f"Preset '{name}' not found")
            
        # Load preset from file
        with open(preset_path, 'r') as f:
            preset = json.load(f)
            
        # Apply preset weights if price data is loaded
        if self.price_data is not None:
            # Check if all assets in preset exist in current price data
            missing_assets = [asset for asset in preset['weights'] if asset not in self.price_data.columns]
            if missing_assets:
                logger.warning(f"Some assets in preset are missing from current data: {missing_assets}")
                
            # Set weights for available assets
            weights = {asset: weight for asset, weight in preset['weights'].items() 
                      if asset in self.price_data.columns}
            self.set_weights(weights)
            
        # Apply preset config
        if 'config' in preset and isinstance(preset['config'], dict):
            for key, value in preset['config'].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    
        logger.info(f"Loaded portfolio preset '{name}' from {preset_path}")
        return preset
    
    def list_presets(self) -> List[Dict[str, Any]]:
        """
        List all available portfolio presets
        
        Returns:
            List of preset information dictionaries
        """
        presets_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'presets')
        os.makedirs(presets_dir, exist_ok=True)
        
        presets = []
        for filename in os.listdir(presets_dir):
            if filename.endswith('.json'):
                preset_path = os.path.join(presets_dir, filename)
                try:
                    with open(preset_path, 'r') as f:
                        preset = json.load(f)
                        presets.append({
                            'name': preset.get('name', filename.replace('.json', '').replace('_', ' ')),
                            'description': preset.get('description', ''),
                            'created_at': preset.get('created_at', ''),
                            'asset_count': len(preset.get('weights', {}))
                        })
                except Exception as e:
                    logger.warning(f"Error loading preset {filename}: {e}")
                    
        return presets
    
    # Real-time updates functionality
    def enable_real_time_updates(self, interval: int = 60, callback: Optional[Callable] = None) -> None:
        """
        Enable real-time portfolio updates
        
        Args:
            interval: Update interval in seconds
            callback: Optional callback function to call on updates
        """
        self.real_time_enabled = True
        self.real_time_interval = interval
        self.real_time_callback = callback
        self._real_time_last_update = time.time()
        
        logger.info(f"Enabled real-time updates with {interval}s interval")
    
    def disable_real_time_updates(self) -> None:
        """Disable real-time portfolio updates"""
        self.real_time_enabled = False
        logger.info("Disabled real-time updates")
    
    def check_for_updates(self) -> bool:
        """
        Check if portfolio needs updating based on real-time settings
        
        Returns:
            True if updates were performed, False otherwise
        """
        if not self.real_time_enabled:
            return False
        
        current_time = time.time()
        if (current_time - self._real_time_last_update) < self.real_time_interval:
            return False
        
        # Perform update
        self._real_time_last_update = current_time
        updated = self._update_portfolio_data()
        
        # Call callback if provided
        if updated and self.real_time_callback:
            try:
                self.real_time_callback(self)
            except Exception as e:
                logger.error(f"Error in real-time update callback: {e}")
        
        return updated
    
    def _update_portfolio_data(self) -> bool:
        """
        Update portfolio data with latest market information
        
        Returns:
            True if data was updated, False otherwise
        """
        # Implementation would depend on data sources
        # This is a placeholder
        logger.info("Updating portfolio data...")
        return True

def load_data(self, price_data: pd.DataFrame) -> None:
    """
    Load price data for portfolio analysis.
    
    Args:
        price_data: DataFrame of asset prices with DatetimeIndex
    """
    # Use more efficient data types
    for col in price_data.columns:
        if price_data[col].dtype == 'float64':
            price_data[col] = price_data[col].astype('float32')
    
    self.price_data = price_data
    
    # Calculate returns more efficiently
    self.returns_data = price_data.pct_change().dropna()
    
    # Pre-calculate covariance matrix for later use
    self._update_covariance_matrix()
    
    logger.info(f"Loaded price data with {len(price_data)} rows and {len(price_data.columns)} assets")

def _update_covariance_matrix(self) -> None:
    """Update the covariance matrix of returns for efficient calculations"""
    if self.returns_data is not None and len(self.returns_data) > 0:
        # Use parallel processing for large datasets
        if len(self.returns_data) > 1000 and self.use_parallel:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Split the calculation into chunks
                chunk_size = len(self.returns_data.columns) // self.max_workers
                chunks = [self.returns_data.iloc[:, i:i+chunk_size] 
                         for i in range(0, len(self.returns_data.columns), chunk_size)]
                
                # Calculate partial covariances
                partial_results = list(executor.map(
                    lambda df: df.cov(), 
                    chunks
                ))
                
                # Combine results (simplified approach)
                self.cov_matrix = pd.concat(partial_results).fillna(0)
        else:
            # Standard calculation for smaller datasets
            self.cov_matrix = self.returns_data.cov()
        
        logger.debug(f"Updated covariance matrix with shape {self.cov_matrix.shape}")
    else:
        self.cov_matrix = None


def run_backtest(self, 
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                initial_investment: float = 10000.0,
                rebalance_frequency: str = 'monthly') -> Dict[str, Any]:
    """
    Run a backtest on the current portfolio.
    
    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_investment: Initial investment amount
        rebalance_frequency: Frequency to rebalance portfolio
        
    Returns:
        Dictionary with backtest results
    """
    if self.price_data is None or self.current_weights is None:
        raise ValueError("Price data and weights must be set before running backtest")
    
    # Use cached result if available
    cache_key = self._cache_key(
        "backtest", 
        start_date, 
        end_date, 
        initial_investment, 
        rebalance_frequency
    )
    cached_result = self._cache_get(cache_key)
    if cached_result is not None:
        logger.info("Using cached backtest results")
        self.backtest_results = cached_result
        return self.backtest_results
    
    # Run backtest with parallel processing for large datasets
    if len(self.price_data) > 1000 and self.use_parallel:
        logger.info("Using parallel processing for large dataset backtest")
        self.backtest_results = self._run_backtest_parallel(
            start_date=start_date,
            end_date=end_date,
            initial_investment=initial_investment,
            rebalance_frequency=rebalance_frequency
        )
    else:
        self.backtest_results = backtest_portfolio(
            price_data=self.price_data,
            weights=self.current_weights,
            start_date=start_date,
            end_date=end_date,
            initial_investment=initial_investment,
            rebalance_frequency=rebalance_frequency,
            config=self.config
        )
    
    # Cache the results
    self._cache_set(cache_key, self.backtest_results)
    
    logger.info(f"Completed backtest: Final value=${self.backtest_results['final_value']:.2f}")
    return self.backtest_results

def _run_backtest_parallel(self,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          initial_investment: float = 10000.0,
                          rebalance_frequency: str = 'monthly') -> Dict[str, Any]:
    """Run backtest with parallel processing for performance"""
    # Split the date range into chunks for parallel processing
    if start_date is None:
        start_date = self.price_data.index[0].strftime('%Y-%m-%d')
    if end_date is None:
        end_date = self.price_data.index[-1].strftime('%Y-%m-%d')
    
    # Filter data to the date range
    mask = (self.price_data.index >= pd.Timestamp(start_date)) & (self.price_data.index <= pd.Timestamp(end_date))
    date_range = self.price_data.index[mask]
    
    # If date range is too small, just run normally
    if len(date_range) < 100:
        return backtest_portfolio(
            price_data=self.price_data,
            weights=self.current_weights,
            start_date=start_date,
            end_date=end_date,
            initial_investment=initial_investment,
            rebalance_frequency=rebalance_frequency,
            config=self.config
        )
    
    # Split into chunks
    chunk_size = len(date_range) // min(self.max_workers, 4)
    chunks = []
    for i in range(0, len(date_range), chunk_size):
        chunk_end = min(i + chunk_size, len(date_range) - 1)
        chunks.append((date_range[i].strftime('%Y-%m-%d'), date_range[chunk_end].strftime('%Y-%m-%d')))
    
    # Process each chunk in parallel
    with ProcessPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as executor:
        futures = []
        for chunk_start, chunk_end in chunks:
            futures.append(executor.submit(
                backtest_portfolio,
                price_data=self.price_data,
                weights=self.current_weights,
                start_date=chunk_start,
                end_date=chunk_end,
                initial_investment=initial_investment,
                rebalance_frequency=rebalance_frequency,
                config=self.config
            ))
        
        # Collect results
        chunk_results = [future.result() for future in futures]
    
    # Combine results
    combined_results = self._combine_backtest_chunks(chunk_results, initial_investment)
    return combined_results

def _combine_backtest_chunks(self, chunk_results: List[Dict[str, Any]], initial_investment: float) -> Dict[str, Any]:
    """Combine backtest chunks into a single result"""
    # Combine portfolio values
    all_values = pd.concat([chunk['portfolio_values'] for chunk in chunk_results])
    all_values = all_values.sort_index()
    
    # Recalculate metrics based on combined values
    start_value = all_values.iloc[0]
    final_value = all_values.iloc[-1]
    total_return = (final_value / start_value) - 1
    
    # Calculate annualized return
    days = (all_values.index[-1] - all_values.index[0]).days
    annual_return = ((1 + total_return) ** (365 / days)) - 1
    
    # Calculate drawdowns
    drawdowns = all_values / all_values.cummax() - 1
    max_drawdown = drawdowns.min()
    
    # Calculate volatility
    returns = all_values.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)  # Annualized
    
    # Calculate Sharpe ratio
    risk_free_rate = self.config.risk_free_rate
    sharpe_ratio = (annual_return - risk_free_rate) / volatility
    
    return {
        'portfolio_values': all_values,
        'returns': returns,
        'drawdowns': drawdowns,
        'initial_investment': initial_investment,
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'start_date': all_values.index[0].strftime('%Y-%m-%d'),
        'end_date': all_values.index[-1].strftime('%Y-%m-%d')
    }