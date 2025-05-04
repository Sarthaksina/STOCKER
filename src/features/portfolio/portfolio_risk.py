"""
Portfolio Risk Analysis Module for STOCKER Pro

This module provides functions for analyzing portfolio risk.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.stats import norm, t
import matplotlib.pyplot as plt
import seaborn as sns

from src.features.portfolio.portfolio_config import PortfolioConfig

# Configure logging
logger = logging.getLogger(__name__)

# --- Helper Functions (Keep these if they are general utilities) ---
def calculate_var(returns: pd.Series, confidence_level: float = 0.95, method: str = 'historical') -> float:
    """
    Calculate Value at Risk (VaR)
    
    Args:
        returns: Series of portfolio returns
        confidence_level: Confidence level for VaR (e.g., 0.95 for 95%)
        method: Method for VaR calculation ('historical', 'parametric_norm', 'parametric_t')
        
    Returns:
        Value at Risk (negative value indicates loss)
    """
    if returns.empty:
        return np.nan
        
    if method == 'historical':
        var = returns.quantile(1 - confidence_level)
    elif method == 'parametric_norm':
        mean = returns.mean()
        std_dev = returns.std()
        var = norm.ppf(1 - confidence_level, loc=mean, scale=std_dev)
    elif method == 'parametric_t':
        mean = returns.mean()
        std_dev = returns.std()
        df = len(returns) - 1
        var = t.ppf(1 - confidence_level, df, loc=mean, scale=std_dev)
    else:
        raise ValueError(f"Unsupported VaR method: {method}")
        
    return var

def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95, method: str = 'historical') -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall (ES)
    
    Args:
        returns: Series of portfolio returns
        confidence_level: Confidence level for CVaR (e.g., 0.95 for 95%)
        method: Method for CVaR calculation ('historical', 'parametric_norm', 'parametric_t')

    Returns:
        Conditional Value at Risk (negative value indicates loss)
    """
    if returns.empty:
        return np.nan

    var = calculate_var(returns, confidence_level, method)
    
    # CVaR is the expected return conditional on the return being less than or equal to VaR
    cvar = returns[returns <= var].mean()
    
    # Note: Parametric CVaR calculations can be more complex, especially for t-distribution.
    # This implementation uses the historical method for CVaR regardless of the VaR method chosen,
    # which is a common simplification but might not be theoretically pure for parametric cases.
    # For more accurate parametric CVaR, specific formulas involving the distribution's PDF/CDF are needed.

    return cvar

def calculate_drawdown(returns: pd.Series) -> Tuple[pd.Series, float, float]:
    """
    Calculate drawdown series, max drawdown, and average drawdown.
    
    Args:
        returns: Series of portfolio returns.
        
    Returns:
        Tuple containing:
        - Drawdown series (percentage from peak)
        - Maximum drawdown
        - Average drawdown
    """
    if returns.empty:
        return pd.Series(dtype=float), np.nan, np.nan

    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    
    max_drawdown = drawdown.min()
    average_drawdown = drawdown.mean()
    
    return drawdown, max_drawdown, average_drawdown

# --- PortfolioRiskAnalyzer Class ---
class PortfolioRiskAnalyzer:
    """
    Provides advanced portfolio risk analysis capabilities.
    """
    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        Initialize risk analyzer.
        
        Args:
            config: Portfolio configuration object.
        """
        self.config = config or PortfolioConfig()

    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate various risk metrics for a given return series.
        
        Args:
            returns: Series of portfolio returns.
            
        Returns:
            Dictionary containing risk metrics.
        """
        if returns.empty:
            logger.warning("Return series is empty. Cannot calculate risk metrics.")
            return {
                'volatility': np.nan,
                'downside_deviation': np.nan,
                'var_historical_95': np.nan,
                'cvar_historical_95': np.nan,
                'var_parametric_norm_95': np.nan,
                'cvar_parametric_norm_95': np.nan, # Note: Uses historical method based on VaR
            }

        metrics = {}
        
        # Volatility (Annualized Standard Deviation)
        metrics['volatility'] = returns.std() * np.sqrt(self.config.annualization_factor)
        
        # Downside Deviation (Annualized)
        downside_returns = returns[returns < self.config.target_return_for_sortino]
        downside_deviation = np.sqrt(np.mean(np.square(downside_returns - self.config.target_return_for_sortino)))
        metrics['downside_deviation'] = downside_deviation * np.sqrt(self.config.annualization_factor)

        # Value at Risk (VaR) - Example for 95% confidence
        metrics['var_historical_95'] = calculate_var(returns, 0.95, 'historical')
        metrics['var_parametric_norm_95'] = calculate_var(returns, 0.95, 'parametric_norm')
        # Add other VaR levels/methods if needed

        # Conditional Value at Risk (CVaR) - Example for 95% confidence
        metrics['cvar_historical_95'] = calculate_cvar(returns, 0.95, 'historical')
        # Note: Parametric CVaR calculation here is simplified
        metrics['cvar_parametric_norm_95'] = calculate_cvar(returns, 0.95, 'parametric_norm') 
        # Add other CVaR levels/methods if needed

        return metrics

    def calculate_drawdown(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate drawdown metrics for a given return series.

        Args:
            returns: Series of portfolio returns.

        Returns:
            Dictionary containing drawdown metrics and series.
        """
        drawdown_series, max_dd, avg_dd = calculate_drawdown(returns)
        
        return {
            'max_drawdown': max_dd,
            'average_drawdown': avg_dd,
            'drawdown_series': drawdown_series
        }

    def run_monte_carlo_simulation(self, 
                                   returns: pd.Series, 
                                   num_simulations: int = 1000, 
                                   num_periods: int = 252) -> pd.DataFrame:
        """
        Run Monte Carlo simulation on portfolio returns.
        
        Args:
            returns: Historical daily returns series.
            num_simulations: Number of simulation paths.
            num_periods: Number of periods to simulate (e.g., 252 for one year).
            
        Returns:
            DataFrame where each column is a simulated price path.
        """
        if returns.empty:
            logger.warning("Return series is empty. Cannot run Monte Carlo simulation.")
            return pd.DataFrame()
            
        mean_return = returns.mean()
        std_dev = returns.std()
        
        # Generate random shocks (assuming normal distribution)
        random_shocks = np.random.normal(mean_return, std_dev, size=(num_periods, num_simulations))
        
        # Simulate price paths (starting from 1)
        simulated_returns = pd.DataFrame(random_shocks)
        simulated_paths = (1 + simulated_returns).cumprod()
        
        # Prepend starting value of 1
        start_values = pd.DataFrame(np.ones((1, num_simulations)), columns=simulated_paths.columns)
        simulated_paths = pd.concat([start_values, simulated_paths], ignore_index=True)

        return simulated_paths

    def run_stress_test(self, 
                        returns: pd.Series, 
                        scenarios: Dict[str, float]) -> Dict[str, float]:
        """
        Run simple stress tests by applying shocks to returns.
        
        Args:
            returns: Historical daily returns series.
            scenarios: Dictionary where keys are scenario names and values are shock factors 
                       (e.g., {'Market Crash': -0.1} for a 10% drop).
                       Note: This is a simplified approach. Real stress tests often involve
                       factor models or specific historical event replication.
                       
        Returns:
            Dictionary with potential portfolio value change under each scenario.
            (This implementation calculates the impact of a single-period shock)
        """
        if returns.empty:
            logger.warning("Return series is empty. Cannot run stress test.")
            return {scenario: np.nan for scenario in scenarios}

        results = {}
        # Simple approach: Apply the shock as a one-period return
        for scenario, shock in scenarios.items():
            # This calculates the immediate impact of the shock
            results[scenario] = shock 
            # A more complex approach would simulate the portfolio value after the shock
            # e.g., portfolio_value * (1 + shock)

        return results

# --- Removed Standalone Duplicate Functions ---
# The following functions were duplicates of methods within the PortfolioRiskAnalyzer class
# and have been removed to maintain a single source of truth.

# def run_stress_test(returns: pd.Series, scenarios: Dict[str, float]) -> Dict[str, float]:
#     """ Duplicate removed """
#     pass 

# def run_monte_carlo_simulation(returns: pd.Series, num_simulations: int = 1000, num_periods: int = 252) -> pd.DataFrame:
#     """ Duplicate removed """
#     pass

# --- MonteCarloSimulator Class (Moved from portfolio/monte_carlo.py) ---
class MonteCarloSimulator:
    """
    Monte Carlo simulator for portfolio analysis
    """
    
    def __init__(self, 
                 returns_data: Optional[pd.DataFrame] = None,
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
        if returns_data is not None:
            self.num_assets = returns_data.shape[1]
            
            # Use equal weights if not provided
            if weights is None:
                self.weights = np.ones(self.num_assets) / self.num_assets
            else:
                # Normalize weights to sum to 1
                self.weights = weights / np.sum(weights)
            
            # Calculate mean returns and covariance matrix
            self.mean_returns = returns_data.mean().values
            self.cov_matrix = returns_data.cov().values
        else:
            self.num_assets = 0
            self.weights = None
            self.mean_returns = None
            self.cov_matrix = None
        
        self.risk_free_rate = risk_free_rate
        self.simulation_years = simulation_years
        self.trading_days = trading_days
        self.num_simulations = num_simulations
        
        # Simulation results
        self.simulation_results = None
    
    def _calculate_portfolio_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio metrics based on historical data
        
        Returns:
            Dictionary with portfolio metrics
        """
        if self.returns_data is None or self.weights is None:
            raise ValueError("Returns data and weights must be set before calculating metrics")
            
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
    
    def run_simulation(self, 
                      returns: Optional[pd.DataFrame] = None,
                      weights: Optional[np.ndarray] = None,
                      initial_investment: float = 10000.0,
                      simulation_length: Optional[int] = None,
                      num_simulations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for the portfolio
        
        Args:
            returns: Optional DataFrame with asset returns to override the one set in constructor
            weights: Optional portfolio weights to override the ones set in constructor
            initial_investment: Initial investment amount
            simulation_length: Optional number of periods to simulate
            num_simulations: Optional number of simulations to run
            
        Returns:
            Dictionary with simulation results
        """
        # Use provided parameters or fall back to instance variables
        if returns is not None:
            self.returns_data = returns
            self.num_assets = returns.shape[1]
            self.mean_returns = returns.mean().values
            self.cov_matrix = returns.cov().values
        
        if weights is not None:
            self.weights = weights / np.sum(weights)
            
        if simulation_length is None:
            simulation_length = self.simulation_years * self.trading_days
            
        if num_simulations is None:
            num_simulations = self.num_simulations
            
        if self.returns_data is None or self.weights is None:
            raise ValueError("Returns data and weights must be provided")
        
        # Calculate time steps
        total_steps = simulation_length
        
        # Initialize array for simulation results
        simulation_results = np.zeros((num_simulations, total_steps + 1))
        simulation_results[:, 0] = 1  # Start with $1
        
        # Generate random returns using multivariate normal distribution
        for sim in range(num_simulations):
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
        for sim in range(num_simulations):
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
            "num_simulations": num_simulations,
            "initial_investment": initial_investment,
            "final_values": (final_values * initial_investment).tolist()
        }
    
    def plot_simulations(self, 
                         num_paths_to_plot: int = 100, 
                         figsize: Tuple[int, int] = (12, 8),
                         title: str = "Monte Carlo Simulation of Portfolio Performance",
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the Monte Carlo simulation results
        
        Args:
            num_paths_to_plot: Number of random paths to plot
            figsize: Figure size
            title: Plot title
            save_path: Path to save the figure (if None, the figure will be displayed)
            
        Returns:
            Matplotlib figure
        """
        if self.simulation_results is None:
            raise ValueError("No simulation results available. Run run_simulation() first.")
        
        fig = plt.figure(figsize=figsize)
        
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
        
        return fig
    
    def plot_distribution(self,
                          figsize: Tuple[int, int] = (12, 8),
                          title: str = "Distribution of Final Portfolio Values",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the distribution of final portfolio values
        
        Args:
            figsize: Figure size
            title: Plot title
            save_path: Path to save the figure (if None, the figure will be displayed)
            
        Returns:
            Matplotlib figure
        """
        if self.simulation_results is None:
            raise ValueError("No simulation results available. Run run_simulation() first.")
        
        final_values = self.simulation_results[:, -1]
        
        fig = plt.figure(figsize=figsize)
        
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
        
        return fig

    def get_extreme_scenarios(self, confidence_level: float = 0.05) -> Dict[str, Any]:
        """
        Get best and worst case scenarios with given confidence level
        
        Args:
            confidence_level: Confidence level for extreme scenarios
            
        Returns:
            Dictionary with extreme scenario information
        """
        if self.simulation_results is None:
            self.run_simulation()
        
        final_values = self.simulation_results[:, -1]
        
        return {
            "best_case": float(np.percentile(final_values, 100*(1-confidence_level))),
            "worst_case": float(np.percentile(final_values, 100*confidence_level)),
            "confidence_level": confidence_level
        }