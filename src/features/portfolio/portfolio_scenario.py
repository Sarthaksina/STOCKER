"""
Portfolio Scenario Analysis Module for STOCKER Pro

This module provides scenario analysis capabilities for portfolio stress testing.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import scipy.stats as stats

from .portfolio_config import PortfolioConfig
from .portfolio_risk import PortfolioRiskAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

class ScenarioAnalyzer:
    """
    Scenario analysis for portfolio stress testing
    """
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        Initialize scenario analyzer
        
        Args:
            config: Portfolio configuration
        """
        self.config = config or PortfolioConfig()
        self.risk_analyzer = PortfolioRiskAnalyzer(config=self.config)
        
    def run_historical_scenario(self,
                              portfolio_returns: pd.Series,
                              scenario_period: Tuple[str, str],
                              initial_investment: float = 10000.0) -> Dict[str, Any]:
        """
        Run a historical scenario analysis
        
        Args:
            portfolio_returns: Series of portfolio returns
            scenario_period: Tuple of (start_date, end_date) for the scenario
            initial_investment: Initial investment amount
            
        Returns:
            Dictionary with scenario analysis results
        """
        start_date, end_date = scenario_period
        
        # Extract scenario period returns
        scenario_returns = portfolio_returns.loc[start_date:end_date]
        
        if len(scenario_returns) == 0:
            raise ValueError(f"No data found for scenario period {start_date} to {end_date}")
        
        # Calculate portfolio value during scenario
        scenario_values = initial_investment * (1 + scenario_returns).cumprod()
        
        # Calculate key metrics
        total_return = (scenario_values.iloc[-1] / scenario_values.iloc[0]) - 1
        max_drawdown = (scenario_values / scenario_values.cummax() - 1).min()
        
        # Calculate risk metrics for the scenario period
        risk_metrics = self.risk_analyzer.calculate_risk_metrics(scenario_returns)
        
        return {
            'scenario_name': f"Historical: {start_date} to {end_date}",
            'scenario_returns': scenario_returns,
            'scenario_values': scenario_values,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'volatility': risk_metrics['volatility'],
            'var_95': risk_metrics['var_95'],
            'cvar_95': risk_metrics['cvar_95'],
            'worst_day': scenario_returns.min(),
            'best_day': scenario_returns.max(),
            'num_days': len(scenario_returns),
            'profitable_days': (scenario_returns > 0).sum() / len(scenario_returns)
        }
    
    def run_monte_carlo_scenario(self,
                               portfolio_returns: pd.Series,
                               weights: np.ndarray,
                               num_simulations: int = 1000,
                               time_horizon: int = 252,  # 1 year
                               initial_investment: float = 10000.0,
                               confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Run a Monte Carlo scenario analysis
        
        Args:
            portfolio_returns: Series of portfolio returns
            weights: Array of portfolio weights
            num_simulations: Number of simulations to run
            time_horizon: Time horizon in days
            initial_investment: Initial investment amount
            confidence_level: Confidence level for VaR and CVaR
            
        Returns:
            Dictionary with scenario analysis results
        """
        # Calculate portfolio mean and covariance
        returns_data = portfolio_returns.values
        mu = np.mean(returns_data)
        sigma = np.std(returns_data)
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        sim_returns = np.random.normal(
            loc=mu,
            scale=sigma,
            size=(time_horizon, num_simulations)
        )
        
        # Calculate cumulative returns
        sim_cumulative_returns = np.cumprod(1 + sim_returns, axis=0)
        
        # Calculate final values
        final_values = initial_investment * sim_cumulative_returns[-1, :]
        
        # Calculate percentiles
        percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        percentile_values = np.percentile(final_values, [p * 100 for p in percentiles])
        
        # Calculate VaR and CVaR
        var_index = int((1 - confidence_level) * num_simulations)
        sorted_returns = np.sort(sim_cumulative_returns[-1, :] - 1)
        var = -sorted_returns[var_index]
        cvar = -np.mean(sorted_returns[:var_index])
        
        # Calculate probability of loss
        prob_loss = np.mean(final_values < initial_investment)
        
        # Create a representative path for each percentile
        paths = {}
        for i, p in enumerate(percentiles):
            target_value = percentile_values[i]
            closest_sim = np.argmin(np.abs(final_values - target_value))
            paths[f"p{int(p*100)}"] = sim_cumulative_returns[:, closest_sim] * initial_investment
        
        return {
            'scenario_name': 'Monte Carlo Simulation',
            'num_simulations': num_simulations,
            'time_horizon': time_horizon,
            'initial_investment': initial_investment,
            'final_values': final_values.tolist(),
            'percentile_values': {f"p{int(p*100)}": val for p, val in zip(percentiles, percentile_values)},
            'var': var,
            'cvar': cvar,
            'prob_loss': prob_loss,
            'paths': {k: v.tolist() for k, v in paths.items()},
            'time_points': list(range(time_horizon))
        }
    
    def run_stress_test(self,
                      portfolio_returns: pd.Series,
                      stress_scenarios: Dict[str, Dict[str, float]],
                      initial_investment: float = 10000.0) -> Dict[str, Any]:
        """
        Run stress tests on portfolio
        
        Args:
            portfolio_returns: Series of portfolio returns
            stress_scenarios: Dictionary mapping scenario names to dictionaries of
                              asset return shocks
            initial_investment: Initial investment amount
            
        Returns:
            Dictionary with stress test results
        """
        # Calculate portfolio statistics
        mean_return = portfolio_returns.mean()
        volatility = portfolio_returns.std()
        
        # Run each stress scenario
        results = {}
        for scenario_name, shocks in stress_scenarios.items():
            # Apply shock to mean return
            shocked_return = mean_return + shocks.get('return_shock', 0)
            
            # Apply shock to volatility
            shocked_vol = volatility * (1 + shocks.get('vol_shock', 0))
            
            # Simulate portfolio value
            num_days = shocks.get('duration', 21)  # Default to 1 month (21 trading days)
            
            # Generate random returns with shocked parameters
            np.random.seed(42)  # For reproducibility
            sim_returns = np.random.normal(
                loc=shocked_return,
                scale=shocked_vol,
                size=num_days
            )
            
            # Calculate cumulative returns
            cum_returns = np.cumprod(1 + sim_returns)
            
            # Calculate final value
            final_value = initial_investment * cum_returns[-1]
            
            # Calculate drawdown
            max_value = initial_investment * np.max(cum_returns)
            max_drawdown = (initial_investment * np.min(cum_returns / np.maximum.accumulate(cum_returns))) - initial_investment
            
            results[scenario_name] = {
                'initial_value': initial_investment,
                'final_value': float(final_value),
                'return': float(cum_returns[-1] - 1),
                'max_value': float(max_value),
                'max_drawdown': float(max_drawdown),
                'max_drawdown_pct': float(max_drawdown / initial_investment),
                'values': (initial_investment * cum_returns).tolist(),
                'time_points': list(range(num_days))
            }
            
        return {
            'scenario_results': results,
            'initial_investment': initial_investment,
            'baseline_mean': mean_return,
            'baseline_volatility': volatility
        }
    
    def define_standard_scenarios(self) -> Dict[str, Dict[str, float]]:
        """
        Define standard stress test scenarios
        
        Returns:
            Dictionary of standard scenarios
        """
        return {
            'Market Crash': {
                'return_shock': -0.15,  # -15% shock to returns
                'vol_shock': 2.0,       # 200% increase in volatility
                'duration': 21          # 21 days (about 1 month)
            },
            'Economic Recession': {
                'return_shock': -0.08,  # -8% shock to returns
                'vol_shock': 1.0,       # 100% increase in volatility
                'duration': 63          # 63 days (about 3 months)
            },
            'Interest Rate Hike': {
                'return_shock': -0.05,  # -5% shock to returns
                'vol_shock': 0.5,       # 50% increase in volatility
                'duration': 21          # 21 days (about 1 month)
            },
            'Tech Bubble Burst': {
                'return_shock': -0.12,  # -12% shock to returns
                'vol_shock': 1.5,       # 150% increase in volatility
                'duration': 42          # 42 days (about 2 months)
            },
            'Commodity Shock': {
                'return_shock': -0.06,  # -6% shock to returns
                'vol_shock': 0.8,       # 80% increase in volatility
                'duration': 21          # 21 days (about 1 month)
            }
        }


# Add this method to ScenarioAnalyzer class
def run_rl_monte_carlo_scenario(self,
                              rl_optimizer,
                              returns_data: pd.DataFrame,
                              num_simulations: int = 1000,
                              time_horizon: int = 252,  # 1 year
                              initial_investment: float = 10000.0) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation with weights from RL optimizer
    
    Args:
        rl_optimizer: Trained RL portfolio optimizer
        returns_data: DataFrame of asset returns
        num_simulations: Number of simulations to run
        time_horizon: Time horizon in days
        initial_investment: Initial investment amount
        
    Returns:
        Dictionary with simulation results
    """
    # Get optimal weights from RL
    optimal_weights = rl_optimizer.get_current_weights()
    
    # Calculate portfolio returns
    portfolio_returns = returns_data.dot(optimal_weights)
    
    # Run Monte Carlo simulation
    return self.run_monte_carlo_scenario(
        portfolio_returns=portfolio_returns,
        weights=optimal_weights,
        num_simulations=num_simulations,
        time_horizon=time_horizon,
        initial_investment=initial_investment
    )