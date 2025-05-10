"""
Portfolio risk analysis functionality for STOCKER Pro.

This module provides risk analysis methods for portfolio risk assessment,
stress testing, and risk decomposition.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import scipy.stats as stats

from src.core.logging import logger
from src.core.exceptions import StockerBaseException


class RiskAnalysisError(StockerBaseException):
    """Exception raised for portfolio risk analysis errors."""
    pass


def calculate_portfolio_risk(returns: pd.DataFrame, weights: Union[Dict[str, float], List[float], np.ndarray]) -> Dict[str, Any]:
    """
    Calculate comprehensive portfolio risk metrics.
    
    Args:
        returns: DataFrame with asset returns
        weights: Asset weights (dictionary, list, or array)
        
    Returns:
        Dictionary with risk metrics
        
    Raises:
        RiskAnalysisError: If calculation fails
    """
    try:
        # Convert weights to array if needed
        if isinstance(weights, dict):
            weight_array = np.array([weights.get(col, 0.0) for col in returns.columns])
        else:
            weight_array = np.array(weights)
        
        # Normalize weights to sum to 1
        weight_array = weight_array / np.sum(weight_array)
        
        # Calculate covariance matrix
        cov_matrix = returns.cov() * 252  # Annualized
        
        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weight_array.T, np.dot(cov_matrix, weight_array)))
        
        # Calculate beta contribution (if first column is market)
        market_col = returns.columns[0]
        market_volatility = np.sqrt(returns[market_col].var() * 252)
        
        betas = {}
        for i, col in enumerate(returns.columns):
            if col == market_col:
                betas[col] = 1.0
            else:
                # Calculate beta = Cov(asset, market) / Var(market)
                beta = returns[col].cov(returns[market_col]) / returns[market_col].var()
                betas[col] = beta
        
        # Calculate marginal contributions to risk
        marginal_contrib = np.dot(cov_matrix, weight_array) / portfolio_volatility
        
        # Calculate component contribution to risk
        component_contrib = {}
        for i, col in enumerate(returns.columns):
            component_contrib[col] = weight_array[i] * marginal_contrib[i]
        
        # Calculate percentage contribution to risk
        percentage_contrib = {}
        for col, contrib in component_contrib.items():
            percentage_contrib[col] = contrib / portfolio_volatility * 100
        
        # Calculate diversification ratio
        weighted_volatilities = 0
        for i, col in enumerate(returns.columns):
            asset_volatility = np.sqrt(returns[col].var() * 252)
            weighted_volatilities += weight_array[i] * asset_volatility
        
        diversification_ratio = weighted_volatilities / portfolio_volatility
        
        # Calculate factor exposures if additional factors available
        # (placeholder for more advanced risk models)
        
        return {
            "volatility": portfolio_volatility,
            "diversification_ratio": diversification_ratio,
            "betas": betas,
            "marginal_contributions": {col: marginal_contrib[i] for i, col in enumerate(returns.columns)},
            "component_contributions": component_contrib,
            "percentage_contributions": percentage_contrib
        }
        
    except Exception as e:
        logger.error(f"Error calculating portfolio risk: {e}")
        raise RiskAnalysisError(f"Failed to calculate portfolio risk: {e}")


def calculate_var(returns: pd.Series, confidence_level: float = 0.95, method: str = 'historical') -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Series with portfolio returns
        confidence_level: Confidence level for VaR (e.g., 0.95 for 95%)
        method: Method for calculation ('historical', 'parametric', 'monte_carlo')
        
    Returns:
        VaR value (positive number representing loss)
        
    Raises:
        RiskAnalysisError: If calculation fails
    """
    try:
        if method == 'historical':
            # Historical VaR (non-parametric)
            var = -np.percentile(returns, 100 * (1 - confidence_level))
            
        elif method == 'parametric':
            # Parametric VaR (assuming normal distribution)
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mean + z_score * std)
            
        elif method == 'monte_carlo':
            # Monte Carlo VaR
            # Basic implementation with normal distribution assumption
            mean = returns.mean()
            std = returns.std()
            
            # Generate random returns
            np.random.seed(42)
            n_simulations = 10000
            simulated_returns = np.random.normal(mean, std, n_simulations)
            
            # Calculate VaR from simulations
            var = -np.percentile(simulated_returns, 100 * (1 - confidence_level))
            
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        return float(var)
        
    except Exception as e:
        logger.error(f"Error calculating VaR: {e}")
        raise RiskAnalysisError(f"Failed to calculate VaR: {e}")


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95, method: str = 'historical') -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall (ES).
    
    Args:
        returns: Series with portfolio returns
        confidence_level: Confidence level for CVaR (e.g., 0.95 for 95%)
        method: Method for calculation ('historical', 'parametric', 'monte_carlo')
        
    Returns:
        CVaR value (positive number representing loss)
        
    Raises:
        RiskAnalysisError: If calculation fails
    """
    try:
        var = calculate_var(returns, confidence_level, method)
        
        if method == 'historical':
            # Historical CVaR
            tail_returns = returns[returns <= -var]
            if len(tail_returns) > 0:
                cvar = -tail_returns.mean()
            else:
                cvar = var  # Fallback if no returns beyond VaR
            
        elif method == 'parametric':
            # Parametric CVaR
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            cvar = -(mean + std * stats.norm.pdf(z_score) / (1 - confidence_level))
            
        elif method == 'monte_carlo':
            # Monte Carlo CVaR
            # Basic implementation with normal distribution assumption
            mean = returns.mean()
            std = returns.std()
            
            # Generate random returns
            np.random.seed(42)
            n_simulations = 10000
            simulated_returns = np.random.normal(mean, std, n_simulations)
            
            # Calculate CVaR from simulations
            tail_returns = simulated_returns[simulated_returns <= -var]
            cvar = -tail_returns.mean()
            
        else:
            raise ValueError(f"Unknown CVaR method: {method}")
        
        return float(cvar)
        
    except Exception as e:
        logger.error(f"Error calculating CVaR: {e}")
        raise RiskAnalysisError(f"Failed to calculate CVaR: {e}")


def calculate_drawdown(returns: pd.Series) -> Dict[str, Any]:
    """
    Calculate drawdown metrics.
    
    Args:
        returns: Series with portfolio returns
        
    Returns:
        Dictionary with drawdown metrics
        
    Raises:
        RiskAnalysisError: If calculation fails
    """
    try:
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate drawdowns
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns / running_max - 1) * 100  # Convert to percentages
        
        # Calculate maximum drawdown
        max_drawdown = drawdowns.min()
        
        # Calculate drawdown duration
        drawdown_started = False
        drawdown_lengths = []
        current_drawdown_length = 0
        
        for dd in drawdowns:
            if dd < 0:
                drawdown_started = True
                current_drawdown_length += 1
            elif drawdown_started:
                drawdown_started = False
                drawdown_lengths.append(current_drawdown_length)
                current_drawdown_length = 0
        
        # If still in drawdown at the end of the period
        if drawdown_started:
            drawdown_lengths.append(current_drawdown_length)
        
        # Calculate average drawdown
        avg_drawdown = drawdowns[drawdowns < 0].mean() if len(drawdowns[drawdowns < 0]) > 0 else 0
        
        # Calculate average drawdown duration
        avg_duration = np.mean(drawdown_lengths) if drawdown_lengths else 0
        
        # Calculate drawdown to recovery ratio
        recovery_lengths = []
        in_drawdown = False
        drawdown_start = 0
        
        for i, dd in enumerate(drawdowns):
            if not in_drawdown and dd < 0:
                in_drawdown = True
                drawdown_start = i
            elif in_drawdown and dd >= 0:
                in_drawdown = False
                recovery_length = i - drawdown_start
                recovery_lengths.append(recovery_length)
        
        avg_recovery = np.mean(recovery_lengths) if recovery_lengths else 0
        
        return {
            "max_drawdown": max_drawdown,
            "avg_drawdown": avg_drawdown,
            "avg_duration": avg_duration,
            "avg_recovery": avg_recovery,
            "recovery_ratio": avg_recovery / avg_duration if avg_duration > 0 else 0,
            "drawdown_series": drawdowns
        }
        
    except Exception as e:
        logger.error(f"Error calculating drawdown: {e}")
        raise RiskAnalysisError(f"Failed to calculate drawdown: {e}")


def stress_test_portfolio(returns: pd.DataFrame, weights: Union[Dict[str, float], List[float], np.ndarray], 
                        scenarios: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
    """
    Perform stress testing on the portfolio.
    
    Args:
        returns: DataFrame with asset returns
        weights: Asset weights (dictionary, list, or array)
        scenarios: Dictionary of scenarios with asset shocks
        
    Returns:
        Dictionary with stress test results
        
    Raises:
        RiskAnalysisError: If test fails
    """
    try:
        # Convert weights to dictionary if not already
        if not isinstance(weights, dict):
            weights = {col: weight for col, weight in zip(returns.columns, weights)}
        
        # Define default scenarios if none provided
        if scenarios is None:
            scenarios = {
                "market_crash": {
                    asset: -0.15 if 'bond' not in asset.lower() else -0.05 
                    for asset in returns.columns
                },
                "inflation_spike": {
                    asset: -0.10 if 'bond' in asset.lower() or 'treasury' in asset.lower() else 0.05
                    for asset in returns.columns
                },
                "tech_bubble": {
                    asset: -0.25 if 'tech' in asset.lower() else -0.05
                    for asset in returns.columns
                },
                "recession": {
                    asset: -0.20 if 'equity' in asset.lower() else 0.05 if 'bond' in asset.lower() else -0.10
                    for asset in returns.columns
                }
            }
        
        # Calculate historical statistics for comparison
        historical_mean = returns.mean()
        historical_std = returns.std()
        historical_corr = returns.corr()
        
        # Calculate portfolio historical metrics
        portfolio_returns = pd.Series(0.0, index=returns.index)
        
        for asset, weight in weights.items():
            if asset in returns.columns:
                portfolio_returns += weight * returns[asset]
        
        historical_portfolio_mean = portfolio_returns.mean()
        historical_portfolio_std = portfolio_returns.std()
        historical_var_95 = calculate_var(portfolio_returns, 0.95)
        historical_cvar_95 = calculate_cvar(portfolio_returns, 0.95)
        
        # Run scenarios
        scenario_results = {}
        
        for scenario_name, shocks in scenarios.items():
            # Apply shocks to calculate scenario impact
            scenario_impact = 0.0
            
            for asset, shock in shocks.items():
                if asset in weights:
                    scenario_impact += weights[asset] * shock
            
            # Calculate scenario metrics
            scenario_results[scenario_name] = {
                "portfolio_impact": scenario_impact * 100,  # Convert to percent
                "worst_asset": max(shocks.items(), key=lambda x: abs(x[1]))[0],
                "worst_shock": max(shocks.items(), key=lambda x: abs(x[1]))[1]
            }
        
        # Calculate extreme scenarios based on historical data
        worst_day_returns = returns.min()
        worst_day_impact = sum(weights.get(asset, 0) * ret for asset, ret in worst_day_returns.items())
        
        best_day_returns = returns.max()
        best_day_impact = sum(weights.get(asset, 0) * ret for asset, ret in best_day_returns.items())
        
        # Combine results
        result = {
            "scenarios": scenario_results,
            "historical": {
                "mean_return": historical_portfolio_mean,
                "volatility": historical_portfolio_std,
                "var_95": historical_var_95,
                "cvar_95": historical_cvar_95,
                "worst_day": worst_day_impact,
                "best_day": best_day_impact
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error performing stress test: {e}")
        raise RiskAnalysisError(f"Failed to perform stress test: {e}")


def tail_risk_analysis(returns: pd.Series, confidence_levels: List[float] = [0.95, 0.99, 0.999]) -> Dict[str, Any]:
    """
    Perform tail risk analysis.
    
    Args:
        returns: Series with portfolio returns
        confidence_levels: List of confidence levels for analysis
        
    Returns:
        Dictionary with tail risk analysis
        
    Raises:
        RiskAnalysisError: If analysis fails
    """
    try:
        # Calculate basic statistics
        mean = returns.mean()
        std = returns.std()
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Calculate VaR and CVaR at different confidence levels
        var_values = {}
        cvar_values = {}
        
        for level in confidence_levels:
            var_values[str(level)] = calculate_var(returns, level)
            cvar_values[str(level)] = calculate_cvar(returns, level)
        
        # Calculate tail risk metrics
        jarque_bera = stats.jarque_bera(returns)
        
        # Test for normality
        normality_test = stats.normaltest(returns)
        
        # Calculate tail correlation
        # (correlation when market is down vs normal times)
        if len(returns) > 0 and len(returns) == len(returns.index):
            # Assuming first column is market
            market_col = returns.index[0]
            market_returns = returns
            
            # Down market and normal market days
            down_market = market_returns < 0
            normal_market = ~down_market
            
            if sum(down_market) > 10 and sum(normal_market) > 10:
                down_market_return = returns[down_market].mean()
                normal_market_return = returns[normal_market].mean()
                
                down_to_normal_ratio = down_market_return / normal_market_return if normal_market_return != 0 else float('inf')
            else:
                down_to_normal_ratio = None
        else:
            down_to_normal_ratio = None
        
        return {
            "basic_stats": {
                "mean": mean,
                "std": std,
                "skewness": skewness,
                "kurtosis": kurtosis
            },
            "var": var_values,
            "cvar": cvar_values,
            "normality": {
                "jarque_bera_statistic": jarque_bera.statistic,
                "jarque_bera_pvalue": jarque_bera.pvalue,
                "normality_test_statistic": normality_test.statistic,
                "normality_test_pvalue": normality_test.pvalue
            },
            "down_to_normal_ratio": down_to_normal_ratio
        }
        
    except Exception as e:
        logger.error(f"Error performing tail risk analysis: {e}")
        raise RiskAnalysisError(f"Failed to perform tail risk analysis: {e}") 