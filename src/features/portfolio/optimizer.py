"""
Portfolio optimization algorithms.

This module implements portfolio optimization algorithms, including
mean-variance optimization and efficient frontier methods.
"""

import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Dict, List, Optional, Tuple, Union, Callable

from src.core.exceptions import PortfolioOptimizationError
from src.core.logging import get_logger

logger = get_logger(__name__)


class PortfolioOptimizer:
    """
    Base class for portfolio optimization algorithms.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
        constraints: Optional[List[Dict]] = None
    ):
        """
        Initialize the PortfolioOptimizer.
        
        Args:
            returns (pd.DataFrame): DataFrame of asset returns with assets as columns.
            risk_free_rate (float, optional): Risk-free rate. Defaults to 0.0.
            constraints (List[Dict], optional): List of constraints for optimization.
        """
        self.returns = returns
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        self.risk_free_rate = risk_free_rate
        self.constraints = constraints or []
        
        # Calculate expected returns and covariance matrix
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        
        # Initialize optimal weights
        self.weights = None
    
    def _portfolio_return(self, weights: np.ndarray) -> float:
        """
        Calculate the expected portfolio return.
        
        Args:
            weights (np.ndarray): Array of asset weights.
            
        Returns:
            float: Expected portfolio return.
        """
        return np.sum(self.mean_returns * weights)
    
    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """
        Calculate the portfolio volatility (standard deviation).
        
        Args:
            weights (np.ndarray): Array of asset weights.
            
        Returns:
            float: Portfolio volatility.
        """
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def _portfolio_sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate the portfolio Sharpe ratio.
        
        Args:
            weights (np.ndarray): Array of asset weights.
            
        Returns:
            float: Portfolio Sharpe ratio.
        """
        portfolio_return = self._portfolio_return(weights)
        portfolio_volatility = self._portfolio_volatility(weights)
        
        if portfolio_volatility == 0:
            return 0
            
        return (portfolio_return - self.risk_free_rate) / portfolio_volatility
    
    def _negative_sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate the negative Sharpe ratio for minimization.
        
        Args:
            weights (np.ndarray): Array of asset weights.
            
        Returns:
            float: Negative portfolio Sharpe ratio.
        """
        return -self._portfolio_sharpe_ratio(weights)
    
    def optimize(self, objective: str = 'sharpe', target_return: Optional[float] = None) -> Dict:
        """
        Optimize the portfolio based on the specified objective.
        
        Args:
            objective (str, optional): Optimization objective.
                Options: 'sharpe', 'min_volatility', 'max_return', 'target_return'.
                Defaults to 'sharpe'.
            target_return (float, optional): Target return for 'target_return' objective.
                
        Returns:
            Dict: Optimization results.
            
        Raises:
            PortfolioOptimizationError: If optimization fails or invalid objective.
        """
        # Default constraints ensure weights sum to 1
        constraints = self.constraints.copy()
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Default bounds (0 to 1 for each asset)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1./self.n_assets] * self.n_assets)
        
        try:
            if objective == 'sharpe':
                # Maximize Sharpe ratio
                result = sco.minimize(
                    self._negative_sharpe_ratio,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                self.weights = result['x']
                
            elif objective == 'min_volatility':
                # Minimize volatility
                result = sco.minimize(
                    self._portfolio_volatility,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                self.weights = result['x']
                
            elif objective == 'max_return':
                # Maximize return
                result = sco.minimize(
                    lambda x: -self._portfolio_return(x),
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                self.weights = result['x']
                
            elif objective == 'target_return':
                if target_return is None:
                    raise PortfolioOptimizationError("Target return must be specified for 'target_return' objective.")
                
                # Add constraint for target return
                target_return_constraint = {
                    'type': 'eq',
                    'fun': lambda x: self._portfolio_return(x) - target_return
                }
                constraints.append(target_return_constraint)
                
                # Minimize volatility subject to target return
                result = sco.minimize(
                    self._portfolio_volatility,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                self.weights = result['x']
                
            else:
                raise PortfolioOptimizationError(f"Invalid optimization objective: {objective}")
            
            if not result['success']:
                raise PortfolioOptimizationError(f"Optimization failed: {result['message']}")
            
            # Calculate portfolio metrics
            portfolio_return = self._portfolio_return(self.weights)
            portfolio_volatility = self._portfolio_volatility(self.weights)
            portfolio_sharpe = self._portfolio_sharpe_ratio(self.weights)
            
            # Create dictionary of results
            optimization_result = {
                'weights': dict(zip(self.assets, self.weights)),
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': portfolio_sharpe,
                'optimization_result': result
            }
            
            return optimization_result
            
        except Exception as e:
            raise PortfolioOptimizationError(f"Portfolio optimization failed: {str(e)}")
    
    def get_efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Generate the efficient frontier.
        
        Args:
            n_points (int, optional): Number of points on the frontier. Defaults to 50.
            
        Returns:
            pd.DataFrame: DataFrame containing volatility, return, and sharpe ratio.
        """
        # Find minimum volatility and maximum return portfolios
        min_vol_result = self.optimize(objective='min_volatility')
        max_ret_result = self.optimize(objective='max_return')
        
        min_ret = min_vol_result['expected_return']
        max_ret = max_ret_result['expected_return']
        
        # Generate target returns between min and max
        target_returns = np.linspace(min_ret, max_ret, n_points)
        efficient_frontier = []
        
        for target_return in target_returns:
            try:
                result = self.optimize(objective='target_return', target_return=target_return)
                efficient_frontier.append({
                    'return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio']
                })
            except Exception as e:
                logger.warning(f"Failed to optimize for target return {target_return}: {str(e)}")
                continue
        
        return pd.DataFrame(efficient_frontier)


class EfficientFrontier(PortfolioOptimizer):
    """
    Efficient Frontier implementation with advanced optimization options.
    
    Extends the base PortfolioOptimizer with additional methods for
    efficient frontier analysis and visualization.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
        constraints: Optional[List[Dict]] = None,
        max_leverage: Optional[float] = None
    ):
        """
        Initialize the EfficientFrontier optimizer.
        
        Args:
            returns (pd.DataFrame): DataFrame of asset returns with assets as columns.
            risk_free_rate (float, optional): Risk-free rate. Defaults to 0.0.
            constraints (List[Dict], optional): List of constraints for optimization.
            max_leverage (float, optional): Maximum allowed leverage. If None, no leverage.
        """
        super().__init__(returns, risk_free_rate, constraints)
        self.max_leverage = max_leverage
        
        # Update bounds if leverage is allowed
        if max_leverage is not None:
            self.bounds = tuple((-max_leverage, max_leverage) for _ in range(self.n_assets))
        else:
            self.bounds = tuple((0, 1) for _ in range(self.n_assets))
    
    def optimize_for_target_risk(self, target_volatility: float) -> Dict:
        """
        Optimize the portfolio for a target level of risk (volatility).
        
        Args:
            target_volatility (float): Target portfolio volatility.
            
        Returns:
            Dict: Optimization results.
            
        Raises:
            PortfolioOptimizationError: If optimization fails.
        """
        constraints = self.constraints.copy()
        constraints.extend([
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: self._portfolio_volatility(x) - target_volatility}
        ])
        
        initial_weights = np.array([1./self.n_assets] * self.n_assets)
        
        try:
            # Maximize return subject to target volatility
            result = sco.minimize(
                lambda x: -self._portfolio_return(x),
                initial_weights,
                method='SLSQP',
                bounds=self.bounds,
                constraints=constraints
            )
            
            if not result['success']:
                raise PortfolioOptimizationError(f"Optimization failed: {result['message']}")
            
            self.weights = result['x']
            
            # Calculate portfolio metrics
            portfolio_return = self._portfolio_return(self.weights)
            portfolio_volatility = self._portfolio_volatility(self.weights)
            portfolio_sharpe = self._portfolio_sharpe_ratio(self.weights)
            
            # Create dictionary of results
            optimization_result = {
                'weights': dict(zip(self.assets, self.weights)),
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': portfolio_sharpe,
                'optimization_result': result
            }
            
            return optimization_result
        
        except Exception as e:
            raise PortfolioOptimizationError(f"Portfolio optimization failed: {str(e)}")
    
    def get_capital_market_line(self, n_points: int = 50) -> pd.DataFrame:
        """
        Generate the Capital Market Line.
        
        Args:
            n_points (int, optional): Number of points on the line. Defaults to 50.
            
        Returns:
            pd.DataFrame: DataFrame containing volatility, return, and allocation.
        """
        # Optimize for maximum Sharpe ratio
        max_sharpe_result = self.optimize(objective='sharpe')
        max_sharpe_return = max_sharpe_result['expected_return']
        max_sharpe_volatility = max_sharpe_result['volatility']
        
        # Generate points along the CML
        volatilities = np.linspace(0, max_sharpe_volatility * 2, n_points)
        cml_points = []
        
        for vol in volatilities:
            # Calculate return using CML equation
            ret = self.risk_free_rate + (max_sharpe_return - self.risk_free_rate) / max_sharpe_volatility * vol
            
            # Calculate allocation to risky portfolio vs risk-free asset
            if vol <= max_sharpe_volatility:
                risky_alloc = vol / max_sharpe_volatility
                rf_alloc = 1 - risky_alloc
            else:
                risky_alloc = vol / max_sharpe_volatility
                rf_alloc = 1 - risky_alloc
            
            cml_points.append({
                'return': ret,
                'volatility': vol,
                'risky_allocation': risky_alloc,
                'risk_free_allocation': rf_alloc
            })
        
        return pd.DataFrame(cml_points)


# Standalone functions

def optimize_portfolio(
    returns: pd.DataFrame,
    objective: str = 'sharpe',
    risk_free_rate: float = 0.0,
    constraints: Optional[List[Dict]] = None,
    target_return: Optional[float] = None,
    target_volatility: Optional[float] = None,
    max_leverage: Optional[float] = None
) -> Dict:
    """
    Optimize a portfolio based on the specified objective.
    
    Args:
        returns (pd.DataFrame): DataFrame of asset returns with assets as columns.
        objective (str, optional): Optimization objective.
            Options: 'sharpe', 'min_volatility', 'max_return', 'target_return', 'target_risk'.
            Defaults to 'sharpe'.
        risk_free_rate (float, optional): Risk-free rate. Defaults to 0.0.
        constraints (List[Dict], optional): List of constraints for optimization.
        target_return (float, optional): Target return for 'target_return' objective.
        target_volatility (float, optional): Target volatility for 'target_risk' objective.
        max_leverage (float, optional): Maximum allowed leverage. If None, no leverage.
            
    Returns:
        Dict: Optimization results.
        
    Raises:
        PortfolioOptimizationError: If optimization fails or invalid objective.
    """
    if objective == 'target_risk' and target_volatility is not None:
        optimizer = EfficientFrontier(
            returns=returns,
            risk_free_rate=risk_free_rate,
            constraints=constraints,
            max_leverage=max_leverage
        )
        return optimizer.optimize_for_target_risk(target_volatility)
    else:
        optimizer = PortfolioOptimizer(
            returns=returns,
            risk_free_rate=risk_free_rate,
            constraints=constraints
        )
        return optimizer.optimize(objective=objective, target_return=target_return) 