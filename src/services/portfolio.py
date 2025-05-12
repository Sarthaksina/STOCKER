"""
Portfolio management service for STOCKER Pro.

This module provides services for portfolio creation, optimization,
risk analysis, and management.
"""
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime

from src.core.config import config
from src.core.logging import logger
from src.core.exceptions import StockerBaseException
from src.db.models import Portfolio, OptimizationMethod
from src.data.manager import DataManager


class PortfolioError(StockerBaseException):
    """Exception raised for portfolio-related errors."""
    pass


class PortfolioService:
    """
    Portfolio management service.
    
    Provides methods for creating, updating, optimizing,
    and analyzing investment portfolios.
    """
    
    def __init__(self):
        """Initialize the portfolio service."""
        from src.db.session import get_mongodb_db
        
        try:
            # Initialize database connection
            db = get_mongodb_db()
            self.portfolio_collection = db[config.database.portfolio_collection]
            
            # Initialize data manager
            self.data_manager = DataManager()
            
            # Try to import optimization and risk modules
            try:
                from src.features.portfolio.optimization import (
                    mean_variance_portfolio, 
                    risk_parity_portfolio,
                    min_variance_portfolio,
                    max_sharpe_portfolio
                )
                from src.features.portfolio.risk import (
                    calculate_portfolio_risk,
                    calculate_var,
                    calculate_cvar,
                    calculate_drawdown,
                    stress_test_portfolio
                )
                
                self.optimization_module = True
                self.mean_variance_portfolio = mean_variance_portfolio
                self.risk_parity_portfolio = risk_parity_portfolio
                self.min_variance_portfolio = min_variance_portfolio
                self.max_sharpe_portfolio = max_sharpe_portfolio
                
                self.calculate_portfolio_risk = calculate_portfolio_risk
                self.calculate_var = calculate_var
                self.calculate_cvar = calculate_cvar
                self.calculate_drawdown = calculate_drawdown
                self.stress_test_portfolio = stress_test_portfolio
                
            except ImportError:
                self.optimization_module = False
                logger.warning("Portfolio optimization modules not available. Some functionality will be limited.")
            
            logger.info("Portfolio service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize portfolio service: {e}")
            raise
    
    def create_portfolio(self, name: str, user_id: str, assets: List[Dict[str, Any]], 
                       description: Optional[str] = None, risk_profile: Optional[str] = None,
                       optimization_method: OptimizationMethod = OptimizationMethod.EFFICIENT_FRONTIER) -> Dict[str, Any]:
        """
        Create a new portfolio.
        
        Args:
            name: Portfolio name
            user_id: User ID
            assets: List of assets with symbol and weight
            description: Portfolio description
            risk_profile: Risk profile (e.g., 'conservative', 'moderate', 'aggressive')
            optimization_method: Optimization method used
            
        Returns:
            Created portfolio
            
        Raises:
            PortfolioError: If portfolio creation fails
        """
        try:
            # Check if assets have required fields
            for asset in assets:
                if 'symbol' not in asset:
                    raise ValueError("Each asset must have a 'symbol' field")
                if 'weight' not in asset:
                    raise ValueError("Each asset must have a 'weight' field")
            
            # Check that weights sum to 1 (allowing for small floating-point errors)
            weights_sum = sum(asset['weight'] for asset in assets)
            if not 0.99 <= weights_sum <= 1.01:
                raise ValueError(f"Asset weights must sum to 1, got {weights_sum}")
            
            # Create portfolio
            portfolio = Portfolio(
                name=name,
                user_id=user_id,
                assets=assets,
                description=description,
                risk_profile=risk_profile,
                optimization_method=optimization_method
            )
            
            # Insert into database
            result = self.portfolio_collection.insert_one(portfolio.dict())
            
            logger.info(f"Portfolio created successfully: {name} for user {user_id}")
            
            return portfolio.dict()
            
        except Exception as e:
            logger.error(f"Portfolio creation failed: {e}")
            raise PortfolioError(f"Failed to create portfolio: {e}")
    
    def get_portfolio(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Get a portfolio by ID.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Portfolio data
            
        Raises:
            PortfolioError: If portfolio not found
        """
        try:
            portfolio = self.portfolio_collection.find_one({"id": portfolio_id})
            
            if not portfolio:
                logger.warning(f"Portfolio not found: {portfolio_id}")
                raise PortfolioError("Portfolio not found")
            
            # Remove MongoDB ObjectId
            portfolio.pop("_id", None)
            
            return portfolio
            
        except PortfolioError:
            raise
        except Exception as e:
            logger.error(f"Failed to get portfolio: {e}")
            raise PortfolioError(f"Failed to get portfolio: {e}")
    
    def get_user_portfolios(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all portfolios for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of portfolios
        """
        try:
            portfolios = list(self.portfolio_collection.find({"user_id": user_id}))
            
            # Remove MongoDB ObjectIds
            for portfolio in portfolios:
                portfolio.pop("_id", None)
            
            return portfolios
            
        except Exception as e:
            logger.error(f"Failed to get user portfolios: {e}")
            raise PortfolioError(f"Failed to get user portfolios: {e}")
    
    def update_portfolio(self, portfolio_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a portfolio.
        
        Args:
            portfolio_id: Portfolio ID
            updates: Fields to update
            
        Returns:
            Updated portfolio
            
        Raises:
            PortfolioError: If portfolio not found or update fails
        """
        try:
            # Don't allow updating critical fields directly
            updates.pop("id", None)
            updates.pop("user_id", None)
            
            # Check assets if provided
            if "assets" in updates:
                assets = updates["assets"]
                for asset in assets:
                    if 'symbol' not in asset:
                        raise ValueError("Each asset must have a 'symbol' field")
                    if 'weight' not in asset:
                        raise ValueError("Each asset must have a 'weight' field")
                
                # Check that weights sum to 1 (allowing for small floating-point errors)
                weights_sum = sum(asset['weight'] for asset in assets)
                if not 0.99 <= weights_sum <= 1.01:
                    raise ValueError(f"Asset weights must sum to 1, got {weights_sum}")
            
            # Update timestamp
            updates["updated_at"] = datetime.now().isoformat()
            
            # Update portfolio
            result = self.portfolio_collection.update_one(
                {"id": portfolio_id},
                {"$set": updates}
            )
            
            if result.matched_count == 0:
                logger.warning(f"Portfolio update failed: Portfolio not found ({portfolio_id})")
                raise PortfolioError("Portfolio not found")
            
            # Return updated portfolio
            updated_portfolio = self.get_portfolio(portfolio_id)
            logger.info(f"Portfolio updated successfully: {portfolio_id}")
            
            return updated_portfolio
            
        except PortfolioError:
            raise
        except Exception as e:
            logger.error(f"Portfolio update failed: {e}")
            raise PortfolioError(f"Failed to update portfolio: {e}")
    
    def delete_portfolio(self, portfolio_id: str) -> bool:
        """
        Delete a portfolio.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            True if successful
            
        Raises:
            PortfolioError: If portfolio not found or deletion fails
        """
        try:
            result = self.portfolio_collection.delete_one({"id": portfolio_id})
            
            if result.deleted_count == 0:
                logger.warning(f"Portfolio deletion failed: Portfolio not found ({portfolio_id})")
                raise PortfolioError("Portfolio not found")
            
            logger.info(f"Portfolio deleted successfully: {portfolio_id}")
            return True
            
        except PortfolioError:
            raise
        except Exception as e:
            logger.error(f"Portfolio deletion failed: {e}")
            raise PortfolioError(f"Failed to delete portfolio: {e}")
    
    def optimize_portfolio(self, symbols: List[str], optimization_method: OptimizationMethod = OptimizationMethod.EFFICIENT_FRONTIER,
                         start_date: Optional[str] = None, end_date: Optional[str] = None, 
                         risk_free_rate: float = 0.02, target_return: Optional[float] = None,
                         target_risk: Optional[float] = None, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize a portfolio.
        
        Args:
            symbols: List of symbols
            optimization_method: Optimization method
            start_date: Start date for historical data
            end_date: End date for historical data
            risk_free_rate: Risk-free rate for calculations
            target_return: Target return (for risk parity)
            target_risk: Target risk (for target risk)
            constraints: Optimization constraints
            
        Returns:
            Optimized portfolio weights and metrics
            
        Raises:
            PortfolioError: If optimization fails
        """
        if not self.optimization_module:
            raise PortfolioError("Portfolio optimization modules not available")
            
        try:
            # Get historical data
            data_dict = self.data_manager.get_stock_data_batch(symbols, start_date, end_date)
            
            # Calculate returns
            returns_dict = {}
            for symbol, df in data_dict.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    returns_dict[symbol] = df['close'].pct_change().dropna()
            
            # Create returns DataFrame
            returns = pd.DataFrame(returns_dict)
            
            # Run optimization based on method
            if optimization_method == OptimizationMethod.EFFICIENT_FRONTIER:
                result = self.mean_variance_portfolio(returns, risk_free_rate)
            elif optimization_method == OptimizationMethod.RISK_PARITY:
                result = self.risk_parity_portfolio(returns)
            elif optimization_method == OptimizationMethod.MINIMUM_VARIANCE:
                result = self.min_variance_portfolio(returns)
            elif optimization_method == OptimizationMethod.MAXIMUM_SHARPE:
                result = self.max_sharpe_portfolio(returns, risk_free_rate)
            else:
                raise ValueError(f"Unsupported optimization method: {optimization_method}")
            
            logger.info(f"Portfolio optimization completed successfully for {len(symbols)} symbols")
            return result
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            raise PortfolioError(f"Failed to optimize portfolio: {e}")
    
    def analyze_portfolio_risk(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Analyze portfolio risk.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Risk metrics
            
        Raises:
            PortfolioError: If analysis fails
        """
        if not self.optimization_module:
            raise PortfolioError("Portfolio risk analysis modules not available")
            
        try:
            # Get portfolio
            portfolio = self.get_portfolio(portfolio_id)
            
            # Extract symbols and weights
            symbols = [asset['symbol'] for asset in portfolio['assets']]
            weights = [asset['weight'] for asset in portfolio['assets']]
            
            # Get historical data
            data_dict = self.data_manager.get_stock_data_batch(symbols)
            
            # Calculate returns
            returns_dict = {}
            for symbol, df in data_dict.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    returns_dict[symbol] = df['close'].pct_change().dropna()
            
            # Create returns DataFrame
            returns = pd.DataFrame(returns_dict)
            
            # Calculate risk metrics
            risk_metrics = self.calculate_portfolio_risk(returns, weights)
            
            # Calculate VaR and CVaR
            portfolio_returns = (returns * weights).sum(axis=1)
            var_95 = self.calculate_var(portfolio_returns, confidence_level=0.95)
            cvar_95 = self.calculate_cvar(portfolio_returns, confidence_level=0.95)
            
            # Calculate drawdown
            drawdown = self.calculate_drawdown(portfolio_returns)
            
            # Combine results
            result = {
                "risk_metrics": risk_metrics,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "max_drawdown": drawdown['max_drawdown'],
                "avg_drawdown": drawdown['avg_drawdown'],
                "drawdown_duration": drawdown['avg_duration']
            }
            
            logger.info(f"Portfolio risk analysis completed successfully for portfolio {portfolio_id}")
            return result
            
        except Exception as e:
            logger.error(f"Portfolio risk analysis failed: {e}")
            raise PortfolioError(f"Failed to analyze portfolio risk: {e}")
    
    def backtest_portfolio(self, portfolio_id: str, start_date: Optional[str] = None,
                         end_date: Optional[str] = None, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Backtest a portfolio.
        
        Args:
            portfolio_id: Portfolio ID
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
            
        Returns:
            Backtest results
            
        Raises:
            PortfolioError: If backtest fails
        """
        try:
            # Import backtest module
            try:
                from src.features.portfolio.portfolio_backtester import backtest_portfolio as run_backtest
            except ImportError:
                raise PortfolioError("Portfolio backtesting module not available")
            
            # Get portfolio
            portfolio = self.get_portfolio(portfolio_id)
            
            # Extract symbols and weights
            symbols = [asset['symbol'] for asset in portfolio['assets']]
            weights = [asset['weight'] for asset in portfolio['assets']]
            
            # Get historical data
            data_dict = self.data_manager.get_stock_data_batch(symbols, start_date, end_date)
            
            # Run backtest
            result = run_backtest(data_dict, weights, initial_capital)
            
            logger.info(f"Portfolio backtest completed successfully for portfolio {portfolio_id}")
            return result
            
        except PortfolioError:
            raise
        except Exception as e:
            logger.error(f"Portfolio backtest failed: {e}")
            raise PortfolioError(f"Failed to backtest portfolio: {e}") 


# Utility functions
def create_portfolio(name: str, user_id: str, assets: List[Dict[str, Any]], 
                  description: Optional[str] = None, risk_profile: Optional[str] = None,
                  optimization_method: OptimizationMethod = OptimizationMethod.EFFICIENT_FRONTIER) -> Dict[str, Any]:
    """
    Create a new portfolio.
    
    This is a convenience wrapper around PortfolioService.create_portfolio.
    
    Args:
        name: Portfolio name
        user_id: User ID
        assets: List of assets with symbol and weight
        description: Portfolio description
        risk_profile: Risk profile (e.g., 'conservative', 'moderate', 'aggressive')
        optimization_method: Optimization method used
        
    Returns:
        Created portfolio
    """
    portfolio_service = get_portfolio_service()
    return portfolio_service.create_portfolio(
        name, user_id, assets, description, risk_profile, optimization_method
    )


def update_portfolio(portfolio_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a portfolio.
    
    This is a convenience wrapper around PortfolioService.update_portfolio.
    
    Args:
        portfolio_id: Portfolio ID
        updates: Fields to update
        
    Returns:
        Updated portfolio
    """
    portfolio_service = get_portfolio_service()
    return portfolio_service.update_portfolio(portfolio_id, updates)


# Singleton instance
_portfolio_service_instance = None


def get_portfolio_service() -> PortfolioService:
    """
    Get the singleton instance of the PortfolioService.
    
    Returns:
        PortfolioService instance
    """
    global _portfolio_service_instance
    
    if _portfolio_service_instance is None:
        _portfolio_service_instance = PortfolioService()
        
    return _portfolio_service_instance