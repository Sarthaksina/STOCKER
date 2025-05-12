"""Portfolio module for STOCKER Pro

This module provides comprehensive portfolio analysis, optimization, and management capabilities.
It implements the Facade pattern through PortfolioFacade to provide a simplified interface to the
portfolio functionality while still allowing direct access to individual components when needed.

Components:
    - core: Core portfolio management functionality
    - analysis: Performance and risk metrics calculation
    - optimizer: Portfolio optimization algorithms
    - visualization: Visualization utilities
    - portfolio_facade: Simplified interface to portfolio functionality

Usage Examples:
    # Using the facade (recommended for most use cases)
    from src.features.portfolio import PortfolioFacade
    portfolio = PortfolioFacade()
    metrics = portfolio.calculate_metrics(returns, weights)
    
    # Direct access to specific components (for advanced use cases)
    from src.features.portfolio import optimize_portfolio
    result = optimize_portfolio(returns, objective='sharpe')
"""

# Import core classes for direct access
from .core import PortfolioManager, get_portfolio_manager
from .portfolio_facade import PortfolioFacade

# Import key functions for direct access
from .core import (
    recommend_portfolio,
    self_assess_portfolio,
    advanced_rebalance_portfolio,
    suggest_high_quality_stocks
)

# Import from optimizer module
from .optimizer import (
    PortfolioOptimizer,
    EfficientFrontier,
    optimize_portfolio
)

# Import from analysis module
from .analysis import (
    calculate_portfolio_statistics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_var,
    calculate_cvar
)

__all__ = [
    # Classes
    'PortfolioManager',
    'PortfolioFacade',
    'PortfolioOptimizer',
    'EfficientFrontier',
    
    # Factory functions
    'get_portfolio_manager',
    
    # Core portfolio functions
    'recommend_portfolio',
    'self_assess_portfolio',
    'advanced_rebalance_portfolio',
    'suggest_high_quality_stocks',
    
    # Optimization functions
    'optimize_portfolio',
    
    # Analysis functions
    'calculate_portfolio_statistics',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'calculate_var',
    'calculate_cvar'
]