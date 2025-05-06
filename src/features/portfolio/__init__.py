"""Portfolio module for STOCKER Pro

This module provides comprehensive portfolio analysis, optimization, and management capabilities.
It implements the Facade pattern through PortfolioFacade to provide a simplified interface to the
portfolio functionality while still allowing direct access to individual components when needed.

Components:
    - portfolio_metrics_consolidated: Performance and risk metrics calculation
    - portfolio_risk: Risk analysis and stress testing
    - portfolio_optimization: Portfolio optimization algorithms
    - portfolio_backtester: Backtesting framework
    - portfolio_visualization: Visualization utilities
    - portfolio_reporting: Report generation
    - portfolio_facade: Simplified interface to portfolio functionality

Usage Examples:
    # Using the facade (recommended for most use cases)
    from src.features.portfolio import PortfolioFacade
    portfolio = PortfolioFacade()
    metrics = portfolio.calculate_metrics(returns, weights)
    
    # Direct access to specific components (for advanced use cases)
    from src.features.portfolio import calculate_portfolio_metrics
    metrics = calculate_portfolio_metrics(returns, weights)

Note: All functionality from the previous portfolio_metrics.py has been consolidated into
portfolio_metrics_consolidated.py for better organization and maintainability.
"""

# Import core classes for direct access
from .portfolio_core import PortfolioManager, get_portfolio_manager
from .portfolio_risk import PortfolioRiskAnalyzer
from .portfolio_backtester import PortfolioBacktester
from .portfolio_visualization import PortfolioVisualizer
from .portfolio_facade import PortfolioFacade

# Import key functions for backward compatibility
from .portfolio_core import (
    recommend_portfolio,
    self_assess_portfolio,
    advanced_rebalance_portfolio
)

# Import metrics functions that were previously in other modules
from .portfolio_metrics_consolidated import (
    # Core metrics
    calculate_portfolio_metrics,
    calculate_rolling_metrics,
    
    # Performance analysis
    peer_compare,
    chart_performance,
    performance_analysis,
    
    # Risk and return metrics
    sharpe_ratio,
    alpha_beta,
    attribution_analysis,
    momentum_analysis,
    
    # Fundamental and sentiment
    valuation_metrics,
    sentiment_agg
)

__all__ = [
    # Classes
    'PortfolioManager',
    'PortfolioRiskAnalyzer',
    'PortfolioBacktester',
    'PortfolioVisualizer',
    'PortfolioFacade',  # Main facade class
    
    # Factory functions
    'get_portfolio_manager',
    
    # Core portfolio functions
    'recommend_portfolio',
    'self_assess_portfolio',
    'advanced_rebalance_portfolio',
    
    # Metrics functions
    'calculate_portfolio_metrics',
    'calculate_rolling_metrics',
    'peer_compare',
    'chart_performance',
    'performance_analysis',
    'sharpe_ratio',
    'alpha_beta',
    'attribution_analysis',
    'momentum_analysis',
    'valuation_metrics',
    'sentiment_agg'
]