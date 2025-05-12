"""
UI module for STOCKER Pro.

This module provides user interface components for the application.
"""

from src.ui.components import (
    create_chart,
    create_performance_dashboard,
    create_portfolio_view,
    create_stock_card
)

from src.ui.dashboard import (
    create_dashboard,
    run_dashboard,
    update_dashboard
)

__all__ = [
    # UI Components
    'create_chart',
    'create_performance_dashboard',
    'create_portfolio_view',
    'create_stock_card',
    
    # Dashboard
    'create_dashboard',
    'run_dashboard',
    'update_dashboard'
]
