"""Module Name: Portfolio Backtester

This module provides comprehensive backtesting capabilities for portfolio strategies.

Classes:
    PortfolioBacktester: Main class for backtesting portfolio strategies

Functions:
    None (deprecated functions have been removed)

Dependencies:
    - portfolio_config.PortfolioConfig: Configuration settings for portfolio operations
    - portfolio_risk.PortfolioRiskAnalyzer: Risk analysis functionality
    - portfolio_metrics_consolidated: Consolidated metrics calculation utilities
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime, timedelta

# Change from absolute to relative imports
from .portfolio_config import PortfolioConfig
from .portfolio_metrics_consolidated import calculate_portfolio_metrics
from .portfolio_risk import PortfolioRiskAnalyzer
from .portfolio_optimization import optimize_portfolio

# Configure logging
logger = logging.getLogger(__name__)