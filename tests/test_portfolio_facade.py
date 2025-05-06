import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Import from the facade module
from src.features.portfolio import (
    recommend_portfolio,
    self_assess_portfolio,
    advanced_rebalance_portfolio,
    suggest_high_quality_stocks,
    peer_compare,
    top_n_recommender,
    chart_performance,
    get_portfolio_manager
)

# Import from specialized modules
from src.features.portfolio.portfolio_core import PortfolioManager


# Test the facade functions
def test_recommend_portfolio():
    # Create mock data
    price_history = {
        'AAPL': [100, 105, 110, 108, 112],
        'MSFT': [200, 202, 205, 207, 210]
    }
    user_info = {"price_history_map": price_history}
    config = {}
    
    # Test the function
    result = recommend_portfolio(user_info, config)
    
    # Verify the result
    assert isinstance(result, dict)


def test_self_assess_portfolio():
    # Create mock data
    user_portfolio = {'AAPL': 0.6, 'MSFT': 0.4}
    stock_data_map = {'AAPL': {'pe': 20}, 'MSFT': {'pe': 25}}
    sector_map = {'AAPL': 'Tech', 'MSFT': 'Tech'}
    asset_class_map = {'AAPL': 'Equity', 'MSFT': 'Equity'}
    price_history_map = {
        'AAPL': [100, 105, 110, 108, 112],
        'MSFT': [200, 202, 205, 207, 210]
    }
    
    # Test the function
    result = self_assess_portfolio(
        user_portfolio,
        stock_data_map,
        sector_map,
        asset_class_map,
        price_history_map
    )
    
    # Verify the result
    assert isinstance(result, dict)


def test_advanced_rebalance_portfolio():
    # Create mock data
    user_portfolio = {'AAPL': 0.6, 'MSFT': 0.4}
    stock_data_map = {'AAPL': {'pe': 20}, 'MSFT': {'pe': 25}}
    sector_map = {'AAPL': 'Tech', 'MSFT': 'Tech'}
    asset_class_map = {'AAPL': 'Equity', 'MSFT': 'Equity'}
    price_history_map = {
        'AAPL': [100, 105, 110, 108, 112],
        'MSFT': [200, 202, 205, 207, 210]
    }
    benchmark_prices = [100, 102, 104, 103, 105]
    config = {'target_risk': 0.15}
    
    # Test the function
    result = advanced_rebalance_portfolio(
        user_portfolio,
        stock_data_map,
        sector_map,
        asset_class_map,
        price_history_map,
        benchmark_prices,
        config
    )
    
    # Verify the result
    assert isinstance(result, dict)


def test_suggest_high_quality_stocks():
    # Create mock data
    config = {}
    market_data = {}
    filters = {}
    
    # Test the function
    result = suggest_high_quality_stocks(config, market_data, filters)
    
    # Verify the result
    assert isinstance(result, list)


def test_peer_compare():
    # Create mock data
    price_history_map = {
        'AAPL': [100, 105, 110, 108, 112],
        'MSFT': [200, 202, 205, 207, 210],
        'GOOG': [1000, 1010, 1020, 1015, 1025]
    }
    target = 'AAPL'
    n = 2
    
    # Test the function
    result = peer_compare(price_history_map, target, n)
    
    # Verify the result
    assert isinstance(result, dict)
    assert 'target' in result
    assert 'peers' in result


def test_top_n_recommender():
    # Create mock data
    df = pd.DataFrame({
        'score': [0.9, 0.8, 0.7, 0.6, 0.5]
    }, index=['A', 'B', 'C', 'D', 'E'])
    
    # Test the function
    result = top_n_recommender(df, 'score', 3)
    
    # Verify the result
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == 'A'  # Highest score


def test_chart_performance():
    # Create mock data
    dates = ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01']
    price_history_map = {
        'AAPL': [100, 105, 110, 108, 112],
        'MSFT': [200, 202, 205, 207, 210]
    }
    
    # Test the function
    result = chart_performance(dates, price_history_map)
    
    # Verify the result
    assert isinstance(result, dict)
    assert 'quarterly' in result
    assert 'yearly' in result


def test_get_portfolio_manager():
    # Test the function
    pm = get_portfolio_manager()
    
    # Verify the result
    assert pm is not None
    assert isinstance(pm, PortfolioManager)