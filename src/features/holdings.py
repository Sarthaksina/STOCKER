"""Holdings analysis module for STOCKER Pro.

This module provides functions for analyzing stock holdings and portfolios.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

from src.core.logging import logger
from src.features.analytics import (
    analyze_returns,
    analyze_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_beta,
    calculate_alpha,
    calculate_var,
    calculate_cvar
)


def analyze_holdings(holdings_data: Dict[str, Any], market_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Analyze stock holdings data and provide performance metrics.
    
    Args:
        holdings_data: Dictionary with holdings information
        market_data: Optional dictionary with market data for each holding
        
    Returns:
        Dictionary with analysis results
    """
    try:
        if not holdings_data:
            logger.warning("No holdings data provided for analysis")
            return {}
            
        # Extract holdings information
        symbols = holdings_data.get('symbols', [])
        quantities = holdings_data.get('quantities', [])
        purchase_prices = holdings_data.get('purchase_prices', [])
        purchase_dates = holdings_data.get('purchase_dates', [])
        current_prices = holdings_data.get('current_prices', [])
        
        # Validate input data
        if not symbols or len(symbols) == 0:
            logger.warning("No symbols found in holdings data")
            return {}
            
        if len(quantities) != len(symbols) or len(purchase_prices) != len(symbols):
            logger.error("Mismatched lengths in holdings data")
            return {}
            
        # Initialize results
        results = {
            'holdings': [],
            'total_value': 0,
            'total_cost': 0,
            'total_gain_loss': 0,
            'total_gain_loss_pct': 0,
            'risk_metrics': {}
        }
        
        # Process each holding
        for i, symbol in enumerate(symbols):
            quantity = quantities[i] if i < len(quantities) else 0
            purchase_price = purchase_prices[i] if i < len(purchase_prices) else 0
            purchase_date = purchase_dates[i] if i < len(purchase_dates) else None
            current_price = current_prices[i] if i < len(current_prices) else 0
            
            # Calculate holding metrics
            cost_basis = quantity * purchase_price
            current_value = quantity * current_price
            gain_loss = current_value - cost_basis
            gain_loss_pct = (gain_loss / cost_basis) * 100 if cost_basis > 0 else 0
            
            # Calculate days held if purchase date is available
            days_held = None
            if purchase_date:
                if isinstance(purchase_date, str):
                    try:
                        purchase_date = datetime.strptime(purchase_date, '%Y-%m-%d')
                    except ValueError:
                        purchase_date = None
                        
                if purchase_date:
                    days_held = (datetime.now() - purchase_date).days
            
            # Add to results
            holding_info = {
                'symbol': symbol,
                'quantity': quantity,
                'purchase_price': purchase_price,
                'current_price': current_price,
                'cost_basis': cost_basis,
                'current_value': current_value,
                'gain_loss': gain_loss,
                'gain_loss_pct': gain_loss_pct,
                'days_held': days_held
            }
            
            # Add market data analysis if available
            if market_data and symbol in market_data:
                symbol_data = market_data[symbol]
                if not symbol_data.empty and 'close' in symbol_data.columns:
                    # Calculate returns metrics
                    returns_metrics = analyze_returns(symbol_data, 'close')
                    volatility_metrics = analyze_volatility(symbol_data, 'close')
                    
                    # Add to holding info
                    holding_info.update({
                        'annualized_return': returns_metrics.get('annualized_return', 0),
                        'sharpe_ratio': returns_metrics.get('sharpe_ratio', 0),
                        'max_drawdown': returns_metrics.get('max_drawdown', 0),
                        'volatility': volatility_metrics.get('annualized_volatility', 0)
                    })
            
            results['holdings'].append(holding_info)
            results['total_value'] += current_value
            results['total_cost'] += cost_basis
        
        # Calculate overall portfolio metrics
        results['total_gain_loss'] = results['total_value'] - results['total_cost']
        results['total_gain_loss_pct'] = (results['total_gain_loss'] / results['total_cost']) * 100 if results['total_cost'] > 0 else 0
        
        # Calculate portfolio risk metrics if market data is available
        if market_data:
            portfolio_returns = calculate_portfolio_returns(symbols, quantities, market_data)
            if len(portfolio_returns) > 0:
                results['risk_metrics'] = {
                    'sharpe_ratio': calculate_sharpe_ratio(portfolio_returns),
                    'sortino_ratio': calculate_sortino_ratio(portfolio_returns),
                    'max_drawdown': calculate_max_drawdown(1 + portfolio_returns.cumsum()),
                    'var_95': calculate_var(portfolio_returns, 0.95),
                    'cvar_95': calculate_cvar(portfolio_returns, 0.95),
                    'volatility': portfolio_returns.std() * np.sqrt(252)
                }
                
                # Calculate benchmark metrics if available
                if 'SPY' in market_data:
                    spy_data = market_data['SPY']
                    if not spy_data.empty and 'close' in spy_data.columns:
                        spy_returns = spy_data['close'].pct_change().dropna()
                        if len(spy_returns) > 0:
                            results['risk_metrics']['beta'] = calculate_beta(portfolio_returns, spy_returns)
                            results['risk_metrics']['alpha'] = calculate_alpha(portfolio_returns, spy_returns)
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing holdings: {e}")
        return {}


def calculate_portfolio_returns(symbols: List[str], weights: List[float], market_data: Dict[str, pd.DataFrame]) -> pd.Series:
    """
    Calculate historical returns for a portfolio.
    
    Args:
        symbols: List of symbols in the portfolio
        weights: List of weights for each symbol
        market_data: Dictionary with market data for each symbol
        
    Returns:
        Series with portfolio returns
    """
    try:
        if not symbols or not weights or not market_data:
            return pd.Series()
            
        # Normalize weights
        total_weight = sum(weights)
        if total_weight <= 0:
            return pd.Series()
            
        normalized_weights = [w / total_weight for w in weights]
        
        # Extract returns for each symbol
        returns_data = {}
        for i, symbol in enumerate(symbols):
            if symbol in market_data:
                symbol_data = market_data[symbol]
                if not symbol_data.empty and 'close' in symbol_data.columns:
                    returns = symbol_data['close'].pct_change().dropna()
                    if len(returns) > 0:
                        returns_data[symbol] = returns
        
        if not returns_data:
            return pd.Series()
            
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        # Handle missing data
        returns_df = returns_df.dropna(how='all')
        returns_df = returns_df.fillna(0)  # Fill remaining NaNs with 0
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0, index=returns_df.index)
        for i, symbol in enumerate(symbols):
            if symbol in returns_df.columns and i < len(normalized_weights):
                portfolio_returns += returns_df[symbol] * normalized_weights[i]
        
        return portfolio_returns
        
    except Exception as e:
        logger.error(f"Error calculating portfolio returns: {e}")
        return pd.Series()


def calculate_portfolio_allocation(holdings_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate portfolio allocation by symbol, sector, and asset class.
    
    Args:
        holdings_data: Dictionary with holdings information
        
    Returns:
        Dictionary with allocation analysis
    """
    try:
        if not holdings_data:
            return {}
            
        # Extract holdings information
        symbols = holdings_data.get('symbols', [])
        quantities = holdings_data.get('quantities', [])
        current_prices = holdings_data.get('current_prices', [])
        sectors = holdings_data.get('sectors', [])
        asset_classes = holdings_data.get('asset_classes', [])
        
        # Calculate total portfolio value
        total_value = 0
        for i, symbol in enumerate(symbols):
            if i < len(quantities) and i < len(current_prices):
                total_value += quantities[i] * current_prices[i]
        
        if total_value <= 0:
            return {}
            
        # Calculate allocation by symbol
        symbol_allocation = {}
        for i, symbol in enumerate(symbols):
            if i < len(quantities) and i < len(current_prices):
                value = quantities[i] * current_prices[i]
                symbol_allocation[symbol] = (value / total_value) * 100
        
        # Calculate allocation by sector
        sector_allocation = {}
        for i, sector in enumerate(sectors):
            if i < len(quantities) and i < len(current_prices):
                value = quantities[i] * current_prices[i]
                if sector not in sector_allocation:
                    sector_allocation[sector] = 0
                sector_allocation[sector] += (value / total_value) * 100
        
        # Calculate allocation by asset class
        asset_class_allocation = {}
        for i, asset_class in enumerate(asset_classes):
            if i < len(quantities) and i < len(current_prices):
                value = quantities[i] * current_prices[i]
                if asset_class not in asset_class_allocation:
                    asset_class_allocation[asset_class] = 0
                asset_class_allocation[asset_class] += (value / total_value) * 100
        
        return {
            'by_symbol': symbol_allocation,
            'by_sector': sector_allocation,
            'by_asset_class': asset_class_allocation
        }
        
    except Exception as e:
        logger.error(f"Error calculating portfolio allocation: {e}")
        return {}


def calculate_dividend_income(holdings_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate expected dividend income from holdings.
    
    Args:
        holdings_data: Dictionary with holdings information
        
    Returns:
        Dictionary with dividend income analysis
    """
    try:
        if not holdings_data:
            return {}
            
        # Extract holdings information
        symbols = holdings_data.get('symbols', [])
        quantities = holdings_data.get('quantities', [])
        current_prices = holdings_data.get('current_prices', [])
        dividend_yields = holdings_data.get('dividend_yields', [])
        dividend_frequencies = holdings_data.get('dividend_frequencies', [])
        
        # Initialize results
        results = {
            'holdings': [],
            'total_annual_income': 0,
            'portfolio_yield': 0,
            'income_by_frequency': {
                'annual': 0,
                'semi_annual': 0,
                'quarterly': 0,
                'monthly': 0
            }
        }
        
        # Calculate total portfolio value
        total_value = 0
        for i, symbol in enumerate(symbols):
            if i < len(quantities) and i < len(current_prices):
                total_value += quantities[i] * current_prices[i]
        
        # Process each holding
        for i, symbol in enumerate(symbols):
            if i < len(quantities) and i < len(current_prices) and i < len(dividend_yields):
                quantity = quantities[i]
                current_price = current_prices[i]
                dividend_yield = dividend_yields[i]
                frequency = dividend_frequencies[i] if i < len(dividend_frequencies) else 'quarterly'
                
                # Calculate dividend income
                annual_rate = (dividend_yield / 100) * current_price
                annual_income = annual_rate * quantity
                
                # Add to results
                results['holdings'].append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'current_price': current_price,
                    'dividend_yield': dividend_yield,
                    'frequency': frequency,
                    'annual_income': annual_income
                })
                
                results['total_annual_income'] += annual_income
                
                # Add to frequency breakdown
                if frequency == 'annual':
                    results['income_by_frequency']['annual'] += annual_income
                elif frequency == 'semi_annual':
                    results['income_by_frequency']['semi_annual'] += annual_income
                elif frequency == 'quarterly':
                    results['income_by_frequency']['quarterly'] += annual_income
                elif frequency == 'monthly':
                    results['income_by_frequency']['monthly'] += annual_income
        
        # Calculate portfolio yield
        if total_value > 0:
            results['portfolio_yield'] = (results['total_annual_income'] / total_value) * 100
        
        return results
        
    except Exception as e:
        logger.error(f"Error calculating dividend income: {e}")
        return {}
