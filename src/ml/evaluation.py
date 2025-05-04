"""
Enhanced evaluation metrics for financial models.

This module provides specialized metrics for evaluating financial prediction models,
including directional accuracy, risk-adjusted returns metrics, and other
finance-specific evaluation techniques.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Callable
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

def calculate_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic regression metrics for model evaluation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }
    
    # Calculate MAPE if no zeros in y_true
    if not np.any(y_true == 0):
        metrics["mape"] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return metrics

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correct directional predictions).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Directional accuracy score (0.0 to 1.0)
    """
    # Calculate directions for true and predicted values
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    
    # Count matches
    matches = np.sum(true_dir == pred_dir)
    
    # Calculate accuracy
    return matches / len(true_dir)

def weighted_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate weighted directional accuracy, giving more weight to larger moves.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Weighted directional accuracy score (0.0 to 1.0)
    """
    # Calculate directions and magnitudes
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    
    true_dir = np.sign(true_diff)
    pred_dir = np.sign(pred_diff)
    
    # Calculate weights based on magnitude of true moves
    weights = np.abs(true_diff) / np.sum(np.abs(true_diff))
    
    # Calculate weighted matches
    matches = (true_dir == pred_dir) * weights
    
    # Sum weighted matches
    return np.sum(matches)

def calculate_returns(prices: np.ndarray, is_log_return: bool = False) -> np.ndarray:
    """
    Calculate returns from a price series.
    
    Args:
        prices: Array of prices
        is_log_return: Whether to calculate log returns
        
    Returns:
        Array of returns
    """
    if is_log_return:
        # Log returns: ln(p_t / p_t-1)
        returns = np.log(prices[1:] / prices[:-1])
    else:
        # Simple returns: (p_t / p_t-1) - 1
        returns = (prices[1:] / prices[:-1]) - 1
        
    return returns

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, annualization: int = 252) -> float:
    """
    Calculate Sharpe ratio (risk-adjusted return).
    
    Args:
        returns: Array of period returns
        risk_free_rate: Risk-free rate (annualized)
        annualization: Number of periods in a year (252 for daily, 12 for monthly)
        
    Returns:
        Sharpe ratio
    """
    # Convert annual risk-free rate to period rate
    period_rf = (1 + risk_free_rate) ** (1 / annualization) - 1
    
    # Calculate excess returns
    excess_returns = returns - period_rf
    
    # Calculate Sharpe ratio
    if np.std(returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(annualization)
    
    return sharpe

def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, annualization: int = 252) -> float:
    """
    Calculate Sortino ratio (downside risk-adjusted return).
    
    Args:
        returns: Array of period returns
        risk_free_rate: Risk-free rate (annualized)
        annualization: Number of periods in a year (252 for daily, 12 for monthly)
        
    Returns:
        Sortino ratio
    """
    # Convert annual risk-free rate to period rate
    period_rf = (1 + risk_free_rate) ** (1 / annualization) - 1
    
    # Calculate excess returns
    excess_returns = returns - period_rf
    
    # Calculate downside returns (only negative returns)
    downside_returns = returns[returns < 0]
    
    # Calculate Sortino ratio
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return float('inf')  # No downside risk
    
    sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(annualization)
    
    return sortino

def maximum_drawdown(prices: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        prices: Array of prices
        
    Returns:
        Maximum drawdown as a positive percentage
    """
    # Calculate the maximum drawdown
    peak = np.maximum.accumulate(prices)
    drawdown = (peak - prices) / peak
    
    return np.max(drawdown)

def calmar_ratio(returns: np.ndarray, prices: np.ndarray, annualization: int = 252) -> float:
    """
    Calculate Calmar ratio (return / maximum drawdown).
    
    Args:
        returns: Array of period returns
        prices: Array of prices (used to calculate maximum drawdown)
        annualization: Number of periods in a year (252 for daily, 12 for monthly)
        
    Returns:
        Calmar ratio
    """
    # Calculate annualized return
    annual_return = np.mean(returns) * annualization
    
    # Calculate maximum drawdown
    max_dd = maximum_drawdown(prices)
    
    # Calculate Calmar ratio
    if max_dd == 0:
        return float('inf')  # No drawdown
    
    calmar = annual_return / max_dd
    
    return calmar

def trading_strategy_returns(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Simulate trading strategy returns based on directional predictions.
    
    Args:
        y_true: True prices
        y_pred: Predicted prices
        initial_capital: Initial capital for simulation
        transaction_cost: Transaction cost as percentage
        
    Returns:
        Tuple of (strategy returns, buy and hold returns, performance metrics)
    """
    # Calculate predicted directions
    pred_dir = np.sign(np.diff(y_pred))
    
    # Initialize position and portfolio value arrays
    position = np.zeros_like(y_true)
    portfolio_value = np.zeros_like(y_true)
    cash = np.zeros_like(y_true)
    
    # Set initial values
    portfolio_value[0] = initial_capital
    cash[0] = initial_capital
    
    # Simulate trading
    for t in range(1, len(y_true)):
        if t < len(pred_dir) + 1:  # Make sure we have a prediction
            # Determine position
            if pred_dir[t-1] > 0:  # Predicted up
                # Buy if not already long
                if position[t-1] <= 0:
                    # Calculate shares to buy
                    shares_to_buy = cash[t-1] / y_true[t]
                    # Apply transaction cost
                    shares_to_buy *= (1 - transaction_cost)
                    
                    # Update position and cash
                    position[t] = shares_to_buy
                    cash[t] = 0
                else:
                    # Maintain position
                    position[t] = position[t-1]
                    cash[t] = cash[t-1]
            elif pred_dir[t-1] < 0:  # Predicted down
                # Sell if holding
                if position[t-1] > 0:
                    # Calculate proceeds
                    proceeds = position[t-1] * y_true[t]
                    # Apply transaction cost
                    proceeds *= (1 - transaction_cost)
                    
                    # Update position and cash
                    position[t] = 0
                    cash[t] = proceeds
                else:
                    # Maintain position
                    position[t] = position[t-1]
                    cash[t] = cash[t-1]
            else:  # No change predicted
                position[t] = position[t-1]
                cash[t] = cash[t-1]
        else:
            # Maintain last position if no more predictions
            position[t] = position[t-1]
            cash[t] = cash[t-1]
        
        # Calculate portfolio value
        portfolio_value[t] = cash[t] + position[t] * y_true[t]
    
    # Calculate strategy returns
    strategy_returns = np.zeros_like(portfolio_value)
    strategy_returns[1:] = (portfolio_value[1:] / portfolio_value[:-1]) - 1
    
    # Calculate buy and hold returns
    buy_and_hold_value = np.zeros_like(y_true)
    buy_and_hold_value[0] = initial_capital
    initial_shares = (initial_capital * (1 - transaction_cost)) / y_true[0]
    buy_and_hold_value[1:] = initial_shares * y_true[1:]
    
    buy_and_hold_returns = np.zeros_like(buy_and_hold_value)
    buy_and_hold_returns[1:] = (buy_and_hold_value[1:] / buy_and_hold_value[:-1]) - 1
    
    # Calculate metrics
    strategy_sharpe = sharpe_ratio(strategy_returns[1:])
    strategy_sortino = sortino_ratio(strategy_returns[1:])
    strategy_max_dd = maximum_drawdown(portfolio_value)
    
    buy_and_hold_sharpe = sharpe_ratio(buy_and_hold_returns[1:])
    buy_and_hold_sortino = sortino_ratio(buy_and_hold_returns[1:])
    buy_and_hold_max_dd = maximum_drawdown(buy_and_hold_value)
    
    # Final portfolio values
    final_strategy_value = portfolio_value[-1]
    final_buy_and_hold_value = buy_and_hold_value[-1]
    
    # Calculate total returns
    total_strategy_return = (final_strategy_value / initial_capital) - 1
    total_buy_and_hold_return = (final_buy_and_hold_value / initial_capital) - 1
    
    # Compile metrics
    metrics = {
        "strategy_sharpe": strategy_sharpe,
        "strategy_sortino": strategy_sortino,
        "strategy_max_drawdown": strategy_max_dd,
        "strategy_total_return": total_strategy_return,
        "strategy_final_value": final_strategy_value,
        
        "buy_and_hold_sharpe": buy_and_hold_sharpe,
        "buy_and_hold_sortino": buy_and_hold_sortino,
        "buy_and_hold_max_drawdown": buy_and_hold_max_dd,
        "buy_and_hold_total_return": total_buy_and_hold_return,
        "buy_and_hold_final_value": final_buy_and_hold_value,
        
        "relative_performance": (total_strategy_return / total_buy_and_hold_return if total_buy_and_hold_return != 0 else float('inf'))
    }
    
    return strategy_returns, buy_and_hold_returns, metrics

def evaluate_financial_model(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    prices: Optional[np.ndarray] = None,
    include_strategy_simulation: bool = True,
    transaction_cost: float = 0.001,
    risk_free_rate: float = 0.0,
    annualization: int = 252
) -> Dict[str, float]:
    """
    Comprehensive evaluation of a financial prediction model.
    
    Args:
        y_true: True values (prices or returns)
        y_pred: Predicted values
        prices: Price series (if y_true is returns)
        include_strategy_simulation: Whether to include trading strategy simulation
        transaction_cost: Transaction cost for strategy simulation
        risk_free_rate: Risk-free rate for Sharpe/Sortino ratios
        annualization: Annualization factor (252 for daily, 12 for monthly)
        
    Returns:
        Dictionary with all evaluation metrics
    """
    # Ensure y_true and y_pred are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate basic regression metrics
    metrics = calculate_basic_metrics(y_true, y_pred)
    
    # Calculate directional accuracy metrics
    metrics["directional_accuracy"] = directional_accuracy(y_true, y_pred)
    metrics["weighted_directional_accuracy"] = weighted_directional_accuracy(y_true, y_pred)
    
    # If we're working with returns, convert to prices for other metrics
    if prices is None:
        # Assume we're working with prices directly
        prices = y_true
    
    # Calculate returns if needed for financial metrics
    returns = calculate_returns(prices)
    
    # Calculate financial metrics
    metrics["sharpe_ratio"] = sharpe_ratio(returns, risk_free_rate, annualization)
    metrics["sortino_ratio"] = sortino_ratio(returns, risk_free_rate, annualization)
    metrics["max_drawdown"] = maximum_drawdown(prices)
    metrics["calmar_ratio"] = calmar_ratio(returns, prices, annualization)
    
    # Simulate trading strategy if requested
    if include_strategy_simulation:
        strategy_returns, buy_and_hold_returns, strategy_metrics = trading_strategy_returns(
            prices, y_pred, transaction_cost=transaction_cost
        )
        metrics.update(strategy_metrics)
    
    return metrics

def plot_financial_evaluation(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Model Evaluation"
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create comprehensive financial evaluation plots.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        dates: Optional date index for x-axis
        title: Plot title
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Use dates for x-axis if provided, otherwise use indices
    x = dates if dates is not None else np.arange(len(y_true))
    
    # Plot 1: True vs Predicted
    axes[0].plot(x, y_true, label="True", color="blue")
    axes[0].plot(x, y_pred, label="Predicted", color="red", linestyle="--")
    axes[0].set_title(f"{title} - True vs Predicted")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Prediction Error
    error = y_true - y_pred
    axes[1].plot(x, error, color="green")
    axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
    axes[1].set_title("Prediction Error")
    axes[1].set_ylabel("Error")
    axes[1].grid(True)
    
    # Plot 3: Directional Accuracy
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    
    # Correct and incorrect predictions
    correct = (true_dir == pred_dir)
    x_dir = x[1:] if isinstance(x, pd.DatetimeIndex) else np.arange(len(true_dir))
    
    # Plot directional correctness
    axes[2].scatter(x_dir[correct], np.ones(np.sum(correct)), 
                   color="green", marker="^", label="Correct Direction")
    axes[2].scatter(x_dir[~correct], np.zeros(np.sum(~correct)), 
                   color="red", marker="v", label="Incorrect Direction")
    
    # Plot true direction
    axes[2].plot(x_dir, (true_dir + 1) / 2, color="blue", alpha=0.3, label="Actual Direction")
    
    axes[2].set_title("Directional Accuracy")
    axes[2].set_ylabel("Direction")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Down", "Up"])
    axes[2].legend()
    axes[2].grid(True)
    
    # Adjust layout and return
    plt.tight_layout()
    return fig, axes 