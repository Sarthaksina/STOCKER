"""
Enhanced evaluation metrics for financial models.

This module provides specialized metrics for evaluating financial prediction models,
including directional accuracy, risk-adjusted returns metrics, and other
finance-specific evaluation techniques.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import logging
import os
import json
from datetime import datetime
import seaborn as sns

logger = logging.getLogger(__name__)

# --- Basic Metrics ---

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

# --- Financial Metrics ---

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
    
    return annual_return / max_dd

# --- Trading Strategy Evaluation ---

def trading_strategy_returns(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Simulate trading strategy returns based on predictions.
    
    Args:
        y_true: True price values
        y_pred: Predicted price values
        initial_capital: Initial capital
        transaction_cost: Transaction cost as percentage
        
    Returns:
        Tuple of (strategy_equity_curve, buy_and_hold_equity_curve, performance_metrics)
    """
    # Calculate predicted price changes
    pred_changes = np.diff(y_pred)
    
    # Generate signals (1 for buy, -1 for sell, 0 for hold)
    signals = np.sign(pred_changes)
    signals = np.append(0, signals)  # No signal for first period
    
    # Calculate true returns
    true_returns = np.diff(y_true) / y_true[:-1]
    true_returns = np.append(0, true_returns)  # No return for first period
    
    # Initialize equity curves
    strategy_equity = np.zeros(len(y_true))
    buy_and_hold_equity = np.zeros(len(y_true))
    
    strategy_equity[0] = initial_capital
    buy_and_hold_equity[0] = initial_capital
    
    # Calculate buy and hold equity
    buy_and_hold_returns = np.cumprod(1 + true_returns)
    buy_and_hold_equity = initial_capital * buy_and_hold_returns
    
    # Calculate strategy equity
    position = 0  # 0 = cash, 1 = long
    entry_price = 0
    
    for i in range(1, len(y_true)):
        # Update strategy equity based on position
        if position == 1:  # If in position
            strategy_equity[i] = strategy_equity[i-1] * (1 + true_returns[i])
        else:  # If in cash
            strategy_equity[i] = strategy_equity[i-1]
        
        # Check for position changes
        if position == 0 and signals[i] > 0:  # Enter long
            position = 1
            entry_price = y_true[i]
            # Apply transaction cost
            strategy_equity[i] *= (1 - transaction_cost)
        elif position == 1 and signals[i] < 0:  # Exit long
            position = 0
            # Apply transaction cost
            strategy_equity[i] *= (1 - transaction_cost)
    
    # Calculate strategy performance metrics
    strategy_returns = np.diff(strategy_equity) / strategy_equity[:-1]
    strategy_returns = np.append(0, strategy_returns)
    
    buy_and_hold_returns_calc = np.diff(buy_and_hold_equity) / buy_and_hold_equity[:-1]
    buy_and_hold_returns_calc = np.append(0, buy_and_hold_returns_calc)
    
    # Calculate metrics
    metrics = {
        "strategy_total_return": (strategy_equity[-1] / initial_capital) - 1,
        "buy_and_hold_total_return": (buy_and_hold_equity[-1] / initial_capital) - 1,
        "strategy_sharpe": sharpe_ratio(strategy_returns[1:]),
        "buy_and_hold_sharpe": sharpe_ratio(buy_and_hold_returns_calc[1:]),
        "strategy_max_drawdown": maximum_drawdown(strategy_equity),
        "buy_and_hold_max_drawdown": maximum_drawdown(buy_and_hold_equity),
    }
    
    return strategy_equity, buy_and_hold_equity, metrics

# --- Comprehensive Evaluation ---

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
        y_true: True values
        y_pred: Predicted values
        prices: Price series (if y_true is returns). If None, y_true is used as prices
        include_strategy_simulation: Whether to include trading strategy simulation
        transaction_cost: Transaction cost for strategy simulation
        risk_free_rate: Risk-free rate for Sharpe/Sortino ratios
        annualization: Annualization factor (252 for daily, 12 for monthly)
        
    Returns:
        Dictionary with all evaluation metrics
    """
    # Basic metrics
    metrics = calculate_basic_metrics(y_true, y_pred)
    
    # Add directional accuracy
    metrics["directional_accuracy"] = directional_accuracy(y_true, y_pred)
    metrics["weighted_directional_accuracy"] = weighted_directional_accuracy(y_true, y_pred)
    
    # Use provided prices or y_true as prices
    price_series = prices if prices is not None else y_true
    
    # Trading strategy simulation
    if include_strategy_simulation:
        strategy_equity, buy_and_hold_equity, strategy_metrics = trading_strategy_returns(
            price_series, y_pred, transaction_cost=transaction_cost
        )
        metrics.update(strategy_metrics)
    
    return metrics

# --- Visualization Utilities ---

def plot_financial_evaluation(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Model Evaluation"
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create visualization of model predictions and evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        dates: Optional date index for x-axis
        title: Plot title
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # X-axis values
    x = dates if dates is not None else np.arange(len(y_true))
    
    # Plot 1: True vs Predicted Values
    axes[0].plot(x, y_true, label="True", color="blue")
    axes[0].plot(x, y_pred, label="Predicted", color="red", linestyle="--")
    axes[0].set_title(f"{title} - True vs Predicted Values")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Error
    error = y_true - y_pred
    axes[1].plot(x, error, color="green")
    axes[1].axhline(y=0, color="black", linestyle="-")
    axes[1].set_title("Prediction Error")
    axes[1].grid(True)
    
    # Plot 3: Directional Accuracy
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    correct_dir = (true_dir == pred_dir)
    
    # X-axis for directional accuracy (one less point)
    x_dir = dates[1:] if dates is not None else np.arange(1, len(y_true))
    
    # Plot with colors by correctness
    for i, (is_correct, x_val) in enumerate(zip(correct_dir, x_dir)):
        color = "green" if is_correct else "red"
        axes[2].plot([x_val, x_val], [0, true_dir[i]], color=color, alpha=0.7)
    
    axes[2].axhline(y=0, color="black", linestyle="-")
    axes[2].set_title("Directional Accuracy (Green = Correct, Red = Incorrect)")
    axes[2].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, axes

# --- Model Evaluator Class ---

class ModelEvaluator:
    """
    Standardized model evaluation for STOCKER Pro.
    
    This class provides methods to evaluate and compare financial prediction models
    using both standard ML metrics and finance-specific metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model evaluator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.results = {}
        self.models_evaluated = []
    
    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prices: Optional[np.ndarray] = None,
        include_strategy_simulation: bool = True,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.0,
        annualization: int = 252
    ) -> Dict[str, float]:
        """
        Evaluate a single model using financial metrics.
        
        Args:
            model_name: Name of the model being evaluated
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
        # Use the comprehensive evaluation function
        metrics = evaluate_financial_model(
            y_true=y_true,
            y_pred=y_pred,
            prices=prices,
            include_strategy_simulation=include_strategy_simulation,
            transaction_cost=transaction_cost,
            risk_free_rate=risk_free_rate,
            annualization=annualization
        )
        
        # Store results for this model
        self.results[model_name] = metrics
        if model_name not in self.models_evaluated:
            self.models_evaluated.append(model_name)
        
        return metrics
    
    def compare_models(
        self,
        models_dict: Dict[str, Any] = None,
        test_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_true: Optional[np.ndarray] = None,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models side by side.
        
        Args:
            models_dict: Dictionary of {model_name: model} pairs
            test_data: Test features (if models need to make predictions)
            y_true: True values (if predictions are already available)
            metrics: Optional list of metrics to include in comparison
            
        Returns:
            DataFrame with comparative metrics
        """
        if len(self.results) == 0 and models_dict is None:
            raise ValueError("No models have been evaluated and no models provided")
        
        comparison_results = {}
        
        # If models_dict provided, evaluate all models
        if models_dict is not None and test_data is not None and y_true is not None:
            for name, model in models_dict.items():
                y_pred = model.predict(test_data)
                metrics_dict = self.evaluate_model(name, y_true, y_pred)
                comparison_results[name] = metrics_dict
        else:
            # Use existing evaluated models
            comparison_results = self.results
        
        # Convert to DataFrame
        df = pd.DataFrame(comparison_results).T
        
        # Filter metrics if specified
        if metrics is not None:
            df = df[metrics]
            
        return df
    
    def plot_comparison(
        self,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create bar plot comparing model performance.
        
        Args:
            metrics: List of metrics to plot (defaults to all)
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if len(self.results) == 0:
            raise ValueError("No models have been evaluated")
            
        # Convert results to DataFrame
        df = pd.DataFrame(self.results).T
        
        # Filter metrics if specified
        if metrics is not None:
            df = df[metrics]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot
        df.plot(kind='bar', ax=ax)
        
        # Add labels and title
        ax.set_ylabel("Metric Value")
        ax.set_title("Model Comparison")
        ax.legend(title="Metrics")
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', rotation=90, padding=3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path is not None:
            plt.savefig(save_path)
        
        return fig
    
    def save_results(self, output_dir: str, filename: Optional[str] = None) -> str:
        """
        Save evaluation results to disk.
        
        Args:
            output_dir: Directory to save results
            filename: Optional filename (defaults to timestamp)
            
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            filename = f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        filepath = os.path.join(output_dir, filename)
        
        # Convert results to serializable format
        serializable_results = {
            model: {k: float(v) if isinstance(v, np.float32) or isinstance(v, np.float64) else v 
                   for k, v in metrics.items()}
            for model, metrics in self.results.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        return filepath

# --- Standalone Utility Function ---

def compare_models(
    models: Dict[str, Any],
    test_data: Union[pd.DataFrame, np.ndarray],
    y_true: np.ndarray,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convenience function to compare multiple models.
    
    Args:
        models: Dictionary of {model_name: model} pairs
        test_data: Test features
        y_true: True values
        metrics: Optional list of metrics to include in comparison
        
    Returns:
        DataFrame with comparative metrics
    """
    evaluator = ModelEvaluator()
    
    for name, model in models.items():
        y_pred = model.predict(test_data)
        evaluator.evaluate_model(name, y_true, y_pred)
    
    return evaluator.compare_models(metrics=metrics) 