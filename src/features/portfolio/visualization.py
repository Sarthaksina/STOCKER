"""
Portfolio visualization functionality for STOCKER Pro.

This module provides visualization tools for portfolio performance,
risk metrics, and optimization results.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from io import BytesIO
import base64
from datetime import datetime, timedelta

from src.core.logging import logger
from src.features.portfolio.risk import calculate_portfolio_risk, calculate_drawdown


def plot_efficient_frontier(results: Dict[str, Any], risk_free_rate: float = 0.02, 
                          show_assets: bool = True, title: str = "Efficient Frontier") -> str:
    """
    Plot the efficient frontier.
    
    Args:
        results: Dictionary with efficient frontier results
        risk_free_rate: Annual risk-free rate
        show_assets: Whether to show individual assets
        title: Plot title
        
    Returns:
        Base64 encoded PNG image
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # Extract portfolio results
        portfolios = results.get("portfolios", [])
        
        # Plot random portfolios
        if portfolios:
            x = [p.get("volatility", 0) for p in portfolios]
            y = [p.get("return", 0) for p in portfolios]
            colors = [p.get("sharpe_ratio", 0) for p in portfolios]
            
            sc = plt.scatter(x, y, c=colors, marker="o", cmap="viridis", alpha=0.5, s=10)
            plt.colorbar(sc, label="Sharpe Ratio")
        
        # Plot minimum volatility and maximum Sharpe portfolios
        ef_data = results.get("efficient_frontier", {})
        
        min_vol = ef_data.get("min_volatility", {})
        max_sharpe = ef_data.get("max_sharpe", {})
        
        if min_vol:
            plt.scatter(min_vol.get("volatility", 0), min_vol.get("return", 0), 
                       marker="*", color="r", s=200, label="Minimum Volatility")
            
        if max_sharpe:
            plt.scatter(max_sharpe.get("volatility", 0), max_sharpe.get("return", 0), 
                       marker="*", color="g", s=200, label="Maximum Sharpe")
        
        # Plot individual assets if requested
        if show_assets and portfolios and len(portfolios) > 0:
            # Get the first portfolio to extract asset names
            first_portfolio = portfolios[0]
            asset_weights = first_portfolio.get("weights", {})
            
            if asset_weights:
                # Get asset returns and volatilities
                # This assumes we have single-asset portfolios in the results
                for asset_name in asset_weights.keys():
                    # Find portfolio where this asset has weight 1.0
                    asset_portfolio = None
                    for p in portfolios:
                        weights = p.get("weights", {})
                        if weights.get(asset_name, 0) > 0.95:  # Almost 100% allocation
                            asset_portfolio = p
                            break
                    
                    if asset_portfolio:
                        asset_vol = asset_portfolio.get("volatility", 0)
                        asset_ret = asset_portfolio.get("return", 0)
                        plt.scatter(asset_vol, asset_ret, marker="o", s=100, label=asset_name)
        
        # Plot Capital Market Line if available
        cml_data = results.get("capital_market_line", {})
        cml_x = cml_data.get("x", [])
        cml_y = cml_data.get("y", [])
        
        if cml_x and cml_y:
            plt.plot(cml_x, cml_y, "k--", label="Capital Market Line")
        
        # Plot risk-free rate
        plt.axhline(y=risk_free_rate, color="k", linestyle="-.", label=f"Risk-Free Rate ({risk_free_rate:.2%})")
        
        # Set labels and title
        plt.xlabel("Annualized Volatility")
        plt.ylabel("Annualized Return")
        plt.title(title)
        
        # Format ticks as percentages
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1%}"))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1%}"))
        
        # Show grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save to BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        
        # Return as base64
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return image_base64
        
    except Exception as e:
        logger.error(f"Error plotting efficient frontier: {e}")
        return ""


def plot_portfolio_weights(weights: Dict[str, float], title: str = "Portfolio Weights") -> str:
    """
    Plot portfolio weights as a pie chart.
    
    Args:
        weights: Dictionary mapping assets to weights
        title: Plot title
        
    Returns:
        Base64 encoded PNG image
    """
    try:
        # Sort weights by value
        sorted_weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)}
        
        # Group small allocations
        threshold = 0.03  # 3%
        other_sum = 0
        major_weights = {}
        
        for asset, weight in sorted_weights.items():
            if weight >= threshold:
                major_weights[asset] = weight
            else:
                other_sum += weight
        
        if other_sum > 0:
            major_weights["Other"] = other_sum
        
        # Create pie chart
        plt.figure(figsize=(10, 7))
        plt.pie(major_weights.values(), labels=major_weights.keys(), autopct="%1.1f%%", 
               startangle=90, shadow=False, wedgeprops={"edgecolor": "w"})
        
        plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title(title)
        
        # Save to BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        
        # Return as base64
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return image_base64
        
    except Exception as e:
        logger.error(f"Error plotting portfolio weights: {e}")
        return ""


def plot_portfolio_performance(portfolio_returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                             title: str = "Portfolio Performance") -> str:
    """
    Plot portfolio performance over time.
    
    Args:
        portfolio_returns: Series with portfolio returns
        benchmark_returns: Optional Series with benchmark returns
        title: Plot title
        
    Returns:
        Base64 encoded PNG image
    """
    try:
        # Calculate cumulative returns
        portfolio_cum_returns = (1 + portfolio_returns).cumprod() - 1
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot portfolio
        plt.plot(portfolio_cum_returns.index, portfolio_cum_returns * 100, label="Portfolio")
        
        # Plot benchmark if provided
        if benchmark_returns is not None:
            benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
            plt.plot(benchmark_cum_returns.index, benchmark_cum_returns * 100, label="Benchmark")
        
        # Set labels and title
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return (%)")
        plt.title(title)
        
        # Format x-axis as dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Show grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save to BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        
        # Return as base64
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return image_base64
        
    except Exception as e:
        logger.error(f"Error plotting portfolio performance: {e}")
        return ""


def plot_drawdown(returns: pd.Series, title: str = "Portfolio Drawdown") -> str:
    """
    Plot portfolio drawdown over time.
    
    Args:
        returns: Series with portfolio returns
        title: Plot title
        
    Returns:
        Base64 encoded PNG image
    """
    try:
        # Calculate drawdown
        drawdown_data = calculate_drawdown(returns)
        drawdown_series = drawdown_data.get("drawdown_series")
        
        if drawdown_series is None:
            logger.error("Drawdown calculation failed")
            return ""
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot drawdown
        plt.fill_between(drawdown_series.index, drawdown_series, 0, 
                        color="red", alpha=0.3, label="Drawdown")
        plt.plot(drawdown_series.index, drawdown_series, color="red", linewidth=1)
        
        # Set labels and title
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.title(title)
        
        # Format x-axis as dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Show grid
        plt.grid(True, alpha=0.3)
        
        # Save to BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        
        # Return as base64
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return image_base64
        
    except Exception as e:
        logger.error(f"Error plotting drawdown: {e}")
        return ""


def plot_risk_contribution(risk_data: Dict[str, Any], title: str = "Risk Contribution") -> str:
    """
    Plot risk contribution of portfolio assets.
    
    Args:
        risk_data: Dictionary with portfolio risk data
        title: Plot title
        
    Returns:
        Base64 encoded PNG image
    """
    try:
        # Extract percentage contributions
        percentage_contrib = risk_data.get("percentage_contributions", {})
        
        if not percentage_contrib:
            logger.error("No percentage contributions found in risk data")
            return ""
        
        # Sort by contribution
        sorted_contrib = {k: v for k, v in sorted(percentage_contrib.items(), key=lambda item: item[1], reverse=True)}
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        
        assets = list(sorted_contrib.keys())
        contributions = list(sorted_contrib.values())
        
        # Get colors based on contribution
        colors = plt.cm.RdYlGn_r(np.array(contributions) / max(contributions))
        
        plt.bar(assets, contributions, color=colors)
        
        # Set labels and title
        plt.xlabel("Asset")
        plt.ylabel("Risk Contribution (%)")
        plt.title(title)
        
        # Rotate x labels if there are many assets
        if len(assets) > 5:
            plt.xticks(rotation=45, ha="right")
        
        # Show grid
        plt.grid(True, alpha=0.3, axis="y")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        
        # Return as base64
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return image_base64
        
    except Exception as e:
        logger.error(f"Error plotting risk contribution: {e}")
        return ""


def plot_correlation_matrix(returns: pd.DataFrame, title: str = "Asset Correlation Matrix") -> str:
    """
    Plot correlation matrix heatmap.
    
    Args:
        returns: DataFrame with asset returns
        title: Plot title
        
    Returns:
        Base64 encoded PNG image
    """
    try:
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", 
                   vmin=-1, vmax=1, center=0, fmt=".2f", 
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title(title)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        
        # Return as base64
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return image_base64
        
    except Exception as e:
        logger.error(f"Error plotting correlation matrix: {e}")
        return ""


def plot_rolling_statistics(returns: pd.Series, window: int = 252, title: str = "Rolling Statistics") -> str:
    """
    Plot rolling statistics of portfolio returns.
    
    Args:
        returns: Series with portfolio returns
        window: Rolling window size
        title: Plot title
        
    Returns:
        Base64 encoded PNG image
    """
    try:
        # Calculate rolling statistics
        rolling_mean = returns.rolling(window=window).mean() * 252  # Annualized
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        rolling_sharpe = rolling_mean / rolling_vol
        
        # Create plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot rolling return
        ax1.plot(rolling_mean.index, rolling_mean * 100, color="blue")
        ax1.set_ylabel("Annualized Return (%)")
        ax1.set_title(f"{title} - {window}-Day Window")
        ax1.grid(True, alpha=0.3)
        
        # Plot rolling volatility
        ax2.plot(rolling_vol.index, rolling_vol * 100, color="red")
        ax2.set_ylabel("Annualized Volatility (%)")
        ax2.grid(True, alpha=0.3)
        
        # Plot rolling Sharpe ratio
        ax3.plot(rolling_sharpe.index, rolling_sharpe, color="green")
        ax3.set_ylabel("Sharpe Ratio")
        ax3.set_xlabel("Date")
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis as dates
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        
        # Return as base64
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return image_base64
        
    except Exception as e:
        logger.error(f"Error plotting rolling statistics: {e}")
        return ""


def create_performance_dashboard(portfolio_returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                               asset_returns: Optional[pd.DataFrame] = None, weights: Optional[Dict[str, float]] = None) -> Dict[str, str]:
    """
    Create comprehensive performance dashboard.
    
    Args:
        portfolio_returns: Series with portfolio returns
        benchmark_returns: Optional Series with benchmark returns
        asset_returns: Optional DataFrame with asset returns
        weights: Optional dictionary with portfolio weights
        
    Returns:
        Dictionary with base64 encoded PNG images for each plot
    """
    try:
        result = {}
        
        # Performance plot
        result["performance"] = plot_portfolio_performance(
            portfolio_returns, 
            benchmark_returns,
            "Portfolio Performance"
        )
        
        # Drawdown plot
        result["drawdown"] = plot_drawdown(
            portfolio_returns,
            "Portfolio Drawdown"
        )
        
        # Rolling statistics plot
        result["rolling_stats"] = plot_rolling_statistics(
            portfolio_returns,
            252,  # One year window
            "Rolling Statistics"
        )
        
        # Weights plot if provided
        if weights:
            result["weights"] = plot_portfolio_weights(
                weights,
                "Portfolio Allocation"
            )
        
        # Risk contribution plot if asset returns provided
        if asset_returns is not None and weights:
            # Convert weights to proper format if needed
            if isinstance(weights, dict):
                weight_array = [weights.get(col, 0.0) for col in asset_returns.columns]
            else:
                weight_array = weights
                
            risk_data = calculate_portfolio_risk(asset_returns, weight_array)
            result["risk_contribution"] = plot_risk_contribution(
                risk_data,
                "Risk Contribution"
            )
            
            # Correlation matrix plot
            result["correlation"] = plot_correlation_matrix(
                asset_returns,
                "Asset Correlation Matrix"
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating performance dashboard: {e}")
        return {} 