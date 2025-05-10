"""
UI components for STOCKER Pro.

This module provides reusable UI components for the application.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Union, Any

def create_chart(
    data: pd.DataFrame,
    title: str = "Stock Price Chart",
    chart_type: str = "candlestick",
    x_title: str = "Date",
    y_title: str = "Price",
    width: int = 800,
    height: int = 500,
    add_volume: bool = True,
    add_indicators: Optional[List[str]] = None
) -> go.Figure:
    """
    Create a financial chart using Plotly.
    
    Args:
        data: DataFrame with OHLCV data
        title: Chart title
        chart_type: Type of chart (line, candlestick, ohlc)
        x_title: X-axis title
        y_title: Y-axis title
        width: Chart width
        height: Chart height
        add_volume: Whether to add volume subplot
        add_indicators: List of indicators to add
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Set up the main chart
    if chart_type == "candlestick":
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="OHLC"
        ))
    elif chart_type == "ohlc":
        fig.add_trace(go.Ohlc(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="OHLC"
        ))
    else:  # line chart
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['close'],
            mode='lines',
            name="Close"
        ))
    
    # Add volume subplot if requested
    if add_volume and 'volume' in data.columns:
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['volume'],
            name="Volume",
            yaxis="y2",
            marker_color='rgba(0, 0, 128, 0.3)'
        ))
        
        # Update layout to include volume
        fig.update_layout(
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )
    
    # Add indicators if requested
    if add_indicators:
        for indicator in add_indicators:
            if indicator in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[indicator],
                    mode='lines',
                    name=indicator
                ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        width=width,
        height=height,
        xaxis_rangeslider_visible=False,  # Hide rangeslider
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_performance_dashboard(
    returns_data: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    metrics: Optional[Dict[str, float]] = None,
    title: str = "Portfolio Performance",
    width: int = 1000,
    height: int = 600
) -> go.Figure:
    """
    Create a portfolio performance dashboard.
    
    Args:
        returns_data: DataFrame with returns data
        benchmark_returns: Optional benchmark returns
        metrics: Optional metrics to display
        title: Dashboard title
        width: Dashboard width
        height: Dashboard height
        
    Returns:
        Plotly figure object
    """
    # Create subplot figure
    fig = go.Figure()
    
    # Calculate cumulative returns
    cum_returns = (1 + returns_data).cumprod() - 1
    
    # Add portfolio returns
    fig.add_trace(go.Scatter(
        x=cum_returns.index,
        y=cum_returns * 100,  # Convert to percentage
        mode='lines',
        name='Portfolio',
        line=dict(color='blue', width=2)
    ))
    
    # Add benchmark if provided
    if benchmark_returns is not None:
        cum_benchmark = (1 + benchmark_returns).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum_benchmark.index,
            y=cum_benchmark * 100,  # Convert to percentage
            mode='lines',
            name='Benchmark',
            line=dict(color='gray', width=2, dash='dash')
        ))
    
    # Add metrics table if provided
    if metrics:
        metrics_text = "<br>".join([f"<b>{k}:</b> {v:.2f}" for k, v in metrics.items()])
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=metrics_text,
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=5
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        width=width,
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def create_portfolio_view(
    weights: Dict[str, float],
    returns: Optional[pd.DataFrame] = None,
    risk_metrics: Optional[Dict[str, float]] = None,
    sector_info: Optional[Dict[str, str]] = None
) -> Dict[str, go.Figure]:
    """
    Create a set of portfolio visualization figures.
    
    Args:
        weights: Dictionary of portfolio weights
        returns: Optional DataFrame with returns
        risk_metrics: Optional risk metrics
        sector_info: Optional sector information
        
    Returns:
        Dictionary of Plotly figures
    """
    figures = {}
    
    # Create pie chart of portfolio weights
    weights_fig = go.Figure(data=[go.Pie(
        labels=list(weights.keys()),
        values=list(weights.values()),
        textinfo='label+percent',
        insidetextorientation='radial'
    )])
    
    weights_fig.update_layout(
        title="Portfolio Allocation",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    figures['weights'] = weights_fig
    
    # Create sector breakdown if sector info is provided
    if sector_info:
        # Group by sector
        sector_weights = {}
        for symbol, weight in weights.items():
            sector = sector_info.get(symbol, "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        sector_fig = go.Figure(data=[go.Pie(
            labels=list(sector_weights.keys()),
            values=list(sector_weights.values()),
            textinfo='label+percent',
            insidetextorientation='radial'
        )])
        
        sector_fig.update_layout(
            title="Sector Allocation",
            height=500,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        figures['sectors'] = sector_fig
    
    # Create risk metrics visualization if provided
    if risk_metrics:
        metrics_text = "<br>".join([f"<b>{k}:</b> {v:.4f}" for k, v in risk_metrics.items()])
        
        metrics_fig = go.Figure()
        
        metrics_fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=metrics_text,
            showarrow=False,
            font=dict(size=14),
            align="center",
            bordercolor="gray",
            borderwidth=1,
            borderpad=10
        )
        
        metrics_fig.update_layout(
            title="Risk Metrics",
            height=300,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        figures['metrics'] = metrics_fig
    
    return figures

def create_stock_card(
    ticker: str,
    price: float,
    change: float,
    metrics: Dict[str, float] = None,
    forecast: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Create a stock information card.
    
    Args:
        ticker: Stock ticker symbol
        price: Current price
        change: Price change
        metrics: Optional metrics to display
        forecast: Optional forecast information
        
    Returns:
        Dictionary with card components
    """
    # Format price and change
    price_formatted = f"${price:.2f}"
    change_pct = f"{change:.2f}%"
    change_color = "green" if change >= 0 else "red"
    change_symbol = "▲" if change >= 0 else "▼"
    
    # Create card elements
    card = {
        "ticker": ticker,
        "price": price_formatted,
        "change": f"{change_symbol} {change_pct}",
        "change_color": change_color,
        "metrics": metrics or {},
        "forecast": forecast or {}
    }
    
    return card 