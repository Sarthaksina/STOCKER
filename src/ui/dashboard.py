"""
Dashboard functionality for STOCKER Pro.

This module provides dashboard creation and management functionality.
"""

import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from src.ui.components import (
    create_chart,
    create_performance_dashboard,
    create_portfolio_view,
    create_stock_card
)

def create_dashboard(
    app_title: str = "STOCKER Pro Dashboard",
    theme: str = "light",
    debug: bool = False
) -> dash.Dash:
    """
    Create a Dash application for the dashboard.
    
    Args:
        app_title: Title of the dashboard
        theme: Dashboard theme (light or dark)
        debug: Whether to run in debug mode
        
    Returns:
        Dash application
    """
    # Create Dash app
    app = dash.Dash(
        __name__,
        title=app_title,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
    )
    
    # Define app layout
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1(app_title),
            html.Div([
                html.Button("Portfolio", id="btn-portfolio", className="nav-button"),
                html.Button("Analysis", id="btn-analysis", className="nav-button"),
                html.Button("Forecasting", id="btn-forecasting", className="nav-button"),
                html.Button("Settings", id="btn-settings", className="nav-button")
            ], className="nav-buttons")
        ], className="header"),
        
        # Main content area
        html.Div([
            # Portfolio section (default view)
            html.Div([
                html.H2("Portfolio Overview"),
                html.Div([
                    dcc.Graph(id="portfolio-performance"),
                    html.Div([
                        dcc.Graph(id="portfolio-allocation"),
                        dcc.Graph(id="portfolio-risk")
                    ], className="chart-row")
                ], className="charts-container")
            ], id="portfolio-section"),
            
            # Analysis section (hidden by default)
            html.Div([
                html.H2("Market Analysis"),
                html.Div([
                    dcc.Dropdown(id="ticker-selector"),
                    dcc.Graph(id="ticker-chart"),
                    html.Div([
                        dcc.Graph(id="correlation-matrix"),
                        dcc.Graph(id="technical-indicators")
                    ], className="chart-row")
                ], className="charts-container")
            ], id="analysis-section", style={"display": "none"}),
            
            # Forecasting section (hidden by default)
            html.Div([
                html.H2("Price Forecasting"),
                html.Div([
                    dcc.Dropdown(id="forecast-ticker-selector"),
                    dcc.Graph(id="forecast-chart"),
                    html.Div([
                        dcc.Graph(id="forecast-metrics"),
                        dcc.Graph(id="forecast-comparison")
                    ], className="chart-row")
                ], className="charts-container")
            ], id="forecasting-section", style={"display": "none"}),
            
            # Settings section (hidden by default)
            html.Div([
                html.H2("Dashboard Settings"),
                html.Div([
                    html.Label("Theme:"),
                    dcc.RadioItems(
                        id="theme-selector",
                        options=[
                            {"label": "Light", "value": "light"},
                            {"label": "Dark", "value": "dark"}
                        ],
                        value=theme
                    ),
                    html.Label("Data Refresh Interval:"),
                    dcc.Slider(
                        id="refresh-interval",
                        min=0,
                        max=60,
                        step=5,
                        value=30,
                        marks={0: "Off", 15: "15s", 30: "30s", 60: "60s"}
                    )
                ], className="settings-container")
            ], id="settings-section", style={"display": "none"})
        ], className="main-content")
    ], className=f"app-container {theme}")
    
    # Register callbacks
    register_callbacks(app)
    
    return app

def register_callbacks(app: dash.Dash) -> None:
    """
    Register callbacks for the dashboard.
    
    Args:
        app: Dash application
    """
    # Navigation callbacks
    @app.callback(
        [Output("portfolio-section", "style"),
         Output("analysis-section", "style"),
         Output("forecasting-section", "style"),
         Output("settings-section", "style")],
        [Input("btn-portfolio", "n_clicks"),
         Input("btn-analysis", "n_clicks"),
         Input("btn-forecasting", "n_clicks"),
         Input("btn-settings", "n_clicks")]
    )
    def toggle_sections(btn1, btn2, btn3, btn4):
        ctx = dash.callback_context
        if not ctx.triggered:
            # Default to portfolio section
            return [{"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}]
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "btn-portfolio":
            return [{"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}]
        elif button_id == "btn-analysis":
            return [{"display": "none"}, {"display": "block"}, {"display": "none"}, {"display": "none"}]
        elif button_id == "btn-forecasting":
            return [{"display": "none"}, {"display": "none"}, {"display": "block"}, {"display": "none"}]
        elif button_id == "btn-settings":
            return [{"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "block"}]
        
        # Default
        return [{"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}]
    
    # Theme callback
    @app.callback(
        Output("app-container", "className"),
        Input("theme-selector", "value")
    )
    def update_theme(theme):
        return f"app-container {theme}"

def run_dashboard(app: dash.Dash, host: str = "0.0.0.0", port: int = 8050, debug: bool = False) -> None:
    """
    Run the dashboard application.
    
    Args:
        app: Dash application
        host: Host address
        port: Port number
        debug: Whether to run in debug mode
    """
    app.run_server(host=host, port=port, debug=debug)

def update_dashboard(
    app: dash.Dash,
    data_provider: Callable[[], Dict[str, Any]],
    interval_seconds: int = 30
) -> None:
    """
    Set up automatic dashboard updates.
    
    Args:
        app: Dash application
        data_provider: Function that returns dashboard data
        interval_seconds: Update interval in seconds
    """
    # Add interval component to layout
    app.layout.children.append(
        dcc.Interval(
            id="auto-refresh",
            interval=interval_seconds * 1000,  # Convert to milliseconds
            n_intervals=0
        )
    )
    
    # Register callback for data updates
    @app.callback(
        [Output("portfolio-performance", "figure"),
         Output("portfolio-allocation", "figure"),
         Output("ticker-chart", "figure")],
        Input("auto-refresh", "n_intervals")
    )
    def update_data(n):
        # Get fresh data
        data = data_provider()
        
        # Update portfolio performance chart
        portfolio_perf = create_performance_dashboard(
            returns_data=data.get("returns", pd.DataFrame()),
            benchmark_returns=data.get("benchmark_returns"),
            metrics=data.get("metrics")
        )
        
        # Update portfolio allocation chart
        if "weights" in data:
            allocation_fig = create_portfolio_view(
                weights=data["weights"],
                sector_info=data.get("sector_info")
            )["weights"]
        else:
            allocation_fig = go.Figure()
        
        # Update ticker chart
        if "price_data" in data and data["price_data"] is not None:
            ticker_chart = create_chart(
                data=data["price_data"],
                title="Stock Price",
                add_volume=True
            )
        else:
            ticker_chart = go.Figure()
        
        return portfolio_perf, allocation_fig, ticker_chart
