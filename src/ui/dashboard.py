"""
Dashboard functionality for STOCKER Pro.

This module provides dashboard creation and management functionality for the STOCKER Pro application.
It integrates with the Alpha Vantage API client to display real-time stock data and technical indicators.
"""

import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import logging

# Import STOCKER modules
from src.data.clients.alpha_vantage import AlphaVantageClient
from src.core.config import get_config
from src.core.logging import get_logger
from src.features.indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands

# Set up logger
logger = get_logger(__name__)

def get_alpha_vantage_client():
    """
    Get an instance of the Alpha Vantage client.
    
    Returns:
        AlphaVantageClient: Configured Alpha Vantage client
    """
    from src.data.clients.alpha_vantage import AlphaVantageClient
    
    # Try to get API key from environment variable
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    
    # If not found, try to get from config
    if not api_key:
        try:
            from src.core.config import get_config
            config = get_config()
            api_key = config.api.alpha_vantage_api_key if hasattr(config, 'api') else None
        except Exception as e:
            logger.warning(f"Could not get Alpha Vantage API key from config: {e}")
    
    # Create client (will use mock data if no API key is provided)
    return AlphaVantageClient(api_key=api_key)

# Create a function to generate sample data when API is not available
def generate_sample_data(symbol: str = "AAPL", days: int = 180):
    """
    Generate sample stock data for demonstration when API is not available.
    
    Args:
        symbol: Stock symbol
        days: Number of days of data to generate
        
    Returns:
        DataFrame with OHLCV data and technical indicators
    """
    logger.info(f"Generating sample data for {symbol} ({days} days)")
    
    # Create sample dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Set seed based on symbol for consistent but different data per symbol
    seed = sum(ord(c) for c in symbol)
    np.random.seed(seed)
    
    # Base price varies by symbol
    base_prices = {
        "AAPL": 150.0,
        "MSFT": 250.0,
        "GOOGL": 120.0,
        "AMZN": 100.0,
        "TSLA": 200.0,
        "META": 180.0,
        "NVDA": 300.0,
        "JPM": 140.0,
        "V": 220.0,
        "JNJ": 160.0
    }
    base_price = base_prices.get(symbol, 100.0)
    
    # Generate random walk for price data
    np.random.seed(seed)
    daily_returns = np.random.normal(0.0005, 0.015, len(dates))
    cumulative_returns = np.cumprod(1 + daily_returns)
    close_prices = base_price * cumulative_returns
    
    # Generate OHLC data
    daily_volatility = 0.015
    high_prices = close_prices * (1 + np.random.uniform(0, daily_volatility, len(dates)))
    low_prices = close_prices * (1 - np.random.uniform(0, daily_volatility, len(dates)))
    open_prices = low_prices + np.random.uniform(0, 1, len(dates)) * (high_prices - low_prices)
    
    # Generate volume data - higher volume on bigger price moves
    volume_base = np.random.randint(1000000, 5000000, len(dates))
    volume_modifier = np.abs(daily_returns) * 50000000  # More volume on bigger moves
    volumes = volume_base + volume_modifier.astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    # Add technical indicators
    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Bollinger Bands
    df['middle_band'] = df['close'].rolling(window=20).mean()
    df['std_dev'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['signal']
    
    return df

def fetch_stock_data(symbol: str, interval: str = 'daily', use_real_data: bool = True):
    """
    Fetch stock data from Alpha Vantage or generate sample data.
    
    Args:
        symbol: Stock symbol
        interval: Data interval (daily, weekly, monthly)
        use_real_data: Whether to use real data from Alpha Vantage
        
    Returns:
        DataFrame with OHLCV data and technical indicators
    """
    if use_real_data:
        try:
            # Get Alpha Vantage client
            client = get_alpha_vantage_client()
            
            # Fetch data based on interval
            if interval == 'daily':
                data = client.get_daily(symbol, outputsize='full')
            elif interval == 'weekly':
                data = client.get_weekly(symbol)
            elif interval == 'monthly':
                data = client.get_monthly(symbol)
            else:
                data = client.get_daily(symbol, outputsize='full')
            
            # If data is empty or None, fall back to sample data
            if data is None or data.empty:
                logger.warning(f"No data returned from Alpha Vantage for {symbol}, using sample data")
                return generate_sample_data(symbol)
            
            # Add technical indicators
            # Moving averages
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
            data['sma_200'] = data['close'].rolling(window=200).mean()
            
            # Bollinger Bands
            data['middle_band'] = data['close'].rolling(window=20).mean()
            data['std_dev'] = data['close'].rolling(window=20).std()
            data['upper_band'] = data['middle_band'] + (data['std_dev'] * 2)
            data['lower_band'] = data['middle_band'] - (data['std_dev'] * 2)
            
            # RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
            data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
            data['macd'] = data['ema_12'] - data['ema_26']
            data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
            data['macd_histogram'] = data['macd'] - data['signal']
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage: {e}")
            logger.info("Falling back to sample data")
    
    # Generate sample data if real data is not available or requested
    return generate_sample_data(symbol)

def create_dashboard(
    app_title: str = "STOCKER Pro Dashboard",
    theme: str = "light",
    debug: bool = False,
    use_real_data: bool = True
) -> dash.Dash:
    """
    Create a Dash application for the dashboard.
    
    Args:
        app_title: Title of the dashboard
        theme: Dashboard theme (light or dark)
        debug: Whether to run in debug mode
        use_real_data: Whether to use real data from Alpha Vantage
        
    Returns:
        Dash application
    """
    logger.info(f"Creating dashboard with theme={theme}, debug={debug}, use_real_data={use_real_data}")
    
    # Create Dash app with suppress_callback_exceptions to avoid errors
    app = dash.Dash(
        __name__,
        title=app_title,
        suppress_callback_exceptions=True,
        external_stylesheets=[
            'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'
        ],
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ]
    )
    
    # Fetch initial data for default stock
    default_symbol = "AAPL"
    df = fetch_stock_data(default_symbol, use_real_data=use_real_data)
    
    # Define app layout
    app.layout = html.Div([
        # Header
        html.Div([
            html.Div([
                html.H1(app_title, className="display-4 text-primary"),
                html.P("Financial Analysis & Stock Tracking Platform", className="lead")
            ], className="col-md-8"),
            html.Div([
                html.Img(src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Pictogram_voting_question.svg/1200px-Pictogram_voting_question.svg.png", 
                        height="50px", className="float-right mt-2")
            ], className="col-md-4")
        ], className="row p-3 bg-light border-bottom"),
        
        # Navigation
        html.Div([
            html.Nav([
                html.Div([
                    html.Button("Stock Data", id="btn-stock-data", n_clicks=0, 
                              className="btn btn-primary mr-2"),
                    html.Button("Technical Analysis", id="btn-tech-analysis", n_clicks=0, 
                                className="btn btn-outline-primary mr-2"),
                    html.Button("Portfolio", id="btn-portfolio", n_clicks=0, 
                                className="btn btn-outline-primary mr-2"),
                    html.Button("Forecasting", id="btn-forecasting", n_clicks=0, 
                                className="btn btn-outline-primary")
                ], className="navbar-nav mr-auto"),
                html.Div([
                    dcc.Dropdown(
                        id="stock-dropdown",
                        options=[
                            {"label": "Apple (AAPL)", "value": "AAPL"},
                            {"label": "Microsoft (MSFT)", "value": "MSFT"},
                            {"label": "Google (GOOGL)", "value": "GOOGL"},
                            {"label": "Amazon (AMZN)", "value": "AMZN"},
                            {"label": "Tesla (TSLA)", "value": "TSLA"},
                            {"label": "Meta (META)", "value": "META"},
                            {"label": "NVIDIA (NVDA)", "value": "NVDA"},
                            {"label": "JPMorgan Chase (JPM)", "value": "JPM"},
                            {"label": "Visa (V)", "value": "V"},
                            {"label": "Johnson & Johnson (JNJ)", "value": "JNJ"}
                        ],
                        value=default_symbol,
                        clearable=False,
                        style={"width": "300px"}
                    )
                ], className="form-inline my-2 my-lg-0")
            ], className="navbar navbar-expand-lg navbar-light bg-light")
        ], className="row"),
        
        # Stock data section
        html.Div([
            html.Div([
                html.H2("Stock Price Data", className="mb-4"),
                
                # Stock info card
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3(id="stock-name", className="card-title"),
                            html.H4(id="stock-price", className="text-primary"),
                            html.P(id="stock-change", className="lead")
                        ], className="col-md-6"),
                        html.Div([
                            html.Table([
                                html.Tr([html.Th("Open"), html.Td(id="stock-open")]),
                                html.Tr([html.Th("High"), html.Td(id="stock-high")]),
                                html.Tr([html.Th("Low"), html.Td(id="stock-low")]),
                                html.Tr([html.Th("Volume"), html.Td(id="stock-volume")])
                            ], className="table table-sm")
                        ], className="col-md-6")
                    ], className="row")
                ], className="card p-3 mb-4"),
                
                # Main price chart
                html.Div([
                    dcc.Graph(
                        id="stock-price-chart",
                        figure={
                            "data": [
                                {
                                    "x": df.index,
                                    "y": df["close"],
                                    "type": "line",
                                    "name": "Stock Price"
                                },
                                {
                                    "x": df.index,
                                    "y": df["sma_20"],
                                    "type": "line",
                                    "name": "20-day SMA",
                                    "line": {"dash": "dash"}
                                },
                                {
                                    "x": df.index,
                                    "y": df["sma_50"],
                                    "type": "line",
                                    "name": "50-day SMA",
                                    "line": {"dash": "dot"}
                                }
                            ],
                            "layout": {
                                "title": f"{default_symbol} Stock Price",
                                "xaxis": {"title": "Date"},
                                "yaxis": {"title": "Price ($)"},
                                "legend": {"orientation": "h", "y": 1.1},
                                "height": 500
                            }
                        },
                        config={"displayModeBar": True, "scrollZoom": True}
                    )
                ], className="mb-4"),
                
                # Chart type selector
                html.Div([
                    html.Label("Chart Type:", className="mr-2"),
                    dcc.RadioItems(
                        id="chart-type",
                        options=[
                            {"label": "Line", "value": "line"},
                            {"label": "Candlestick", "value": "candlestick"},
                            {"label": "OHLC", "value": "ohlc"}
                        ],
                        value="line",
                        labelStyle={"display": "inline-block", "margin-right": "10px"},
                        className="mb-3"
                    )
                ], className="mb-4"),
                
                # Lower charts
                html.Div([
                    html.Div([
                        html.H4("Volume", className="card-header"),
                        dcc.Graph(
                            id="volume-chart",
                            figure={
                                "data": [
                                    {
                                        "x": df.index,
                                        "y": df["volume"],
                                        "type": "bar",
                                        "name": "Volume",
                                        "marker": {"color": "rgba(0, 0, 128, 0.3)"}
                                    }
                                ],
                                "layout": {
                                    "margin": {"t": 10, "b": 30},
                                    "xaxis": {"title": "Date"},
                                    "yaxis": {"title": "Volume"},
                                    "height": 250
                                }
                            },
                            config={"displayModeBar": False}
                        )
                    ], className="col-md-6 mb-4"),
                    html.Div([
                        html.H4("Moving Averages", className="card-header"),
                        dcc.Graph(
                            id="ma-chart",
                            figure={
                                "data": [
                                    {
                                        "x": df.index,
                                        "y": df["sma_20"],
                                        "type": "line",
                                        "name": "20-day SMA"
                                    },
                                    {
                                        "x": df.index,
                                        "y": df["sma_50"],
                                        "type": "line",
                                        "name": "50-day SMA"
                                    },
                                    {
                                        "x": df.index,
                                        "y": df["sma_200"],
                                        "type": "line",
                                        "name": "200-day SMA"
                                    }
                                ],
                                "layout": {
                                    "margin": {"t": 10, "b": 30},
                                    "xaxis": {"title": "Date"},
                                    "yaxis": {"title": "Price ($)"},
                                    "height": 250
                                }
                            },
                            config={"displayModeBar": False}
                        )
                    ], className="col-md-6 mb-4")
                ], className="row")
            ], className="container-fluid")
        ], id="stock-data-section", className="mt-4"),
        
        # Technical analysis section (hidden by default)
        html.Div([
            html.Div([
                html.H2("Technical Analysis", className="mb-4"),
                
                # Technical indicators tabs
                dcc.Tabs([
                    dcc.Tab(label="Moving Averages", children=[
                        html.Div([
                            html.Div([
                                html.H4("Moving Averages", className="mb-3"),
                                dcc.Graph(
                                    id="ma-analysis-chart",
                                    figure={
                                        "data": [
                                            {
                                                "x": df.index,
                                                "y": df["close"],
                                                "type": "line",
                                                "name": "Price"
                                            },
                                            {
                                                "x": df.index,
                                                "y": df["sma_20"],
                                                "type": "line",
                                                "name": "20-day SMA",
                                                "line": {"dash": "dash"}
                                            },
                                            {
                                                "x": df.index,
                                                "y": df["sma_50"],
                                                "type": "line",
                                                "name": "50-day SMA",
                                                "line": {"dash": "dot"}
                                            },
                                            {
                                                "x": df.index,
                                                "y": df["sma_200"],
                                                "type": "line",
                                                "name": "200-day SMA",
                                                "line": {"dash": "dashdot"}
                                            }
                                        ],
                                        "layout": {
                                            "title": "Moving Average Analysis",
                                            "xaxis": {"title": "Date"},
                                            "yaxis": {"title": "Price ($)"},
                                            "height": 500
                                        }
                                    }
                                )
                            ], className="mb-4"),
                            
                            html.Div([
                                html.H5("Moving Average Analysis", className="card-header"),
                                html.Div([
                                    html.P(id="ma-analysis-text", className="card-text")
                                ], className="card-body")
                            ], className="card mb-4")
                        ], className="p-4")
                    ]),
                    
                    dcc.Tab(label="Bollinger Bands", children=[
                        html.Div([
                            html.Div([
                                html.H4("Bollinger Bands", className="mb-3"),
                                dcc.Graph(
                                    id="bollinger-chart",
                                    figure={
                                        "data": [
                                            {
                                                "x": df.index,
                                                "y": df["close"],
                                                "type": "line",
                                                "name": "Price"
                                            },
                                            {
                                                "x": df.index,
                                                "y": df["middle_band"],
                                                "type": "line",
                                                "name": "Middle Band (20-day SMA)",
                                                "line": {"dash": "dash"}
                                            },
                                            {
                                                "x": df.index,
                                                "y": df["upper_band"],
                                                "type": "line",
                                                "name": "Upper Band (+2σ)",
                                                "line": {"dash": "dot"}
                                            },
                                            {
                                                "x": df.index,
                                                "y": df["lower_band"],
                                                "type": "line",
                                                "name": "Lower Band (-2σ)",
                                                "line": {"dash": "dot"}
                                            }
                                        ],
                                        "layout": {
                                            "title": "Bollinger Bands",
                                            "xaxis": {"title": "Date"},
                                            "yaxis": {"title": "Price ($)"},
                                            "height": 500
                                        }
                                    }
                                )
                            ], className="mb-4"),
                            
                            html.Div([
                                html.H5("Bollinger Bands Analysis", className="card-header"),
                                html.Div([
                                    html.P(id="bollinger-analysis-text", className="card-text")
                                ], className="card-body")
                            ], className="card mb-4")
                        ], className="p-4")
                    ])
                ], className="mb-4")
            ], className="container-fluid")
        ], id="tech-analysis-section", style={"display": "none"}, className="mt-4"),
        
        # Portfolio section (hidden by default)
        html.Div([
            html.Div([
                html.H2("Portfolio Analysis", className="mb-4"),
                html.P("Portfolio analysis features will be implemented in future updates.", className="lead")
            ], className="container-fluid")
        ], id="portfolio-section", style={"display": "none"}, className="mt-4"),
        
        # Forecasting section (hidden by default)
        html.Div([
            html.Div([
                html.H2("Stock Price Forecasting", className="mb-4"),
                html.P("Forecasting features will be implemented in future updates.", className="lead")
            ], className="container-fluid")
        ], id="forecasting-section", style={"display": "none"}, className="mt-4"),
        
        # Footer
        html.Footer([
            html.Div([
                html.P("STOCKER Pro Dashboard 2023", className="text-center"),
                html.P("Powered by Alpha Vantage API", className="text-center text-muted")
            ], className="container")
        ], className="mt-5 pt-3 pb-3 bg-light border-top")
    ])
    
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
        [Output("stock-data-section", "style"),
         Output("tech-analysis-section", "style"),
         Output("portfolio-section", "style"),
         Output("forecasting-section", "style")],
        [Input("btn-stock-data", "n_clicks"),
         Input("btn-tech-analysis", "n_clicks"),
         Input("btn-portfolio", "n_clicks"),
         Input("btn-forecasting", "n_clicks")]
    )
    def toggle_sections(stock_clicks, tech_clicks, portfolio_clicks, forecast_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            # Default to stock data section
            return [{"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}]
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "btn-stock-data":
            return [{"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}]
        elif button_id == "btn-tech-analysis":
            return [{"display": "none"}, {"display": "block"}, {"display": "none"}, {"display": "none"}]
        elif button_id == "btn-portfolio":
            return [{"display": "none"}, {"display": "none"}, {"display": "block"}, {"display": "none"}]
        elif button_id == "btn-forecasting":
            return [{"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "block"}]
        
        # Default
        return [{"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}]
    
    # Update stock info based on dropdown selection
    @app.callback(
        [Output("stock-name", "children"),
         Output("stock-price", "children"),
         Output("stock-change", "children"),
         Output("stock-change", "className"),
         Output("stock-open", "children"),
         Output("stock-high", "children"),
         Output("stock-low", "children"),
         Output("stock-volume", "children")],
        [Input("stock-dropdown", "value")]
    )
    def update_stock_info(symbol):
        # Get the latest data for the selected stock
        df = fetch_stock_data(symbol, use_real_data=True)
        
        # Get the latest values
        latest = df.iloc[-1]
        prev_close = df.iloc[-2]["close"] if len(df) > 1 else latest["close"]
        
        # Calculate change and percent change
        change = latest["close"] - prev_close
        pct_change = (change / prev_close) * 100
        
        # Format values
        price = f"${latest['close']:.2f}"
        change_text = f"{change:.2f} ({pct_change:.2f}%)"
        change_class = "lead text-success" if change >= 0 else "lead text-danger"
        
        # Format volume with commas for thousands
        volume = f"{int(latest['volume']):,}"
        
        return [
            symbol,  # stock name
            price,   # stock price
            change_text,  # stock change
            change_class,  # class for stock change
            f"${latest['open']:.2f}",  # open
            f"${latest['high']:.2f}",  # high
            f"${latest['low']:.2f}",   # low
            volume  # volume
        ]
    
    # Update stock charts based on dropdown selection and chart type
    @app.callback(
        Output("stock-price-chart", "figure"),
        [Input("stock-dropdown", "value"),
         Input("chart-type", "value")]
    )
    def update_stock_chart(symbol, chart_type):
        # Get data for the selected stock
        df = fetch_stock_data(symbol, use_real_data=True)
        
        # Create figure based on chart type
        if chart_type == "line":
            fig = {
                "data": [
                    {
                        "x": df.index,
                        "y": df["close"],
                        "type": "line",
                        "name": "Price"
                    },
                    {
                        "x": df.index,
                        "y": df["sma_20"],
                        "type": "line",
                        "name": "20-day SMA",
                        "line": {"dash": "dash"}
                    },
                    {
                        "x": df.index,
                        "y": df["sma_50"],
                        "type": "line",
                        "name": "50-day SMA",
                        "line": {"dash": "dot"}
                    }
                ],
                "layout": {
                    "title": f"{symbol} Stock Price",
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": "Price ($)"},
                    "legend": {"orientation": "h", "y": 1.1},
                    "height": 500
                }
            }
        elif chart_type == "candlestick":
            fig = {
                "data": [
                    {
                        "x": df.index,
                        "open": df["open"],
                        "high": df["high"],
                        "low": df["low"],
                        "close": df["close"],
                        "type": "candlestick",
                        "name": symbol
                    }
                ],
                "layout": {
                    "title": f"{symbol} Stock Price",
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": "Price ($)"},
                    "height": 500
                }
            }
        elif chart_type == "ohlc":
            fig = {
                "data": [
                    {
                        "x": df.index,
                        "open": df["open"],
                        "high": df["high"],
                        "low": df["low"],
                        "close": df["close"],
                        "type": "ohlc",
                        "name": symbol
                    }
                ],
                "layout": {
                    "title": f"{symbol} Stock Price",
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": "Price ($)"},
                    "height": 500
                }
            }
        
        return fig
    
    # Update volume chart based on dropdown selection
    @app.callback(
        Output("volume-chart", "figure"),
        [Input("stock-dropdown", "value")]
    )
    def update_volume_chart(symbol):
        # Get data for the selected stock
        df = fetch_stock_data(symbol, use_real_data=True)
        
        # Create volume chart
        fig = {
            "data": [
                {
                    "x": df.index,
                    "y": df["volume"],
                    "type": "bar",
                    "name": "Volume",
                    "marker": {"color": "rgba(0, 0, 128, 0.3)"}
                }
            ],
            "layout": {
                "margin": {"t": 10, "b": 30},
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Volume"},
                "height": 250
            }
        }
        
        return fig
    
    # Update moving averages chart based on dropdown selection
    @app.callback(
        Output("ma-chart", "figure"),
        [Input("stock-dropdown", "value")]
    )
    def update_ma_chart(symbol):
        # Get data for the selected stock
        df = fetch_stock_data(symbol, use_real_data=True)
        
        # Create MA chart
        fig = {
            "data": [
                {
                    "x": df.index,
                    "y": df["sma_20"],
                    "type": "line",
                    "name": "20-day SMA"
                },
                {
                    "x": df.index,
                    "y": df["sma_50"],
                    "type": "line",
                    "name": "50-day SMA"
                },
                {
                    "x": df.index,
                    "y": df["sma_200"],
                    "type": "line",
                    "name": "200-day SMA"
                }
            ],
            "layout": {
                "margin": {"t": 10, "b": 30},
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Price ($)"},
                "height": 250
            }
        }
        
        return fig
    
    # Update technical analysis charts and text based on dropdown selection
    @app.callback(
        [Output("ma-analysis-chart", "figure"),
         Output("ma-analysis-text", "children"),
         Output("bollinger-chart", "figure"),
         Output("bollinger-analysis-text", "children"),
         Output("rsi-chart", "figure"),
         Output("rsi-analysis-text", "children"),
         Output("macd-chart", "figure"),
         Output("macd-analysis-text", "children")],
        [Input("stock-dropdown", "value")]
    )
    def update_technical_analysis(symbol):
        # Get data for the selected stock
        df = fetch_stock_data(symbol, use_real_data=True)
        
        # Get the latest values for analysis
        latest = df.iloc[-1]
        
        # Moving Average Analysis
        ma_fig = {
            "data": [
                {
                    "x": df.index,
                    "y": df["close"],
                    "type": "line",
                    "name": "Price"
                },
                {
                    "x": df.index,
                    "y": df["sma_20"],
                    "type": "line",
                    "name": "20-day SMA",
                    "line": {"dash": "dash"}
                },
                {
                    "x": df.index,
                    "y": df["sma_50"],
                    "type": "line",
                    "name": "50-day SMA",
                    "line": {"dash": "dot"}
                },
                {
                    "x": df.index,
                    "y": df["sma_200"],
                    "type": "line",
                    "name": "200-day SMA",
                    "line": {"dash": "dashdot"}
                }
            ],
            "layout": {
                "title": "Moving Average Analysis",
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Price ($)"},
                "height": 500
            }
        }
        
        # MA Analysis Text
        if latest["close"] > latest["sma_50"] and latest["sma_20"] > latest["sma_50"]:
            ma_analysis = f"Bullish: {symbol} is trading above its 50-day moving average, and the 20-day MA is above the 50-day MA, suggesting positive momentum."
        elif latest["close"] < latest["sma_50"] and latest["sma_20"] < latest["sma_50"]:
            ma_analysis = f"Bearish: {symbol} is trading below its 50-day moving average, and the 20-day MA is below the 50-day MA, suggesting negative momentum."
        else:
            ma_analysis = f"Neutral: {symbol} is showing mixed signals based on moving average analysis."
        
        # Bollinger Bands Analysis
        bollinger_fig = {
            "data": [
                {
                    "x": df.index,
                    "y": df["close"],
                    "type": "line",
                    "name": "Price"
                },
                {
                    "x": df.index,
                    "y": df["middle_band"],
                    "type": "line",
                    "name": "Middle Band (20-day SMA)",
                    "line": {"dash": "dash"}
                },
                {
                    "x": df.index,
                    "y": df["upper_band"],
                    "type": "line",
                    "name": "Upper Band (+2σ)",
                    "line": {"dash": "dot"}
                },
                {
                    "x": df.index,
                    "y": df["lower_band"],
                    "type": "line",
                    "name": "Lower Band (-2σ)",
                    "line": {"dash": "dot"}
                }
            ],
            "layout": {
                "title": "Bollinger Bands",
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Price ($)"},
                "height": 500
            }
        }
        
        # Bollinger Analysis Text
        if latest["close"] > latest["upper_band"]:
            bollinger_analysis = f"Overbought: {symbol} is trading above the upper Bollinger Band, suggesting it may be overbought."
        elif latest["close"] < latest["lower_band"]:
            bollinger_analysis = f"Oversold: {symbol} is trading below the lower Bollinger Band, suggesting it may be oversold."
        else:
            bollinger_analysis = f"Normal Range: {symbol} is trading within its Bollinger Bands, suggesting normal volatility."
        
        # RSI Analysis
        rsi_fig = {
            "data": [
                {
                    "x": df.index,
                    "y": df["rsi"],
                    "type": "line",
                    "name": "RSI"
                }
            ],
            "layout": {
                "title": "Relative Strength Index (14-day)",
                "xaxis": {"title": "Date"},
                "yaxis": {
                    "title": "RSI",
                    "range": [0, 100]
                },
                "shapes": [
                    {
                        "type": "line",
                        "x0": df.index[0],
                        "x1": df.index[-1],
                        "y0": 70,
                        "y1": 70,
                        "line": {
                            "color": "red",
                            "width": 2,
                            "dash": "dash"
                        }
                    },
                    {
                        "type": "line",
                        "x0": df.index[0],
                        "x1": df.index[-1],
                        "y0": 30,
                        "y1": 30,
                        "line": {
                            "color": "green",
                            "width": 2,
                            "dash": "dash"
                        }
                    }
                ],
                "height": 500
            }
        }
        
        # RSI Analysis Text
        if latest["rsi"] > 70:
            rsi_analysis = f"Overbought: {symbol} has an RSI above 70, suggesting it may be overbought and could experience a price correction."
        elif latest["rsi"] < 30:
            rsi_analysis = f"Oversold: {symbol} has an RSI below 30, suggesting it may be oversold and could experience a price bounce."
        else:
            rsi_analysis = f"Neutral: {symbol} has an RSI of {latest['rsi']:.2f}, which is within the normal range."
        
        # MACD Analysis
        macd_fig = {
            "data": [
                {
                    "x": df.index,
                    "y": df["macd"],
                    "type": "line",
                    "name": "MACD"
                },
                {
                    "x": df.index,
                    "y": df["signal"],
                    "type": "line",
                    "name": "Signal Line",
                    "line": {"dash": "dash"}
                },
                {
                    "x": df.index,
                    "y": df["macd_histogram"],
                    "type": "bar",
                    "name": "Histogram",
                    "marker": {"color": "rgba(0, 0, 128, 0.3)"}
                }
            ],
            "layout": {
                "title": "MACD (12-26-9)",
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "MACD"},
                "height": 500
            }
        }
        
        # MACD Analysis Text
        if latest["macd"] > latest["signal"] and latest["macd_histogram"] > 0:
            macd_analysis = f"Bullish: {symbol}'s MACD is above the signal line, suggesting positive momentum."
        elif latest["macd"] < latest["signal"] and latest["macd_histogram"] < 0:
            macd_analysis = f"Bearish: {symbol}'s MACD is below the signal line, suggesting negative momentum."
        else:
            macd_analysis = f"Neutral: {symbol}'s MACD is showing mixed signals."
        
        return ma_fig, ma_analysis, bollinger_fig, bollinger_analysis, rsi_fig, rsi_analysis, macd_fig, macd_analysis

def update_dashboard(app: dash.Dash, interval_seconds: int = 60) -> None:
    """
    Set up automatic dashboard updates at specified intervals.
    
    Args:
        app: Dash application
        interval_seconds: Update interval in seconds
    """
    # Add an interval component for automatic updates
    from dash import dcc
    from dash.dependencies import Input, Output
    
    # Add interval component to the layout if it doesn't exist
    if not any(component.id == 'interval-component' for component in app.layout.children):
        app.layout.children.append(
            dcc.Interval(
                id='interval-component',
                interval=interval_seconds * 1000,  # in milliseconds
                n_intervals=0
            )
        )
    
    # Register callback to update data on interval
    @app.callback(
        [Output("stock-price-chart", "figure"),
         Output("volume-chart", "figure"),
         Output("ma-chart", "figure")],
        [Input("interval-component", "n_intervals"),
         Input("stock-dropdown", "value"),
         Input("chart-type", "value")]
    )
    def update_charts_on_interval(n_intervals, symbol, chart_type):
        """Update charts at regular intervals."""
        # Get fresh data
        df = fetch_stock_data(symbol, use_real_data=True)
        
        # Update main chart based on chart type
        if chart_type == "line":
            main_chart = {
                "data": [
                    {
                        "x": df.index,
                        "y": df["close"],
                        "type": "line",
                        "name": "Price"
                    },
                    {
                        "x": df.index,
                        "y": df["sma_20"],
                        "type": "line",
                        "name": "20-day SMA",
                        "line": {"dash": "dash"}
                    },
                    {
                        "x": df.index,
                        "y": df["sma_50"],
                        "type": "line",
                        "name": "50-day SMA",
                        "line": {"dash": "dot"}
                    }
                ],
                "layout": {
                    "title": f"{symbol} Stock Price",
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": "Price ($)"},
                    "legend": {"orientation": "h", "y": 1.1},
                    "height": 500
                }
            }
        elif chart_type == "candlestick":
            main_chart = {
                "data": [
                    {
                        "x": df.index,
                        "open": df["open"],
                        "high": df["high"],
                        "low": df["low"],
                        "close": df["close"],
                        "type": "candlestick",
                        "name": symbol
                    }
                ],
                "layout": {
                    "title": f"{symbol} Stock Price",
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": "Price ($)"},
                    "height": 500
                }
            }
        elif chart_type == "ohlc":
            main_chart = {
                "data": [
                    {
                        "x": df.index,
                        "open": df["open"],
                        "high": df["high"],
                        "low": df["low"],
                        "close": df["close"],
                        "type": "ohlc",
                        "name": symbol
                    }
                ],
                "layout": {
                    "title": f"{symbol} Stock Price",
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": "Price ($)"},
                    "height": 500
                }
            }
        else:
            # Default to line chart
            main_chart = {
                "data": [
                    {
                        "x": df.index,
                        "y": df["close"],
                        "type": "line",
                        "name": "Price"
                    }
                ],
                "layout": {
                    "title": f"{symbol} Stock Price",
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": "Price ($)"},
                    "height": 500
                }
            }
        
        # Update volume chart
        volume_chart = {
            "data": [
                {
                    "x": df.index,
                    "y": df["volume"],
                    "type": "bar",
                    "name": "Volume",
                    "marker": {"color": "rgba(0, 0, 128, 0.3)"}
                }
            ],
            "layout": {
                "margin": {"t": 10, "b": 30},
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Volume"},
                "height": 250
            }
        }
        
        # Update MA chart
        ma_chart = {
            "data": [
                {
                    "x": df.index,
                    "y": df["sma_20"],
                    "type": "line",
                    "name": "20-day SMA"
                },
                {
                    "x": df.index,
                    "y": df["sma_50"],
                    "type": "line",
                    "name": "50-day SMA"
                },
                {
                    "x": df.index,
                    "y": df["sma_200"],
                    "type": "line",
                    "name": "200-day SMA"
                }
            ],
            "layout": {
                "margin": {"t": 10, "b": 30},
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Price ($)"},
                "height": 250
            }
        }
        
        return main_chart, volume_chart, ma_chart

def run_dashboard(app: dash.Dash, host: str = "0.0.0.0", port: int = 8050, debug: bool = False) -> None:
    """
    Run the dashboard application.
    
    Args:
        app: Dash application
        host: Host address
        port: Port number
        debug: Whether to run in debug mode
    """
    print(f"Starting dashboard on {host}:{port} with debug={debug}")
    try:
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        print(f"Error running dashboard: {e}")
        import traceback
        traceback.print_exc()

def update_dashboard(
    app: dash.Dash,
    interval_seconds: int = 30
) -> None:
    """
    Set up automatic dashboard updates.
    
    Args:
        app: Dash application
        interval_seconds: Update interval in seconds
    """
    # Add interval component to layout if it doesn't exist
    if not any(isinstance(child, dcc.Interval) and child.id == 'auto-refresh' 
               for child in app.layout.children):
        app.layout.children.append(
            dcc.Interval(
                id="auto-refresh",
                interval=interval_seconds * 1000,  # Convert to milliseconds
                n_intervals=0
            )
        )
    
    # Update volume chart based on dropdown selection
    @app.callback(
        Output("volume-chart", "figure"),
        [Input("stock-dropdown", "value")]
    )
    def update_volume_chart(selected_stock):
        # Generate sample data
        df = generate_sample_data()
        
        return {
            "data": [
                {
                    "x": df.index,
                    "y": df["volume"],
                    "type": "bar",
                    "name": "Volume",
                    "marker": {"color": "rgba(0, 0, 128, 0.3)"}
                }
            ],
            "layout": {
                "title": f"{selected_stock} Trading Volume",
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Volume"}
            }
        }
    
    # Update moving averages chart based on dropdown selection
    @app.callback(
        Output("ma-chart", "figure"),
        [Input("stock-dropdown", "value")]
    )
    def update_ma_chart(selected_stock):
        # Generate sample data
        df = generate_sample_data()
        
        return {
            "data": [
                {
                    "x": df.index,
                    "y": df["sma_20"],
                    "type": "line",
                    "name": "20-day SMA"
                },
                {
                    "x": df.index,
                    "y": df["sma_50"],
                    "type": "line",
                    "name": "50-day SMA"
                }
            ],
            "layout": {
                "title": f"{selected_stock} Moving Averages",
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Price ($)"}
            }
        }
    
    # Update forecast chart based on dropdown selection
    @app.callback(
        Output("forecast-chart", "figure"),
        [Input("stock-dropdown", "value")]
    )
    def update_forecast_chart(selected_stock):
        # Generate sample data
        df = generate_sample_data()
        last_price = df["close"].iloc[-1]
        
        # Generate forecast data
        np.random.seed(42)  # For reproducibility
        forecast_values = [last_price * (1 + np.random.normal(0.0005, 0.005) * i) for i in range(1, 31)]
        
        return {
            "data": [
                {
                    "x": list(range(1, 31)),
                    "y": forecast_values,
                    "type": "line",
                    "name": "Forecast"
                }
            ],
            "layout": {
                "title": f"{selected_stock} Price Forecast (30 Days)",
                "xaxis": {"title": "Days Ahead"},
                "yaxis": {"title": "Price ($)"}
            }
        }
