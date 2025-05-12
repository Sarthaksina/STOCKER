"""
Alpha Vantage API client for STOCKER Pro.
This module provides access to financial data from Alpha Vantage.
"""
import requests
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timedelta

from src.core.config import config
from src.data.clients.base import BaseDataClient
from src.core.exceptions import DataAccessError, DataNotFoundError, DataSourceRateLimitError, InvalidDataError
from src.core.logging import logger

class AlphaVantageClient(BaseDataClient):
    """
    Client for accessing Alpha Vantage API.
    
    Provides methods for fetching time series data, fundamental data,
    and technical indicators with caching support.
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "cache"):
        """
        Initialize the Alpha Vantage client.
        
        Args:
            api_key: Alpha Vantage API key (defaults to config if not provided)
            cache_dir: Directory for caching data
        """
        super().__init__("alpha_vantage", cache_dir)
        
        # Try to read the API key directly from the .env file
        import os
        from pathlib import Path
        from dotenv import load_dotenv
        
        # Try to load from .env file in project root
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        env_path = project_root / ".env"
        load_dotenv(dotenv_path=env_path)
        
        # Get the API key from various sources
        env_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        config_key = config.api.alpha_vantage_api_key
        
        # Log available keys for debugging
        logger.info(f"Environment API key: {env_key and 'Set' or 'Not set'}")
        logger.info(f"Config API key: {config_key and 'Set' or 'Not set'}")
        
        # Try to read directly from .env file as a last resort
        direct_key = None
        if env_path.exists():
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.startswith('ALPHA_VANTAGE_API_KEY='):
                            direct_key = line.strip().split('=', 1)[1].strip('"')
                            break
                logger.info(f"Direct .env file API key: {direct_key and 'Set' or 'Not set'}")
            except Exception as e:
                logger.warning(f"Error reading .env file directly: {e}")
        
        # Use the first available API key
        self.api_key = api_key or config_key or env_key or direct_key
        self.use_mock_data = False
        
        if not self.api_key:
            logger.warning("No Alpha Vantage API key provided. Using mock data instead.")
            self.use_mock_data = True
        else:
            logger.info(f"Using Alpha Vantage API key: {self.api_key[:4]}...{self.api_key[-4:]}")
        
        # Track API call count to avoid rate limits
        self.call_count = 0
        self.last_call_time = 0
        
    def _handle_rate_limit(self):
        """Handle API rate limiting."""
        # Alpha Vantage free tier allows 5 calls per minute
        current_time = time.time()
        if current_time - self.last_call_time < 60 and self.call_count >= 5:
            # Wait until a minute has passed since first call
            sleep_time = 60 - (current_time - self.last_call_time)
            logger.info(f"Rate limit reached, waiting {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            self.call_count = 0
            self.last_call_time = time.time()
        elif current_time - self.last_call_time >= 60:
            # Reset counter if more than a minute has passed
            self.call_count = 0
            self.last_call_time = current_time
            
        self.call_count += 1
        
    def _generate_mock_data(self, function: str, symbol: str, **params) -> Dict[str, Any]:
        """
        Generate mock data for testing when no API key is available.
        
        Args:
            function: Alpha Vantage function name
            symbol: Stock symbol
            params: Additional parameters
            
        Returns:
            Mock API response as dictionary
        """
        logger.info(f"Generating mock data for {function} - {symbol}")
        
        # Generate mock time series data
        if function in ['TIME_SERIES_DAILY_ADJUSTED', 'TIME_SERIES_WEEKLY_ADJUSTED', 'TIME_SERIES_MONTHLY_ADJUSTED']:
            # Determine the appropriate time series key based on the function
            if function == 'TIME_SERIES_DAILY_ADJUSTED':
                time_series_key = 'Time Series (Daily)'
                date_format = '%Y-%m-%d'
                days_increment = 1
            elif function == 'TIME_SERIES_WEEKLY_ADJUSTED':
                time_series_key = 'Weekly Adjusted Time Series'
                date_format = '%Y-%m-%d'
                days_increment = 7
            else:  # Monthly
                time_series_key = 'Monthly Adjusted Time Series'
                date_format = '%Y-%m-%d'
                days_increment = 30
            
            # Generate 100 data points by default, or more if 'full' outputsize is requested
            num_points = 1000 if params.get('outputsize') == 'full' else 100
            
            # Start from today and go backwards
            end_date = datetime.now()
            
            # Generate time series data
            time_series = {}
            current_date = end_date
            base_price = 100.0  # Starting price
            
            # Add some randomness based on the symbol to make different stocks look different
            import hashlib
            symbol_hash = int(hashlib.md5(symbol.encode()).hexdigest(), 16) % 100
            base_price += symbol_hash
            
            for i in range(num_points):
                # Generate price data with some randomness
                close_price = base_price * (1 + 0.0001 * (np.random.randn() * 10))
                open_price = close_price * (1 + 0.0001 * (np.random.randn() * 5))
                high_price = max(close_price, open_price) * (1 + 0.0001 * abs(np.random.randn() * 3))
                low_price = min(close_price, open_price) * (1 - 0.0001 * abs(np.random.randn() * 3))
                volume = int(1000000 * (1 + np.random.randn() * 0.3))
                
                # Format date as string
                date_str = current_date.strftime(date_format)
                
                # Create data point
                time_series[date_str] = {
                    '1. open': f"{open_price:.4f}",
                    '2. high': f"{high_price:.4f}",
                    '3. low': f"{low_price:.4f}",
                    '4. close': f"{close_price:.4f}",
                    '5. adjusted close': f"{close_price:.4f}",
                    '6. volume': str(volume),
                    '7. dividend amount': "0.0000",
                    '8. split coefficient': "1.0000"
                }
                
                # Move to previous date
                current_date -= timedelta(days=days_increment)
                
                # Adjust base price with a slight trend and randomness
                base_price *= (1 + 0.0001 * (np.random.randn() * 5 - 1))
            
            # Create the full response
            return {
                'Meta Data': {
                    '1. Information': f"Mock {function.replace('_', ' ')} for {symbol}",
                    '2. Symbol': symbol,
                    '3. Last Refreshed': end_date.strftime('%Y-%m-%d'),
                    '4. Output Size': params.get('outputsize', 'compact'),
                    '5. Time Zone': "US/Eastern"
                },
                time_series_key: time_series
            }
            
        # Generate mock company overview data
        elif function == 'OVERVIEW':
            return {
                'Symbol': symbol,
                'AssetType': 'Common Stock',
                'Name': f"{symbol} Corporation",
                'Description': f"Mock company description for {symbol}",
                'Exchange': 'NYSE',
                'Currency': 'USD',
                'Country': 'USA',
                'Sector': 'Technology',
                'Industry': 'Software',
                'MarketCapitalization': '1000000000',
                'EBITDA': '100000000',
                'PERatio': '20.5',
                'PEGRatio': '1.5',
                'BookValue': '50.5',
                'DividendPerShare': '1.0',
                'DividendYield': '0.01',
                'EPS': '5.5',
                'RevenuePerShareTTM': '25.5',
                'ProfitMargin': '0.15',
                'OperatingMarginTTM': '0.2',
                'ReturnOnAssetsTTM': '0.1',
                'ReturnOnEquityTTM': '0.15',
                'RevenueTTM': '1000000000',
                'GrossProfitTTM': '500000000',
                'DilutedEPSTTM': '5.5',
                'QuarterlyEarningsGrowthYOY': '0.1',
                'QuarterlyRevenueGrowthYOY': '0.15',
                'AnalystTargetPrice': '150.5',
                '52WeekHigh': '160.5',
                '52WeekLow': '90.5',
                '50DayMovingAverage': '130.5',
                '200DayMovingAverage': '125.5',
                'SharesOutstanding': '50000000',
                'SharesFloat': '45000000',
                'SharesShort': '1000000',
                'SharesShortPriorMonth': '900000',
                'ShortRatio': '2.5',
                'ShortPercentOutstanding': '0.02',
                'ShortPercentFloat': '0.022',
                'PercentInsiders': '0.1',
                'PercentInstitutions': '0.7',
                'ForwardPE': '19.5',
                'TrailingPE': '20.5',
                'PriceToSalesRatioTTM': '5.5',
                'PriceToBookRatio': '3.5',
                'EVToRevenue': '6.5',
                'EVToEBITDA': '15.5',
                'Beta': '1.2',
                '52WeekChange': '0.25',
                'S&P500_52WeekChange': '0.15',
                'LastDividendDate': '2023-05-15',
                'LastSplitFactor': '2:1',
                'LastSplitDate': '2020-08-31'
            }
            
        # Generate mock technical indicator data
        elif function in ['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS']:
            # Determine the appropriate indicator key
            indicator_key = f"Technical Analysis: {function}"
            
            # Generate 100 data points by default
            num_points = 100
            
            # Start from today and go backwards
            end_date = datetime.now()
            
            # Generate indicator data
            indicator_data = {}
            current_date = end_date
            
            for i in range(num_points):
                # Format date as string
                date_str = current_date.strftime('%Y-%m-%d')
                
                # Create indicator value based on the function
                if function == 'SMA' or function == 'EMA':
                    indicator_data[date_str] = {
                        function: f"{100 + 10 * np.sin(i/10) + np.random.randn() * 5:.4f}"
                    }
                elif function == 'RSI':
                    indicator_data[date_str] = {
                        'RSI': f"{50 + 20 * np.sin(i/10) + np.random.randn() * 5:.4f}"
                    }
                elif function == 'MACD':
                    indicator_data[date_str] = {
                        'MACD': f"{np.sin(i/10) * 5 + np.random.randn():.4f}",
                        'MACD_Signal': f"{np.sin((i+3)/10) * 5 + np.random.randn():.4f}",
                        'MACD_Hist': f"{np.sin(i/10) * 2 - np.sin((i+3)/10) * 2 + np.random.randn() * 0.5:.4f}"
                    }
                elif function == 'BBANDS':
                    mid = 100 + 10 * np.sin(i/10)
                    indicator_data[date_str] = {
                        'Real Middle Band': f"{mid:.4f}",
                        'Real Upper Band': f"{mid + 20 + np.random.randn() * 2:.4f}",
                        'Real Lower Band': f"{mid - 20 + np.random.randn() * 2:.4f}"
                    }
                
                # Move to previous date
                current_date -= timedelta(days=1)
            
            # Create the full response
            return {
                'Meta Data': {
                    '1. Information': f"Mock {function} for {symbol}",
                    '2. Symbol': symbol,
                    '3. Last Refreshed': end_date.strftime('%Y-%m-%d'),
                    '4. Interval': params.get('interval', 'daily'),
                    '5. Time Period': params.get('time_period', 14),
                    '6. Series Type': params.get('series_type', 'close'),
                    '7. Time Zone': "US/Eastern"
                },
                indicator_key: indicator_data
            }
            
        # Generate mock search results
        elif function == 'SYMBOL_SEARCH':
            keywords = params.get('keywords', '')
            return {
                'bestMatches': [
                    {
                        '1. symbol': symbol,
                        '2. name': f"{symbol} Corporation",
                        '3. type': 'Equity',
                        '4. region': 'United States',
                        '5. marketOpen': '09:30',
                        '6. marketClose': '16:00',
                        '7. timezone': 'UTC-04',
                        '8. currency': 'USD',
                        '9. matchScore': '1.0000'
                    },
                    {
                        '1. symbol': f"{symbol}A",
                        '2. name': f"{symbol} America Inc.",
                        '3. type': 'Equity',
                        '4. region': 'United States',
                        '5. marketOpen': '09:30',
                        '6. marketClose': '16:00',
                        '7. timezone': 'UTC-04',
                        '8. currency': 'USD',
                        '9. matchScore': '0.8000'
                    },
                    {
                        '1. symbol': f"{symbol}B",
                        '2. name': f"{symbol} International",
                        '3. type': 'Equity',
                        '4. region': 'United States',
                        '5. marketOpen': '09:30',
                        '6. marketClose': '16:00',
                        '7. timezone': 'UTC-04',
                        '8. currency': 'USD',
                        '9. matchScore': '0.7500'
                    }
                ]
            }
            
        # Generate mock intraday data
        elif function == 'TIME_SERIES_INTRADAY':
            interval = params.get('interval', '5min')
            time_series_key = f"Time Series ({interval})"
            
            # Generate 100 data points by default, or more if 'full' outputsize is requested
            num_points = 1000 if params.get('outputsize') == 'full' else 100
            
            # Start from now and go backwards
            end_datetime = datetime.now().replace(second=0, microsecond=0)
            
            # Determine minutes to subtract based on interval
            if interval == '1min':
                minutes_increment = 1
            elif interval == '5min':
                minutes_increment = 5
            elif interval == '15min':
                minutes_increment = 15
            elif interval == '30min':
                minutes_increment = 30
            else:  # 60min
                minutes_increment = 60
            
            # Generate time series data
            time_series = {}
            current_datetime = end_datetime
            base_price = 100.0  # Starting price
            
            # Add some randomness based on the symbol
            import hashlib
            symbol_hash = int(hashlib.md5(symbol.encode()).hexdigest(), 16) % 100
            base_price += symbol_hash
            
            for i in range(num_points):
                # Only include points during market hours (9:30 AM - 4:00 PM ET, Monday-Friday)
                if current_datetime.weekday() < 5 and (
                    (current_datetime.hour == 9 and current_datetime.minute >= 30) or
                    (current_datetime.hour > 9 and current_datetime.hour < 16) or
                    (current_datetime.hour == 16 and current_datetime.minute == 0)
                ):
                    # Generate price data with some randomness
                    close_price = base_price * (1 + 0.0001 * (np.random.randn() * 10))
                    open_price = close_price * (1 + 0.0001 * (np.random.randn() * 5))
                    high_price = max(close_price, open_price) * (1 + 0.0001 * abs(np.random.randn() * 3))
                    low_price = min(close_price, open_price) * (1 - 0.0001 * abs(np.random.randn() * 3))
                    volume = int(100000 * (1 + np.random.randn() * 0.3))
                    
                    # Format datetime as string
                    datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Create data point
                    time_series[datetime_str] = {
                        '1. open': f"{open_price:.4f}",
                        '2. high': f"{high_price:.4f}",
                        '3. low': f"{low_price:.4f}",
                        '4. close': f"{close_price:.4f}",
                        '5. volume': str(volume)
                    }
                    
                    # Adjust base price with a slight trend and randomness
                    base_price *= (1 + 0.0001 * (np.random.randn() * 5 - 1))
                
                # Move to previous time point
                current_datetime -= timedelta(minutes=minutes_increment)
            
            # Create the full response
            return {
                'Meta Data': {
                    '1. Information': f"Mock Intraday Time Series with {interval} interval",
                    '2. Symbol': symbol,
                    '3. Last Refreshed': end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    '4. Interval': interval,
                    '5. Output Size': params.get('outputsize', 'compact'),
                    '6. Time Zone': "US/Eastern"
                },
                time_series_key: time_series
            }
            
        # Default mock data for unsupported functions
        else:
            return {
                'Meta Data': {
                    '1. Information': f"Mock data for {function}",
                    '2. Symbol': symbol,
                    '3. Last Refreshed': datetime.now().strftime('%Y-%m-%d'),
                    '4. Time Zone': "US/Eastern"
                },
                'Note': f"Mock data for {function} is not fully implemented. This is placeholder data."
            }
    
    def _make_request(self, function: str, symbol: str, **params) -> Dict[str, Any]:
        """
        Make a request to Alpha Vantage API with caching.
        
        Args:
            function: Alpha Vantage function name
            symbol: Stock symbol
            params: Additional parameters
            
        Returns:
            API response as dictionary
        """
        # Check cache first
        cached_data = self._get_from_cache(f"{function}_{symbol}", **params)
        if cached_data:
            logger.debug(f"Using cached data for {symbol} ({function})")
            return cached_data
            
        # If using mock data, generate synthetic data
        if self.use_mock_data:
            logger.info(f"Using mock data for {symbol} ({function})")
            mock_data = self._generate_mock_data(function, symbol, **params)
            
            # Cache the mock data
            self._save_to_cache(mock_data, f"{function}_{symbol}", **params)
            
            return mock_data
        
        # Handle rate limiting
        self._handle_rate_limit()
        
        # Prepare request parameters
        request_params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_key,
            'datatype': 'json',
            **params
        }
        
        # Make request
        try:
            logger.debug(f"Fetching {function} data for {symbol}")
            response = requests.get(self.BASE_URL, params=request_params)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise DataAccessError(f"Alpha Vantage API error: {data['Error Message']}", "alpha_vantage")
            
            if 'Note' in data and 'API call frequency' in data['Note']:
                logger.warning(f"Alpha Vantage rate limit warning: {data['Note']}")
                raise DataSourceRateLimitError("alpha_vantage", 60)
            
            # Cache the response
            self._save_to_cache(data, f"{function}_{symbol}", **params)
            
            return data
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise DataAccessError(f"Failed to fetch data from Alpha Vantage: {e}", "alpha_vantage")
    
    def get_historical_data(self, 
                           symbol: str, 
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None,
                           interval: str = 'daily') -> pd.DataFrame:
        """
        Get historical price data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval ('daily', 'weekly', 'monthly')
            
        Returns:
            DataFrame with historical price data
            
        Raises:
            DataAccessError: If data retrieval fails
        """
        # Map interval to Alpha Vantage function
        interval_map = {
            'daily': 'TIME_SERIES_DAILY_ADJUSTED',
            'weekly': 'TIME_SERIES_WEEKLY_ADJUSTED',
            'monthly': 'TIME_SERIES_MONTHLY_ADJUSTED'
        }
        
        if interval not in interval_map:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {list(interval_map.keys())}")
        
        function = interval_map[interval]
        outputsize = 'full' if start_date and (datetime.now() - datetime.strptime(start_date, '%Y-%m-%d')).days > 100 else 'compact'
        
        # Get data from Alpha Vantage
        data = self._make_request(function, symbol, outputsize=outputsize)
        
        # Parse time series data
        time_series_key = next((k for k in data.keys() if 'Time Series' in k), None)
        if not time_series_key or not data.get(time_series_key):
            raise DataNotFoundError(f"No {interval} data found", symbol, "alpha_vantage")
        
        time_series = data.get(time_series_key, {})
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Rename columns
        df.columns = [col.split('. ')[1] for col in df.columns]
        
        # Sort by date
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        # Filter by date range if provided
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        return df
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company information
            
        Raises:
            DataAccessError: If data retrieval fails
        """
        data = self._make_request('OVERVIEW', symbol)
        
        if not data or len(data) <= 1:  # Alpha Vantage returns a small dict with error info if no data
            raise DataNotFoundError(f"No company information found", symbol, "alpha_vantage")
            
        return data
    
    def get_technical_indicator(self, symbol: str, indicator: str, interval: str = 'daily', 
                               time_period: int = 14, series_type: str = 'close') -> pd.DataFrame:
        """
        Get technical indicator values for a symbol.
        
        Args:
            symbol: Stock symbol
            indicator: Technical indicator (e.g., 'SMA', 'EMA', 'RSI')
            interval: Data interval ('daily', 'weekly', 'monthly')
            time_period: Number of data points to calculate the indicator
            series_type: Price series to use ('close', 'open', 'high', 'low')
            
        Returns:
            DataFrame with indicator values
            
        Raises:
            DataAccessError: If data retrieval fails
        """
        # Map interval to Alpha Vantage interval string
        interval_map = {
            'daily': 'daily',
            'weekly': 'weekly',
            'monthly': 'monthly'
        }
        
        if interval not in interval_map:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {list(interval_map.keys())}")
            
        av_interval = interval_map[interval]
        
        # Call Alpha Vantage
        data = self._make_request(
            indicator,
            symbol,
            interval=av_interval,
            time_period=time_period,
            series_type=series_type
        )
        
        # Parse the response
        indicator_key = f"Technical Analysis: {indicator}"
        if not data.get(indicator_key):
            raise DataNotFoundError(f"No {indicator} data found", symbol, "alpha_vantage")
            
        indicator_data = data.get(indicator_key, {})
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(indicator_data, orient='index')
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        # Set index to datetime
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        return df
    
    def search_symbols(self, keywords: str) -> List[Dict[str, str]]:
        """
        Search for symbols matching keywords.
        
        Args:
            keywords: Search keywords
            
        Returns:
            List of matching symbols with metadata
            
        Raises:
            DataAccessError: If search fails
        """
        data = self._make_request('SYMBOL_SEARCH', None, keywords=keywords)
        
        if 'bestMatches' not in data:
            raise DataAccessError("Failed to get search results", "alpha_vantage")
            
        return data['bestMatches']
    
    def get_intraday_data(self, symbol: str, interval: str = '5min', 
                           outputsize: str = 'compact') -> pd.DataFrame:
        """
        Get intraday price data for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: Time interval between data points ('1min', '5min', '15min', '30min', '60min')
            outputsize: 'compact' (100 data points) or 'full' (all available data)
            
        Returns:
            DataFrame with intraday price data
            
        Raises:
            DataAccessError: If data retrieval fails
        """
        valid_intervals = ['1min', '5min', '15min', '30min', '60min']
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {valid_intervals}")
            
        data = self._make_request('TIME_SERIES_INTRADAY', symbol, interval=interval, outputsize=outputsize)
        
        # Parse time series data
        time_series_key = f"Time Series ({interval})"
        if not data.get(time_series_key):
            raise DataNotFoundError(f"No intraday data found", symbol, "alpha_vantage")
            
        time_series = data.get(time_series_key, {})
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        # Rename columns
        df.columns = [col.split('. ')[1] for col in df.columns]
        
        # Set index to datetime
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        return df