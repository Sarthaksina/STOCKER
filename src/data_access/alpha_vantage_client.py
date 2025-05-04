"""
Alpha Vantage API client for STOCKER Pro.
This module provides access to financial data from Alpha Vantage.
"""
import requests
import pandas as pd
import numpy as np
import os
import json
import time
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
from datetime import datetime, timedelta

from src.configuration.config import StockerConfig
from src.data_access.base_client import BaseDataClient
from src.data_access.exceptions import DataAccessError, DataNotFoundError, DataSourceRateLimitError, InvalidDataError

logger = logging.getLogger(__name__)

class AlphaVantageClient(BaseDataClient):
    """
    Client for accessing Alpha Vantage API.
    
    Provides methods for fetching time series data, fundamental data,
    and technical indicators with caching support.
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: str, cache_dir: str = "cache"):
        """
        Initialize the Alpha Vantage client.
        
        Args:
            api_key: Alpha Vantage API key
            cache_dir: Directory for caching data
        """
        super().__init__("alpha_vantage", cache_dir)
        
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
        
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
        cache_key = self._get_cache_key(f"{function}_{symbol}", **params)
        cached_data = self._get_from_cache(f"{function}_{symbol}", **params)
        if cached_data:
            logger.debug(f"Using cached data for {symbol} ({function})")
            return cached_data
        
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
        
        if not data or 'Symbol' not in data:
            raise DataNotFoundError("Company information not found", symbol, "alpha_vantage")
        
        return data
    
    # Additional methods from the original class can be added here
    # with appropriate modifications to match the BaseDataClient interface
    
    def get_daily_adjusted(self, symbol: str, outputsize: str = 'compact') -> pd.DataFrame:
        """
        Get daily adjusted time series for a symbol.
        
        Args:
            symbol: Stock symbol
            outputsize: 'compact' (100 data points) or 'full' (20+ years)
            
        Returns:
            DataFrame with daily adjusted time series
        """
        # This is now a wrapper around get_historical_data for backward compatibility
        return self.get_historical_data(symbol, interval='daily')