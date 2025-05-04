"""
MongoDB client for STOCKER Pro.

This module provides access to MongoDB data sources, with caching support.
"""
import logging
from typing import Dict, Any, Optional, List, Union
import pandas as pd
from pymongo import MongoClient
import json

from src.entity.config_entity import DatabaseConfig
from src.constant.constants import STOCK_COLLECTION, SUPPORTED_EXCHANGES
from src.data_access.base_client import BaseDataClient
from src.data_access.exceptions import DataNotFoundError, DataSourceConnectionError

logger = logging.getLogger(__name__)


class MongoDBClient(BaseDataClient):
    """
    Client for accessing data from MongoDB.
    
    This class provides methods to retrieve various types of financial data
    from MongoDB collections, with caching support.
    """
    
    def __init__(self, config: DatabaseConfig, cache_dir: str = "cache"):
        """
        Initialize the MongoDB client.
        
        Args:
            config: MongoDB configuration
            cache_dir: Directory for caching data
        """
        super().__init__("mongodb", cache_dir)
        self.config = config
        self._client = None
        logger.debug(f"Initialized MongoDB client with database: {config.db_name}")
    
    @property
    def client(self):
        """
        Get the MongoDB client, creating it if necessary.
        
        Returns:
            MongoDB client
            
        Raises:
            DataSourceConnectionError: If connection fails
        """
        if self._client is None:
            try:
                self._client = MongoClient(self.config.uri)
                # Test connection
                self._client.admin.command('ping')
            except Exception as e:
                raise DataSourceConnectionError("mongodb", str(e))
        return self._client
    
    @property
    def db(self):
        """
        Get the MongoDB database.
        
        Returns:
            MongoDB database
        """
        return self.client[self.config.db_name]
    
    def get_historical_data(self, 
                           symbol: str, 
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None,
                           interval: str = 'daily') -> pd.DataFrame:
        """
        Get historical price data for a symbol from MongoDB.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval ('daily', 'weekly', 'monthly')
            
        Returns:
            DataFrame with historical price data
            
        Raises:
            DataNotFoundError: If no data found for the symbol
            DataSourceConnectionError: If connection fails
        """
        # Check cache first
        cached_data = self._get_from_cache("get_historical_data", 
                                          symbol=symbol, 
                                          start_date=start_date, 
                                          end_date=end_date, 
                                          interval=interval)
        if cached_data:
            return pd.DataFrame(cached_data)
        
        # Build query
        query = {"symbol": symbol}
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = start_date
            if end_date:
                date_query["$lte"] = end_date
            if date_query:
                query["date"] = date_query
        
        try:
            collection = self.db[STOCK_COLLECTION]
            data = list(collection.find(query))
            
            if not data:
                raise DataNotFoundError("No historical data found", symbol, "mongodb")
            
            df = pd.DataFrame(data)
            
            # Cache the results
            self._save_to_cache(df.to_dict(orient="records"), 
                               "get_historical_data", 
                               symbol=symbol, 
                               start_date=start_date, 
                               end_date=end_date, 
                               interval=interval)
            
            return df
        except DataNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving historical data for {symbol}: {e}")
            raise DataSourceConnectionError("mongodb", str(e))
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information for a symbol from MongoDB.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company information
            
        Raises:
            DataNotFoundError: If no data found for the symbol
            DataSourceConnectionError: If connection fails
        """
        # Check cache first
        cached_data = self._get_from_cache("get_company_info", symbol=symbol)
        if cached_data:
            return cached_data
        
        try:
            collection = self.db["company_info"]
            data = collection.find_one({"symbol": symbol})
            
            if not data:
                raise DataNotFoundError("No company information found", symbol, "mongodb")
            
            # Cache the results
            self._save_to_cache(data, "get_company_info", symbol=symbol)
            
            return data
        except DataNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving company info for {symbol}: {e}")
            raise DataSourceConnectionError("mongodb", str(e))
    
    def get_news_data(self, symbol: str) -> pd.DataFrame:
        """
        Get news data for a symbol from MongoDB.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with news data
            
        Raises:
            DataNotFoundError: If no data found for the symbol
            DataSourceConnectionError: If connection fails
        """
        # Check cache first
        cached_data = self._get_from_cache("get_news_data", symbol=symbol)
        if cached_data:
            return pd.DataFrame(cached_data)
        
        try:
            collection = self.db[f"{symbol.replace('.', '_')}_news"]
            data = list(collection.find())
            
            if not data:
                raise DataNotFoundError("No news data found", symbol, "mongodb")
            
            df = pd.DataFrame(data)
            
            # Cache the results
            self._save_to_cache(df.to_dict(orient="records"), "get_news_data", symbol=symbol)
            
            return df
        except DataNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving news data for {symbol}: {e}")
            raise DataSourceConnectionError("mongodb", str(e))
    
    def get_holdings_data(self, symbol: str) -> pd.DataFrame:
        """
        Get holdings data for a symbol from MongoDB.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with holdings data
            
        Raises:
            DataNotFoundError: If no data found for the symbol
            DataSourceConnectionError: If connection fails
        """
        # Check cache first
        cached_data = self._get_from_cache("get_holdings_data", symbol=symbol)
        if cached_data:
            return pd.DataFrame(cached_data)
        
        try:
            collection = self.db[f"{symbol.replace('.', '_')}_holdings"]
            data = list(collection.find())
            
            if not data:
                raise DataNotFoundError("No holdings data found", symbol, "mongodb")
            
            df = pd.DataFrame(data)
            
            # Cache the results
            self._save_to_cache(df.to_dict(orient="records"), "get_holdings_data", symbol=symbol)
            
            return df
        except DataNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving holdings data for {symbol}: {e}")
            raise DataSourceConnectionError("mongodb", str(e))
    
    def get_quarterly_results(self, symbol: str) -> pd.DataFrame:
        """
        Get quarterly results for a symbol from MongoDB.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with quarterly results
            
        Raises:
            DataNotFoundError: If no data found for the symbol
            DataSourceConnectionError: If connection fails
        """
        # Check cache first
        cached_data = self._get_from_cache("get_quarterly_results", symbol=symbol)
        if cached_data:
            return pd.DataFrame(cached_data)
        
        try:
            collection = self.db[f"{symbol.replace('.', '_')}_quarterly"]
            data = list(collection.find())
            
            if not data:
                raise DataNotFoundError("No quarterly results found", symbol, "mongodb")
            
            df = pd.DataFrame(data)
            
            # Cache the results
            self._save_to_cache(df.to_dict(orient="records"), "get_quarterly_results", symbol=symbol)
            
            return df
        except DataNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving quarterly results for {symbol}: {e}")
            raise DataSourceConnectionError("mongodb", str(e))