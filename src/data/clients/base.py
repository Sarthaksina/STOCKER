"""
Base client for data access in STOCKER Pro.

This module provides a base class for all data source clients,
ensuring a consistent interface and behavior.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import pandas as pd
from datetime import datetime
import os
import json

from src.core.utils import save_to_cache, load_from_cache
from src.core.exceptions import DataAccessError, DataNotFoundError
from src.core.logging import logger

class BaseDataClient(ABC):
    """
    Abstract base class for all data source clients.
    
    This class defines the interface that all data source clients must implement,
    and provides common functionality like caching.
    """
    
    def __init__(self, name: str, cache_dir: str = "cache"):
        """
        Initialize the base data client.
        
        Args:
            name: Name of the data source
            cache_dir: Directory for caching data
        """
        self.name = name
        self.cache_dir = os.path.join(cache_dir, name)
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.debug(f"Initialized {name} client with cache directory: {self.cache_dir}")
    
    def _get_cache_key(self, function_name: str, **params) -> str:
        """
        Generate a cache key for the given function and parameters.
        
        Args:
            function_name: Name of the function being cached
            **params: Parameters to the function
            
        Returns:
            Cache key string
        """
        cache_data = {"function": function_name, "source": self.name}
        cache_data.update(params)
        return json.dumps(cache_data, sort_keys=True)
    
    def _get_from_cache(self, function_name: str, **params) -> Optional[Any]:
        """
        Get data from cache if available.
        
        Args:
            function_name: Name of the function being cached
            **params: Parameters to the function
            
        Returns:
            Cached data if available, None otherwise
        """
        cache_key = self._get_cache_key(function_name, **params)
        return load_from_cache(self.cache_dir, cache_key)
    
    def _save_to_cache(self, data: Any, function_name: str, **params) -> None:
        """
        Save data to cache.
        
        Args:
            data: Data to cache
            function_name: Name of the function being cached
            **params: Parameters to the function
        """
        cache_key = self._get_cache_key(function_name, **params)
        save_to_cache(self.cache_dir, cache_key, data)
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass