"""
Data access manager for STOCKER Pro.

This module provides a unified interface for retrieving financial data from various sources.
"""
import logging
from typing import Dict, Any, Optional, List, Union
import pandas as pd
from datetime import datetime, timedelta

from src.configuration.config import StockerConfig
from src.entity.config_entity import DatabaseConfig
from src.data_access.exceptions import DataAccessError, DataNotFoundError
from src.data_access.mongodb_client import MongoDBClient
from src.data_access.alpha_vantage_client import AlphaVantageClient

logger = logging.getLogger(__name__)


class DataAccessManager:
    """
    Unified interface for accessing financial data from various sources.
    
    This class serves as the primary interface for all data retrieval operations,
    abstracting away the specific data sources and providing a consistent API.
    """
    
    def __init__(self, config: StockerConfig):
        """
        Initialize the data access manager.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.clients = {}
        self.cache_dir = config.cache_dir
        
        # Initialize MongoDB client if configured
        if hasattr(config, 'db_config') and config.db_config:
            self.clients['mongodb'] = MongoDBClient(config.db_config, self.cache_dir)
        
        # Initialize Alpha Vantage client if configured
        if hasattr(config, 'use_alpha_vantage') and config.use_alpha_vantage and config.alpha_vantage_api_key:
            self.clients['alpha_vantage'] = AlphaVantageClient(config.alpha_vantage_api_key, self.cache_dir)
        
        # Add other data source clients here as needed
        
        if not self.clients:
            logger.warning("No data source clients configured. Data access will be limited.")
    
    def get_historical_data(self, 
                           symbol: str, 
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None,
                           interval: str = 'daily',
                           preferred_source: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical price data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format (default: 5 years ago)
            end_date: End date in YYYY-MM-DD format (default: today)
            interval: Data interval ('daily', 'weekly', 'monthly')
            preferred_source: Preferred data source (if None, tries all available sources)
            
        Returns:
            DataFrame with historical price data
            
        Raises:
            DataAccessError: If data retrieval fails from all sources
        """
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        errors = []
        
        # Try preferred source first if specified
        if preferred_source and preferred_source in self.clients:
            try:
                logger.info(f"Retrieving data for {symbol} from preferred source: {preferred_source}")
                return self.clients[preferred_source].get_historical_data(
                    symbol, start_date, end_date, interval
                )
            except Exception as e:
                logger.warning(f"Failed to retrieve data from preferred source {preferred_source}: {e}")
                errors.append(f"{preferred_source}: {str(e)}")
        
        # Try all available sources
        for source_name, client in self.clients.items():
            if source_name == preferred_source:
                continue  # Skip preferred source as it was already tried
                
            try:
                logger.info(f"Retrieving data for {symbol} from {source_name}")
                return client.get_historical_data(symbol, start_date, end_date, interval)
            except Exception as e:
                logger.warning(f"Failed to retrieve data from {source_name}: {e}")
                errors.append(f"{source_name}: {str(e)}")
        
        # If we get here, all sources failed
        error_msg = f"Failed to retrieve data for {symbol} from all sources: {'; '.join(errors)}"
        logger.error(error_msg)
        raise DataAccessError(error_msg)
    
    def get_company_info(self, symbol: str, preferred_source: Optional[str] = None) -> Dict[str, Any]:
        """
        Get company information for a symbol.
        
        Args:
            symbol: Stock symbol
            preferred_source: Preferred data source
            
        Returns:
            Dictionary with company information
            
        Raises:
            DataAccessError: If data retrieval fails from all sources
        """
        errors = []
        
        # Try preferred source first if specified
        if preferred_source and preferred_source in self.clients:
            try:
                logger.info(f"Retrieving company info for {symbol} from preferred source: {preferred_source}")
                return self.clients[preferred_source].get_company_info(symbol)
            except Exception as e:
                logger.warning(f"Failed to retrieve company info from preferred source {preferred_source}: {e}")
                errors.append(f"{preferred_source}: {str(e)}")
        
        # Try all available sources
        for source_name, client in self.clients.items():
            if source_name == preferred_source:
                continue  # Skip preferred source as it was already tried
                
            try:
                logger.info(f"Retrieving company info for {symbol} from {source_name}")
                return client.get_company_info(symbol)
            except Exception as e:
                logger.warning(f"Failed to retrieve company info from {source_name}: {e}")
                errors.append(f"{source_name}: {str(e)}")
        
        # If we get here, all sources failed
        error_msg = f"Failed to retrieve company info for {symbol} from all sources: {'; '.join(errors)}"
        logger.error(error_msg)
        raise DataAccessError(error_msg)
    
    def get_news_data(self, symbol: str, preferred_source: Optional[str] = None) -> pd.DataFrame:
        """
        Get news data for a symbol.
        
        Args:
            symbol: Stock symbol
            preferred_source: Preferred data source
            
        Returns:
            DataFrame with news data
            
        Raises:
            DataAccessError: If data retrieval fails from all sources
        """
        errors = []
        
        # Try preferred source first if specified
        if preferred_source and preferred_source in self.clients:
            if hasattr(self.clients[preferred_source], 'get_news_data'):
                try:
                    logger.info(f"Retrieving news data for {symbol} from preferred source: {preferred_source}")
                    return self.clients[preferred_source].get_news_data(symbol)
                except Exception as e:
                    logger.warning(f"Failed to retrieve news data from preferred source {preferred_source}: {e}")
                    errors.append(f"{preferred_source}: {str(e)}")
        
        # Try all available sources
        for source_name, client in self.clients.items():
            if source_name == preferred_source:
                continue  # Skip preferred source as it was already tried
                
            if hasattr(client, 'get_news_data'):
                try:
                    logger.info(f"Retrieving news data for {symbol} from {source_name}")
                    return client.get_news_data(symbol)
                except Exception as e:
                    logger.warning(f"Failed to retrieve news data from {source_name}: {e}")
                    errors.append(f"{source_name}: {str(e)}")
        
        # If we get here, all sources failed
        error_msg = f"Failed to retrieve news data for {symbol} from all sources: {'; '.join(errors)}"
        logger.error(error_msg)
        raise DataAccessError(error_msg)
    
    def get_holdings_data(self, symbol: str, preferred_source: Optional[str] = None) -> pd.DataFrame:
        """
        Get holdings data for a symbol.
        
        Args:
            symbol: Stock symbol
            preferred_source: Preferred data source
            
        Returns:
            DataFrame with holdings data
            
        Raises:
            DataAccessError: If data retrieval fails from all sources
        """
        errors = []
        
        # Try preferred source first if specified
        if preferred_source and preferred_source in self.clients:
            if hasattr(self.clients[preferred_source], 'get_holdings_data'):
                try:
                    logger.info(f"Retrieving holdings data for {symbol} from preferred source: {preferred_source}")
                    return self.clients[preferred_source].get_holdings_data(symbol)
                except Exception as e:
                    logger.warning(f"Failed to retrieve holdings data from preferred source {preferred_source}: {e}")
                    errors.append(f"{preferred_source}: {str(e)}")
        
        # Try all available sources
        for source_name, client in self.clients.items():
            if source_name == preferred_source:
                continue  # Skip preferred source as it was already tried
                
            if hasattr(client, 'get_holdings_data'):
                try:
                    logger.info(f"Retrieving holdings data for {symbol} from {source_name}")
                    return client.get_holdings_data(symbol)
                except Exception as e:
                    logger.warning(f"Failed to retrieve holdings data from {source_name}: {e}")
                    errors.append(f"{source_name}: {str(e)}")
        
        # If we get here, all sources failed
        error_msg = f"Failed to retrieve holdings data for {symbol} from all sources: {'; '.join(errors)}"
        logger.error(error_msg)
        raise DataAccessError(error_msg)
    
    def get_quarterly_results(self, symbol: str, preferred_source: Optional[str] = None) -> pd.DataFrame:
        """
        Get quarterly results for a symbol.
        
        Args:
            symbol: Stock symbol
            preferred_source: Preferred data source
            
        Returns:
            DataFrame with quarterly results
            
        Raises:
            DataAccessError: If data retrieval fails from all sources
        """
        errors = []
        
        # Try preferred source first if specified
        if preferred_source and preferred_source in self.clients:
            if hasattr(self.clients[preferred_source], 'get_quarterly_results'):
                try:
                    logger.info(f"Retrieving quarterly results for {symbol} from preferred source: {preferred_source}")
                    return self.clients[preferred_source].get_quarterly_results(symbol)
                except Exception as e:
                    logger.warning(f"Failed to retrieve quarterly results from preferred source {preferred_source}: {e}")
                    errors.append(f"{preferred_source}: {str(e)}")
        
        # Try all available sources
        for source_name, client in self.clients.items():
            if source_name == preferred_source:
                continue  # Skip preferred source as it was already tried
                
            if hasattr(client, 'get_quarterly_results'):
                try:
                    logger.info(f"Retrieving quarterly results for {symbol} from {source_name}")
                    return client.get_quarterly_results(symbol)
                except Exception as e:
                    logger.warning(f"Failed to retrieve quarterly results from {source_name}: {e}")
                    errors.append(f"{source_name}: {str(e)}")
        
        # If we get here, all sources failed
        error_msg = f"Failed to retrieve quarterly results for {symbol} from all sources: {'; '.join(errors)}"
        logger.error(error_msg)
        raise DataAccessError(error_msg)