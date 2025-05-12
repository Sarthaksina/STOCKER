"""
Data manager module for STOCKER Pro.

This module provides a unified interface for accessing and managing data
from various sources and ensures data consistency and caching.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import logging
import concurrent.futures
import time

from src.core.config import config
from src.data.clients.alpha_vantage import AlphaVantageClient
from src.data.clients.mongodb import MongoDBClient
from src.data.ingestion import DataIngestion
from src.core.exceptions import DataAccessError, DataNotFoundError
from src.core.logging import logger

class DataManager:
    """
    Unified interface for data access and management.
    
    This class coordinates data access from various sources, handles caching,
    and provides utilities for common data operations.
    """
    
    def __init__(self, alpha_vantage_client: Optional[AlphaVantageClient] = None,
                 mongodb_client: Optional[MongoDBClient] = None,
                 data_ingestion: Optional[DataIngestion] = None):
        """
        Initialize the data manager.
        
        Args:
            alpha_vantage_client: Alpha Vantage client (created if not provided)
            mongodb_client: MongoDB client (created if not provided)
            data_ingestion: Data ingestion service (created if not provided)
        """
        self.alpha_vantage = alpha_vantage_client or AlphaVantageClient()
        self.mongodb = mongodb_client or MongoDBClient()
        self.ingestion = data_ingestion or DataIngestion(self.alpha_vantage, self.mongodb)
        
    def get_stock_data(self, symbol: str, start_date: Optional[str] = None,
                      end_date: Optional[str] = None, interval: str = 'daily',
                      force_refresh: bool = False) -> pd.DataFrame:
        """
        Get stock price data, fetching from source if necessary.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('daily', 'weekly', 'monthly')
            force_refresh: Force refresh from source even if data exists
            
        Returns:
            DataFrame with stock price data
            
        Raises:
            DataAccessError: If data access fails
        """
        try:
            # Try to get data from database first
            if not force_refresh:
                try:
                    return self.mongodb.get_historical_data(symbol, start_date, end_date, interval)
                except DataNotFoundError:
                    logger.info(f"No data found in database for {symbol}, fetching from source")
                except Exception as e:
                    logger.warning(f"Error retrieving data from database: {e}, fetching from source")
            
            # Fetch and store data using ingestion service
            return self.ingestion.fetch_and_store_stock_data(
                symbol, start_date, end_date, interval, force_refresh
            )
            
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {e}")
            raise DataAccessError(f"Failed to get stock data: {e}", "data_manager")
    
    def get_company_info(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get company information, fetching from source if necessary.
        
        Args:
            symbol: Stock symbol
            force_refresh: Force refresh from source even if data exists
            
        Returns:
            Dictionary with company information
            
        Raises:
            DataAccessError: If data access fails
        """
        try:
            # Try to get data from database first
            if not force_refresh:
                try:
                    return self.mongodb.get_company_info(symbol)
                except DataNotFoundError:
                    logger.info(f"No company info found in database for {symbol}, fetching from source")
                except Exception as e:
                    logger.warning(f"Error retrieving company info from database: {e}, fetching from source")
            
            # Fetch and store data using ingestion service
            return self.ingestion.fetch_and_store_company_info(symbol, force_refresh)
            
        except Exception as e:
            logger.error(f"Error getting company info for {symbol}: {e}")
            raise DataAccessError(f"Failed to get company info: {e}", "data_manager")
    
    def get_stock_data_batch(self, symbols: List[str], start_date: Optional[str] = None,
                           end_date: Optional[str] = None, interval: str = 'daily',
                           force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Get stock price data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('daily', 'weekly', 'monthly')
            force_refresh: Force refresh from source even if data exists
            
        Returns:
            Dictionary mapping symbols to DataFrames
            
        Raises:
            DataAccessError: If data access fails
        """
        results = {}
        
        # Get data in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(
                    self.get_stock_data, 
                    symbol, start_date, end_date, interval, force_refresh
                ): symbol for symbol in symbols
            }
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"Error getting data for {symbol}: {e}")
                    # Store error message instead of raising
                    results[symbol] = str(e)
        
        return results
    
    def get_technical_indicators(self, symbol: str, indicators: List[str] = None,
                               interval: str = 'daily', time_period: int = 14,
                               force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Get technical indicators for a stock.
        
        Args:
            symbol: Stock symbol
            indicators: List of indicators to fetch (defaults to a standard set)
            interval: Data interval ('daily', 'weekly', 'monthly')
            time_period: Time period for indicators
            force_refresh: Force refresh from source even if data exists
            
        Returns:
            Dictionary mapping indicator names to DataFrames
            
        Raises:
            DataAccessError: If data access fails
        """
        try:
            return self.ingestion.fetch_and_store_technical_indicators(
                symbol, indicators, interval, time_period, force_refresh
            )
        except Exception as e:
            logger.error(f"Error getting technical indicators for {symbol}: {e}")
            raise DataAccessError(f"Failed to get technical indicators: {e}", "data_manager")
    
    def get_correlation_matrix(self, symbols: List[str], start_date: Optional[str] = None,
                             end_date: Optional[str] = None, column: str = 'close') -> pd.DataFrame:
        """
        Calculate correlation matrix for a list of symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            column: Price column to use ('open', 'high', 'low', 'close')
            
        Returns:
            DataFrame with correlation matrix
            
        Raises:
            DataAccessError: If data access fails
        """
        try:
            # Get data for all symbols
            data_dict = self.get_stock_data_batch(symbols, start_date, end_date)
            
            # Extract the specified column and create a combined DataFrame
            price_data = {}
            
            for symbol, df in data_dict.items():
                if isinstance(df, pd.DataFrame) and not df.empty and column in df.columns:
                    price_data[symbol] = df[column]
                    
            if not price_data:
                raise DataAccessError("No valid price data available for correlation", "data_manager")
                
            # Create a DataFrame with all price series
            combined_df = pd.DataFrame(price_data)
            
            # Calculate correlation matrix
            corr_matrix = combined_df.corr()
            
            return corr_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            raise DataAccessError(f"Failed to calculate correlation matrix: {e}", "data_manager")
    
    def get_returns(self, symbols: List[str], start_date: Optional[str] = None,
                   end_date: Optional[str] = None, period: str = 'daily') -> pd.DataFrame:
        """
        Calculate returns for a list of symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Return period ('daily', 'weekly', 'monthly', 'annual')
            
        Returns:
            DataFrame with returns for each symbol
            
        Raises:
            DataAccessError: If data access fails
        """
        try:
            # Get data for all symbols
            data_dict = self.get_stock_data_batch(symbols, start_date, end_date)
            
            # Calculate returns
            returns_dict = {}
            
            for symbol, df in data_dict.items():
                if isinstance(df, pd.DataFrame) and not df.empty and 'close' in df.columns:
                    # Calculate returns based on period
                    if period == 'daily':
                        returns = df['close'].pct_change()
                    elif period == 'weekly':
                        returns = df['close'].resample('W').last().pct_change()
                    elif period == 'monthly':
                        returns = df['close'].resample('M').last().pct_change()
                    elif period == 'annual':
                        returns = df['close'].resample('Y').last().pct_change()
                    else:
                        raise ValueError(f"Invalid period: {period}")
                        
                    returns_dict[symbol] = returns
                    
            if not returns_dict:
                raise DataAccessError("No valid price data available for returns calculation", "data_manager")
                
            # Create a DataFrame with all return series
            returns_df = pd.DataFrame(returns_dict)
            
            return returns_df
            
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            raise DataAccessError(f"Failed to calculate returns: {e}", "data_manager")
    
    def get_market_data(self, market_symbol: str = 'SPY', start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get market benchmark data.
        
        Args:
            market_symbol: Market benchmark symbol (default: SPY)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with market data
            
        Raises:
            DataAccessError: If data access fails
        """
        try:
            return self.get_stock_data(market_symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Error getting market data for {market_symbol}: {e}")
            raise DataAccessError(f"Failed to get market data: {e}", "data_manager")
    
    def get_economic_indicators(self, indicator: str, start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get economic indicator data.
        
        Args:
            indicator: Economic indicator code
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with economic indicator data
            
        Raises:
            DataAccessError: If data access fails
            NotImplementedError: This method needs to be implemented with actual API
        """
        # This would need to be implemented with a client for economic data
        # For now, we'll raise a NotImplementedError
        raise NotImplementedError("Economic indicators not yet implemented")
    
    def search_symbols(self, query: str) -> List[Dict[str, str]]:
        """
        Search for symbols matching a query.
        
        Args:
            query: Search query
            
        Returns:
            List of matching symbols with metadata
            
        Raises:
            DataAccessError: If search fails
        """
        try:
            return self.alpha_vantage.search_symbols(query)
        except Exception as e:
            logger.error(f"Error searching symbols for '{query}': {e}")
            raise DataAccessError(f"Failed to search symbols: {e}", "data_manager")
    
    def save_dataframe(self, name: str, df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a DataFrame to the database.
        
        Args:
            name: Name/identifier for the DataFrame
            df: DataFrame to save
            metadata: Additional metadata about the DataFrame
            
        Returns:
            ID of the saved DataFrame
            
        Raises:
            DataAccessError: If save fails
        """
        try:
            collection = self.mongodb.get_collection("dataframes")
            
            # Convert to records
            records = df.reset_index().to_dict('records')
            
            # Create document
            document = {
                'name': name,
                'data': records,
                'created_at': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            # Save to MongoDB
            result = collection.replace_one({'name': name}, document, upsert=True)
            
            logger.info(f"Saved DataFrame '{name}' to database")
            
            return str(result.upserted_id or result.matched_count)
            
        except Exception as e:
            logger.error(f"Error saving DataFrame '{name}': {e}")
            raise DataAccessError(f"Failed to save DataFrame: {e}", "data_manager")
    
    def load_dataframe(self, name: str) -> pd.DataFrame:
        """
        Load a DataFrame from the database.
        
        Args:
            name: Name/identifier of the DataFrame
            
        Returns:
            Loaded DataFrame
            
        Raises:
            DataAccessError: If load fails
            DataNotFoundError: If DataFrame not found
        """
        try:
            collection = self.mongodb.get_collection("dataframes")
            
            document = collection.find_one({'name': name})
            
            if not document:
                raise DataNotFoundError(f"DataFrame '{name}' not found", name, "data_manager")
                
            # Convert records to DataFrame
            df = pd.DataFrame(document['data'])
            
            # Set index if 'index' column exists
            if 'index' in df.columns:
                df = df.set_index('index')
                
            logger.info(f"Loaded DataFrame '{name}' from database")
            
            return df
            
        except DataNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error loading DataFrame '{name}': {e}")
            raise DataAccessError(f"Failed to load DataFrame: {e}", "data_manager") 


# Singleton instance of the DataManager
_data_manager_instance = None


def get_data_manager() -> DataManager:
    """
    Get the singleton instance of the DataManager.
    
    This function ensures that only one instance of the DataManager is created,
    which helps with resource management and consistency.
    
    Returns:
        DataManager: The singleton instance of the DataManager
    """
    global _data_manager_instance
    if _data_manager_instance is None:
        logger.info("Creating new DataManager instance")
        _data_manager_instance = DataManager()
    return _data_manager_instance