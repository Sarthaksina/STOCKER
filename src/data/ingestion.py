"""
Data ingestion module for STOCKER pro.

This module handles data ingestion from various sources and stores it in a consistent format.
"""
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import os
import json
import time
import concurrent.futures

from src.core.config import config
from src.data.clients.alpha_vantage import AlphaVantageClient
from src.data.clients.mongodb import MongoDBClient
from src.core.exceptions import DataIngestionError, DataNotFoundError, DataSourceRateLimitError
from src.core.logging import logger

class DataIngestion:
    """
    Data ingestion class for handling data acquisition and storage.
    
    This class provides methods for ingesting data from various sources,
    transforming it to a consistent format, and storing it in the database.
    """
    
    def __init__(self, alpha_vantage_client: Optional[AlphaVantageClient] = None,
                 mongodb_client: Optional[MongoDBClient] = None):
        """
        Initialize the data ingestion service.
        
        Args:
            alpha_vantage_client: Alpha Vantage client (created if not provided)
            mongodb_client: MongoDB client (created if not provided)
        """
        self.alpha_vantage = alpha_vantage_client or AlphaVantageClient()
        self.mongodb = mongodb_client or MongoDBClient()
        
    def fetch_and_store_stock_data(self, symbol: str, start_date: Optional[str] = None,
                                   end_date: Optional[str] = None, interval: str = 'daily',
                                   force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch stock data and store it in the database.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('daily', 'weekly', 'monthly')
            force_refresh: Force refresh from source even if data exists
            
        Returns:
            DataFrame with the fetched data
            
        Raises:
            DataIngestionError: If data ingestion fails
        """
        try:
            # Set default dates if not provided
            if not start_date:
                start_date = config.data.default_start_date
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            logger.info(f"Fetching {interval} data for {symbol} from {start_date} to {end_date}")
            
            # Check if data already exists in the database
            if not force_refresh:
                try:
                    df = self.mongodb.get_historical_data(symbol, start_date, end_date, interval)
                    logger.info(f"Found existing data for {symbol} in database")
                    return df
                except DataNotFoundError:
                    logger.info(f"No existing data for {symbol} in database, fetching from source")
                except Exception as e:
                    logger.warning(f"Error checking existing data: {e}, fetching from source")
            
            # Fetch data from Alpha Vantage
            df = self.alpha_vantage.get_historical_data(symbol, start_date, end_date, interval)
            
            if df.empty:
                raise DataNotFoundError(f"No data returned for {symbol}", symbol)
            
            # Store data in MongoDB
            self.mongodb.store_stock_data(symbol, df, {
                'source': 'alpha_vantage',
                'interval': interval,
                'ingestion_date': datetime.now().isoformat()
            })
            
            return df
            
        except DataSourceRateLimitError as e:
            logger.warning(f"Rate limit hit fetching {symbol}: {e}")
            raise DataIngestionError(f"Rate limit reached: {e}")
        except Exception as e:
            logger.error(f"Error ingesting data for {symbol}: {e}")
            raise DataIngestionError(f"Failed to ingest data for {symbol}: {e}")
    
    def fetch_and_store_batch(self, symbols: List[str], start_date: Optional[str] = None,
                             end_date: Optional[str] = None, interval: str = 'daily',
                             max_workers: int = 5) -> Dict[str, Union[pd.DataFrame, str]]:
        """
        Fetch and store data for multiple symbols in parallel.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('daily', 'weekly', 'monthly')
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping symbols to DataFrames or error messages
        """
        results = {}
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_symbol = {
                executor.submit(
                    self.fetch_and_store_stock_data, 
                    symbol, start_date, end_date, interval
                ): symbol for symbol in symbols
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    results[symbol] = str(e)
                    
                # Small delay to avoid rate limiting
                time.sleep(0.1)
        
        return results
    
    def fetch_and_store_company_info(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetch company information and store it in the database.
        
        Args:
            symbol: Stock symbol
            force_refresh: Force refresh from source even if data exists
            
        Returns:
            Dictionary with company information
            
        Raises:
            DataIngestionError: If data ingestion fails
        """
        try:
            logger.info(f"Fetching company information for {symbol}")
            
            # Check if data already exists in the database
            if not force_refresh:
                try:
                    company_info = self.mongodb.get_company_info(symbol)
                    logger.info(f"Found existing company info for {symbol} in database")
                    return company_info
                except DataNotFoundError:
                    logger.info(f"No existing company info for {symbol} in database, fetching from source")
                except Exception as e:
                    logger.warning(f"Error checking existing company info: {e}, fetching from source")
            
            # Fetch data from Alpha Vantage
            company_info = self.alpha_vantage.get_company_info(symbol)
            
            # Add metadata
            company_info['source'] = 'alpha_vantage'
            company_info['ingestion_date'] = datetime.now().isoformat()
            company_info['type'] = 'company_info'
            
            # Store in MongoDB
            collection = self.mongodb.get_collection(config.database.stock_data_collection)
            collection.replace_one(
                {'symbol': symbol, 'type': 'company_info'}, 
                company_info, 
                upsert=True
            )
            
            return company_info
            
        except Exception as e:
            logger.error(f"Error ingesting company info for {symbol}: {e}")
            raise DataIngestionError(f"Failed to ingest company info for {symbol}: {e}")
    
    def fetch_and_store_technical_indicators(self, symbol: str, indicators: List[str] = None,
                                           interval: str = 'daily', time_period: int = 14,
                                           force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Fetch technical indicators and store them in the database.
        
        Args:
            symbol: Stock symbol
            indicators: List of indicators to fetch (defaults to a standard set)
            interval: Data interval ('daily', 'weekly', 'monthly')
            time_period: Time period for indicators
            force_refresh: Force refresh from source even if data exists
            
        Returns:
            Dictionary mapping indicator names to DataFrames
            
        Raises:
            DataIngestionError: If data ingestion fails
        """
        # Default indicators if not specified
        if indicators is None:
            indicators = ['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS']
            
        results = {}
        
        try:
            logger.info(f"Fetching technical indicators for {symbol}")
            
            for indicator in indicators:
                try:
                    # Fetch indicator from Alpha Vantage
                    df = self.alpha_vantage.get_technical_indicator(
                        symbol, indicator, interval, time_period
                    )
                    
                    if not df.empty:
                        # Add metadata columns
                        df['symbol'] = symbol
                        df['indicator'] = indicator
                        
                        # Store in MongoDB (in a technical indicators collection)
                        collection = self.mongodb.get_collection(f"technical_indicators")
                        
                        # Convert to records and store
                        records = df.reset_index().to_dict('records')
                        
                        # Use bulk operations for efficiency
                        for record in records:
                            # Convert date to string if needed
                            if isinstance(record.get('date'), pd.Timestamp):
                                record['date'] = record['date'].isoformat()
                                
                            record['updated_at'] = datetime.now().isoformat()
                            
                            # Upsert the record
                            collection.replace_one(
                                {
                                    'symbol': symbol,
                                    'indicator': indicator,
                                    'date': record['date']
                                },
                                record,
                                upsert=True
                            )
                        
                        results[indicator] = df
                    
                except Exception as e:
                    logger.error(f"Error fetching {indicator} for {symbol}: {e}")
            
            if not results:
                raise DataIngestionError(f"Failed to fetch any indicators for {symbol}")
                
            return results
            
        except Exception as e:
            logger.error(f"Error ingesting technical indicators for {symbol}: {e}")
            raise DataIngestionError(f"Failed to ingest technical indicators for {symbol}: {e}")
    
    def ingest_all_data(self, symbols: List[str], start_date: Optional[str] = None,
                        end_date: Optional[str] = None, include_company_info: bool = True,
                        include_indicators: bool = True) -> Dict[str, Any]:
        """
        Ingest all data for a list of symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_company_info: Whether to include company info
            include_indicators: Whether to include technical indicators
            
        Returns:
            Dictionary with results
        """
        results = {
            'price_data': {},
            'company_info': {},
            'indicators': {}
        }
        
        # Fetch price data
        logger.info(f"Ingesting price data for {len(symbols)} symbols")
        results['price_data'] = self.fetch_and_store_batch(symbols, start_date, end_date)
        
        # Fetch company info
        if include_company_info:
            logger.info(f"Ingesting company info for {len(symbols)} symbols")
            for symbol in symbols:
                try:
                    results['company_info'][symbol] = self.fetch_and_store_company_info(symbol)
                except Exception as e:
                    logger.error(f"Error ingesting company info for {symbol}: {e}")
                    results['company_info'][symbol] = str(e)
        
        # Fetch technical indicators
        if include_indicators:
            logger.info(f"Ingesting technical indicators for {len(symbols)} symbols")
            for symbol in symbols:
                try:
                    results['indicators'][symbol] = self.fetch_and_store_technical_indicators(symbol)
                except Exception as e:
                    logger.error(f"Error ingesting indicators for {symbol}: {e}")
                    results['indicators'][symbol] = str(e)
        
        return results 


# Standalone functions for backward compatibility

def ingest_stock_data(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, 
                    interval: str = 'daily', force_refresh: bool = False) -> pd.DataFrame:
    """
    Standalone function to ingest stock data for a symbol.
    
    This is a wrapper around DataIngestion.fetch_and_store_stock_data for backward compatibility.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval ('daily', 'weekly', 'monthly')
        force_refresh: Force refresh from source even if data exists
        
    Returns:
        DataFrame with the fetched data
    """
    from src.data.manager import get_data_manager
    data_manager = get_data_manager()
    return data_manager.ingestion.fetch_and_store_stock_data(
        symbol, start_date, end_date, interval, force_refresh
    )


def ingest_financial_data(symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Standalone function to ingest financial data for a symbol.
    
    This is a wrapper around DataIngestion.fetch_and_store_company_info for backward compatibility.
    
    Args:
        symbol: Stock symbol
        force_refresh: Force refresh from source even if data exists
        
    Returns:
        Dictionary with company information
    """
    from src.data.manager import get_data_manager
    data_manager = get_data_manager()
    return data_manager.ingestion.fetch_and_store_company_info(symbol, force_refresh)


def batch_ingest(symbols: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None,
                include_company_info: bool = True, include_indicators: bool = True) -> Dict[str, Any]:
    """
    Standalone function to ingest all data for a list of symbols.
    
    This is a wrapper around DataIngestion.ingest_all_data for backward compatibility.
    
    Args:
        symbols: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        include_company_info: Whether to include company info
        include_indicators: Whether to include technical indicators
        
    Returns:
        Dictionary with results
    """
    from src.data.manager import get_data_manager
    data_manager = get_data_manager()
    return data_manager.ingestion.ingest_all_data(
        symbols, start_date, end_date, include_company_info, include_indicators
    )