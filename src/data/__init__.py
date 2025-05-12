"""
Data module for STOCKER Pro.

This module contains data access, ingestion, and management functionality.
"""

from src.data.manager import DataManager, get_data_manager
from src.data.ingestion import ingest_stock_data, ingest_financial_data, batch_ingest
from src.data.clients.alpha_vantage import AlphaVantageClient
from src.data.clients.mongodb import MongoDBClient

__all__ = [
    # Classes
    'DataManager',
    'AlphaVantageClient',
    'MongoDBClient',
    
    # Factory functions
    'get_data_manager',
    
    # Ingestion functions
    'ingest_stock_data',
    'ingest_financial_data',
    'batch_ingest'
]
