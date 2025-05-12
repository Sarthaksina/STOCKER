"""
Database session management for STOCKER Pro.

This module provides utilities for connecting to databases and
managing database sessions throughout the application.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pymongo import MongoClient, errors
from typing import Generator, Optional, Dict, Any, List, Union
import logging
import sys
from contextlib import contextmanager

from src.core.config import config
from src.core.exceptions import DatabaseConnectionError
from src.core.logging import logger

# SQLAlchemy setup - commented out as we're using MongoDB primarily
# For compatibility with existing code, we'll keep the Base and SessionLocal objects
Base = declarative_base()

# Create a dummy engine and session for compatibility
engine = None
SessionLocal = sessionmaker()

# We'll initialize these properly if SQL database is needed
try:
    # Only create SQLAlchemy engine if SQL database is configured
    if hasattr(config.database, 'sql_connection_string') and config.database.sql_connection_string:
        engine = create_engine(config.database.sql_connection_string)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("SQLAlchemy engine created")
except Exception as e:
    logger.warning(f"Failed to create SQLAlchemy engine: {e}. Using MongoDB only.")


# MongoDB setup
mongodb_client = None

def get_mongodb_client() -> MongoClient:
    """
    Get MongoDB client, creating it if necessary.
    
    Returns:
        MongoDB client
        
    Raises:
        DatabaseConnectionError: If MongoDB connection fails
    """
    global mongodb_client
    
    try:
        if mongodb_client is None:
            mongodb_client = MongoClient(config.database.mongodb_connection_string)
            # Test the connection
            mongodb_client.admin.command('ping')
            logger.info("Connected to MongoDB")
        
        return mongodb_client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise DatabaseConnectionError(str(e))

def get_mongodb_db():
    """
    Get MongoDB database.
    
    Returns:
        MongoDB database
        
    Raises:
        DatabaseConnectionError: If MongoDB connection fails
    """
    client = get_mongodb_client()
    return client[config.database.mongodb_database_name]

def get_collection(collection_name: str, db_name: Optional[str] = None):
    """
    Get MongoDB collection.
    
    Args:
        collection_name: Name of the collection
        db_name: Optional database name (uses default if None)
        
    Returns:
        MongoDB collection
        
    Raises:
        DatabaseConnectionError: If MongoDB connection fails
    """
    client = get_mongodb_client()
    db = client[db_name or config.database.mongodb_database_name]
    return db[collection_name]

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Get a database session as a context manager.
    
    Yields:
        SQLAlchemy session
        
    Raises:
        Exception: Any exception that occurs during session use
        DatabaseConnectionError: If SQL database is not configured
    """
    if engine is None:
        raise DatabaseConnectionError("SQL database is not configured")
        
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db() -> Generator[Session, None, None]:
    """
    Get a database session as a dependency.
    
    Yields:
        SQLAlchemy session
        
    Raises:
        Exception: Any exception that occurs during session use
        DatabaseConnectionError: If SQL database is not configured
    """
    if engine is None:
        raise DatabaseConnectionError("SQL database is not configured")
        
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_session() -> Session:
    """
    Get a database session.
    
    Returns:
        SQLAlchemy session
        
    Raises:
        DatabaseConnectionError: If SQL database is not configured
    """
    if engine is None:
        raise DatabaseConnectionError("SQL database is not configured")
        
    return SessionLocal()

def init_db():
    """
    Initialize the database schema.
    
    Creates all tables defined in SQLAlchemy models.
    """
    if engine is not None:
        logger.info("Creating database tables")
        Base.metadata.create_all(bind=engine)
    else:
        logger.info("Skipping SQL table creation as no engine is configured")

def close_db_connections():
    """
    Close all database connections.
    
    Should be called when the application is shutting down.
    """
    global mongodb_client
    
    if mongodb_client is not None:
        mongodb_client.close()
        mongodb_client = None
        logger.info("Closed MongoDB connections")
        
    if engine is not None:
        engine.dispose()
        logger.info("Disposed SQLAlchemy engine")

def test_connection():
    """
    Test MongoDB connection by inserting, fetching, and deleting a test document.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        client = get_mongodb_client()
        db = client[config.database.mongodb_database_name]
        collection = db['test_collection']
        test_doc = {'message': 'MongoDB connection successful!'}
        result = collection.insert_one(test_doc)
        logger.info(f"Inserted document with _id: {result.inserted_id}")
        doc = collection.find_one({'_id': result.inserted_id})
        logger.info(f'Fetched document: {doc}')
        collection.delete_one({'_id': result.inserted_id})
        logger.info('Test document deleted.')
        return True
    except Exception as e:
        logger.error(f'Database connection test failed: {e}')
        return False

def create_indexes():
    """
    Create MongoDB indexes.
    
    Creates indexes for commonly queried fields to improve performance.
    """
    try:
        db = get_mongodb_db()
        
        # Stock data collection
        db[config.database.stock_data_collection].create_index([("symbol", 1), ("date", 1)], unique=True)
        
        # Company info collection
        db[config.database.stock_data_collection].create_index([("symbol", 1), ("type", 1)], unique=True)
        
        # Models collection
        db[config.database.models_collection].create_index("model_id")
        
        # Portfolio collection
        db[config.database.portfolio_collection].create_index([("user_id", 1), ("name", 1)])
        
        # User collection
        db[config.database.user_collection].create_index("username", unique=True)
        db[config.database.user_collection].create_index("email", unique=True)
        
        # News collection
        db[config.database.news_collection].create_index([("symbol", 1), ("published_at", -1)])
        
        logger.info("Created MongoDB indexes")
    except Exception as e:
        logger.error(f"Failed to create MongoDB indexes: {e}")
        raise DatabaseConnectionError(f"Failed to create indexes: {e}")

def setup_database():
    """
    Set up database connections and schema.
    
    Should be called when the application starts.
    """
    # Connect to MongoDB
    get_mongodb_client()
    
    # Create indexes
    create_indexes()
    
    # Initialize SQLAlchemy tables
    init_db()
    
    logger.info("Database setup complete")

# CRUD operations for various collections

def fetch_stock_data(symbol: str, db_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch stock data for a given symbol.
    
    Args:
        symbol: Stock symbol
        db_name: Optional database name
        
    Returns:
        List of stock data documents
    """
    collection = get_collection('stocks', db_name)
    return list(collection.find({'symbol': symbol}))

def save_analysis(symbol: str, analysis_dict: Dict[str, Any], db_name: Optional[str] = None) -> None:
    """
    Save analysis data for a given symbol.
    
    Args:
        symbol: Stock symbol
        analysis_dict: Analysis data to save
        db_name: Optional database name
    """
    collection = get_collection('analysis', db_name)
    collection.update_one({'symbol': symbol}, {'$set': analysis_dict}, upsert=True)

def fetch_holdings(symbol: str, db_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch holdings data for a given symbol.
    
    Args:
        symbol: Stock symbol
        db_name: Optional database name
        
    Returns:
        List of holdings documents
    """
    collection = get_collection('holdings', db_name)
    return list(collection.find({'symbol': symbol}))

def save_holdings(symbol: str, holdings_dict: Dict[str, Any], db_name: Optional[str] = None) -> None:
    """
    Save holdings data for a given symbol.
    
    Args:
        symbol: Stock symbol
        holdings_dict: Holdings data to save
        db_name: Optional database name
    """
    collection = get_collection('holdings', db_name)
    collection.update_one({'symbol': symbol}, {'$set': holdings_dict}, upsert=True)

def fetch_news(symbol: str, db_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch news articles for a given symbol.
    
    Args:
        symbol: Stock symbol
        db_name: Optional database name
        
    Returns:
        List of news article documents
    """
    collection = get_collection('news', db_name)
    return list(collection.find({'symbol': symbol}, {'_id': 0}))

def save_news(symbol: str, news_list: List[Dict[str, Any]], db_name: Optional[str] = None) -> None:
    """
    Save news articles for a given symbol.
    
    Args:
        symbol: Stock symbol
        news_list: List of news articles to save
        db_name: Optional database name
    """
    collection = get_collection('news', db_name)
    for article in news_list:
        doc = {**article, 'symbol': symbol}
        collection.update_one({'symbol': symbol, 'url': doc.get('url')}, {'$set': doc}, upsert=True)

def fetch_events(symbol: str, db_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch corporate events for a given symbol.
    
    Args:
        symbol: Stock symbol
        db_name: Optional database name
        
    Returns:
        List of event documents
    """
    collection = get_collection('events', db_name)
    return list(collection.find({'symbol': symbol}, {'_id': 0}))

def save_events(symbol: str, events_list: List[Dict[str, Any]], db_name: Optional[str] = None) -> None:
    """
    Save corporate events for a given symbol.
    
    Args:
        symbol: Stock symbol
        events_list: List of events to save
        db_name: Optional database name
    """
    collection = get_collection('events', db_name)
    for event in events_list:
        doc = {**event, 'symbol': symbol}
        key = {'symbol': symbol, 'event_id': doc.get('event_id')} if doc.get('event_id') else {'symbol': symbol, 'timestamp': doc.get('timestamp')}
        collection.update_one(key, {'$set': doc}, upsert=True)

def fetch_portfolio(user_id: str, db_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch portfolio for a given user.
    
    Args:
        user_id: User ID
        db_name: Optional database name
        
    Returns:
        Portfolio document or empty dict if not found
    """
    collection = get_collection('portfolios', db_name)
    return collection.find_one({'user_id': user_id}, {'_id': 0}) or {}

def save_portfolio(user_id: str, portfolio_dict: Dict[str, Any], db_name: Optional[str] = None) -> None:
    """
    Save portfolio for a given user.
    
    Args:
        user_id: User ID
        portfolio_dict: Portfolio data to save
        db_name: Optional database name
    """
    collection = get_collection('portfolios', db_name)
    collection.update_one({'user_id': user_id}, {'$set': portfolio_dict}, upsert=True)

def fetch_price_history(symbol: str, db_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch price history for a given symbol.
    
    Args:
        symbol: Stock symbol
        db_name: Optional database name
        
    Returns:
        List of price history documents sorted by date
    """
    collection = get_collection('price_history', db_name)
    return list(collection.find({'symbol': symbol}, {'_id': 0}).sort('date', 1))

def save_price_history(symbol: str, history: List[Dict[str, Any]], db_name: Optional[str] = None) -> None:
    """
    Save price history for a given symbol.
    
    Args:
        symbol: Stock symbol
        history: List of price history records to save
        db_name: Optional database name
    """
    collection = get_collection('price_history', db_name)
    for record in history:
        doc = {**record, 'symbol': symbol}
        key = {'symbol': symbol, 'date': doc.get('date')}
        collection.update_one(key, {'$set': doc}, upsert=True)