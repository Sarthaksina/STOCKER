"""
Database session management for STOCKER Pro.

This module provides utilities for connecting to databases and
managing database sessions throughout the application.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pymongo import MongoClient
from typing import Generator, Optional, Dict, Any
import logging
from contextlib import contextmanager

from src.core.config import config
from src.core.exceptions import DatabaseConnectionError
from src.core.logging import logger

# SQLAlchemy setup
SQLALCHEMY_DATABASE_URL = config.database.mongodb_connection_string.replace("mongodb://", "postgresql://")
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

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

def get_collection(collection_name: str):
    """
    Get MongoDB collection.
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        MongoDB collection
        
    Raises:
        DatabaseConnectionError: If MongoDB connection fails
    """
    db = get_mongodb_db()
    return db[collection_name]

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Get a database session as a context manager.
    
    Yields:
        SQLAlchemy session
        
    Raises:
        Exception: Any exception that occurs during session use
    """
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
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Initialize the database schema.
    
    Creates all tables defined in SQLAlchemy models.
    """
    logger.info("Creating database tables")
    Base.metadata.create_all(bind=engine)

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
        
    engine.dispose()
    logger.info("Disposed SQLAlchemy engine")

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