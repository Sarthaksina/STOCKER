"""
MongoDB client for STOCKER Pro.
This module provides a client for storing and retrieving data from MongoDB.
"""
import logging
from typing import Dict, Any, Optional, List, Union
import pandas as pd
from datetime import datetime
import pymongo
from pymongo import MongoClient, IndexModel, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

from src.core.config import config
from src.data.clients.base import BaseDataClient
from src.core.exceptions import DataAccessError, DataNotFoundError, DatabaseConnectionError
from src.core.logging import logger

class MongoDBClient(BaseDataClient):
    """
    Client for accessing MongoDB.
    
    Provides methods for CRUD operations on MongoDB collections with
    specialized methods for financial data and model artifacts.
    """
    
    def __init__(self, connection_string: Optional[str] = None, 
                 database_name: Optional[str] = None, cache_dir: str = "cache"):
        """
        Initialize the MongoDB client.
        
        Args:
            connection_string: MongoDB connection string (defaults to config if not provided)
            database_name: MongoDB database name (defaults to config if not provided)
            cache_dir: Directory for caching data
        """
        super().__init__("mongodb", cache_dir)
        
        self.connection_string = connection_string or config.database.mongodb_connection_string
        self.database_name = database_name or config.database.mongodb_database_name
        
        if not self.connection_string:
            raise ValueError("MongoDB connection string is required")
            
        if not self.database_name:
            raise ValueError("MongoDB database name is required")
            
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.database_name]
            logger.info(f"Connected to MongoDB: {self.database_name}")
        except PyMongoError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise DatabaseConnectionError(str(e))
            
    def __del__(self):
        """Close the MongoDB connection when the object is deleted."""
        if hasattr(self, 'client'):
            self.client.close()
            logger.debug("MongoDB connection closed")
    
    def get_collection(self, collection_name: str):
        """
        Get a MongoDB collection.
        
        Args:
            collection_name: Collection name
            
        Returns:
            pymongo.collection.Collection: MongoDB collection
        """
        return self.db[collection_name]
    
    def ensure_index(self, collection_name: str, keys: List[tuple], unique: bool = False):
        """
        Ensure an index exists on a collection.
        
        Args:
            collection_name: Collection name
            keys: List of (field, direction) tuples 
                e.g. [("symbol", pymongo.ASCENDING), ("date", pymongo.DESCENDING)]
            unique: Whether the index should be unique
        """
        try:
            collection = self.get_collection(collection_name)
            collection.create_index(keys, unique=unique)
            logger.debug(f"Created index on {collection_name}: {keys}")
        except PyMongoError as e:
            logger.error(f"Failed to create index on {collection_name}: {e}")
            raise DataAccessError(f"Failed to create index: {e}", "mongodb")
    
    def insert_one(self, collection_name: str, document: Dict[str, Any]) -> str:
        """
        Insert a single document into a collection.
        
        Args:
            collection_name: Collection name
            document: Document to insert
            
        Returns:
            Inserted document ID
            
        Raises:
            DataAccessError: If insertion fails
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.insert_one(document)
            logger.debug(f"Inserted document in {collection_name}: {result.inserted_id}")
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"Failed to insert document in {collection_name}: {e}")
            raise DataAccessError(f"Failed to insert document: {e}", "mongodb")
    
    def insert_many(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Insert multiple documents into a collection.
        
        Args:
            collection_name: Collection name
            documents: Documents to insert
            
        Returns:
            List of inserted document IDs
            
        Raises:
            DataAccessError: If insertion fails
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.insert_many(documents)
            logger.debug(f"Inserted {len(result.inserted_ids)} documents in {collection_name}")
            return [str(id) for id in result.inserted_ids]
        except PyMongoError as e:
            logger.error(f"Failed to insert documents in {collection_name}: {e}")
            raise DataAccessError(f"Failed to insert documents: {e}", "mongodb")
    
    def find_one(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find a single document in a collection.
        
        Args:
            collection_name: Collection name
            query: Query filter
            
        Returns:
            Document if found, None otherwise
            
        Raises:
            DataAccessError: If query fails
        """
        try:
            collection = self.get_collection(collection_name)
            document = collection.find_one(query)
            return document
        except PyMongoError as e:
            logger.error(f"Failed to find document in {collection_name}: {e}")
            raise DataAccessError(f"Failed to query collection: {e}", "mongodb")
    
    def find_many(self, collection_name: str, query: Dict[str, Any], 
                  projection: Optional[Dict[str, int]] = None, 
                  sort: Optional[List[tuple]] = None,
                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find multiple documents in a collection.
        
        Args:
            collection_name: Collection name
            query: Query filter
            projection: Fields to include/exclude
            sort: List of (field, direction) tuples for sorting
            limit: Maximum number of documents to return
            
        Returns:
            List of matching documents
            
        Raises:
            DataAccessError: If query fails
        """
        try:
            collection = self.get_collection(collection_name)
            cursor = collection.find(query, projection)
            
            if sort:
                cursor = cursor.sort(sort)
                
            if limit:
                cursor = cursor.limit(limit)
                
            documents = list(cursor)
            logger.debug(f"Found {len(documents)} documents in {collection_name}")
            return documents
        except PyMongoError as e:
            logger.error(f"Failed to find documents in {collection_name}: {e}")
            raise DataAccessError(f"Failed to query collection: {e}", "mongodb")
    
    def update_one(self, collection_name: str, query: Dict[str, Any], 
                   update: Dict[str, Any], upsert: bool = False) -> bool:
        """
        Update a single document in a collection.
        
        Args:
            collection_name: Collection name
            query: Query filter
            update: Update operations
            upsert: Whether to insert if document doesn't exist
            
        Returns:
            True if a document was modified, False otherwise
            
        Raises:
            DataAccessError: If update fails
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_one(query, update, upsert=upsert)
            logger.debug(f"Updated document in {collection_name}: {result.modified_count} modified")
            return result.modified_count > 0 or result.upserted_id is not None
        except PyMongoError as e:
            logger.error(f"Failed to update document in {collection_name}: {e}")
            raise DataAccessError(f"Failed to update document: {e}", "mongodb")
    
    def update_many(self, collection_name: str, query: Dict[str, Any], 
                    update: Dict[str, Any], upsert: bool = False) -> int:
        """
        Update multiple documents in a collection.
        
        Args:
            collection_name: Collection name
            query: Query filter
            update: Update operations
            upsert: Whether to insert if documents don't exist
            
        Returns:
            Number of documents modified
            
        Raises:
            DataAccessError: If update fails
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_many(query, update, upsert=upsert)
            logger.debug(f"Updated documents in {collection_name}: {result.modified_count} modified")
            return result.modified_count
        except PyMongoError as e:
            logger.error(f"Failed to update documents in {collection_name}: {e}")
            raise DataAccessError(f"Failed to update documents: {e}", "mongodb")
    
    def delete_one(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """
        Delete a single document from a collection.
        
        Args:
            collection_name: Collection name
            query: Query filter
            
        Returns:
            True if a document was deleted, False otherwise
            
        Raises:
            DataAccessError: If deletion fails
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_one(query)
            logger.debug(f"Deleted document from {collection_name}: {result.deleted_count} deleted")
            return result.deleted_count > 0
        except PyMongoError as e:
            logger.error(f"Failed to delete document from {collection_name}: {e}")
            raise DataAccessError(f"Failed to delete document: {e}", "mongodb")
    
    def delete_many(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        Delete multiple documents from a collection.
        
        Args:
            collection_name: Collection name
            query: Query filter
            
        Returns:
            Number of documents deleted
            
        Raises:
            DataAccessError: If deletion fails
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_many(query)
            logger.debug(f"Deleted documents from {collection_name}: {result.deleted_count} deleted")
            return result.deleted_count
        except PyMongoError as e:
            logger.error(f"Failed to delete documents from {collection_name}: {e}")
            raise DataAccessError(f"Failed to delete documents: {e}", "mongodb")
    
    def count_documents(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        Count documents in a collection.
        
        Args:
            collection_name: Collection name
            query: Query filter
            
        Returns:
            Document count
            
        Raises:
            DataAccessError: If count fails
        """
        try:
            collection = self.get_collection(collection_name)
            count = collection.count_documents(query)
            logger.debug(f"Counted {count} documents in {collection_name}")
            return count
        except PyMongoError as e:
            logger.error(f"Failed to count documents in {collection_name}: {e}")
            raise DataAccessError(f"Failed to count documents: {e}", "mongodb")
    
    # Specialized methods for stock data
    
    def store_stock_data(self, symbol: str, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store stock price data in MongoDB.
        
        Args:
            symbol: Stock symbol
            data: DataFrame of stock price data
            metadata: Additional metadata about the stock data
            
        Returns:
            True if successful
            
        Raises:
            DataAccessError: If storage fails
        """
        try:
            if data.empty:
                logger.warning(f"Empty DataFrame for {symbol}")
                return False
                
            # Ensure index
            self.ensure_index(config.database.stock_data_collection, 
                             [("symbol", ASCENDING), ("date", ASCENDING)], 
                             unique=True)
                
            # Convert to records
            data = data.reset_index() if 'date' not in data.columns else data
            data = data.rename(columns={'index': 'date'}) if 'index' in data.columns else data
            data['date'] = pd.to_datetime(data['date'])
            
            # Create documents
            documents = []
            for _, row in data.iterrows():
                doc = row.to_dict()
                doc['symbol'] = symbol
                doc['date'] = doc['date'].isoformat()
                
                if metadata:
                    doc['metadata'] = metadata
                    
                doc['updated_at'] = datetime.now().isoformat()
                documents.append(doc)
                
            # Use bulk operations for efficiency
            collection = self.get_collection(config.database.stock_data_collection)
            bulk_ops = []
            
            for doc in documents:
                bulk_ops.append(
                    pymongo.UpdateOne(
                        {'symbol': doc['symbol'], 'date': doc['date']},
                        {'$set': doc},
                        upsert=True
                    )
                )
                
            if bulk_ops:
                result = collection.bulk_write(bulk_ops)
                logger.info(f"Stored {len(documents)} price points for {symbol}")
                return True
            
            return False
            
        except PyMongoError as e:
            logger.error(f"Failed to store stock data for {symbol}: {e}")
            raise DataAccessError(f"Failed to store stock data: {e}", "mongodb")
    
    def get_historical_data(self, symbol: str, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None, interval: str = 'daily') -> pd.DataFrame:
        """
        Get historical price data for a symbol from MongoDB.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (currently only 'daily' is supported)
            
        Returns:
            DataFrame with historical price data
            
        Raises:
            DataAccessError: If retrieval fails
        """
        try:
            # Build query
            query = {'symbol': symbol}
            
            if start_date or end_date:
                query['date'] = {}
                
            if start_date:
                query['date']['$gte'] = start_date
                
            if end_date:
                query['date']['$lte'] = end_date
                
            # Get data from collection
            collection = self.get_collection(config.database.stock_data_collection)
            cursor = collection.find(query).sort('date', ASCENDING)
            
            # Convert to DataFrame
            data = list(cursor)
            
            if not data:
                logger.warning(f"No data found for {symbol}")
                raise DataNotFoundError(f"No data found", symbol, "mongodb")
                
            df = pd.DataFrame(data)
            
            # Clean up DataFrame
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
                
            if 'metadata' in df.columns:
                df = df.drop('metadata', axis=1)
                
            # Convert date column to datetime index
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            return df
            
        except PyMongoError as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise DataAccessError(f"Failed to retrieve stock data: {e}", "mongodb")
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information for a symbol from MongoDB.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company information
            
        Raises:
            DataAccessError: If retrieval fails
            DataNotFoundError: If company info not found
        """
        try:
            collection = self.get_collection(config.database.stock_data_collection)
            company_info = collection.find_one({'symbol': symbol, 'type': 'company_info'})
            
            if not company_info:
                raise DataNotFoundError(f"No company information found", symbol, "mongodb")
                
            # Remove MongoDB _id field
            if '_id' in company_info:
                del company_info['_id']
                
            return company_info
            
        except PyMongoError as e:
            logger.error(f"Failed to get company info for {symbol}: {e}")
            raise DataAccessError(f"Failed to retrieve company info: {e}", "mongodb")
    
    def store_model_artifact(self, model_id: str, artifact_type: str, 
                            artifact_data: Dict[str, Any]) -> str:
        """
        Store a model artifact in MongoDB.
        
        Args:
            model_id: Model identifier
            artifact_type: Type of artifact ('weights', 'metrics', 'config', etc.)
            artifact_data: Artifact data to store
            
        Returns:
            ID of the stored artifact
            
        Raises:
            DataAccessError: If storage fails
        """
        try:
            collection = self.get_collection(config.database.models_collection)
            
            document = {
                'model_id': model_id,
                'type': artifact_type,
                'data': artifact_data,
                'created_at': datetime.now().isoformat()
            }
            
            result = collection.insert_one(document)
            logger.info(f"Stored {artifact_type} artifact for model {model_id}")
            
            return str(result.inserted_id)
            
        except PyMongoError as e:
            logger.error(f"Failed to store model artifact: {e}")
            raise DataAccessError(f"Failed to store model artifact: {e}", "mongodb")
    
    def get_model_artifact(self, model_id: str, artifact_type: str) -> Dict[str, Any]:
        """
        Get a model artifact from MongoDB.
        
        Args:
            model_id: Model identifier
            artifact_type: Type of artifact
            
        Returns:
            Artifact data
            
        Raises:
            DataAccessError: If retrieval fails
            DataNotFoundError: If artifact not found
        """
        try:
            collection = self.get_collection(config.database.models_collection)
            
            document = collection.find_one({
                'model_id': model_id,
                'type': artifact_type
            })
            
            if not document:
                raise DataNotFoundError(f"No {artifact_type} artifact found for model {model_id}", 
                                       model_id, "mongodb")
                
            # Remove MongoDB _id field
            if '_id' in document:
                del document['_id']
                
            return document.get('data', {})
            
        except PyMongoError as e:
            logger.error(f"Failed to get model artifact: {e}")
            raise DataAccessError(f"Failed to retrieve model artifact: {e}", "mongodb")