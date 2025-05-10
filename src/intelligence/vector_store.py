"""
Vector store for STOCKER Pro.

This module provides vector database functionality for storing and retrieving
document embeddings for the RAG system.
"""

import os
import json
import shutil
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import logging

from src.core.config import Config
from src.core.exceptions import VectorDBError
from src.core.logging import logger

class VectorStore:
    """
    Vector store for embeddings and semantic search.
    
    Provides an interface to a vector database (ChromaDB) for storing and
    retrieving document embeddings with semantic search capabilities.
    """
    
    def __init__(self, config=None):
        """
        Initialize the vector store.
        
        Args:
            config: Optional configuration for the vector store
        """
        self.config = config or {}
        self.embedding_model = None  # Will be initialized on demand
        self.db = None  # Will be initialized on demand
        self.collection = None  # Will be initialized on demand
        
        self.persist_directory = self.config.get('persist_directory', './data/vector_db')
        self.collection_name = self.config.get('collection_name', 'stocker_documents')
        self.embedding_dimension = self.config.get('embedding_dimension', 1536)  # For OpenAI embeddings
    
    def initialize(self):
        """Initialize the vector store if not already initialized."""
        if self.db is None:
            self._initialize_chromadb()
        
        if self.embedding_model is None:
            self._initialize_embeddings()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB
            self.db = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.db.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
            except:
                self.collection = self.db.create_collection(
                    name=self.collection_name,
                    metadata={"description": "STOCKER document collection"}
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
            
        except ImportError:
            raise VectorDBError("ChromaDB is not installed. Please install it with 'pip install chromadb'.")
        except Exception as e:
            raise VectorDBError(f"Failed to initialize ChromaDB: {e}")
    
    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        try:
            embedding_provider = self.config.get('embedding_provider', 'openai').lower()
            
            if embedding_provider == 'openai':
                from src.intelligence.llm import get_embedding_model
                self.embedding_model = get_embedding_model(self.config)
            elif embedding_provider == 'huggingface':
                # Add support for HuggingFace models
                from sentence_transformers import SentenceTransformer
                model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
                self.embedding_model = SentenceTransformer(model_name)
            else:
                raise VectorDBError(f"Unsupported embedding provider: {embedding_provider}")
            
            logger.info(f"Initialized embedding model: {embedding_provider}")
            
        except ImportError as e:
            raise VectorDBError(f"Failed to import embedding dependencies: {e}")
        except Exception as e:
            raise VectorDBError(f"Failed to initialize embedding model: {e}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        self.initialize()
        
        try:
            # Prepare document batches (ChromaDB has limits on batch size)
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Extract document components
                ids = [doc.get('id', str(i + idx)) for idx, doc in enumerate(batch)]
                texts = [doc['text'] for doc in batch]
                metadatas = [doc.get('metadata', {}) for doc in batch]
                
                # Generate embeddings if not using ChromaDB's built-in embedding
                embeddings = None
                if self.embedding_model is not None:
                    embeddings = self._generate_embeddings(texts)
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise VectorDBError(f"Failed to add documents: {e}")
    
    def search(self, query: str, 
              filter_criteria: Optional[Dict[str, Any]] = None, 
              limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the vector store for documents matching the query.
        
        Args:
            query: Query string
            filter_criteria: Optional metadata filter criteria
            limit: Maximum number of results to return
            
        Returns:
            List of matching documents
        """
        self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = None
            if self.embedding_model is not None:
                query_embedding = self._generate_embeddings([query])[0]
            
            # Search collection
            results = self.collection.query(
                query_texts=[query] if not query_embedding else None,
                query_embeddings=[query_embedding] if query_embedding else None,
                where=filter_criteria,
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            documents = []
            for i in range(len(results['ids'][0])):
                documents.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise VectorDBError(f"Failed to search: {e}")
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            ids: List of document IDs to delete
        """
        self.initialize()
        
        try:
            # Delete in batches (ChromaDB has limits on batch size)
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch = ids[i:i + batch_size]
                self.collection.delete(ids=batch)
            
            logger.info(f"Deleted {len(ids)} documents from vector store")
            
        except Exception as e:
            logger.error(f"Error deleting documents from vector store: {e}")
            raise VectorDBError(f"Failed to delete documents: {e}")
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.initialize()
        
        try:
            self.db.delete_collection(name=self.collection_name)
            self.collection = None
            logger.info(f"Deleted collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise VectorDBError(f"Failed to delete collection: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        self.initialize()
        
        try:
            # Get all documents (not efficient for large collections)
            results = self.collection.get()
            
            return {
                'collection_name': self.collection_name,
                'document_count': len(results['ids']),
                'metadata_fields': self._get_metadata_fields(results['metadatas'])
            }
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            raise VectorDBError(f"Failed to get stats: {e}")
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if self.embedding_model is None:
            raise VectorDBError("Embedding model not initialized")
        
        try:
            embedding_provider = self.config.get('embedding_provider', 'openai').lower()
            
            if embedding_provider == 'openai':
                # OpenAI embeddings come as a list of embeddings
                return self.embedding_model(texts)
            elif embedding_provider == 'huggingface':
                # SentenceTransformer returns tensor, convert to list
                embeddings = self.embedding_model.encode(texts)
                return embeddings.tolist()
            else:
                raise VectorDBError(f"Unsupported embedding provider: {embedding_provider}")
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise VectorDBError(f"Failed to generate embeddings: {e}")
    
    def _get_metadata_fields(self, metadatas: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get count of metadata fields across all documents."""
        field_counts = {}
        for metadata in metadatas:
            for field in metadata.keys():
                field_counts[field] = field_counts.get(field, 0) + 1
        return field_counts
    
    def backup(self, backup_path: Optional[str] = None) -> str:
        """
        Backup the vector store.
        
        Args:
            backup_path: Optional path for backup
            
        Returns:
            Path to backup directory
        """
        self.initialize()
        
        # Use default path if none provided
        if backup_path is None:
            backup_path = f"./data/backups/vector_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Copy the entire persist directory
            shutil.copytree(self.persist_directory, backup_path)
            
            logger.info(f"Backed up vector store to {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error backing up vector store: {e}")
            raise VectorDBError(f"Failed to backup vector store: {e}")
    
    def restore(self, backup_path: str) -> None:
        """
        Restore the vector store from a backup.
        
        Args:
            backup_path: Path to backup directory
        """
        if not os.path.exists(backup_path):
            raise VectorDBError(f"Backup not found at {backup_path}")
        
        try:
            # Close current DB connection if open
            self.collection = None
            self.db = None
            
            # Remove current persist directory
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            
            # Copy backup to persist directory
            shutil.copytree(backup_path, self.persist_directory)
            
            # Reinitialize
            self._initialize_chromadb()
            
            logger.info(f"Restored vector store from {backup_path}")
            
        except Exception as e:
            logger.error(f"Error restoring vector store: {e}")
            raise VectorDBError(f"Failed to restore vector store: {e}")
    
    def reset(self) -> None:
        """Reset the vector store by deleting all data."""
        try:
            # Close current DB connection if open
            self.collection = None
            self.db = None
            
            # Remove persist directory
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            
            # Reinitialize
            self._initialize_chromadb()
            
            logger.info("Reset vector store")
            
        except Exception as e:
            logger.error(f"Error resetting vector store: {e}")
            raise VectorDBError(f"Failed to reset vector store: {e}") 