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
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
import logging
import faiss

from src.core.config import Config
from src.core.exceptions import VectorDBError
from src.core.logging import logger

class FaissVectorStore:
    """
    Simple FAISS-based vector store using L2 distance.
    Stores vectors and their corresponding IDs.
    """
    def __init__(self, config=None):
        """
        Initialize the FAISS vector store.
        
        Args:
            config: Optional configuration for the vector store
        """
        self.config = config or {}
        self.dim = self.config.get('embedding_dimension', 1536)  # Default to OpenAI embedding dimension
        
        # Get paths for FAISS index and ID mapping file
        data_dir = self.config.get('data_dir', './data/vector_db')
        os.makedirs(data_dir, exist_ok=True)
        self.index_path = os.path.join(data_dir, 'faiss.index')
        self.ids_path = os.path.join(data_dir, 'faiss_ids.json')
        
        # Load existing index and IDs if available
        if os.path.exists(self.index_path) and os.path.exists(self.ids_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.ids_path, 'r', encoding='utf-8') as f:
                self.ids = json.load(f)
            logger.info(f"Loaded FAISS index ({self.index.ntotal} vectors).")
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.ids = []
            logger.info(f"Created new FAISS index (dim={self.dim}).")
        
        self.embedding_model = None  # Will be initialized on demand
    
    def initialize(self):
        """Initialize the embedding model if not already initialized."""
        if self.embedding_model is None:
            self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        try:
            from src.intelligence.llm import get_embedding_model
            self.embedding_model = get_embedding_model(self.config)
            logger.info(f"Initialized embedding model for FAISS vector store")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise VectorDBError(f"Failed to initialize embedding model: {e}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        self.initialize()
        
        if not documents:
            logger.warning("No documents to add")
            return
        
        try:
            # Extract text and IDs
            texts = [doc.get('text', '') for doc in documents]
            ids = [doc.get('id', str(i)) for i, doc in enumerate(documents)]
            
            # Generate embeddings
            embeddings = self._generate_embeddings(texts)
            
            # Add to FAISS index
            vectors = np.array(embeddings).astype('float32')
            self.index.add(vectors)
            self.ids.extend(ids)
            
            # Save to disk
            self._save()
            
            logger.info(f"Added {len(documents)} documents to FAISS vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to FAISS vector store: {e}")
            raise VectorDBError(f"Failed to add documents to FAISS vector store: {e}")
    
    def search(self, query: str, 
              filter_criteria: Optional[Dict[str, Any]] = None, 
              limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the vector store for documents matching the query.
        
        Args:
            query: Query string
            filter_criteria: Optional metadata filter criteria (not used in FAISS)
            limit: Maximum number of results to return
            
        Returns:
            List of matching documents with IDs and distances
        """
        self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])[0]
            query_vector = np.array(query_embedding).reshape(1, -1).astype('float32')
            
            # Search FAISS index
            D, I = self.index.search(query_vector, limit)
            
            results = []
            for idx, dist in zip(I[0], D[0]):
                if idx < len(self.ids):
                    results.append({
                        'id': self.ids[idx],
                        'distance': float(dist),
                        'metadata': {}
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching FAISS vector store: {e}")
            raise VectorDBError(f"Failed to search FAISS vector store: {e}")
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            ids: List of document IDs to delete
        
        Note: FAISS doesn't support direct deletion. This implementation rebuilds the index.
        """
        if not ids:
            return
        
        try:
            # FAISS doesn't support direct deletion, so we need to rebuild the index
            # First, identify indices to keep
            keep_indices = [i for i, doc_id in enumerate(self.ids) if doc_id not in ids]
            
            if len(keep_indices) == len(self.ids):  # Nothing to delete
                return
            
            # Create a new index
            new_index = faiss.IndexFlatL2(self.dim)
            
            # Extract vectors to keep
            if len(keep_indices) > 0:
                vectors_to_keep = np.vstack([self.index.reconstruct(i) for i in keep_indices])
                new_index.add(vectors_to_keep)
            
            # Update IDs list
            self.ids = [self.ids[i] for i in keep_indices]
            
            # Replace old index
            self.index = new_index
            
            # Save to disk
            self._save()
            
            logger.info(f"Deleted {len(ids)} documents from FAISS vector store")
            
        except Exception as e:
            logger.error(f"Error deleting documents from FAISS vector store: {e}")
            raise VectorDBError(f"Failed to delete documents from FAISS vector store: {e}")
    
    def reset(self) -> None:
        """Reset the vector store by deleting all data."""
        try:
            # Create new empty index
            self.index = faiss.IndexFlatL2(self.dim)
            self.ids = []
            
            # Save to disk
            self._save()
            
            logger.info("Reset FAISS vector store")
            
        except Exception as e:
            logger.error(f"Error resetting FAISS vector store: {e}")
            raise VectorDBError(f"Failed to reset FAISS vector store: {e}")
    
    def _save(self) -> None:
        """Persist index and ID list to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.ids_path, 'w', encoding='utf-8') as f:
            json.dump(self.ids, f)
        logger.info(f"Saved FAISS index ({self.index.ntotal} vectors). IDs saved: {len(self.ids)}.")
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        self.initialize()
        
        try:
            return self.embedding_model(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise VectorDBError(f"Failed to generate embeddings: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        return {
            "type": "FAISS",
            "document_count": self.index.ntotal,
            "dimension": self.dim
        }


class VectorStore:
    """
    Vector store for embeddings and semantic search.
    
    Provides an interface to a vector database (ChromaDB or FAISS) for storing and
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
        self.faiss_store = None  # FAISS store if using FAISS
        
        self.backend = self.config.get('vector_db_backend', 'chromadb').lower()
        self.persist_directory = self.config.get('persist_directory', './data/vector_db')
        self.collection_name = self.config.get('collection_name', 'stocker_documents')
        self.embedding_dimension = self.config.get('embedding_dimension', 1536)  # For OpenAI embeddings
    
    def initialize(self):
        """Initialize the vector store if not already initialized."""
        if self.backend == 'chromadb':
            if self.db is None:
                self._initialize_chromadb()
        elif self.backend == 'faiss':
            if self.faiss_store is None:
                self._initialize_faiss()
        else:
            raise VectorDBError(f"Unsupported vector database backend: {self.backend}")
        
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
            
    def _initialize_faiss(self):
        """Initialize FAISS vector store."""
        try:
            self.faiss_store = FaissVectorStore(self.config)
            logger.info("Initialized FAISS vector store")
        except Exception as e:
            raise VectorDBError(f"Failed to initialize FAISS vector store: {e}")
    
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
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        self.initialize()
        
        if not documents:
            logger.warning("No documents to add")
            return
        
        try:
            if self.backend == 'chromadb':
                # Extract document data
                ids = [doc.get('id', str(i)) for i, doc in enumerate(documents)]
                texts = [doc.get('text', '') for doc in documents]
                metadatas = [doc.get('metadata', {}) for doc in documents]
                
                # Log metadata fields for debugging
                metadata_fields = self._get_metadata_fields(metadatas)
                logger.debug(f"Metadata fields: {metadata_fields}")
                
                # Generate embeddings
                embeddings = self._generate_embeddings(texts)
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
                
                logger.info(f"Added {len(documents)} documents to ChromaDB vector store")
            elif self.backend == 'faiss':
                self.faiss_store.add_documents(documents)
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise VectorDBError(f"Failed to add documents to vector store: {e}")
    
    def search(self, query: str, 
              filter_criteria: Optional[Dict[str, Any]] = None, 
              limit: int = 5):
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
            if self.backend == 'chromadb':
                # Generate query embedding
                query_embedding = self._generate_embeddings([query])[0]
                
                # Perform search
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                    where=filter_criteria
                )
                
                # Process results
                documents = []
                for i in range(len(results['ids'][0])):
                    doc_id = results['ids'][0][i]
                    document = results['documents'][0][i]
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else None
                    
                    documents.append({
                        'id': doc_id,
                        'text': document,
                        'metadata': metadata,
                        'distance': distance
                    })
                
                return documents
            elif self.backend == 'faiss':
                return self.faiss_store.search(query, filter_criteria, limit)
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise VectorDBError(f"Failed to search vector store: {e}")
    
    def delete_documents(self, ids: List[str]):
        """
        Delete documents from the vector store.
        
        Args:
            ids: List of document IDs to delete
        """
        self.initialize()
        
        if not ids:
            return
        
        try:
            if self.backend == 'chromadb':
                self.collection.delete(
                    ids=ids
                )
                logger.info(f"Deleted {len(ids)} documents from ChromaDB vector store")
            elif self.backend == 'faiss':
                self.faiss_store.delete_documents(ids)
            
        except Exception as e:
            logger.error(f"Error deleting documents from vector store: {e}")
            raise VectorDBError(f"Failed to delete documents from vector store: {e}")
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.initialize()
        
        try:
            if self.backend == 'chromadb':
                self.db.delete_collection(name=self.collection_name)
                self.collection = None
                logger.info(f"Deleted ChromaDB collection: {self.collection_name}")
            elif self.backend == 'faiss':
                self.faiss_store.reset()
                logger.info("Reset FAISS vector store")
            
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise VectorDBError(f"Failed to delete collection: {e}")
    
    def get_stats(self):
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        self.initialize()
        
        try:
            if self.backend == 'chromadb':
                # Get collection count
                count = self.collection.count()
                
                # Get collection info
                collection_info = {
                    "type": "ChromaDB",
                    "name": self.collection_name,
                    "document_count": count,
                    "embedding_dimension": self.embedding_dimension
                }
                
                return collection_info
            elif self.backend == 'faiss':
                return self.faiss_store.get_stats()
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            raise VectorDBError(f"Failed to get vector store stats: {e}")
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        self.initialize()
        
        try:
            embedding_provider = self.config.get('embedding_provider', 'openai').lower()
            
            if embedding_provider == 'openai':
                from src.intelligence.llm import get_embedding_model
                return get_embedding_model(self.config)(texts)
            elif embedding_provider == 'huggingface':
                # Add support for HuggingFace models
                from sentence_transformers import SentenceTransformer
                model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
                model = SentenceTransformer(model_name)
                return model.encode(texts).tolist()
            else:
                raise VectorDBError(f"Unsupported embedding provider: {embedding_provider}")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise VectorDBError(f"Failed to generate embeddings: {e}")
    
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
            if self.backend == 'chromadb':
                shutil.copytree(self.persist_directory, backup_path)
            elif self.backend == 'faiss':
                shutil.copytree(os.path.dirname(self.faiss_store.index_path), backup_path)
            
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
            if self.backend == 'chromadb':
                shutil.copytree(backup_path, self.persist_directory)
            elif self.backend == 'faiss':
                shutil.copytree(backup_path, os.path.dirname(self.faiss_store.index_path))
            
            # Reinitialize
            self._initialize_chromadb()
            
            logger.info(f"Restored vector store from {backup_path}")
            
        except Exception as e:
            logger.error(f"Error restoring vector store: {e}")
            raise VectorDBError(f"Failed to restore vector store: {e}")
    
    def reset(self):
        """Reset the vector store by deleting all data."""
        try:
            if self.backend == 'chromadb':
                # Close current DB connection if open
                self.collection = None
                self.db = None
                
                # Remove persist directory
                if os.path.exists(self.persist_directory):
                    shutil.rmtree(self.persist_directory)
                
                # Reinitialize
                self._initialize_chromadb()
                
                logger.info("Reset ChromaDB vector store")
            elif self.backend == 'faiss':
                self.faiss_store.reset()
            
        except Exception as e:
            logger.error(f"Error resetting vector store: {e}")
            raise VectorDBError(f"Failed to reset vector store: {e}")

# Get or create a singleton vector store instance
_vector_store = None
_faiss_store = None

def get_vector_store(config=None):
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(config)
    return _vector_store


def get_faiss_db(dim: int = 1536) -> FaissVectorStore:
    """
    Get a singleton FAISS vector store instance.
    
    Args:
        dim: Dimension of the embedding vectors
        
    Returns:
        FaissVectorStore instance
    """
    global _faiss_store
    if _faiss_store is None:
        _faiss_store = FaissVectorStore({'embedding_dimension': dim})
    return _faiss_store