"""
ChromaDB implementation for vector storage in the RAG system.
"""
import os
import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from src.configuration.config import settings

# Initialize logger
logger = logging.getLogger(__name__)

# Define paths for persistence
DATA_DIR = os.path.join(settings.data_dir, "chroma_db")
os.makedirs(DATA_DIR, exist_ok=True)

class ChromaDBManager:
    """
    ChromaDB vector store for financial news, reports, and other text data.
    Supports embedding storage, retrieval, and similarity search.
    """
    def __init__(self, 
                 collection_name: str = "financial_data",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize ChromaDB with specified collection and embedding model.
        
        Args:
            collection_name: Name of the collection to use
            embedding_model: HuggingFace model name for embeddings
        """
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=DATA_DIR)
        
        # Set up embedding function from HuggingFace
        self.embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Found existing collection '{collection_name}'")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Created new collection '{collection_name}'")
    
    def add_documents(self, 
                      documents: List[str], 
                      ids: List[str], 
                      metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add documents to the collection with their corresponding IDs and metadata.
        
        Args:
            documents: List of document texts
            ids: List of unique IDs for each document
            metadatas: Optional list of metadata dictionaries
        """
        try:
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB collection")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise
    
    def search(self, 
               query: str, 
               n_results: int = 5, 
               filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Text query to search for
            n_results: Number of results to return
            filter_criteria: Optional metadata filter criteria
        
        Returns:
            Dictionary with search results including documents, ids, and distances
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_criteria
            )
            logger.info(f"Found {len(results['documents'][0])} matching documents for query")
            return results
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            raise
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id: ID of the document to retrieve
        
        Returns:
            Dictionary with document data or None if not found
        """
        try:
            result = self.collection.get(ids=[doc_id])
            if result and result["documents"]:
                return {
                    "document": result["documents"][0],
                    "metadata": result["metadatas"][0] if result["metadatas"] else None
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
            return None
    
    def delete_documents(self, doc_ids: List[str]) -> None:
        """
        Delete documents by their IDs.
        
        Args:
            doc_ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents from ChromaDB collection")
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "document_count": count,
            "collection_name": self.collection.name
        }


# Convenience function to get ChromaDB manager instance
_chroma_instance = None

def get_chroma_db(collection_name: str = "financial_data") -> ChromaDBManager:
    """
    Get or create a ChromaDB manager instance.
    
    Args:
        collection_name: Name of the collection to use
    
    Returns:
        ChromaDBManager instance
    """
    global _chroma_instance
    if _chroma_instance is None:
        _chroma_instance = ChromaDBManager(collection_name=collection_name)
    return _chroma_instance 